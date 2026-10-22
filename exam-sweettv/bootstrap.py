#!/usr/bin/env python3
"""First measured interval-join and baseline pass. No CBF/ALS training yet.

Uses the competition Data page's >=80% screen-time rule per airing and ranks
relevant items by the number of qualifying airings. This is a documented local
reconstruction, not a claim to have the organiser's exact private label code.
"""

from __future__ import annotations

import csv
import json
import platform
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from audit import parse_time
from protocol import Airing, View, join_interactions, map_at_k, rank_top_k

BASE = Path(__file__).resolve().parent
RAW = BASE / "data/raw"
OUT = BASE / "artifacts"
EPOCH = datetime(2020, 1, 1)
TRAIN_END = int((datetime(2020, 5, 18) - EPOCH).total_seconds())
VALID_END = int((datetime(2020, 7, 27) - EPOCH).total_seconds())


def seconds(value: str) -> int:
    return int((parse_time(value) - EPOCH).total_seconds())


def read_airings(name: str):
    result, metadata, candidates = [], {}, set()
    with (RAW / name).open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            item = row["tv_show_id"]
            if item == "0":
                continue
            start, duration = seconds(row["start_time"]), int(row["duration"])
            if duration <= 0:
                raise ValueError(f"Invalid labelled schedule duration in {name}")
            result.append(Airing(sys.intern(row["channel_id"]), item, start, start + duration))
            candidates.add(item)
            metadata.setdefault(item, {"title": row["tv_show_title"], "category": row["tv_show_category"]})
    return result, metadata, candidates


def dump_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    started = time.perf_counter()
    train_airings, train_metadata, _ = read_airings("export_arh_11-20-final.csv")
    valid_airings, valid_metadata, _ = read_airings("export_arh_21-30-final.csv")
    # File names are not temporal boundaries: the first grid extends into May 18.
    # Include its crossing airings in validation, with views clipped at May 18.
    all_known_airings = train_airings + valid_airings
    valid_airings = [a for a in all_known_airings if a.start < VALID_END and a.stop > TRAIN_END]
    candidates = {a.item for a in valid_airings if a.start < VALID_END and a.stop > TRAIN_END}
    print(f"Loaded labelled airings: train={len(train_airings)}, validation={len(valid_airings)}", flush=True)
    channels = {a.channel for a in train_airings + valid_airings}
    views, raw_count, out_of_period, no_label_channel, clipped_end = [], 0, 0, 0, 0
    with (RAW / "dataset11-30.csv").open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            raw_count += 1
            if row["vsetv_id"] not in channels:
                no_label_channel += 1
                continue
            start, stop = seconds(row["start_time"]), seconds(row["stop_time"])
            if start >= VALID_END:
                out_of_period += 1
                continue
            clipped_end += int(stop > VALID_END)
            if stop <= start:
                raise ValueError("Invalid viewing interval")
            views.append(View(sys.intern(row["user_id"]), sys.intern(row["vsetv_id"]), start, min(stop, VALID_END)))
    print(f"Loaded {len(views)} applicable viewing intervals, joining train", flush=True)
    train = join_interactions(views, train_airings, cutoff=TRAIN_END, min_seconds=0, min_fraction=0.8)
    print(f"Train user-item pairs: {len(train)}; joining validation", flush=True)
    valid_views = (View(v.user, v.channel, max(v.start, TRAIN_END), v.stop)
                   for v in views if v.stop > TRAIN_END)
    valid = join_interactions(valid_views, valid_airings, cutoff=VALID_END, min_seconds=0, min_fraction=0.8)
    print(f"Validation user-item pairs: {len(valid)}", flush=True)
    del views

    train_by_user, valid_by_user = defaultdict(dict), defaultdict(dict)
    popularity, frequency = Counter(), Counter()
    for r in train:
        train_by_user[r.user][r.item] = r
        popularity[r.item] += 1
        frequency[r.item] += r.airings
    for r in valid:
        valid_by_user[r.user][r.item] = r
    with (RAW / "submission.csv").open(encoding="utf-8-sig", newline="") as stream:
        target_users = [row["user_id"] for row in csv.DictReader(stream)]
    truth = {u: [r.item for r in sorted(valid_by_user[u].values(), key=lambda r: (-r.airings, -r.seconds, int(r.item)))[:5]]
             for u in target_users}
    cohort = [u for u in target_users if truth[u]]
    if not cohort:
        raise ValueError("No eligible validation users in the target template; revisit cohort explicitly")
    full_five = [u for u in cohort if len(truth[u]) == 5]
    cold = [u for u in cohort if not train_by_user[u]]
    warm = [u for u in cohort if train_by_user[u]]
    fallback = sorted(candidates, key=lambda item: (-popularity[item], int(item)))
    predictions = {name: {} for name in ["popularity_unique_users", "popularity_full_airings", "personal_repeat_frequency"]}
    for u in target_users:
        predictions["popularity_unique_users"][u] = rank_top_k(popularity, candidates, fallback)
        predictions["popularity_full_airings"][u] = rank_top_k(frequency, candidates, fallback)
        repeat = {item: r.airings for item, r in train_by_user[u].items()}
        predictions["personal_repeat_frequency"][u] = rank_top_k(repeat, candidates, fallback)
    scores = {}
    for name, pred in predictions.items():
        scores[name] = {"map_at_5_eligible": map_at_k(truth, pred, cohort),
                        "map_at_5_all_template_zero_if_no_truth": map_at_k(truth, pred, target_users),
                        "map_at_5_warm": map_at_k(truth, pred, warm) if warm else None,
                        "map_at_5_cold": map_at_k(truth, pred, cold) if cold else None,
                        "map_at_5_exactly_five_truth": map_at_k(truth, pred, full_five) if full_five else None}
    selected = []
    if warm:
        selected.append(max(warm, key=lambda u: (len(train_by_user[u]), u)))
    if cold:
        selected.append(sorted(cold)[0])
    elif len(warm) > 1:
        selected.append(min((u for u in warm if u not in selected), key=lambda u: (len(train_by_user[u]), u)))
    metadata = {**valid_metadata, **train_metadata}

    def describe(items):
        return [{"id": item, **metadata.get(item, {"title": "<unknown>", "category": "<unknown>"})} for item in items]

    examples = [{"user_id": u, "selection": "heavy" if i == 0 and train_by_user[u] else "cold" if not train_by_user[u] else "light",
                 "train_unique_items": len(train_by_user[u]),
                 "train_top5_by_full_airings": describe([r.item for r in sorted(train_by_user[u].values(), key=lambda r: (-r.airings, -r.seconds, int(r.item)))[:5]]),
                 "validation_truth": describe(truth[u]),
                 "recommendations": {name: describe(pred[u]) for name, pred in predictions.items()}}
                for i, u in enumerate(selected)]
    output = {"status": "MEASURED_BASELINES_ONLY", "protocol": {
        "train_start_source": "2020-03-09", "train_cutoff_exclusive": "2020-05-18T00:00:00",
        "validation_end_exclusive": "2020-07-27T00:00:00", "source_clock": "naive common wall clock; no asserted UTC conversion",
        "join": "same channel; union of overlapping views per user/channel; half-open intervals; clipped BEFORE threshold",
        "watch_threshold": "union overlap seconds / original airing duration >= 0.8; no absolute seconds minimum",
        "truth": "top 5 distinct IDs by qualifying airing count; tie by seconds desc, then numeric ID asc",
        "repeats_allowed": True, "primary_cohort": "template users with >=1 relevant validation item, including cold train users",
        "fallback": "train unique-user popularity restricted to the common validation schedule; ID ascending for zero-history ties",
        "caveat": "Local reconstruction of published rules; exact organiser overlap, deduplication and tie code is not available."},
        "counts": {"raw_view_rows": raw_count, "views_discarded_channel_without_labelled_training_airings": no_label_channel,
                   "views_start_at_or_after_validation_end": out_of_period, "views_clipped_at_validation_end": clipped_end,
                   "train_user_item_pairs": len(train), "validation_user_item_pairs": len(valid),
                   "train_users_with_high_screen_history": sum(bool(history) for history in train_by_user.values()),
                   "template_users": len(target_users), "eligible_validation_users": len(cohort),
                   "users_without_validation_truth": len(target_users)-len(cohort),
                   "eligible_cold_users": len(cold), "eligible_warm_users": len(warm),
                   "eligible_users_with_exactly_five_truth": len(full_five), "common_validation_candidates": len(candidates)},
        "metrics": scores, "examples": examples,
        "seconds": round(time.perf_counter()-started, 3),
        "environment": {"python": platform.python_version(), "platform": platform.platform()},
        "not_yet_done": ["CBF", "implicit ALS", "held-out stability checks", "Kaggle submission or score", "student personal verification"]}
    dump_json(OUT / "bootstrap-baselines.json", output)
    dump_json(OUT / "validation-cohort.json", {"users": cohort, "truth": truth})
    processed = BASE / "data/processed"
    processed.mkdir(parents=True, exist_ok=True)
    for name, rows in [("train-high-screen", train), ("validation-high-screen", valid)]:
        with (processed / f"{name}.jsonl").open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(asdict(row), ensure_ascii=False) + "\n")
    print(json.dumps({"counts": output["counts"], "metrics": scores, "seconds": output["seconds"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
