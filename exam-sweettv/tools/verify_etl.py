#!/usr/bin/env python3
"""Real-data ETL crosscheck, future-removal and independent seconds oracle."""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from etl import Clock, EXAMPLES, aggregate, clip_union, load_schedule, load_views, match_events, metadata_before, read_events
from repro import ROOT, dump, load, sha256


def event_signature(events):
    digest = hashlib.sha256()
    for row in sorted(events):
        digest.update(json.dumps(row, ensure_ascii=False, separators=(",", ":")).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def main(output, full_leakage):
    started = time.perf_counter()
    base = ROOT / "data/processed/elapsed-v2/main"
    proto = load(base/"protocol.json")
    clock, raw = Clock(), ROOT/"data/raw"
    eu, ei, ec, arr = read_events(base/"train-events.npz", full_only=False)
    full = arr[arr[:, 5] >= .8*arr[:, 7]]
    got = defaultdict(lambda: [0, 0])
    for u, i, _, _, _, seconds, _, _ in full:
        r = got[(eu[u], ei[i])]; r[0] += 1; r[1] += int(seconds)
    reference = {(r["user"], r["item"]): [r["airings"], r["seconds"]]
                 for r in map(json.loads, (ROOT/"data/processed/train-high-screen.jsonl").open())}
    changes = [{"user_id": u, "item_id": i, "naive": reference.get((u, i)), "elapsed": got.get((u, i))}
               for u, i in sorted(set(got) | set(reference)) if reference.get((u, i)) != got.get((u, i))]
    old = load(ROOT/"artifacts/runs/reference-naive/bootstrap-baselines.json")
    original = load(ROOT/"evidence/preparation/bootstrap-baselines.json")
    assert all(old[k] == original[k] for k in ("counts", "metrics", "examples"))
    assert load(base/"truth.json") == load(ROOT/"artifacts/runs/reference-naive/validation-cohort.json")["truth"]
    # A real, partial high-screen airing for the preselected heavy user.
    uidx = eu.index(EXAMPLES[0])
    choices = full[(full[:, 0] == uidx) & (full[:, 7] <= 7200) & (full[:, 7] >= 900) & (full[:, 5] < full[:, 7])]
    event = choices[0]
    _, i, c, a, b, expected, last, duration = map(int, event)
    raw_intervals = []
    for r in csv.DictReader((raw/"dataset11-30.csv").open(encoding="utf-8-sig", newline="")):
        if r["user_id"] == EXAMPLES[0] and r["vsetv_id"] == ec[c]:
            left, right = clock.seconds(r["start_time"]), clock.seconds(r["stop_time"])
            if left < b and right > a:
                raw_intervals.append((left, right))
    covered_seconds = set()
    for left, right in raw_intervals:
        covered_seconds.update(range(max(a, left), min(b, right, proto["cutoff_elapsed_seconds"])))
    assert len(covered_seconds) == expected
    overlap = {"user_id": EXAMPLES[0], "item_id": ei[i], "channel": ec[c],
               "start": clock.local(a).isoformat(), "stop": clock.local(b).isoformat(),
               "airing_elapsed_seconds": duration, "raw_views": [{"start": clock.local(x).isoformat(), "stop": clock.local(y).isoformat()} for x, y in raw_intervals],
               "independent_second_set_size": len(covered_seconds), "etl_seconds": expected,
               "fraction": expected/duration, "qualifies": expected/duration >= .8}
    result = {"status": "PASS", "reference_reproduced_exactly": ["counts", "metrics", "examples"],
              "elapsed_vs_naive_changed_train_pairs": len(changes), "changed_examples": changes[:12],
              "validation_truth_unchanged": True, "real_overlap_seconds_oracle": overlap}
    if full_leakage:
        schedule, metadata, _ = load_schedule(raw, clock)
        groups, _ = load_views(raw, clock)
        start, cutoff = clock.seconds(proto["train_start"]), proto["cutoff_elapsed_seconds"]
        training_only = {key: clipped for key, spans in groups.items() if (clipped := clip_union(spans, start, cutoff))}
        trimmed, _ = match_events(training_only, [a for a in schedule if a.start < cutoff], start, cutoff)
        saved = [(eu[u], ei[i], ec[c], int(a), int(b), int(s), int(last), int(dur)) for u, i, c, a, b, s, last, dur in arr]
        saved_hash, trimmed_hash = event_signature(saved), event_signature(trimmed)
        assert saved_hash == trimmed_hash
        # Append impossible-to-fit future-only users/items/metadata; historical docs unchanged.
        modified = ({**r, "tv_show_title": "FUTURE_SENTINEL_NOT_ALLOWED"} if r["start"] >= cutoff else r for r in metadata)
        assert metadata_before(modified, cutoff) == load(base/"metadata-train.json")
        result["full_leakage_check"] = {"status": "PASS", "views_after_cutoff_removed_before_join": True,
                                         "future_airings_removed_from_training_join": True,
                                         "train_event_rows": len(trimmed), "canonical_events_sha256": saved_hash,
                                         "future_content_perturbation": "historical documents exactly unchanged",
                                         "consequence": "identical event data imply identical full/recency CSR; seeded model reproducibility checked separately"}
    result["seconds"] = time.perf_counter()-started
    dump(output, result)
    print(json.dumps({k: v for k, v in result.items() if k != "changed_examples"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--full-leakage", action="store_true")
    a = p.parse_args()
    main(a.output, a.full_leakage)
