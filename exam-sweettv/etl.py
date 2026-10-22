"""Cutoff-safe interval ETL, elapsed-time clock, frozen evaluation contracts.

No future viewing is exposed to model fitting. Metadata documents contain only
pre-cutoff schedule rows; future rows supply availability (ID/channel/time) only.
The chosen IANA zone is an evidenced clock hypothesis, not a source declaration.
"""
from __future__ import annotations

import argparse
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
import csv
from datetime import datetime
import math
from pathlib import Path
import sys
import time
from zoneinfo import ZoneInfo

import numpy as np
from scipy import sparse

from audit import parse_time
from protocol import Airing, merge_intervals
from repro import ROOT, dump, load, now, sha256

FILES = ["export_arh_11-20-final.csv", "export_arh_21-30-final.csv", "export_arh_31-42-final.csv"]
FIELDS = ["tv_show_title", "tv_show_category", "tv_show_genre_1", "tv_show_genre_2",
          "tv_show_genre_3", "year_of_production", "director", "actors"]
SPLITS = {"main": ("2020-03-09", "2020-05-18", "2020-07-27"),
          "early": ("2020-03-09", "2020-04-20", "2020-06-29"),
          "final": ("2020-03-09", "2020-07-27", "2020-10-19")}
EXAMPLES = ["13571199046623737884", "10120801182216576239"]


class Clock:
    def __init__(self, policy="Europe/Kyiv"):
        self.policy = policy
        self.zone = ZoneInfo(policy) if policy != "naive" else None

    def seconds(self, text):
        dt = parse_time(text)
        if self.zone is None:
            return int((dt - datetime(1970, 1, 1)).total_seconds())
        aware = dt.replace(tzinfo=self.zone)
        result = int(aware.timestamp())
        if datetime.fromtimestamp(result, self.zone).replace(tzinfo=None) != dt:
            raise ValueError(f"Nonexistent source local time: {text} ({self.policy})")
        return result

    def local(self, seconds):
        return datetime.fromtimestamp(seconds, self.zone) if self.zone else datetime.fromtimestamp(seconds, ZoneInfo("UTC"))


def clip_union(intervals, start, end):
    return merge_intervals((max(a, start), min(b, end)) for a, b in intervals if a < end and b > start)


def load_schedule(raw, clock):
    labelled, metadata, zeros = [], [], 0
    transition = defaultdict(list)
    samples = {"actors": Counter(), "director": Counter(), "year_of_production": Counter()}
    for name in FILES:
        with (raw / name).open(encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                if row["start_time"].startswith("29.03.2020"):
                    transition[row["channel_id"]].append((row["start_time"], int(row["duration"]), row["tv_show_id"]))
                if row["tv_show_id"] == "0":
                    zeros += 1
                    continue
                duration = int(row["duration"])
                if duration <= 0:
                    raise ValueError("Labelled airing must have a positive duration")
                start = clock.seconds(row["start_time"])
                item, channel = sys.intern(row["tv_show_id"]), sys.intern(row["channel_id"])
                labelled.append(Airing(channel, item, start, start + duration))
                metadata.append({"item": item, "start": start, **{f: row[f] for f in FIELDS}})
                if name == FILES[0]:
                    for field in samples:
                        if row[field].strip():
                            samples[field][row[field]] += 1
        print(f"schedule {name}: cumulative labelled={len(labelled)}", flush=True)
    unique = sorted(set(labelled), key=lambda a: (a.channel, a.start, a.stop, int(a.item)))
    overlaps, latest = [], {}
    for a in unique:
        if a.start < latest.get(a.channel, -math.inf):
            overlaps.append({"channel": a.channel, "item": a.item,
                             "start": clock.local(a.start).isoformat(),
                             "previous_stop": clock.local(latest[a.channel]).isoformat()})
        latest[a.channel] = max(a.stop, latest.get(a.channel, a.stop))
    gaps = []
    naive = Clock("naive")
    for channel, rows in sorted(transition.items()):
        rows.sort(key=lambda row: naive.seconds(row[0]))
        for row, nxt in zip(rows, rows[1:]):
            wall_gap = naive.seconds(nxt[0]) - naive.seconds(row[0]) - row[1]
            elapsed_gap = clock.seconds(nxt[0]) - clock.seconds(row[0]) - row[1]
            if wall_gap == 3600 and elapsed_gap == 0:
                gaps.append({"channel": channel, "item": row[2], "start": row[0],
                             "duration": row[1], "next_start": nxt[0],
                             "naive_gap_seconds": wall_gap, "elapsed_gap_seconds": elapsed_gap})
    return unique, metadata, {"dropped_zero_id_rows": zeros, "duplicate_labelled_airings": len(labelled)-len(unique),
                              "labelled_overlaps": overlaps, "dst_gaps_reconciled": len(gaps),
                              "dst_examples": gaps[:8],
                              "train_metadata_separator_examples": {f: c.most_common(5) for f, c in samples.items()}}


def load_views(raw, clock):
    groups = defaultdict(list)
    count = mismatches = wall_mismatches = 0
    naive = Clock("naive")
    for row in csv.DictReader((raw / "dataset11-30.csv").open(encoding="utf-8-sig", newline="")):
        count += 1
        start, stop = clock.seconds(row["start_time"]), clock.seconds(row["stop_time"])
        duration = int(row["duraton"])
        mismatches += stop-start != duration
        # Only March intervals can differ between naive and elapsed clocks in these data.
        if row["start_time"][:7] == "2020-03":
            wall_mismatches += naive.seconds(row["stop_time"]) - naive.seconds(row["start_time"]) != duration
        if stop <= start:
            raise ValueError("Invalid viewing interval")
        groups[(sys.intern(row["user_id"]), sys.intern(row["vsetv_id"]))].append((start, stop))
    merged = {key: merge_intervals(intervals) for key, intervals in groups.items()}
    del groups
    return merged, {"raw_rows": count, "elapsed_duration_mismatches": mismatches,
                    "naive_duration_mismatches_in_march": wall_mismatches,
                    "union_intervals": sum(len(v) for v in merged.values()),
                    "union_user_channel_seconds": sum(b-a for rows in merged.values() for a, b in rows)}


def match_events(groups, airings, start, end):
    """Independent optimized implementation; checked against reference and second oracle."""
    by_channel = defaultdict(list)
    for a in airings:
        if a.item != "0" and a.start < end and a.stop > start:
            by_channel[a.channel].append(a)
    indices = {}
    for channel, rows in by_channel.items():
        rows.sort(key=lambda a: (a.start, a.stop, int(a.item)))
        indices[channel] = (rows, [a.start for a in rows], np.maximum.accumulate([a.stop for a in rows]).tolist())
    events, total_seconds, matched_seconds, union_count = [], 0, 0, 0
    for (user, channel), intervals in groups.items():
        clipped = clip_union(intervals, start, end)
        total_seconds += sum(b-a for a, b in clipped)
        union_count += len(clipped)
        if channel not in indices:
            continue
        rows, starts, stops = indices[channel]
        coverage = {}
        for a, b in clipped:
            for j in range(bisect_right(stops, a), bisect_left(starts, b)):
                airing = rows[j]
                left, right = max(a, airing.start), min(b, airing.stop)
                if right > left:
                    seconds, last = coverage.get(j, (0, 0))
                    coverage[j] = (seconds+right-left, max(last, right))
        for j, (seconds, last) in coverage.items():
            a = rows[j]
            if seconds > a.stop-a.start:
                raise AssertionError("Overlap double-counted")
            matched_seconds += seconds
            if seconds >= 60 or seconds/(a.stop-a.start) >= 0.8:
                events.append((user, a.item, channel, a.start, a.stop, seconds, last, a.stop-a.start))
    return events, {"union_intervals": union_count, "union_user_channel_seconds": total_seconds,
                    "labelled_matched_user_channel_seconds": matched_seconds,
                    "labelled_share_of_user_channel_seconds": matched_seconds/total_seconds if total_seconds else 0,
                    "stored_full_or_60_second_events": len(events)}


def save_events(path, events):
    users = sorted({e[0] for e in events})
    items = sorted({e[1] for e in events}, key=int)
    channels = sorted({e[2] for e in events}, key=int)
    um, im, cm = ({v: i for i, v in enumerate(values)} for values in (users, items, channels))
    values = np.array([(um[u], im[i], cm[c], a, b, s, last, dur)
                       for u, i, c, a, b, s, last, dur in events], dtype=np.int64).reshape(-1, 8)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, users=np.array(users), items=np.array(items), channels=np.array(channels), events=values)


def read_events(path, full_only=True):
    with np.load(path, allow_pickle=False) as z:
        users, items, channels, events = (z[k] for k in ("users", "items", "channels", "events"))
    if full_only:
        events = events[events[:, 5] >= .8*events[:, 7]]
    return users.tolist(), items.tolist(), channels.tolist(), events


def aggregate(events):
    result = defaultdict(lambda: [0, 0, 0])
    for user, item, _, _, _, seconds, last, duration in events:
        if seconds/duration < .8:
            continue
        r = result[(user, item)]
        r[0] += 1
        r[1] += seconds
        r[2] = max(r[2], last)
    return result


def metadata_before(rows, cutoff):
    docs = {}
    for row in rows:
        if row["start"] >= cutoff:
            continue
        item = row["item"]
        doc = docs.setdefault(item, {f: set() for f in FIELDS})
        for field in FIELDS:
            if row[field].strip():
                doc[field].add(row[field].strip())
    return {item: {f: sorted(v) for f, v in doc.items()} for item, doc in sorted(docs.items(), key=lambda p: int(p[0]))}


def build_habits(path, groups, airings, target_users, candidates, start, cutoff, end, clock):
    channels = sorted({c for _, c in groups} | {a.channel for a in airings}, key=int)
    cm, um, im = ({x: i for i, x in enumerate(values)} for values in (channels, target_users, candidates))
    channel_seconds = np.zeros((len(target_users), len(channels)), dtype=np.float64)
    slots = np.zeros((len(target_users), 8), dtype=np.float64)

    def segments(a, b):
        while a < b:
            local = clock.local(a)
            slot = local.hour//3
            next_boundary = a + (3-local.hour%3)*3600-local.minute*60-local.second
            stop = min(b, next_boundary)
            yield slot, stop-a
            a = stop

    for (user, channel), intervals in groups.items():
        if user not in um:
            continue
        u, c = um[user], cm[channel]
        for a, b in clip_union(intervals, start, cutoff):
            channel_seconds[u, c] += b-a
            for slot, seconds in segments(a, b):
                slots[u, slot] += seconds
    rr, cc, vv = [], [], []
    for a in airings:
        if a.item in im and a.start < end and a.stop > cutoff:
            for slot, seconds in segments(max(a.start, cutoff), min(a.stop, end)):
                rr.append(im[a.item]); cc.append(cm[a.channel]*8+slot); vv.append(seconds/3600)
    opportunities = sparse.csr_matrix((vv, (rr, cc)), shape=(len(candidates), len(channels)*8))
    sparse.save_npz(path / "opportunity-slots.npz", opportunities)
    np.savez_compressed(path / "habits.npz", channel_seconds=channel_seconds, slot_seconds=slots)


def prepare(raw, output, splits=("main", "early", "final"), clock_policy="Europe/Kyiv"):
    if (output / "preparation.json").exists():
        raise FileExistsError("Prepared dataset is immutable; use a new output directory")
    started = time.perf_counter()
    clock = Clock(clock_policy)
    schedule, metadata, schedule_audit = load_schedule(raw, clock)
    groups, views_audit = load_views(raw, clock)
    users = [r["user_id"] for r in csv.DictReader((raw/"submission.csv").open(encoding="utf-8-sig", newline=""))]
    summaries = {}
    for name in splits:
        print(f"preparing {name}", flush=True)
        path = output / name
        if path.exists() and any(path.iterdir()):
            raise FileExistsError(path)
        start_text, cutoff_text, end_text = SPLITS[name]
        start, cutoff, end = map(clock.seconds, (start_text, cutoff_text, end_text))
        train_events, train_audit = match_events(groups, schedule, start, cutoff)
        training = aggregate(train_events)
        save_events(path / "train-events.npz", train_events)
        del train_events
        popularity = Counter()
        histories = defaultdict(dict)
        for (u, item), (count, seconds, last) in training.items():
            popularity[item] += 1
            histories[u][item] = {"airings": count, "seconds": seconds, "last_stop": last}
        candidates = sorted({a.item for a in schedule if a.start < end and a.stop > cutoff}, key=int)
        fallback = sorted(candidates, key=lambda i: (-popularity[i], int(i)))
        truth, val_audit = {u: [] for u in users}, None
        if name != "final":
            valid_events, val_audit = match_events(groups, schedule, cutoff, end)
            valid = aggregate(valid_events)
            save_events(path / "validation-events.npz", valid_events)
            del valid_events
            per_user = defaultdict(list)
            for (u, item), (count, seconds, last) in valid.items():
                per_user[u].append((item, count, seconds))
            for u in users:
                truth[u] = [v[0] for v in sorted(per_user[u], key=lambda r: (-r[1], -r[2], int(r[0])))[:5]]
        main = [u for u in users if truth[u]]
        cohorts = {"all_template": users, "main": main,
                   "warm": [u for u in main if histories[u]], "cold": [u for u in main if not histories[u]],
                   "exactly_five_truth": [u for u in main if len(truth[u]) == 5]}
        dump(path / "cohorts.json", cohorts)
        dump(path / "truth.json", truth)
        dump(path / "histories.json", histories)
        dump(path / "metadata-train.json", metadata_before(metadata, cutoff))
        # Display metadata is never fed to features. It can describe future recommendations.
        display = {}
        for row in sorted(metadata, key=lambda r: (r["start"], r["tv_show_title"])):
            display.setdefault(row["item"], {"title": row["tv_show_title"], "category": row["tv_show_category"]})
        dump(path / "metadata-display-only.json", display)
        dump(path / "fallback.json", fallback)
        (path / "candidates.txt").write_text("\n".join(candidates)+"\n", encoding="utf-8")
        train_opps, future_opps = Counter(), Counter()
        for a in schedule:
            if a.start < cutoff and a.stop > start:
                train_opps[a.item] += 1
            if a.start < end and a.stop > cutoff:
                future_opps[a.item] += 1
        dump(path / "opportunities.json", {"train": train_opps, "future": future_opps})
        build_habits(path, groups, schedule, users, candidates, start, cutoff, end, clock)
        protocol = {"schema_version": 2, "status": "FROZEN", "split": name,
                    "clock": clock_policy, "clock_is_inferred_not_source_declared": True,
                    "clock_choice_reason": "elapsed durations agree for all views and DST schedule gaps close; chosen before model scores",
                    "train_start": start_text, "train_cutoff_exclusive": cutoff_text,
                    "forecast_end_exclusive": end_text, "cutoff_elapsed_seconds": cutoff,
                    "join": "same channel; half-open intervals; union user/channel; clip views to window before overlap",
                    "labels": "union seconds / original elapsed airing duration >= 0.8; top5 by count, seconds desc, numeric ID asc",
                    "labels_are_reconstruction": True, "zero_id_policy": "drop before positive duration validation",
                    "candidates": "all labelled airings intersecting window from the combined grid, not nominal filenames",
                    "metadata": "aggregate unique pre-cutoff values; no future content features",
                    "repeats_allowed": True, "k": 5, "fallback": "train distinct-user popularity; numeric ID asc",
                    "cold_gate": "all methods return exactly the shared fallback if no >=80% training history",
                    "early_window_caveat": "early and main temporal windows overlap and are not independent",
                    "example_users": EXAMPLES,
                    "input_hashes": {f.name: sha256(f) for f in sorted(raw.glob("*.csv"))},
                    "frozen_hashes": {f.name: sha256(f) for f in sorted(path.iterdir()) if f.is_file()}}
        dump(path / "protocol.json", protocol)
        summaries[name] = {"train_pairs": len(training), "train_positive_users": len({u for u, _ in training}),
                           "candidate_items": len(candidates), "cohorts": {k: len(v) for k, v in cohorts.items()},
                           "train_audit": train_audit, "validation_audit": val_audit,
                           "protocol_sha256": sha256(path / "protocol.json")}
        print(summaries[name], flush=True)
    dump(output / "preparation.json", {"started_at": now(), "clock": clock_policy, "schedule_audit": schedule_audit,
                                        "view_audit": views_audit, "splits": summaries,
                                        "seconds": time.perf_counter()-started})
    return summaries


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/raw")
    parser.add_argument("--output", type=Path, default=ROOT / "data/processed/elapsed-v2")
    parser.add_argument("--splits", nargs="+", choices=list(SPLITS), default=list(SPLITS))
    parser.add_argument("--clock", default="Europe/Kyiv", choices=["Europe/Kyiv", "naive"])
    a = parser.parse_args()
    prepare(a.data_dir, a.output, a.splits, a.clock)
