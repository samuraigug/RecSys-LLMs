"""Reference contracts for SWEET.TV, using only the Python standard library.

Timestamps below are seconds in one common, explicitly chosen clock. CSV time
parsing, time-zone decisions, model fitting and production-scale ETL are separate
steps. These functions currently have synthetic verification only.
"""

from __future__ import annotations

import csv
import hashlib
import math
from bisect import bisect_left, bisect_right
from collections import defaultdict
from dataclasses import dataclass
from itertools import zip_longest
from pathlib import Path
from typing import Iterable, Mapping, Sequence


def programme_id(value: str) -> str:
    """Keep IDs lossless: never convert a 64-bit identifier through a float."""
    if not isinstance(value, str) or not value.isascii() or not value.isdigit():
        raise ValueError(f"Programme ID must be a decimal string: {value!r}")
    if int(value) <= 0:
        raise ValueError("Programme ID 0 is unlabelled airtime, not an item")
    return value


def overlap_seconds(a: float, b: float, c: float, d: float) -> float:
    """Length of [a,b) intersect [c,d); touching endpoints have no overlap."""
    if not all(math.isfinite(t) for t in (a, b, c, d)):
        raise ValueError("Interval boundaries must be finite")
    if b < a or d < c:
        raise ValueError("Interval ends before it starts")
    return max(0.0, min(b, d) - max(a, c))


def merge_intervals(intervals: Iterable[tuple[float, float]]) -> list[tuple[float, float]]:
    """Union of a user's intervals on ONE channel; no double-counted seconds."""
    result: list[tuple[float, float]] = []
    for start, stop in sorted(intervals):
        overlap_seconds(start, stop, start, stop)
        if stop == start:
            continue
        if result and start <= result[-1][1]:
            result[-1] = (result[-1][0], max(stop, result[-1][1]))
        else:
            result.append((start, stop))
    return result


@dataclass(frozen=True)
class View:
    user: str
    channel: str
    start: float
    stop: float


@dataclass(frozen=True)
class Airing:
    channel: str
    item: str
    start: float
    stop: float


@dataclass(frozen=True)
class Interaction:
    user: str
    item: str
    seconds: float
    airings: int
    last_stop: float


def join_interactions(
    views: Iterable[View],
    airings: Iterable[Airing],
    *,
    cutoff: float,
    min_seconds: float = 60.0,
    min_fraction: float = 0.0,
) -> list[Interaction]:
    """Reference interval join, clipped at cutoff BEFORE thresholding.

Replays are retained as one user-item pair with multiple airings. Exact duplicate
schedule rows are removed. Conflicting different programmes in the same channel
slot need an explicit ETL audit; this reference does not silently choose a label.
The fraction denominator is the original programme duration, not the view length.
"""
    if not math.isfinite(cutoff):
        raise ValueError("cutoff must be finite")
    if not math.isfinite(min_seconds) or min_seconds < 0 or not 0 <= min_fraction <= 1:
        raise ValueError("Invalid watch threshold")

    by_channel: dict[str, list[Airing]] = defaultdict(list)
    for a in set(airings):
        # Unlabelled rows are not programmes, including the raw zero-duration row.
        if a.item == "0":
            continue
        overlap_seconds(a.start, a.stop, a.start, a.stop)
        if a.stop <= a.start:
            raise ValueError("Airing duration must be positive")
        programme_id(a.item)
        if a.start < cutoff:
            by_channel[a.channel].append(a)

    indices = {}
    for channel, group in by_channel.items():
        group.sort(key=lambda a: (a.start, a.stop, int(a.item)))
        starts, prefix_stops = [], []
        latest = -math.inf
        for a in group:
            starts.append(a.start)
            latest = max(latest, min(a.stop, cutoff))
            prefix_stops.append(latest)
        indices[channel] = (group, starts, prefix_stops)

    user_channels: dict[tuple[str, str], list[tuple[float, float]]] = defaultdict(list)
    for v in views:
        overlap_seconds(v.start, v.stop, v.start, v.stop)
        if v.start < cutoff and v.stop > v.start:
            user_channels[(v.user, v.channel)].append((v.start, min(v.stop, cutoff)))

    combined: dict[tuple[str, str], list[float]] = {}
    for (user, channel), intervals in user_channels.items():
        if channel not in indices:
            continue
        group, starts, prefix_stops = indices[channel]
        per_airing: dict[int, list[float]] = {}
        for start, stop in merge_intervals(intervals):
            lo, hi = bisect_right(prefix_stops, start), bisect_left(starts, stop)
            for j in range(lo, hi):
                a = group[j]
                seconds = overlap_seconds(start, stop, a.start, min(a.stop, cutoff))
                if seconds > 0:
                    acc = per_airing.setdefault(j, [0.0, -math.inf])
                    acc[0] += seconds
                    acc[1] = max(acc[1], min(stop, a.stop, cutoff))
        for j, (seconds, last_stop) in per_airing.items():
            a = group[j]
            if seconds < min_seconds or seconds / (a.stop - a.start) < min_fraction:
                continue
            acc = combined.setdefault((user, a.item), [0.0, 0.0, -math.inf])
            acc[0] += seconds
            acc[1] += 1
            acc[2] = max(acc[2], last_stop)
    return [Interaction(user, item, v[0], int(v[1]), v[2])
            for (user, item), v in sorted(combined.items())]


def ap_at_k(actual: Iterable[str], predicted: Sequence[str], k: int = 5) -> float:
    """Binary AP@K / min(K, number of relevant IDs); duplicate hits count once.

Empty truth is 0 here. Selection of the primary eligible cohort is deliberately
outside this function and must be frozen and recorded before model comparison.
"""
    if k <= 0:
        raise ValueError("k must be positive")
    truth = set(actual)
    if not truth:
        return 0.0
    seen, hits, total = set(), 0, 0.0
    for rank, item in enumerate(predicted[:k], 1):
        if item in truth and item not in seen:
            hits += 1
            total += hits / rank
        seen.add(item)
    return total / min(k, len(truth))


def map_at_k(
    actual: Mapping[str, Iterable[str]],
    predicted: Mapping[str, Sequence[str]],
    users: Sequence[str],
    k: int = 5,
) -> float:
    if not users or len(set(users)) != len(users):
        raise ValueError("Evaluation requires a nonempty, unique, fixed user list")
    if any(user not in actual or user not in predicted for user in users):
        raise ValueError("Truth and predictions must cover every evaluated user")
    return sum(ap_at_k(actual[u], predicted[u], k) for u in users) / len(users)


def rank_top_k(
    scores: Mapping[str, float],
    candidates: Iterable[str],
    fallback: Sequence[str],
    k: int = 5,
) -> list[str]:
    """Keep seen items eligible; fill missing scores using one common fallback."""
    pool = {programme_id(item) for item in candidates}
    if k <= 0 or len(pool) < k:
        raise ValueError("Candidate pool cannot fill the required recommendation list")
    scored = []
    for item, score in scores.items():
        if item in pool:
            if not math.isfinite(score):
                raise ValueError("Nonfinite model score")
            scored.append((item, score))
    result = [item for item, _ in sorted(scored, key=lambda pair: (-pair[1], int(pair[0]), pair[0]))[:k]]
    selected = set(result)
    for item in list(fallback) + sorted(pool, key=lambda value: (int(value), value)):
        if item in pool and item not in selected and len(result) < k:
            result.append(item)
            selected.add(item)
    return result


def validate_submission(
    template: Path,
    submission: Path,
    candidates: Iterable[str],
    *,
    expected_rows: int = 1260,
) -> dict:
    """Strictly preserve the template's header and user order, including 64-bit IDs."""
    pool = {programme_id(item) for item in candidates}
    users, count = set(), 0
    with template.open(newline="", encoding="utf-8-sig") as ta, submission.open(newline="", encoding="utf-8-sig") as su:
        a, b = csv.reader(ta), csv.reader(su)
        if next(a, None) != ["user_id", "tv_show_id"] or next(b, None) != ["user_id", "tv_show_id"]:
            raise ValueError("Header must be exactly user_id,tv_show_id in both files")
        for expected, row in zip_longest(a, b):
            count += 1
            if expected is None or row is None or len(expected) != 2 or len(row) != 2:
                raise ValueError(f"Row {count}: missing/extra row or wrong column count")
            if row[0] != expected[0] or row[0] in users or not row[0]:
                raise ValueError(f"Row {count}: user ID mismatch, duplicate or empty ID")
            users.add(row[0])
            ids = row[1].split()
            if len(ids) != 5 or len(set(ids)) != 5:
                raise ValueError(f"Row {count}: require five distinct programme IDs")
            if any(programme_id(item) not in pool for item in ids):
                raise ValueError(f"Row {count}: programme absent from candidate schedule")
    if count != expected_rows:
        raise ValueError(f"Expected {expected_rows} users, got {count}")
    with submission.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"status": "PASS", "rows": count, "k": 5, "sha256": digest}
