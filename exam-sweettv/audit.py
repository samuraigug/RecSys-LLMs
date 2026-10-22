"""Streaming CSV inventory, NOT a completed dataset or interaction audit."""

import csv
import hashlib
import time
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

VIEW_COLUMNS = {"user_id", "vsetv_id", "start_time", "stop_time", "duraton"}
SCHEDULE_COLUMNS = {"channel_id", "start_time", "duration", "tv_show_id"}


def parse_time(value: str) -> datetime:
    # The downloaded files use two DIFFERENT date orders. Never join raw strings.
    if len(value) >= 10 and value[2:3] == "." and value[5:6] == ".":
        value = value[6:10] + "-" + value[3:5] + "-" + value[:2] + value[10:]
    return datetime.fromisoformat(value)


def histogram_quantile(histogram: Counter, fraction: float) -> float | None:
    total = sum(histogram.values())
    if not total:
        return None
    target = max(1, int(__import__("math").ceil(total * fraction)))
    seen = 0
    for value, count in sorted(histogram.items()):
        seen += count
        if seen >= target:
            return value
    raise AssertionError("Unreachable quantile")


def audit_file(path: Path) -> dict:
    start = time.perf_counter()
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    csv.field_size_limit(16 * 1024 * 1024)
    with path.open(encoding="utf-8-sig", newline="") as stream:
        sample = stream.read(32768)
        stream.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
        except csv.Error as error:
            raise ValueError(f"Cannot infer CSV delimiter for {path.name}; inspect it explicitly") from error
        reader = csv.DictReader(stream, dialect=dialect)
        columns = reader.fieldnames or []
        if len(set(columns)) != len(columns):
            raise ValueError(f"Duplicate header in {path.name}")
        role = ("views" if VIEW_COLUMNS <= set(columns) else
                "schedule" if SCHEDULE_COLUMNS <= set(columns) else
                "template" if columns == ["user_id", "tv_show_id"] else "unknown")
        rows, malformed, blank_ids, zero_ids = 0, 0, 0, 0
        items, users, channels, categories = set(), set(), set(), Counter()
        temporal_examples = []
        durations, weeks = Counter(), Counter()
        min_start, max_start, max_stop = None, None, None
        invalid_temporal, duration_mismatch = 0, 0
        for row in reader:
            rows += 1
            malformed += int(None in row or any(value is None for value in row.values()))
            if role == "schedule":
                item = row.get("tv_show_id", "")
                zero_ids += int(item == "0")
                blank_ids += int(not item)
                if item and item != "0":
                    items.add(item)
                channels.add(row.get("channel_id"))
                categories[row.get("tv_show_category", "<missing>")] += 1
            if role in {"views", "template"}:
                users.add(row.get("user_id"))
            if role == "views":
                channels.add(row.get("vsetv_id"))
            if role in {"views", "schedule"}:
                try:
                    begin = parse_time(row["start_time"])
                    duration = int(row["duraton"] if role == "views" else row["duration"])
                    end = parse_time(row["stop_time"]) if role == "views" else begin + timedelta(seconds=duration)
                    if duration <= 0 or end <= begin:
                        invalid_temporal += 1
                    duration_mismatch += int(role == "views" and (end - begin).total_seconds() != duration)
                    durations[duration] += 1
                    week = begin.isocalendar()
                    weeks[f"{week.year}-W{week.week:02d}"] += 1
                    min_start = min(min_start, begin) if min_start is not None else begin
                    max_start = max(max_start, begin) if max_start is not None else begin
                    max_stop = max(max_stop, end) if max_stop is not None else end
                except (ValueError, TypeError, KeyError):
                    invalid_temporal += 1
            if len(temporal_examples) < 3 and role in {"views", "schedule"}:
                # No user histories or identifiers are included in these examples.
                temporal_examples.append({key: row.get(key) for key in ("start_time", "stop_time", "duration", "duraton") if key in row})
    result = {"file": path.name, "bytes": path.stat().st_size, "sha256": digest,
              "role": role, "delimiter": dialect.delimiter, "columns": columns,
              "rows": rows, "malformed_rows": malformed, "unique_items_nonzero": len(items),
              "unique_users": len(users), "unique_channels": len(channels),
              "categories": dict(categories), "unlabelled_id_zero": zero_ids,
              "blank_programme_ids": blank_ids, "raw_temporal_examples": temporal_examples,
              "seconds": round(time.perf_counter() - start, 4)}
    if role in {"views", "schedule"}:
        result.update({"min_start": min_start.isoformat() if min_start else None,
                       "max_start": max_start.isoformat() if max_start else None,
                       "max_stop": max_stop.isoformat() if max_stop else None,
                       "invalid_temporal_rows": invalid_temporal, "view_duration_mismatches": duration_mismatch,
                       "duration_seconds": {"min": min(durations) if durations else None,
                                            "p50_nearest_rank": histogram_quantile(durations, 0.5),
                                            "p90_nearest_rank": histogram_quantile(durations, 0.9),
                                            "p99_nearest_rank": histogram_quantile(durations, 0.99),
                                            "max": max(durations) if durations else None},
                       "iso_week_counts_by_start": dict(sorted(weeks.items())),
                       "time_zone": "naive source wall clock; no time-zone conversion asserted"})
    if role == "schedule":
        result["labelled_fraction"] = (rows - zero_ids - blank_ids) / rows if rows else 0
    return result


def inventory(data_dir: Path) -> dict:
    files = sorted(data_dir.glob("*.csv")) if data_dir.is_dir() else []
    if not files:
        return {"status": "MISSING_DATA", "data_dir": str(data_dir.resolve()),
                "files": [], "next": "Download and extract the official archive after accepting competition rules."}
    results = [audit_file(path) for path in files]
    roles = Counter(result["role"] for result in results)
    complete = roles["views"] == 1 and roles["schedule"] == 3 and roles["template"] == 1 and roles["unknown"] == 0
    valid = all(result["rows"] > 0 and result["malformed_rows"] == 0
                and result.get("invalid_temporal_rows", 0) == 0
                and result.get("view_duration_mismatches", 0) == 0 for result in results)
    return {"status": "INVENTORIED" if complete and valid else "REVIEW_REQUIRED",
            "files": results, "roles": dict(roles),
            "limitations": ["Time units, cutoffs, duplicates and interval join have not been audited by this inventory.",
                            "Counts from assignment text are not treated as measured results."]}

