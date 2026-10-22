#!/usr/bin/env python3
"""Train, evaluate and retain immutable SWEET.TV experiment bundles."""
import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback

import numpy as np
from threadpoolctl import threadpool_limits

from models import Context, METHODS
from protocol import ap_at_k
from repro import ROOT, code_hashes, dump, environment, load, now, peak_memory_mb, sha256


def register_extensions():
    """Keep extension runs reproducible through the same recorded CLI."""
    import importlib
    for name in ('advanced_temporal', 'advanced_collaborative', 'advanced_ranker', 'advanced_schedule',
                 'advanced_factorization', 'advanced_hourly_schedule', 'advanced_lightfm', 'advanced_content',
                 'advanced_availability', 'advanced_weekly_sessions', 'advanced_collective', 'advanced_leafwise',
                 'advanced_full_fusion', 'advanced_semantic', 'advanced_depth6',
                 'advanced_discovery_specialist', 'advanced_weekly_transition', 'advanced_content_discovery',
                 'advanced_session_survival', 'advanced_session_mixture', 'advanced_posterior_predictive',
                 'advanced_user_gate', 'advanced_user_gate_recovery', 'advanced_tifu_knn'):
        if not (ROOT / (name + '.py')).exists():
            continue
        module = importlib.import_module(name)
        if name in ('advanced_collaborative', 'advanced_schedule', 'advanced_hourly_schedule', 'advanced_content',
                    'advanced_weekly_sessions'):
            module.register(METHODS)
        else:
            module.register()


def finalize_scores(scores, ctx):
    if scores.shape != (len(ctx.targets), len(ctx.candidates)):
        raise ValueError("Wrong scoring orientation")
    if np.isinf(scores).any():
        raise ValueError("Infinite native scores")
    scores[ctx.cold_mask, :] = np.nan
    rank_indices = np.full((len(ctx.targets), 100), -1, dtype=np.int32)
    rank_scores = np.full((len(ctx.targets), 100), np.nan, dtype=np.float32)
    predictions, diagnostics = {}, {}
    numeric_ids = np.array([int(i) for i in ctx.candidates], dtype=np.int64)
    for j, u in enumerate(ctx.targets):
        available = np.flatnonzero(np.isfinite(scores[j]))
        ranked = available[np.lexsort((numeric_ids[available], -scores[j, available]))][:100]
        rank_indices[j, :len(ranked)] = ranked
        rank_scores[j, :len(ranked)] = scores[j, ranked]
        native = [ctx.candidates[i] for i in ranked[:5]]
        result = native[:]
        for item in ctx.fallback:
            if len(result) == 5:
                break
            if item not in result:
                result.append(item)
        assert len(result) == len(set(result)) == 5 and "0" not in result
        if ctx.cold_mask[j]:
            assert result == ctx.fallback[:5]
        predictions[u] = result
        diagnostics[u] = {"native_scored_candidates": len(available), "native_positions": len(native),
                          "fallback_positions": 5-len(native), "cold_gate": bool(ctx.cold_mask[j])}
    return predictions, diagnostics, rank_indices, rank_scores


def evaluate(ctx, predictions, diagnostics):
    aps = {u: ap_at_k(ctx.truth[u], predictions[u]) for u in ctx.targets}
    segment = lambda users: {"users": len(users), "map_at_5": float(np.mean([aps[u] for u in users])) if users else None}
    m = {"map_at_5_main": segment(ctx.cohorts["main"])["map_at_5"], "main_users": len(ctx.cohorts["main"]),
         "map_at_5_all_template": segment(ctx.targets)["map_at_5"], "all_template_users": len(ctx.targets),
         **{key: segment(ctx.cohorts[key]) for key in ("warm", "cold", "exactly_five_truth")}}
    m["recall_at_5_main"] = float(np.mean([len(set(predictions[u]) & set(ctx.truth[u]))/len(ctx.truth[u]) for u in ctx.cohorts["main"]])) if ctx.cohorts["main"] else None
    m["catalogue_items_in_top5"] = len({i for row in predictions.values() for i in row})
    m["repeat_share_top5"] = sum(i in ctx.history.get(u, {}) for u in ctx.targets for i in predictions[u])/(5*len(ctx.targets))
    m["fallback_share_top5"] = sum(r["fallback_positions"] for r in diagnostics.values())/(5*len(ctx.targets))
    return m, aps


def examples(ctx, predictions, diagnostics):
    def describe(ids):
        return [{"id": item, **ctx.display.get(item, {"title": "unknown", "category": "unknown"})} for item in ids]
    return [{"user_id": u, "selection": "heavy" if n == 0 else "cold",
             "train_unique_items": len(ctx.history.get(u, {})),
             "train_top5": describe(sorted(ctx.history.get(u, {}), key=lambda i: (-ctx.history[u][i]["airings"], -ctx.history[u][i]["seconds"], int(i)))[:5]),
             "truth": describe(ctx.truth[u]), "top5": describe(predictions[u]),
             "ap_at_5": ap_at_k(ctx.truth[u], predictions[u]), "diagnostics": diagnostics[u]}
            for n, u in enumerate(ctx.protocol["example_users"])]


def run(split, run_id, method, parameters, data_root=None, runs_root=None):
    if method not in METHODS:
        register_extensions()
    data_root = data_root or ROOT / "data/processed/elapsed-v2"
    runs_root = runs_root or ROOT / "artifacts/runs"
    path = runs_root / split / run_id
    if path.exists():
        raise FileExistsError("Never overwrite an experiment: " + str(path))
    path.mkdir(parents=True)
    started, timestamp = time.perf_counter(), now()
    ctx = Context(data_root / split)
    record = {"run_id": run_id, "status": "FAILED", "method": method, "started_at": timestamp,
              "command": [sys.executable, "experiments.py", "--split", split, "--run-id", run_id,
                          "--method", method, "--parameters", json.dumps(parameters, sort_keys=True)],
              "code_sha256": code_hashes(), "protocol_sha256": ctx.protocol_hash,
              "input_hashes": ctx.protocol["input_hashes"], "parameters": parameters,
              "seed": parameters.get("seed"), "environment": environment(),
              "selection_notes": "Full catalogue, fixed cohorts, common cold/fallback gate. Offline selection only."}
    try:
        with threadpool_limits(limits=1):
            fit_started = time.perf_counter()
            scores, details = METHODS[method](ctx, parameters, path)
            record["fit_and_score_seconds"] = time.perf_counter()-fit_started
            predictions, diagnostics, indices, native_scores = finalize_scores(scores, ctx)
        metrics, aps = evaluate(ctx, predictions, diagnostics)
        np.savez_compressed(path / "native-top100.npz", indices=indices, scores=native_scores)
        dump(path / "top5.json", predictions)
        dump(path / "fallback-diagnostics.json", diagnostics)
        dump(path / "native-score-coverage.json", {"users": len(ctx.targets), "candidates": len(ctx.candidates),
                                                   "min": min(r["native_scored_candidates"] for r in diagnostics.values()),
                                                   "max": max(r["native_scored_candidates"] for r in diagnostics.values()),
                                                   "mean": float(np.mean([r["native_scored_candidates"] for r in diagnostics.values()])),
                                                   "native_scores_are_not_probabilities": True})
        with (path / "per-user-ap.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["user_id", "ap_at_5", "main", "cold", "truth_items"])
            main, cold = set(ctx.cohorts["main"]), set(ctx.cohorts["cold"])
            writer.writerows((u, aps[u], int(u in main), int(u in cold), len(ctx.truth[u])) for u in ctx.targets)
        dump(path / "two-user-examples.json", examples(ctx, predictions, diagnostics))
        record.update(status="SUCCEEDED", metrics=metrics, model_details=details,
                      runtime_seconds=time.perf_counter()-started, peak_memory_mb=peak_memory_mb(),
                      artifacts={"stdout_stderr": str(path / "execution.log"), "top5": str(path / "top5.json"),
                                 "per_user_ap": str(path / "per-user-ap.csv"),
                                 "two_user_examples": str(path / "two-user-examples.json"),
                                 "fallback_diagnostics": str(path / "fallback-diagnostics.json"),
                                 "native_score_coverage": str(path / "native-score-coverage.json")})
        # There are deliberately no unknown-future test metrics.
        if split == "final":
            record["status"] = "REFIT_WITHOUT_TEST_LABELS"
        dump(path / "record.json", record)
        dump(path / "execution.log", {"command": record["command"], "started_at": timestamp,
                                      "finished_at": now(), "status": record["status"], "model_details": details})
        print(json.dumps({"run_id": run_id, "method": method, "map_at_5": metrics["map_at_5_main"],
                          "seconds": record["runtime_seconds"], "peak_mb": record["peak_memory_mb"]}), flush=True)
    except Exception:
        record.pop("model_details", None)
        record.update(status="FAILED", failure=traceback.format_exc(), runtime_seconds=time.perf_counter()-started,
                      peak_memory_mb=peak_memory_mb(), artifacts={})
        dump(path / "record.json", record)
        raise
    return record


def suite(path, split, data_root=None, runs_root=None):
    configs = load(path)
    root = runs_root or ROOT / "artifacts/runs"
    summary = []
    for config in configs:
        command = [sys.executable, "experiments.py", "--split", split, "--run-id", config["run_id"],
                   "--method", config["method"], "--parameters", json.dumps(config.get("parameters", {}))]
        if data_root:
            command += ["--data-root", str(data_root)]
        if runs_root:
            command += ["--runs-root", str(runs_root)]
        logs = root / split / "suite-logs"
        logs.mkdir(parents=True, exist_ok=True)
        log = logs / (config["run_id"] + ".log")
        with log.open("x") as stream:
            result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=False)
        if result.returncode:
            print(log.read_text(), file=sys.stderr)
            raise RuntimeError(f"Experiment failed; retained at {log}")
        print(log.read_text(), end="", flush=True)
        summary.append(load(root / split / config["run_id"] / "record.json"))
    return summary


if __name__ == "__main__":
    register_extensions()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split", choices=["main", "early", "final"], default="main")
    p.add_argument("--suite", type=Path)
    p.add_argument("--run-id")
    p.add_argument("--method", choices=list(METHODS))
    p.add_argument("--parameters", default="{}")
    p.add_argument("--data-root", type=Path)
    p.add_argument("--runs-root", type=Path)
    a = p.parse_args()
    if a.suite:
        suite(a.suite, a.split, a.data_root, a.runs_root)
    elif a.run_id and a.method:
        run(a.split, a.run_id, a.method, json.loads(a.parameters), a.data_root, a.runs_root)
    else:
        p.error("Pass --suite, or --run-id and --method")
