#!/usr/bin/env python3
"""Independent rational AP recalculation, schema checks and ranking contracts."""
import argparse
import csv
from fractions import Fraction
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import jsonschema
from repro import ROOT, dump, load, sha256


def independent_ap(truth, ranking):
    relevant = set(truth)
    if not relevant:
        return Fraction(0)
    return sum((Fraction(len(set(ranking[:r]) & relevant), r)
                for r, i in enumerate(ranking[:5], 1) if i in relevant and i not in ranking[:r-1]), Fraction(0))/min(5, len(relevant))


def verify(split, output, data_root, runs_root):
    source = data_root/split
    truth, cohorts = load(source/"truth.json"), load(source/"cohorts.json")
    protocol_hash = sha256(source/"protocol.json")
    candidates = (source/"candidates.txt").read_text().splitlines()
    pool = set(candidates)
    fallback = load(source/"fallback.json")
    schema = load(ROOT/"docs/experiment-record.schema.json")
    results = []
    for record_path in sorted((runs_root/split).glob("*/record.json")):
        record = load(record_path)
        if record["status"] != "SUCCEEDED":
            continue
        jsonschema.validate(record, schema)
        assert record["protocol_sha256"] == protocol_hash
        path = record_path.parent
        pred = load(path/"top5.json")
        diag = load(path/"fallback-diagnostics.json")
        with np.load(path/"native-top100.npz", allow_pickle=False) as z:
            indices = z["indices"]
        assert list(pred) == cohorts["all_template"]
        stored = {r["user_id"]: float(r["ap_at_5"]) for r in csv.DictReader((path/"per-user-ap.csv").open())}
        aps = {}
        for n, user in enumerate(cohorts["all_template"]):
            ranking = pred[user]
            assert len(ranking) == len(set(ranking)) == 5 and set(ranking) <= pool and "0" not in ranking
            aps[user] = float(independent_ap(truth[user], ranking))
            assert abs(aps[user]-stored[user]) < 1e-12
            native = [candidates[int(i)] for i in indices[n, :5] if i >= 0]
            assert ranking[:len(native)] == native
            assert diag[user]["native_positions"] == len(native)
            if user in cohorts["cold"]:
                assert ranking == fallback[:5] and not native
        main = sum(aps[u] for u in cohorts["main"])/len(cohorts["main"])
        assert abs(main-record["metrics"]["map_at_5_main"]) < 1e-12
        for key in ("warm", "cold", "exactly_five_truth"):
            ids = cohorts[key]
            metric = sum(aps[u] for u in ids)/len(ids) if ids else None
            assert metric is None or abs(metric-record["metrics"][key]["map_at_5"]) < 1e-12
        results.append({"run_id": record["run_id"], "method": record["method"], "independent_map": main,
                        "users_checked": len(pred), "record_sha256": sha256(record_path)})
    dump(output, {"status": "PASS", "protocol_sha256": protocol_hash, "runs_checked": len(results),
                  "method": "Fraction-based independent AP; full users/order, pool, native rank and common cold gate; JSON schema", "results": results})
    print(f"Independent verification PASS: {len(results)} runs × {len(cohorts['all_template'])} users")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split", default="main")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--data-root", type=Path, default=ROOT/"data/processed/elapsed-v2")
    p.add_argument("--runs-root", type=Path, default=ROOT/"artifacts/runs")
    a = p.parse_args()
    verify(a.split, a.output, a.data_root, a.runs_root)
