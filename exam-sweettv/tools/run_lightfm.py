#!/usr/bin/env python3
"""Run LightFM experiments through the common frozen evaluator."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
MODEL_SOURCE = (ROOT / "advanced_lightfm.py").read_text(encoding="utf-8")
RUNNER_SOURCE = Path(__file__).read_text(encoding="utf-8")
import experiments
from advanced_lightfm import register
from repro import dump, load, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["main", "early", "final"], default="main")
    parser.add_argument("--suite", type=Path, required=True)
    args = parser.parse_args()
    register()
    rows = []
    for config in load(args.suite):
        record = experiments.run(args.split, config["run_id"], config["method"], config.get("parameters", {}))
        path = ROOT / "artifacts/runs" / args.split / config["run_id"]
        (path / "advanced_lightfm-source.py").write_text(MODEL_SOURCE, encoding="utf-8")
        (path / "run_lightfm-source.py").write_text(RUNNER_SOURCE, encoding="utf-8")
        record["framework_generated_command"] = record["command"]
        record["command"] = [sys.executable, "tools/run_lightfm.py", *sys.argv[1:]]
        record["loaded_model_source_sha256"] = sha256(path / "advanced_lightfm-source.py")
        record["loaded_runner_source_sha256"] = sha256(path / "run_lightfm-source.py")
        dump(path / "record.json", record)
        rows.append({"run_id": record["run_id"], "map_at_5_main": record["metrics"]["map_at_5_main"],
                     "seconds": record["runtime_seconds"]})
    out = ROOT / "artifacts/runs/lightfm-research" / (args.suite.stem + "-" + args.split + "-summary.json")
    if out.exists():
        raise FileExistsError(out)
    dump(out, rows)


if __name__ == "__main__":
    main()
