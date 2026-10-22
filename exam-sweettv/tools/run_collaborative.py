#!/usr/bin/env python3
"""Register advanced train-only methods and run immutable experiment bundles."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
MODEL_SOURCE = (ROOT / "advanced_collaborative.py").read_text(encoding="utf-8")
RUNNER_SOURCE = Path(__file__).read_text(encoding="utf-8")
import experiments
from advanced_collaborative import register
from repro import dump, load, now, sha256


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split", choices=["main", "early", "final"], default="main")
    p.add_argument("--suite", type=Path, required=True)
    p.add_argument("--data-root", type=Path)
    p.add_argument("--runs-root", type=Path)
    a = p.parse_args()
    register(experiments.METHODS)
    root = a.runs_root or ROOT / "artifacts/runs"
    summary = []
    for config in load(a.suite):
        run_id = config["run_id"]
        record = experiments.run(a.split, run_id, config["method"], config.get("parameters", {}),
                                 a.data_root, a.runs_root)
        record["framework_generated_command"] = record["command"]
        record["command"] = [sys.executable, "tools/run_collaborative.py", *sys.argv[1:]]
        record["runner_sha256"] = sha256(Path(__file__))
        model_snapshot = root / a.split / run_id / "advanced_collaborative-source.py"
        runner_snapshot = root / a.split / run_id / "run_collaborative-source.py"
        model_snapshot.write_text(MODEL_SOURCE, encoding="utf-8")
        runner_snapshot.write_text(RUNNER_SOURCE, encoding="utf-8")
        record["loaded_model_source_sha256"] = sha256(model_snapshot)
        record["loaded_runner_source_sha256"] = sha256(runner_snapshot)
        dump(root / a.split / run_id / "record.json", record)
        summary.append({"run_id": run_id, "map_at_5_main": record["metrics"]["map_at_5_main"],
                        "protocol_sha256": record["protocol_sha256"]})
    evidence = ROOT / "artifacts/runs/collaborative-research"
    evidence.mkdir(parents=True, exist_ok=True)
    filename = a.suite.stem + "-" + a.split + "-summary.json"
    if (evidence / filename).exists():
        raise FileExistsError("Never overwrite a summary: " + filename)
    dump(evidence / filename, {"time": now(), "suite": str(a.suite), "runs": summary})


if __name__ == "__main__":
    main()
