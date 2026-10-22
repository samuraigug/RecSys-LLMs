#!/usr/bin/env python3
"""Initial local tools. Model training is not implemented yet; see PLAN.md."""

import argparse
import json
from pathlib import Path

from audit import inventory
from protocol import validate_submission


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    audit = subs.add_parser("audit")
    audit.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    audit.add_argument("--output", type=Path, default=Path("artifacts/raw-audit.json"))
    check = subs.add_parser("check-submission")
    check.add_argument("--template", type=Path, required=True)
    check.add_argument("--submission", type=Path, required=True)
    check.add_argument("--candidates", type=Path, required=True, help="UTF-8 file with one scheduled programme ID per line")
    args = parser.parse_args()
    if args.command == "audit":
        result = inventory(args.data_dir)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps({"status": result["status"], "output": str(args.output)}, ensure_ascii=False))
        return 0 if result["status"] == "INVENTORIED" else 2
    result = validate_submission(args.template, args.submission, args.candidates.read_text().splitlines())
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

