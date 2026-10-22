#!/usr/bin/env python3
"""Run a real command, preserving its exact output and exit status as evidence."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True, help="Unique output prefix")
    parser.add_argument("--expected-exit", type=int, default=0)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("A command is required after --")
    paths = [Path(str(args.log) + suffix) for suffix in (".json", ".stdout.log", ".stderr.log")]
    if any(path.exists() for path in paths):
        raise FileExistsError("Use a new log prefix; previous evidence is immutable")
    args.log.parent.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now(timezone.utc).isoformat()
    start = time.perf_counter()
    with paths[1].open("xb") as out, paths[2].open("xb") as err:
        result = subprocess.run(command, stdout=out, stderr=err, check=False)
    record = {"command": command, "cwd": str(Path.cwd()), "started_at": started_at,
              "finished_at": datetime.now(timezone.utc).isoformat(),
              "seconds": time.perf_counter() - start, "exit_code": result.returncode,
              "expected_exit_code": args.expected_exit,
              "status": "PASS" if result.returncode == args.expected_exit else "FAIL",
              "stdout": str(paths[1]), "stderr": str(paths[2])}
    paths[0].write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
    sys.stdout.buffer.write(paths[1].read_bytes())
    sys.stderr.buffer.write(paths[2].read_bytes())
    print(json.dumps(record, ensure_ascii=False, indent=2))
    return 0 if result.returncode == args.expected_exit else 1


if __name__ == "__main__":
    raise SystemExit(main())
