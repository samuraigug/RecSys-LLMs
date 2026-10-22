#!/usr/bin/env python3
"""Prepare frozen historical pseudo-futures or run through shared experiment API."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from repro import ROOT
from advanced_ranker import FOLDS, FOLD_ROOT, register


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepare-folds', action='store_true')
    p.add_argument('--split', choices=['main', 'early', 'final'], default='main')
    p.add_argument('--run-id')
    p.add_argument('--parameters', default='{}')
    p.add_argument('--data-root', type=Path)
    p.add_argument('--runs-root', type=Path)
    a = p.parse_args()
    if a.prepare_folds:
        import etl
        etl.SPLITS.update(FOLDS)
        etl.prepare(ROOT/'data/raw', FOLD_ROOT, list(FOLDS))
    else:
        if not a.run_id:
            p.error('--run-id required')
        register()
        import experiments
        experiments.run(a.split, a.run_id, 'learned_ranker', json.loads(a.parameters), a.data_root, a.runs_root)


if __name__ == '__main__':
    main()
