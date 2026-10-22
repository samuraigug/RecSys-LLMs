#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from advanced_tifu_knn import register, RESEARCH
from experiments import run
register()
if __name__ == '__main__':
    split = sys.argv[1]
    for name, parameters in [('TIFU-control', {'temporal': False, 'seed': 42}),
                             ('TIFU-temporal', {'temporal': True, 'seed': 42}),
                             ('FULL-shfl4-TIFU005', {'temporal': True, 'blend': True, 'seed': 42})]:
        run(split, name, 'tifu_knn', parameters, runs_root=RESEARCH/'runs')
