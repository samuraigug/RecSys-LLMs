#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from advanced_weekly_transition import register
from experiments import run
register();split=sys.argv[1]
for symmetric,label in ((False,'directed'),(True,'symmetric')):
    run(split,f'WT-{label}','weekly_transition',dict(symmetric=symmetric,seed=42))
    run(split,f'SHFL4-WT-{label}-005','hybrid',dict(k0=60,components=[dict(run_id='FULL-shfl4',weight=1),dict(run_id=f'WT-{label}',weight=.05)]))
