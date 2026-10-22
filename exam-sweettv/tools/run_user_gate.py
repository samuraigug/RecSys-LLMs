#!/usr/bin/env python3
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from advanced_user_gate import register
from experiments import run
register();split=sys.argv[1]
source=json.loads(Path(f'configs/ranker-allusers-strong-{split}-fit.json').read_text())[0]['parameters']
p={k:source[k] for k in ('fold_root','folds','training_raw_dir')}
p.update(seed=42,iterations=300,depth=4,learning_rate=.05,l2_leaf_reg=10,loss_function='MultiRMSE')
run(split,'UG-AP3-depth4-300','user_gate',p)
run(split,'UG-FULL-shfl4-replacement','user_gate_bundle',dict(source_run='UG-AP3-depth4-300'))
