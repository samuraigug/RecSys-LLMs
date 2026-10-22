#!/usr/bin/env python3
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from advanced_discovery_specialist import register
from experiments import run
register();split=sys.argv[1]
p=json.loads(Path(f'configs/ranker-allusers-strong-{split}-fit.json').read_text())[0]['parameters']
run(split,'DS-nonrepeat-binary300','discovery_specialist',p)
run(split,'DS-nonrepeat-logit05','discovery_specialist_interpolation',dict(source_run='DS-nonrepeat-binary300',weight=.5))
