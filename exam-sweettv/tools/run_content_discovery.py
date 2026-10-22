#!/usr/bin/env python3
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from advanced_content_discovery import register
from experiments import run
register();split=sys.argv[1]
p=json.loads(Path(f'configs/ranker-allusers-strong-{split}-fit.json').read_text())[0]['parameters']
run(split,'CDS-blocks-binary300','content_discovery_specialist',p)
run(split,'CDS-blocks-logit05','content_discovery_interpolation',dict(source_run='CDS-blocks-binary300',weight=.5))
run(split,'SHFL4CDS-rrf-equal','hybrid',dict(k0=60,components=[dict(run_id='SHF-PPR-recency-max-1',weight=1),dict(run_id='L4-logloss-allusers300',weight=.25),dict(run_id='CDS-blocks-binary300',weight=.25)]))
