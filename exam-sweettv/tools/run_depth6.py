#!/usr/bin/env python3
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from advanced_depth6 import register
from experiments import run
register()
split=sys.argv[1]
run(split,'LW-depth6-binary-L5-300','depth6_ranker',json.loads(Path(f'research/depth6-20261022/parameters-{split}.json').read_text()))
