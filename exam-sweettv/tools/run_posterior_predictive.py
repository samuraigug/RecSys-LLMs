#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from advanced_posterior_predictive import register
from experiments import run
register();split=sys.argv[1]
run(split,'BP-mean256','posterior_predictive',dict(seed=42,samples=256))
run(split,'BP-greedyMAP256','posterior_greedy',dict(source_run='BP-mean256'))
