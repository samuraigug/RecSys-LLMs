#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from advanced_session_survival import register
from experiments import run
register();split=sys.argv[1]
run(split,'SS-empirical-gap0','session_survival',dict(seed=42))
run(split,'SS-FULL-shfl4-replacement','session_survival_bundle',dict(source_run='SS-empirical-gap0'))
