#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from advanced_session_mixture import register
from experiments import run
register();split=sys.argv[1]
run(split,'SS-duration-mixture-gap0','session_mixture',dict(seed=42))
run(split,'SS-mixture-FULL-shfl4-replacement','session_mixture_bundle',dict(source_run='SS-duration-mixture-gap0'))
