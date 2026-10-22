#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from advanced_user_gate_recovery import register
from advanced_user_gate import register as register_gate
from experiments import run
register();register_gate()
run('main','UG-AP3-depth4-300-recovered','user_gate_recovery',dict(source_run='UG-AP3-depth4-300'))
run('main','UG-FULL-shfl4-replacement','user_gate_bundle',dict(source_run='UG-AP3-depth4-300-recovered'))
