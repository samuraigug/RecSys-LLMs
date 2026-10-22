#!/usr/bin/env python3
"""Run immutable content ablations with common candidate/cohort/fallback gates."""
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
SOURCE=(ROOT/'advanced_content.py').read_text(encoding='utf-8')
import experiments
from advanced_content import register
from repro import dump,load,sha256

def main():
    p=argparse.ArgumentParser();p.add_argument('--suite',type=Path,required=True);p.add_argument('--split',default='main')
    args=p.parse_args();register(experiments.METHODS)
    for c in load(args.suite):
        record=experiments.run(args.split,c['run_id'],c['method'],c.get('parameters',{}))
        path=ROOT/'artifacts/runs'/args.split/c['run_id'];snapshot=path/'advanced_content-source.py'
        snapshot.write_text(SOURCE,encoding='utf-8');record['loaded_model_source_sha256']=sha256(snapshot)
        record['framework_generated_command']=record['command'];record['command']=[sys.executable,'tools/run_content.py',*sys.argv[1:]]
        record['runner_sha256']=sha256(__file__);dump(path/'record.json',record)

if __name__=='__main__':main()
