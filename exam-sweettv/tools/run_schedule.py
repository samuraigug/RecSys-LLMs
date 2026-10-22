#!/usr/bin/env python3
"""Run immutable schedule hypotheses through the shared experiment framework."""
import argparse
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
SOURCE=(ROOT/'advanced_schedule.py').read_text(encoding='utf-8')
import experiments
from advanced_schedule import register
from repro import load,dump,sha256

def main():
    p=argparse.ArgumentParser(); p.add_argument('--split',default='main');p.add_argument('--suite',type=Path,required=True)
    args=p.parse_args();register(experiments.METHODS)
    for conf in load(args.suite):
        record=experiments.run(args.split,conf['run_id'],conf['method'],conf.get('parameters',{}))
        path=ROOT/'artifacts/runs'/args.split/conf['run_id']
        snapshot=path/'advanced_schedule-source.py';snapshot.write_text(SOURCE,encoding='utf-8')
        record['framework_generated_command']=record['command']
        record['command']=[sys.executable,'tools/run_schedule.py',*sys.argv[1:]]
        record['loaded_model_source_sha256']=sha256(snapshot)
        record['runner_sha256']=sha256(__file__)
        dump(path/'record.json',record)

if __name__=='__main__':main()
