#!/usr/bin/env python3
"""Immutable common-protocol semantic experiments and measured fusions."""
import argparse,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import experiments
from advanced_semantic import register
from repro import dump,load,sha256

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--suite',type=Path,required=True);parser.add_argument('--split',default='main')
    args=parser.parse_args();register();experiments.register_extensions()
    snapshots={name:(ROOT/name).read_text() for name in ('advanced_semantic.py',)}
    results=[]
    for config in load(args.suite):
        result=experiments.run(args.split,config['run_id'],config['method'],config.get('parameters',{}))
        path=ROOT/'artifacts/runs'/args.split/config['run_id']
        for name,source in snapshots.items():(path/(name[:-3]+'-source.py')).write_text(source)
        result['framework_generated_command']=result['command'];result['command']=[sys.executable,'tools/run_semantic.py',*sys.argv[1:]]
        result['runner_sha256']=sha256(__file__);result['loaded_model_source_sha256']=sha256(path/'advanced_semantic-source.py')
        dump(path/'record.json',result);results.append({'run_id':config['run_id'],'map_at_5':result['metrics']['map_at_5_main']})
    summary=ROOT/'artifacts/semantic-research'/(args.suite.stem+'-'+args.split+'-summary.json')
    if summary.exists():raise FileExistsError(summary)
    dump(summary,results)

if __name__=='__main__':main()
