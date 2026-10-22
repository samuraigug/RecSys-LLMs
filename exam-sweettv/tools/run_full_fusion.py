#!/usr/bin/env python3
"""Two preregistered full-catalogue ablations; never train or search weights."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SOURCE = (ROOT/'advanced_full_fusion.py').read_text()
RUNNER = Path(__file__).read_text()
import experiments
from advanced_full_fusion import register
from repro import dump, sha256

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split', choices=['main', 'early'], required=True)
    args = parser.parse_args()
    register()
    result = []
    for bundle in ('shfl4', 'shfl45'):
        run_id = 'FULL-'+bundle
        record = experiments.run(args.split, run_id, 'full_catalogue_fusion', {'bundle': bundle})
        path = ROOT/'artifacts/runs'/args.split/run_id
        (path/'advanced_full_fusion-source.py').write_text(SOURCE)
        (path/'run_full_fusion-source.py').write_text(RUNNER)
        record['framework_generated_command'] = record['command']
        record['command'] = [sys.executable, 'tools/run_full_fusion.py', *sys.argv[1:]]
        record['loaded_model_source_sha256'] = sha256(path/'advanced_full_fusion-source.py')
        record['loaded_runner_source_sha256'] = sha256(path/'run_full_fusion-source.py')
        dump(path/'record.json', record)
        result.append({'run_id': run_id, 'metrics': record['metrics']})
    dest = ROOT/'artifacts/runs/full-fusion-research'/(args.split+'-summary.json')
    assert not dest.exists(), 'Do not overwrite saved results'
    dump(dest, result)
