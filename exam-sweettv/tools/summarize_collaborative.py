#!/usr/bin/env python3
"""Verify this research branch and retain descriptive paired uncertainty."""
from pathlib import Path
import csv
import argparse
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from repro import dump,load,now,sha256
from tools.verify_experiments import verify

PREFIX=('CFC-','SC-','SCF-','SCV2-','SCFV2-','SCFL1-','SCFL0-','SH-','SHF-','SHV-','SHFL3-')
OUT=ROOT/'artifacts/runs/collaborative-research'


def aps(path):
    return {r['user_id']:float(r['ap_at_5']) for r in csv.DictReader((path/'per-user-ap.csv').open())}


def paired(split,a,b):
    root=ROOT/'artifacts/runs'/split
    cohort=load(ROOT/'data/processed/elapsed-v2'/split/'cohorts.json')['main']
    aa,bb=aps(root/a),aps(root/b)
    delta=np.array([aa[u]-bb[u] for u in cohort])
    rng=np.random.default_rng(2026)
    draws=np.mean(delta[rng.integers(0,len(delta),size=(5000,len(delta)))],axis=1)
    return {'a':a,'b':b,'users':len(delta),'delta_map':float(delta.mean()),
        'bootstrap_95_percentile':np.quantile(draws,[.025,.975]).tolist(),
        'users_improved':int((delta>0).sum()),'users_worse':int((delta<0).sum()),
        'users_equal':int((delta==0).sum()),'bootstrap_samples':5000,'seed':2026,
        'interpretation':'Descriptive conditional on tuned offline configs; not unbiased generalization or independent temporal test.'}


def equal(split,a,b):
    root=ROOT/'artifacts/runs'/split
    assert load(root/a/'top5.json')==load(root/b/'top5.json')
    with np.load(root/a/'native-top100.npz') as aa, np.load(root/b/'native-top100.npz') as bb:
        assert all(np.array_equal(aa[k],bb[k],equal_nan=True) for k in ['indices','scores'])
    return {'a':a,'b':b,'top5_equal':True,'native_top100_indices_and_scores_equal':True,
            'top5_sha256':sha256(root/a/'top5.json')}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--tag',default='batch-summary-v2');args=parser.parse_args()
    results={}
    for split in ['main','early']:
        rows=[]
        for rp in sorted((ROOT/'artifacts/runs'/split).glob('*/record.json')):
            r=load(rp)
            if r['run_id'].startswith(PREFIX) and r['status']=='SUCCEEDED':
                rows.append({'run_id':r['run_id'],'method':r['method'],
                    'map_at_5_main':r['metrics']['map_at_5_main'],
                    'map_at_5_all_template':r['metrics']['map_at_5_all_template'],
                    'protocol_sha256':r['protocol_sha256'],'runtime_seconds':r['runtime_seconds']})
        with tempfile.TemporaryDirectory(prefix='sweettv-collaborative-verify-') as directory:
            target=Path(directory)/split;target.mkdir()
            for r in rows:
                (target/r['run_id']).symlink_to(ROOT/'artifacts/runs'/split/r['run_id'],target_is_directory=True)
            verify(split,OUT/f'independent-{split}-{args.tag}.json',ROOT/'data/processed/elapsed-v2',Path(directory))
        rows.sort(key=lambda r:r['map_at_5_main'],reverse=True)
        results[split]={'runs':len(rows),'ranking':rows,
            'paired_vs_repeat':paired(split,'SCFV2-PPR-max05','R2-rate-prior100'),
            'cache_v2_replay':equal(split,'SC-joint05','SCV2-joint05'),
            'fusion_replay':equal(split,'SCF-PPR-max-0.5','SCFV2-PPR-max05')}
        if (ROOT/'artifacts/runs'/split/'SHF-validated-replay/record.json').exists():
            results[split]['hourly_replay']=equal(split,'SH-j05-sm25-hl28','SHV-hourly28-replay')
            results[split]['hourly_fusion_replay']=equal(split,'SHF-PPR-recency-max-1','SHF-validated-replay')
    results['main']['paired_vs_previous_best']=paired('main','SCFV2-PPR-max05','F14-conservative-habits')
    results['main']['ppr_replay']=equal('main','CFC-PPRF-a02','CFC-PPR-best-replay')
    selected=results['main']['ranking'][0]['run_id']
    results['main']['selected_vs_previous_best']=paired('main',selected,'F14-conservative-habits')
    for split in ['main','early']:
        record=load(ROOT/'artifacts/runs'/split/selected/'record.json')
        results[split]['selected_by_main']={'run_id':selected,'map_at_5_main':record['metrics']['map_at_5_main']}
    dump(OUT/f'{args.tag}.json',{'checked_at':now(),'status':'VERIFIED','results':results,
        'selection':'Highest measured main configuration in this branch, same run ID checked on early. Early overlaps main and is diagnostic only.',
        'limits':['No Kaggle score; no future labels or views used for fitting.',
            'Partial-only items/users absent from Context full-view CF universe; parent explores expanded graph.',
            'Many main configurations were compared; bootstrap is descriptive and early overlaps main.',
            'EASE/itemKNN candidates unsupported by train model use common fallback.',
            'Older schedule cache-v1/runs retained unchanged; current v2 source+protocol keys and NPZ/raw hashes checked.'],
        'source_sha256':sha256(__file__)})
    print('Batch verification and summary PASS',flush=True)


if __name__=='__main__':main()
