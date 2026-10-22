#!/usr/bin/env python3
"""Paired user bootstrap on fixed saved outputs, never used for model fitting."""
import argparse,csv,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from repro import ROOT,load,dump,sha256
p=argparse.ArgumentParser();p.add_argument('--split',default='main');p.add_argument('--baseline',required=True);p.add_argument('--candidate',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
base=ROOT/'artifacts/runs'/a.split
left,right=(base/n for n in [a.baseline,a.candidate])
l,r=(load(d/'record.json') for d in [left,right]);assert l['protocol_sha256']==r['protocol_sha256']
def ap(d):
 with (d/'per-user-ap.csv').open() as f:return {x['user_id']:float(x['ap_at_5']) for x in csv.DictReader(f) if x['main']=='1'}
x,y=ap(left),ap(right);assert x.keys()==y.keys()
d=np.array([y[u]-x[u] for u in x]);rng=np.random.default_rng(42)
means=np.concatenate([d[rng.integers(0,len(d),size=(500,len(d)))].mean(axis=1) for _ in range(20)])
result={'split':a.split,'baseline':a.baseline,'candidate':a.candidate,'protocol_sha256':l['protocol_sha256'],'users':len(d),'baseline_map':np.mean(list(x.values())),'candidate_map':np.mean(list(y.values())),'mean_difference':float(d.mean()),'paired_bootstrap_ci_95':np.quantile(means,[.025,.975]).tolist(),'seed':42,'resamples':10000,'win_tie_loss_users':[int((d>0).sum()),int((d==0).sum()),int((d<0).sum())],'scope':'post-selection uncertainty on this fixed offline cohort; not independent replication, not a future score guarantee','record_hashes':{a.baseline:sha256(left/'record.json'),a.candidate:sha256(right/'record.json')}}
dump(a.output,result);print(result)
