#!/usr/bin/env python3
"""Run scoped collective-WARP hypotheses with shared frozen evaluation."""
import argparse
from pathlib import Path
import sys
import pickle
import numpy as np
from scipy import sparse
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
SOURCE=(ROOT/'advanced_collective.py').read_text()
RUNNER=Path(__file__).read_text()
import experiments
from advanced_collective import register
from repro import load,dump,sha256


def discovery_mix(ctx,p,path):
    """Fixed repeat/discovery interleaving; no validation targets used."""
    base=path.parent/p['base_run'];discovery=path.parent/p['discovery_run']
    for source in [base,discovery]:
        if load(source/'record.json')['protocol_sha256']!=ctx.protocol_hash:
            raise ValueError('Cannot combine different frozen protocols')
    mapping=load(discovery/'collective-mapping.json')
    if mapping['candidate_items']!=ctx.candidates:
        raise ValueError('Collective candidate mapping mismatch')
    # Reopen only our own freshly trained model from a successful local run.
    with (discovery/'collective-model.pickle').open('rb') as f:model=pickle.load(f)
    cf=sparse.load_npz(discovery/'candidate-features.npz')
    ub,uf=model.get_user_representations();ib,factors=model.get_item_representations(cf)
    um={u:j for j,u in enumerate(mapping['users'])};tr=np.array([um[u] for u in ctx.targets])
    raw=uf[tr]@factors.T+ib[None,:]+ub[tr,None]
    valid_features=np.asarray(cf.getnnz(axis=1)).ravel()>0
    with np.load(base/'native-top100.npz',allow_pickle=False) as z:base_ranks=z['indices']
    scores=np.full_like(raw,np.nan,dtype=np.float32)
    numeric=np.array([int(i) for i in ctx.candidates],np.int64)
    for j,user in enumerate(ctx.targets):
        repeated=np.array([i in ctx.history.get(user,{}) for i in ctx.candidates])
        ranked=base_ranks[j];ranked=ranked[ranked>=0];ranked=ranked[repeated[ranked]]
        scores[j,ranked]=p.get('repeat_weight',1)/(1+np.arange(len(ranked)))
        novel=np.flatnonzero(~repeated&valid_features)
        novel=novel[np.lexsort((numeric[novel],-raw[j,novel]))]
        scores[j,novel]=p.get('discovery_weight',.5)/(1+np.arange(len(novel)))
    return scores,{'algorithm':'separate repeat/discovery heads; fixed reciprocal rank interleaving',
        'base_run':p['base_run'],'discovery_run':p['discovery_run'],
        'repeat_weight':p.get('repeat_weight',1),'discovery_weight':p.get('discovery_weight',.5),
        'scores_are_probabilities':False,'validation_truth_used':False,
        'base_native_hash':sha256(base/'native-top100.npz'),'discovery_model_hash':sha256(discovery/'collective-model.pickle'),
        'discovery_features_hash':sha256(discovery/'candidate-features.npz')}

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--split',default='main');parser.add_argument('--suite',type=Path,required=True);args=parser.parse_args()
    register();experiments.METHODS['collective_discovery_mix']=discovery_mix;rows=[]
    for conf in load(args.suite):
        record=experiments.run(args.split,conf['run_id'],conf['method'],conf.get('parameters',{}))
        path=ROOT/'artifacts/runs'/args.split/conf['run_id']
        (path/'advanced_collective-source.py').write_text(SOURCE);(path/'run_collective-source.py').write_text(RUNNER)
        record['framework_generated_command']=record['command'];record['command']=[sys.executable,'tools/run_collective.py',*sys.argv[1:]]
        record['loaded_model_source_sha256']=sha256(path/'advanced_collective-source.py');record['loaded_runner_source_sha256']=sha256(path/'run_collective-source.py')
        dump(path/'record.json',record);rows.append({'run_id':conf['run_id'],'map':record['metrics']['map_at_5_main']})
    dest=ROOT/'artifacts/runs/collective-research'/(args.suite.stem+'-'+args.split+'.json')
    if dest.exists():raise FileExistsError(dest)
    dump(dest,rows)
