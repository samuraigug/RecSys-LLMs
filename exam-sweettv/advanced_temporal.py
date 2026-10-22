"""Train-only weak-view/exposure and temporal-schedule ranking ablations.

Inspired by recency-weighted candidate generation; no validation fitting or
future content. Candidate IDs/channel/time availability follow frozen protocol.
"""
from pathlib import Path
import argparse
import json
import sys

import numpy as np
from scipy import sparse

from etl import read_events, Clock
from models import METHODS
from repro import load, dump, sha256


def temporal_repeat(ctx, p, path):
    eu, ei, _, e = read_events(ctx.path/'train-events.npz', full_only=False)
    um = {u:j for j,u in enumerate(ctx.targets)}
    ur = np.array([um.get(u,-1) for u in eu])
    ic = np.array([ctx.candidate_map.get(i,-1) for i in ei])
    keep=(ur[e[:,0]]>=0)&(ic[e[:,1]]>=0)
    e=e[keep]; u=ur[e[:,0]]; i=ic[e[:,1]]
    assert np.all(e[:,6] <= ctx.protocol['cutoff_elapsed_seconds'])
    fraction=e[:,5]/e[:,7]
    full=fraction>=.8
    weak=p.get('weak_weight',.25)*fraction**p.get('fraction_power',2)
    weights=np.where(fraction>=p.get('feature_full_threshold',.8),1,weak)
    if p.get('full_weight')=='seconds':weights*=np.sqrt(e[:,5]/3600)
    if p.get('half_life_days'):
        age=(ctx.protocol['cutoff_elapsed_seconds']-e[:,6])/86400
        weights*=np.exp2(-age/p['half_life_days'])
    if p.get('day_cap'):
        # Cap within each user/item/day before aggregating days, train only.
        k=np.rec.fromarrays([u,i,e[:,3]//86400],names='u,i,d')
        _,inv=np.unique(k,return_inverse=True)
        totals=np.bincount(inv,weights=weights)
        weights*=np.minimum(1,p['day_cap']/np.maximum(totals[inv],1e-9))
    counts=sparse.csr_matrix((weights,(u,i)),shape=(len(ctx.targets),len(ctx.candidates))).toarray()
    opp=load(ctx.path/'opportunities.json')
    tr=np.array([opp['train'].get(i,0) for i in ctx.candidates],float)
    fu=np.array([opp['future'].get(i,0) for i in ctx.candidates],float)
    prior=p.get('prior_strength',50)
    scores=counts / (tr+prior)[None,:]**p.get('train_power',1) * fu[None,:]**p.get('future_power',1)
    if p.get('log_counts'): scores=np.log1p(counts)/(tr+prior)[None,:]**p.get('train_power',1)*fu[None,:]**p.get('future_power',1)
    scores[scores<=0]=np.nan
    return scores.astype(np.float32), {'fit_cutoff':ctx.protocol['train_cutoff_exclusive'],'view_input_sha256':sha256(ctx.path/'train-events.npz'),'weak_events':int((~full).sum()),'full_events':int(full.sum()),'future_information':'ID/channel/time opportunity counts only','algorithm':'weighted full/partial counts with exposure normalization'}


def calibrated_mix(ctx,p,path):
    # Within-user L1 calibrated nonnegative signals, unlike global calibration.
    scores=np.zeros((len(ctx.targets),len(ctx.candidates)),np.float64)
    touched=np.zeros_like(scores,dtype=bool)
    sources=[]
    for c in p['components']:
        rp=path.parent/c['run_id'];rec=load(rp/'record.json')
        if rec['protocol_sha256']!=ctx.protocol_hash:raise ValueError('Protocol mismatch')
        with np.load(rp/'native-top100.npz',allow_pickle=False) as z:
            ix=z['indices'];raw=z['scores'].astype(float)
        valid=ix>=0
        for u in range(len(ctx.targets)):
            ids=ix[u,valid[u]];v=raw[u,valid[u]]
            if not len(ids):continue
            if p.get('normalization')=='softmax':
                scale=max(float(np.std(v)),1e-8);v=np.exp((v-v.max())/scale/p.get('temperature',1))
            else:
                v=np.maximum(v,0)**p.get('power',1)
            if v.sum()<=0:continue
            scores[u,ids]+=c['weight']*v/v.sum();touched[u,ids]=True
        sources.append({'run_id':c['run_id'],'weight':c['weight'],'hash':sha256(rp/'native-top100.npz')})
    scores[~touched]=np.nan
    return scores.astype(np.float32),{'sources':sources,'fusion':'within-user normalized native-score fusion, fallback once'}


def joint_habits(ctx,p,path):
    eu, ei, channels, e=read_events(ctx.path/'train-events.npz',full_only=False)
    um={u:j for j,u in enumerate(ctx.targets)};ur=np.array([um.get(u,-1) for u in eu])
    keep=ur[e[:,0]]>=0;e=e[keep];u=ur[e[:,0]]
    # opportunity-slots is channels x 8 in sorted numeric channels of all grid/log.
    with np.load(ctx.path/'habits.npz',allow_pickle=False) as z:
        channel=z['channel_seconds'];slot=z['slot_seconds']
    # The exact mapping includes zero-ID-only channels, so derive it from raw
    # channel IDs only (no reading view timestamps or future content).
    import csv
    raw=ctx.path.parents[2]/'raw'
    allch=set()
    for fn in ['dataset11-30.csv','export_arh_11-20-final.csv','export_arh_21-30-final.csv','export_arh_31-42-final.csv']:
        with (raw/fn).open(encoding='utf-8-sig',newline='') as f:
            allch.update(r['vsetv_id' if fn.startswith('dataset') else 'channel_id'] for r in csv.DictReader(f)
                         if fn.startswith('dataset') or r['tv_show_id'] != '0')
    cm={c:j for j,c in enumerate(sorted(allch,key=int))}
    assert len(cm)==channel.shape[1]
    # seconds associated with actually viewed labelled airings, 8 bins by local last-view stop.
    clock=Clock(ctx.protocol['clock'])
    bins=np.array([clock.local(int(t)-1).hour//3 for t in e[:,6]])
    cc=np.array([cm[c] for c in channels])[e[:,2]]*8+bins
    weight=e[:,5].astype(float)
    if p.get('half_life_days'):weight*=np.exp2(-(ctx.protocol['cutoff_elapsed_seconds']-e[:,6])/86400/p['half_life_days'])
    joint=sparse.csr_matrix((weight,(u,cc)),shape=(len(ctx.targets),len(cm)*8)).toarray()
    joint/=np.maximum(joint.sum(axis=1,keepdims=True),1)
    marg=(channel/np.maximum(channel.sum(axis=1,keepdims=True),1))[:,:,None]*(slot/np.maximum(slot.sum(axis=1,keepdims=True),1))[:,None,:]
    prof=p.get('joint_weight',.8)*joint+(1-p.get('joint_weight',.8))*marg.reshape(len(ctx.targets),-1)
    opp=sparse.load_npz(ctx.path/'opportunity-slots.npz')
    if p.get('count_opportunities'):
        opportunities=load(ctx.path/'opportunities.json')['future']
        total_hours=np.asarray(opp.sum(axis=1)).ravel()
        factor=np.array([opportunities.get(i,0) for i in ctx.candidates])/np.maximum(total_hours,1e-9)
        opp=opp.multiply(factor[:,None]).tocsr()
    scores=np.asarray(prof@opp.T)
    scores[scores<=0]=np.nan
    np.save(path/'full-context-scores.npy',scores.astype(np.float32))
    return scores.astype(np.float32),{'context':'joint channel/time from pre-cutoff labelled events, shrunk to full-view marginals','future_information':'availability only','fit_cutoff':ctx.protocol['train_cutoff_exclusive']}


def full_temporal_context(ctx,p,path):
    source=path.parent/p['context_run']
    record=load(source/'record.json')
    if record['protocol_sha256']!=ctx.protocol_hash:raise ValueError('Protocol mismatch')
    context=np.load(source/'full-context-scores.npy')
    personal,details=temporal_repeat(ctx,p.get('temporal',{'weak_weight':1,'prior_strength':100}),path)
    context=np.nan_to_num(context).astype(float);personal=np.nan_to_num(personal).astype(float)
    power=p.get('context_power',1)
    context=context**power
    context/=np.maximum(context.sum(axis=1,keepdims=True),1e-20)
    personal/=np.maximum(personal.sum(axis=1,keepdims=True),1e-20)
    scores=personal+p.get('context_weight',.3)*context
    scores[scores<=0]=np.nan
    return scores.astype(np.float32),{'context_scores_sha256':sha256(source/'full-context-scores.npy'),
                                     'context_record_sha256':sha256(source/'record.json'),
                                     'full_catalogue_scoring':True,'temporal_details':details}


def register():
    METHODS.update(temporal_repeat=temporal_repeat,calibrated_mix=calibrated_mix,joint_habits=joint_habits,
                   full_temporal_context=full_temporal_context)

if __name__=='__main__':
    from experiments import run
    register()
    a=argparse.ArgumentParser();a.add_argument('--suite',type=Path,required=True);a.add_argument('--split',default='main');args=a.parse_args()
    results=[]
    for conf in load(args.suite):
        out=Path(__file__).parent/'artifacts/runs'/args.split/conf['run_id']
        if out.exists():continue
        results.append(run(args.split,conf['run_id'],conf['method'],conf.get('parameters',{})))
    print('SUITE FINISHED',len(results),flush=True)
