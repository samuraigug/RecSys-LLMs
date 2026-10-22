"""Cutoff-safe hourly weekly viewing routines; sparse marginal smoothing.

Reuses the verified v2 schedule inventory/airings, but reconstructs exact raw
training intervals at hourly precision. Existing three-hour caches are immutable.
"""
from collections import defaultdict
import csv
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.preprocessing import normalize

import advanced_schedule as coarse
from etl import Clock, clip_union
from repro import ROOT, dump, load, sha256

SOURCE_SHA256=sha256(__file__)
SEMANTIC_VERSION='hourly-union-weekly-counts-v1'


def _build(ctx,path,cache):
    # This call checks coarse source/protocol/raw-manifest and all source NPZ hashes.
    coarse.schedule_joint(ctx,{'joint_weight':.5},path)
    coarse_ref=load(path/'schedule-cache-reference.json')
    coarse_path=ROOT/coarse_ref['path']
    manifest=load(coarse_path/'manifest.json')
    channels=manifest['channels']; cm={c:j for j,c in enumerate(channels)}
    clock=Clock(ctx.protocol['clock']);start=clock.seconds(ctx.protocol['train_start'])
    cutoff=ctx.protocol['cutoff_elapsed_seconds'];end=clock.seconds(ctx.protocol['forecast_end_exclusive'])
    users={u:j for j,u in enumerate(ctx.targets)}; groups=defaultdict(list)
    raw=Path(getattr(ctx, 'raw_input_dir', ROOT/'data/raw'))/'dataset11-30.csv'
    if sha256(raw)!=ctx.protocol['input_hashes'][raw.name]:raise ValueError('Raw viewing log hash mismatch')
    with raw.open(encoding='utf-8-sig',newline='') as stream:
        for row in csv.DictReader(stream):
            if row['user_id'] not in users or row['start_time']>=ctx.protocol['train_cutoff_exclusive'] or row['stop_time']<=ctx.protocol['train_start']:
                continue
            a,b=max(start,clock.seconds(row['start_time'])),min(cutoff,clock.seconds(row['stop_time']))
            if a<b:groups[(row['user_id'],row['vsetv_id'])].append((a,b))
    rr,cc,vv,ages=[],[],[],[];maxstop=0;unions=0
    for (user,channel),intervals in groups.items():
        for a,b in clip_union(intervals,start,cutoff):
            assert b<=cutoff;maxstop=max(maxstop,b);unions+=1
            for weekday,_,secs,segstart in coarse.segments(clock,a,b):
                hour=clock.local(segstart).hour
                rr.append(users[user]);cc.append(cm[channel]*168+weekday*24+hour)
                vv.append(secs);ages.append((cutoff-segstart-secs)/86400)
    np.savez_compressed(cache/'train-hour-segments.npz',rows=np.array(rr,np.int32),cols=np.array(cc,np.int32),
                        seconds=np.array(vv,np.float32),age_days=np.array(ages,np.float64))
    r,c,v=[],[],[]
    with np.load(coarse_path/'future-airings.npz',allow_pickle=False) as z:
        for item,ch,a,b in zip(z['item'],z['channel'],z['start'],z['stop']):
            for wd,_,secs,segstart in coarse.segments(clock,max(a,cutoff),min(b,end)):
                r.append(item);c.append(ch*168+wd*24+clock.local(segstart).hour);v.append(secs/(b-a))
    sparse.save_npz(cache/'future-hour-counts.npz',sparse.csr_matrix((v,(r,c)),
                    shape=(len(ctx.candidates),len(channels)*168),dtype=np.float32))
    dump(cache/'manifest.json',{'protocol_sha256':ctx.protocol_hash,'source_sha256':SOURCE_SHA256,
        'semantic_version':SEMANTIC_VERSION,'coarse_source_sha256':coarse.BUILDER_SOURCE_SHA256,
        'coarse_manifest_sha256':sha256(coarse_path/'manifest.json'),'coarse_cache':str(coarse_path),
        'channels':channels,'unions':unions,'max_training_stop':maxstop,
        'input_hashes':ctx.protocol['input_hashes'],
        'cache_data_sha256':{f.name:sha256(f) for f in sorted(cache.glob('*.npz'))}})


def _marginal(joint,channels,axis):
    width=channels*168;ids=np.arange(width)
    ch=ids//168; wd=(ids%168)//24; hour=ids%24
    if axis=='hour':dest=ch*24+hour;small=channels*24;scale=1/7
    elif axis=='weekday':dest=ch*7+wd;small=channels*7;scale=1/24
    else:dest=ch;small=channels;scale=1/168
    reduce=sparse.csr_matrix((np.full(width,scale,np.float32),(ids,dest)),shape=(width,small))
    expand=sparse.csr_matrix((np.ones(width,np.float32),(dest,ids)),shape=(small,width))
    return (joint@reduce@expand).tocsr()


def smooth_hours(profile,channels,weight):
    """Adjacent hourly bins wrap across days and Sunday/Monday, not within a day."""
    width=channels*168;ids=np.arange(width);base=(ids//168)*168;t=ids%168
    rows=np.concatenate([ids,ids,ids]);cols=np.concatenate([ids,base+(t-1)%168,base+(t+1)%168])
    values=np.concatenate([np.full(width,1-weight),np.full(width,weight/2),np.full(width,weight/2)])
    shift=sparse.csr_matrix((values,(rows,cols)),shape=(width,width),dtype=np.float32)
    return (profile@shift).tocsr()


def schedule_hourly(ctx,p,path):
    cache=ROOT/'artifacts/runs/hourly-schedule-cache-v1'/ctx.protocol_hash/SOURCE_SHA256/coarse.BUILDER_SOURCE_SHA256
    if not (cache/'manifest.json').exists():
        cache.mkdir(parents=True,exist_ok=True);_build(ctx,path,cache)
    manifest=load(cache/'manifest.json')
    if manifest['protocol_sha256']!=ctx.protocol_hash or manifest['source_sha256']!=SOURCE_SHA256 or manifest['semantic_version']!=SEMANTIC_VERSION or manifest['coarse_source_sha256']!=coarse.BUILDER_SOURCE_SHA256:
        raise ValueError('Hourly schedule cache provenance mismatch')
    if manifest['input_hashes']!=ctx.protocol['input_hashes']:raise ValueError('Hourly cache input mismatch')
    if sha256(ROOT/manifest['coarse_cache']/'manifest.json')!=manifest['coarse_manifest_sha256']:
        raise ValueError('Hourly source inventory changed')
    for name,expected in manifest['cache_data_sha256'].items():
        if sha256(cache/name)!=expected:raise ValueError('Hourly schedule cache artifact changed: '+name)
    assert manifest['max_training_stop']<=ctx.protocol['cutoff_elapsed_seconds']
    with np.load(cache/'train-hour-segments.npz',allow_pickle=False) as z:
        seconds=z['seconds']
        if p.get('half_life_days'):seconds=seconds*np.exp2(-z['age_days']/p['half_life_days'])
        channels=len(manifest['channels'])
        joint=sparse.csr_matrix((seconds,(z['rows'],z['cols'])),shape=(len(ctx.targets),channels*168),dtype=np.float32)
    cw=p.get('ct_weight',.7)
    marginal=.9*(cw*_marginal(joint,channels,'hour')+(1-cw)*_marginal(joint,channels,'weekday'))+.1*_marginal(joint,channels,'channel')
    profile=p.get('joint_weight',.5)*joint+(1-p.get('joint_weight',.5))*marginal
    if p.get('neighbor_smooth'):profile=smooth_hours(profile,channels,p['neighbor_smooth'])
    profile=normalize(profile,norm='l1')
    scores=(profile@sparse.load_npz(cache/'future-hour-counts.npz').T).toarray()
    scores[scores<=0]=np.nan
    dump(path/'hourly-cache-reference.json',{'path':str(cache),'manifest_sha256':sha256(cache/'manifest.json')})
    return scores.astype(np.float32),{'algorithm':'raw union channel×weekday×hour; sparse marginal and adjacent-hour smoothing',
        'fit_cutoff':ctx.protocol['train_cutoff_exclusive'],'cache':str(cache),'max_training_stop':manifest['max_training_stop'],
        'future_features':'ID/channel/start/duration only; per-airing weighted counts','future_views_used':False,'future_content_used':False,
        'joint_profile_nnz':joint.nnz,'final_profile_nnz':profile.nnz,'union_intervals':manifest['unions']}


def register(methods):methods['schedule_hourly']=schedule_hourly
