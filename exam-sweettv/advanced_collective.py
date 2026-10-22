"""Collective WARP: programme identities and shared raw-view channel features.

Cold programmes have no identity feature. Known future channel/time/duration may
activate features trained by historical programme and auxiliary channel views.
"""
from collections import defaultdict
import csv
import importlib.metadata
import pickle

import numpy as np
from scipy import sparse

from advanced_lightfm import interactions
from etl import Clock, FILES, clip_union
from models import METHODS
from repro import ROOT, dump, load, sha256

SOURCE_SHA = sha256(__file__)


def build_context(ctx, raw):
    """All users' actual pre-cutoff union seconds; no future viewing parsed."""
    clock = Clock(ctx.protocol['clock'])
    start, cutoff, end = clock.seconds(ctx.protocol['train_start']), ctx.protocol['cutoff_elapsed_seconds'], clock.seconds(ctx.protocol['forecast_end_exclusive'])
    groups = defaultdict(list)
    with (raw/'dataset11-30.csv').open(encoding='utf-8-sig', newline='') as stream:
        for row in csv.DictReader(stream):
            if row['start_time'] >= ctx.protocol['train_cutoff_exclusive'] or row['stop_time'] <= ctx.protocol['train_start']:
                continue
            a,b = max(start,clock.seconds(row['start_time'])), min(cutoff,clock.seconds(row['stop_time']))
            if a < b:
                groups[(row['user_id'],row['vsetv_id'])].append((a,b))
    train_items = set(ctx.items)
    past, future = set(), set()
    for filename in FILES:
        with (raw/filename).open(encoding='utf-8-sig',newline='') as stream:
            for row in csv.DictReader(stream):
                if row['tv_show_id']=='0':
                    continue
                # These are the only future schedule fields read.
                item,ch = row['tv_show_id'],row['channel_id']
                a=clock.seconds(row['start_time']);b=a+int(row['duration'])
                if a<cutoff and b>start and item in train_items:
                    past.add((item,ch,a,b))
                if a<end and b>cutoff and item in ctx.candidate_map:
                    future.add((item,ch,a,b))
    users=sorted(set(ctx.targets)|{u for u,c in groups}|{u for u,h in ctx.history.items() if h})
    channels=sorted({c for u,c in groups}|{a[1] for a in past},key=int)
    um,cm={u:j for j,u in enumerate(users)},{c:j for j,c in enumerate(channels)}
    rr,cc,vv=[],[],[]
    max_stop=0
    for (u,ch),intervals in groups.items():
        seconds=0
        for a,b in clip_union(intervals,start,cutoff):
            assert b<=cutoff
            seconds+=b-a;max_stop=max(max_stop,b)
        if seconds:
            rr.append(um[u]);cc.append(cm[ch]);vv.append(seconds)
    raw_seconds=sparse.csr_matrix((vv,(rr,cc)),shape=(len(users),len(channels)),dtype=np.float64)
    return users,channels,raw_seconds,sorted(past),sorted(future),max_stop


def feature_matrix(items, channel_vocab, identity_items, airings, start, end, *, shared=True, exposure=False):
    """Future-only programme IDs never receive an untrained identity column."""
    im,cm={i:j for j,i in enumerate(items)},{c:j for j,c in enumerate(channel_vocab)}
    identity={i:j for j,i in enumerate(identity_items)}
    width=len(identity_items)+(len(channel_vocab) if shared else 0)+(2 if exposure else 0)
    rows,cols,values=[],[],[]
    for item,r in im.items():
        if item in identity:
            rows.append(r);cols.append(identity[item]);values.append(1.)
    channel_counts=defaultdict(lambda:defaultdict(float))
    counts,duration=np.zeros(len(items)),np.zeros(len(items))
    for item,ch,a,b in airings:
        if item not in im or a>=end or b<=start:
            continue
        r=im[item];fraction=(min(b,end)-max(a,start))/(b-a)
        counts[r]+=fraction;duration[r]+=fraction*(b-a)
        if ch in cm:
            channel_counts[r][cm[ch]]+=fraction
    if shared:
        for r,weights in channel_counts.items():
            total=sum(weights.values())
            for ch,value in weights.items():
                rows.append(r);cols.append(len(identity_items)+ch);values.append(value/total)
    if exposure:
        # Learned numeric availability features; identical definitions in past
        # fitting and future scoring, fitted latent directions only from history.
        days=(end-start)/86400
        for r in range(len(items)):
            if counts[r]:
                rows.extend([r,r]);cols.extend([width-2,width-1])
                values.extend([np.log1p(counts[r]/days),np.log1p(duration[r]/counts[r]/3600)])
    return sparse.csr_matrix((values,(rows,cols)),shape=(len(items),width),dtype=np.float32)


def _context(ctx):
    cache=ROOT/'artifacts/runs/collective-cache-v1'/ctx.protocol_hash/SOURCE_SHA
    if not (cache/'manifest.json').exists():
        raw=ROOT/'data/raw'
        actual={n:sha256(raw/n) for n in ctx.protocol['input_hashes']}
        if actual!=ctx.protocol['input_hashes']:
            raise ValueError('Raw data changed after protocol freeze')
        cache.mkdir(parents=True,exist_ok=True)
        users,channels,seconds,past,future,maxstop=build_context(ctx,raw)
        dump(cache/'schema.json',{'users':users,'channels':channels,'past':past,'future':future})
        sparse.save_npz(cache/'raw-channel-seconds.npz',seconds)
        dump(cache/'manifest.json',{'protocol_sha256':ctx.protocol_hash,'source_sha256':SOURCE_SHA,
            'raw_hashes':actual,'max_training_stop':maxstop,
            'files':{p.name:sha256(p) for p in cache.iterdir() if p.is_file()}})
    manifest=load(cache/'manifest.json')
    assert manifest['protocol_sha256']==ctx.protocol_hash and manifest['source_sha256']==SOURCE_SHA
    assert manifest['max_training_stop']<=ctx.protocol['cutoff_elapsed_seconds']
    assert manifest['raw_hashes']==ctx.protocol['input_hashes']
    assert all(sha256(cache/n)==h for n,h in manifest['files'].items())
    schema=load(cache/'schema.json')
    return schema,sparse.load_npz(cache/'raw-channel-seconds.npz'),manifest


def collective_warp(ctx,p,path):
    from lightfm import LightFM
    schema,raw_seconds,manifest=_context(ctx)
    users,channels=schema['users'],schema['channels']
    user_map={u:j for j,u in enumerate(users)}
    old_users,programmes,positive,weights,detail=interactions(ctx,{'weighting':'log_count'})
    rr=np.array([user_map[old_users[u]] for u in weights.row],np.int32)
    cc=weights.col.copy();vv=weights.data.copy()
    shared=p.get('shared_channels',True)
    exposure=p.get('exposure_features',False)
    clock=Clock(ctx.protocol['clock']);start=clock.seconds(ctx.protocol['train_start']);cutoff=ctx.protocol['cutoff_elapsed_seconds'];end=clock.seconds(ctx.protocol['forecast_end_exclusive'])
    programme_features=feature_matrix(programmes,channels,programmes,schema['past'],start,cutoff,shared=shared,exposure=exposure)
    candidate_features=feature_matrix(ctx.candidates,channels,programmes,schema['future'],cutoff,end,shared=shared,exposure=exposure)
    if shared:
        channel_features=sparse.hstack([sparse.csr_matrix((len(channels),len(programmes))),sparse.eye(len(channels),format='csr'),sparse.csr_matrix((len(channels),2 if exposure else 0))],format='csr',dtype=np.float32)
        features=sparse.vstack([programme_features,channel_features],format='csr')
        auxiliary=raw_seconds.tocoo()
        rr=np.concatenate([rr,auxiliary.row]);cc=np.concatenate([cc,auxiliary.col+len(programmes)])
        vv=np.concatenate([vv,p.get('auxiliary_weight',.1)*np.log1p(auxiliary.data/3600)])
    else:
        features=programme_features
    weighted=sparse.csr_matrix((vv,(rr,cc)),shape=(len(users),features.shape[0]),dtype=np.float32)
    weighted.eliminate_zeros();sample=weighted.tocoo();positive=sample.copy();positive.data=np.ones(positive.nnz,np.float32)
    model=LightFM(loss='warp',no_components=p.get('factors',32),random_state=p.get('seed',42),learning_rate=.05,max_sampled=10)
    model.fit(positive,sample_weight=sample,item_features=features,epochs=p.get('epochs',15),num_threads=1)
    # Mask channels which never occurred in a positive training representation.
    active=np.asarray(features[weighted.getnnz(axis=0)>0].sum(axis=0)).ravel()>0
    candidate_features=candidate_features.multiply(active[None,:]).tocsr()
    valid=np.asarray(candidate_features.getnnz(axis=1)).ravel()>0
    ub,uf=model.get_user_representations();ib,it=model.get_item_representations(candidate_features)
    tr=np.array([user_map[u] for u in ctx.targets],np.int32)
    scores=(uf[tr]@it.T+ib[None,:]+ub[tr,None]).astype(np.float32)
    scores[:,~valid]=np.nan
    warm=np.asarray(weighted.getnnz(axis=1)).ravel()>0
    scores[~warm[tr]]=np.nan
    probe=np.flatnonzero(valid)[:20].astype(np.int32);user=int(tr[np.flatnonzero(warm[tr])[0]])
    native=model.predict(user,probe,item_features=candidate_features,num_threads=1)
    np.testing.assert_allclose(native,uf[user]@it[probe].T+ib[probe]+ub[user],rtol=1e-5,atol=1e-5)
    with (path/'collective-model.pickle').open('wb') as f:pickle.dump(model,f,pickle.HIGHEST_PROTOCOL)
    with (path/'collective-model.pickle').open('rb') as f:restored=pickle.load(f)
    np.testing.assert_array_equal(native,restored.predict(user,probe,item_features=candidate_features,num_threads=1))
    sparse.save_npz(path/'candidate-features.npz',candidate_features)
    sparse.save_npz(path/'train-features.npz',features)
    sparse.save_npz(path/'sample-weights.npz',weighted)
    dump(path/'collective-mapping.json',{'users':users,'identity_programmes':programmes,'shared_channels':channels,'candidate_items':ctx.candidates})
    cold=np.array([i not in set(programmes) for i in ctx.candidates])
    assert candidate_features[cold,:len(programmes)].nnz==0
    return scores,{**detail,'algorithm':'collective identity/channel-feature WARP','shared_channels':shared,
        'auxiliary_weight':p.get('auxiliary_weight',.1) if shared else 0,'exposure_features':exposure,
        'shape':list(weighted.shape),'nnz':weighted.nnz,'epochs':p.get('epochs',15),'seed':p.get('seed',42),
        'future_views_used':False,'future_content_used':False,'future_information':'ID/channel/start/duration only',
        'cold_programme_identity_nnz':0,'cold_programmes_with_trained_shared_features':int(np.sum(cold&valid)),
        'cold_programmes':int(cold.sum()),'raw_train_seconds':float(raw_seconds.sum()),
        'raw_context_users':len(users),'max_training_stop':manifest['max_training_stop'],
        'saved_model_reload_verified':True,'library_predict_crosscheck':True,'lightfm_version':importlib.metadata.version('lightfm')}


def register():
    METHODS['collective_warp']=collective_warp
