"""Historical non-repeat expert with conditional IP sampling and common labels.

This is a piecewise binary relevance predictor, not a normalized mode decoder.
Both heads predict inclusion in the SAME historical top-five truth. No quota and
no outer validation labels enter fit or prediction. Cross-head calibration is an
empirical limitation; no claim of calibrated probabilities is made.
"""
from pathlib import Path
import hashlib
import numpy as np
from etl import read_events
from models import Context
from repro import ROOT, dump, load, sha256
from advanced_depth6 import load_engine, ENGINE_PATH, ENGINE_SHA256


def sample_discovery(features, rng, negatives=30):
    x=[]; y=[]; groups=[]; weights=[]; audit=[]
    for j,user in enumerate(features.ctx.targets):
        truth=features.ctx.truth[user]
        if not truth or features.ctx.cold_mask[j]:
            continue
        eligible=np.flatnonzero(features.count[j] == 0)
        domain=set(eligible.tolist())
        if not domain:
            continue
        ids={features.c[i] for i in truth if i in features.c}&domain
        for signal,k in zip(features.retrievers,(100,80,60,40,40)):
            ids.update(set(np.argpartition(signal[j],-k)[-k:].tolist())&domain)
        outside=np.array(sorted(domain-ids),dtype=np.int32)
        random_ids=rng.choice(outside,min(negatives,len(outside)),replace=False)
        sampled_weight=len(outside)/len(random_ids) if len(random_ids) else 1.
        random_ids=set(random_ids.tolist());ids.update(random_ids)
        ids=np.array(sorted(ids),dtype=np.int32)
        if not len(ids):
            continue
        target=set(truth)
        labels=np.array([float(features.ctx.candidates[i] in target) for i in ids],np.float32)
        weight=np.array([sampled_weight if i in random_ids else 1. for i in ids],np.float32)
        assert np.all(features.count[j,ids] == 0)
        assert np.all(weight[labels>0] == 1)
        assert np.isclose(weight.sum(),len(domain),rtol=1e-6)
        x.append(features.matrix(j,ids));y.append(labels);groups.append(np.full(len(ids),j,np.int64));weights.append(weight)
        audit.append(dict(user=user,eligible=len(domain),sampled=len(ids),positive=int(labels.sum()),outside=len(outside),random_sample=len(random_ids),weight_sum=float(weight.sum())))
    return np.vstack(x),np.concatenate(y),np.concatenate(groups),np.concatenate(weights),audit


def specialist(ctx,p,path):
    from catboost import CatBoostClassifier,Pool
    from advanced_full_fusion import prepare_scores
    cache,manifest=prepare_scores(ctx)
    baseline=np.load(cache/'l5.npy',allow_pickle=False)
    engine=load_engine(); rng=np.random.default_rng(p.get('seed',42));blocks=[];sources=[]
    for name in p['folds']:
        folder=ROOT/Path(p['fold_root'])/name; train=Context(folder)
        assert train.protocol['forecast_end_exclusive'] <= ctx.protocol['train_cutoff_exclusive']
        _,_,_,events=read_events(folder/'validation-events.npz')
        assert not len(events) or events[:,6].max() <= ctx.protocol['cutoff_elapsed_seconds']
        f=engine(train,p,path/'feature-sources'/name)
        names=f.names
        x,y,g,w,audit=sample_discovery(f,rng,p.get('random_negatives',30))
        blocks.append((x,y,w))
        dump(path/f'{name}-sampling-audit.json',audit)
        sources.append(dict(fold=name,protocol_sha256=train.protocol_hash,truth_sha256=sha256(folder/'truth.json'),
            label_end=train.protocol['forecast_end_exclusive'],input_hashes=train.protocol['input_hashes'],
            rows=len(y),positives=int(y.sum()),sampling_audit_sha256=sha256(path/f'{name}-sampling-audit.json'),
            feature_sources=f.details))
        del f,train,events,g
    x,y,w=(np.concatenate([b[k] for b in blocks]) for k in range(3));del blocks
    config=dict(iterations=300,depth=6,learning_rate=.05,l2_leaf_reg=10,random_seed=42,thread_count=1,
                loss_function='Logloss',allow_writing_files=False,verbose=100)
    model=CatBoostClassifier(**config);model.fit(Pool(x,y,weight=w,feature_names=names))
    model.save_model(str(path/'discovery.cbm'))
    dump(path/'training-provenance.json',dict(config=config,sources=sources,engine_sha256=ENGINE_SHA256,
        source_sha256=sha256(__file__),features=names,rows=len(y),positives=int(y.sum()),
        array_sha256={k:hashlib.sha256(a.tobytes()).hexdigest() for k,a in zip(('x','y','weights'),(x,y,w))},
        outer_truth_used=False,sampling='uniform negatives in nonrepeat domain outside deterministic candidates; inverse inclusion weights',
        target='same historical top-five inclusion indicator as L5; restriction to train-known nonrepeat domain',
        baseline_cache_manifest_sha256=sha256(cache/'manifest.json'),baseline_scores_sha256=sha256(cache/'l5.npy')))
    del x,y,w
    f=engine(ctx,p,path/'feature-sources'/'evaluation-history-only')
    assert f.names==names
    restored=CatBoostClassifier();restored.load_model(str(path/'discovery.cbm'))
    scores=baseline.copy();mask=f.count==0
    for j in range(len(ctx.targets)):
        ids=np.flatnonzero(mask[j]);matrix=f.matrix(j,ids)
        if len(ids):
            scores[j,ids]=model.predict_proba(matrix)[:,1]
            if j in (0,len(ctx.targets)-1):
                np.testing.assert_array_equal(model.predict_proba(matrix),restored.predict_proba(matrix))
    np.save(path/'piecewise-scores.npy',scores,allow_pickle=False)
    np.save(path/'nonrepeat-mask.npy',mask,allow_pickle=False)
    dump(path/'score-manifest.json',dict(protocol_sha256=ctx.protocol_hash,baseline=str(cache/'l5.npy'),
        baseline_sha256=sha256(cache/'l5.npy'),scores_sha256=sha256(path/'piecewise-scores.npy'),mask_sha256=sha256(path/'nonrepeat-mask.npy')))
    return scores,dict(expert='conditional nonrepeat binary classifier, L5 unchanged on repeats',
        no_forced_discovery_quota=True,calibration='common target and IP correction; cross-head calibration not guaranteed',
        evaluation_truth_used_for_fit=False,model_reload_verified=True)


def specialist_interpolation(ctx,p,path):
    source=path.parent/p['source_run']; manifest=load(source/'score-manifest.json')
    assert manifest['protocol_sha256']==ctx.protocol_hash
    for file,key in ((source/'piecewise-scores.npy','scores_sha256'),(source/'nonrepeat-mask.npy','mask_sha256'),(Path(manifest['baseline']),'baseline_sha256')):
        assert sha256(file)==manifest[key]
    result=np.load(source/'piecewise-scores.npy',allow_pickle=False)
    mask=np.load(source/'nonrepeat-mask.npy',allow_pickle=False)
    baseline=np.load(manifest['baseline'],allow_pickle=False)
    weight=p['weight']; assert 0<=weight<=1
    def logit(x):
        x=np.clip(x.astype(np.float64),1e-9,1-1e-9)
        return np.log(x)-np.log1p(-x)
    combined=(1-weight)*logit(baseline[mask])+weight*logit(result[mask])
    result[mask]=1/(1+np.exp(-combined))
    np.testing.assert_array_equal(result[~mask],baseline[~mask])
    return result,dict(source_manifest_sha256=sha256(source/'score-manifest.json'),
        weight=weight,interpolation='nonrepeat logits only; repeat scores identical to L5')


def register():
    from models import METHODS
    METHODS['discovery_specialist']=specialist
    METHODS['discovery_specialist_interpolation']=specialist_interpolation
