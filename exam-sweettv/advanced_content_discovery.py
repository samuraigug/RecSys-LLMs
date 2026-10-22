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
from advanced_depth6 import load_engine as load_base_engine, ENGINE_PATH, ENGINE_SHA256


BLOCKS = ('category', 'genres', 'actors', 'director')

def add_content_features(f, feature_path):
    from scipy import sparse
    from sklearn.preprocessing import normalize
    from sklearn.feature_extraction.text import TfidfTransformer
    from models import content_blocks
    ctx=f.ctx
    eu,ei,_,events=read_events(ctx.path/'train-events.npz',full_only=False)
    cutoff=ctx.protocol['cutoff_elapsed_seconds']
    if len(events) and (events[:,6].max()>cutoff or events[:,3].max()>cutoff):
        raise ValueError('Post-cutoff content history')
    items=sorted(ctx.metadata,key=int); imap={i:j for j,i in enumerate(items)}
    ur=np.array([f.u.get(u,-1) for u in eu])[events[:,0]]
    ir=np.array([imap.get(i,-1) for i in ei])[events[:,1]]
    valid=(ur>=0)&(ir>=0)
    coverage=events[:,5]/events[:,7]
    # A complete airing contributes one; a weak airing contributes coverage^2.
    # 56-day decay and log1p repeated evidence reduce dominance of long serials.
    weight=np.where(coverage>=.8,1.,coverage**2)*np.exp2(-(cutoff-events[:,6])/86400/56)
    hist=sparse.csr_matrix((weight[valid],(ur[valid],ir[valid])),shape=(f.n,len(items)),dtype=np.float32)
    hist.data=np.log1p(hist.data)
    schemas={}
    for field in BLOCKS:
        block_items,vectors,schema=content_blocks(ctx.metadata,{field:1.})
        assert block_items==items
        idf=TfidfTransformer(norm='l2',smooth_idf=True)
        vectors=idf.fit_transform(vectors).astype(np.float32)
        profile=normalize(hist@vectors)
        cand=sparse.vstack([vectors[imap[i]] if i in imap else sparse.csr_matrix((1,vectors.shape[1])) for i in ctx.candidates],format='csr')
        scores=(profile@cand.T).toarray().astype(np.float32)
        f.add('content_'+field+'_score',scores)
        f.add('content_'+field+'_rank',f.rank(scores))
        schemas[field]=dict(schema=schema[field],idf=idf.idf_.tolist())
    feature_path=Path(feature_path);feature_path.mkdir(parents=True,exist_ok=True)
    dump(feature_path/'content-schema.json',dict(blocks=schemas,metadata_sha256=sha256(ctx.path/'metadata-train.json'),
        events_sha256=sha256(ctx.path/'train-events.npz'),cutoff=cutoff,last_stop=int(events[:,6].max()) if len(events) else None,
        history='full1 else coverage^2; decay56days; log1p per user-programme; l2 profile',
        source_sha256=sha256(__file__),features=f.names))
    f.details['content_blocks']=dict(schema_sha256=sha256(feature_path/'content-schema.json'),blocks=list(BLOCKS),
        future_metadata_used=False,evaluation_truth_used=False)
    return f

def load_engine():
    base=load_base_engine()
    def make(ctx,p,path):
        return add_content_features(base(ctx,p,path),path)
    return make


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
    METHODS['content_discovery_specialist']=specialist
    METHODS['content_discovery_interpolation']=specialist_interpolation
