"""Chronological user-level AP gate between three fixed unsupervised experts."""
from pathlib import Path
import numpy as np
from etl import Clock,read_events
from models import Context
from protocol import ap_at_k
from repro import ROOT,load,dump,sha256
from advanced_full_fusion import max_fusion,rrf,PPR,HOURLY
import advanced_collaborative as collaborative
import advanced_hourly_schedule as hourly

EXPERTS=['core','ppr','hourly']


def fixed_experts(ctx,p,path):
    if 'training_raw_dir' in p:
        raw=ROOT/p['training_raw_dir']
        assert {name:sha256(raw/name) for name in ctx.protocol['input_hashes']}==ctx.protocol['input_hashes']
        ctx.raw_input_dir=raw
    path.mkdir(parents=True,exist_ok=True)
    values={};details={}
    for key,fn,params in [('ppr',collaborative.ppr,PPR),('hourly',hourly.schedule_hourly,HOURLY)]:
        folder=path/key;folder.mkdir(exist_ok=True)
        values[key],details[key]=fn(ctx,params,folder)
    values['core']=max_fusion([(values['ppr'],1.),(values['hourly'],1.)])
    return [values[key] for key in EXPERTS],details


def entropy(values):
    values=np.asarray(values,dtype=float);s=values.sum()
    if s<=0:return 0.
    q=values[values>0]/s
    return float(-np.sum(q*np.log(q)))


def gate_features(ctx,experts,predictions):
    # Deliberately no ctx.truth or display access.
    eu,ei,channels,e=read_events(ctx.path/'train-events.npz',full_only=False)
    assert not len(e) or e[:,6].max()<=ctx.protocol['cutoff_elapsed_seconds']
    um={u:j for j,u in enumerate(ctx.targets)};mapping=np.array([um.get(u,-1) for u in eu])
    rows=mapping[e[:,0]];keep=rows>=0;e=e[keep];rows=rows[keep]
    order=np.argsort(rows,kind='stable');e=e[order];rows=rows[order]
    bounds=np.searchsorted(rows,np.arange(len(ctx.targets)+1))
    cutoff=ctx.protocol['cutoff_elapsed_seconds'];clock=Clock(ctx.protocol['clock'])
    train_days=(cutoff-clock.seconds(ctx.protocol['train_start']))/86400
    forecast_days=(clock.seconds(ctx.protocol['forecast_end_exclusive'])-cutoff)/86400
    names=['log_full_count','log_unique_full','programme_entropy','max_programme_share','last_full_age',
           'log_active_full_days','recent7_full_fraction','recent28_full_fraction','weak_coverage_per_full',
           'full_fraction_of_events','channel_entropy','train_days','forecast_days','horizon_ratio','log_history_available']
    for key in EXPERTS:names.extend([key+'_log_positive_coverage',key+'_top5_repeat_share',key+'_relative_topgap',key+'_score_entropy'])
    names.append('ppr_hourly_top5_overlap')
    features=[]
    for j,u in enumerate(ctx.targets):
        hist=ctx.history.get(u,{});count=np.array([x['airings'] for x in hist.values()],float);total=count.sum()
        events=e[bounds[j]:bounds[j+1]];full=events[:,5]*5>=events[:,7]*4
        fullstops=events[full,6];age=(cutoff-fullstops)/86400
        coverage=events[:,5]/events[:,7]
        channelmass=np.bincount(events[:,2],weights=events[:,5],minlength=len(channels))
        values=[np.log1p(total),np.log1p(len(hist)),entropy(count),float(count.max(initial=0)/max(total,1)),
            float(age.min()) if len(age) else 1000.,np.log1p(len(np.unique(fullstops//86400))),
            float(np.mean(age<=7)) if len(age) else 0.,float(np.mean(age<=28)) if len(age) else 0.,
            float(np.sum(coverage[~full]**2)/max(total,1)),float(np.mean(full)) if len(full) else 0.,
            entropy(channelmass),train_days,forecast_days,forecast_days/max(train_days,1),
            np.log1p(sum(i in ctx.candidate_map for i in hist))]
        for index,key in enumerate(EXPERTS):
            row=experts[index][j];positive=row[np.isfinite(row)&(row>0)]
            top=np.sort(positive)[-5:]
            values.extend([np.log1p(len(positive)),sum(i in hist for i in predictions[index][u])/5,
                float((top[-1]-top[0])/top[-1]) if len(top) else 0.,entropy(positive)])
        values.append(len(set(predictions[1][u])&set(predictions[2][u]))/5)
        features.append(values)
    x=np.asarray(features,np.float32);assert x.shape==(len(ctx.targets),len(names)) and np.isfinite(x).all()
    return x,names


def choose_rows(experts,predicted_ap):
    predicted_ap=np.asarray(predicted_ap)
    if predicted_ap.shape!=(experts[0].shape[0],len(experts)) or not np.isfinite(predicted_ap).all():
        raise ValueError('Invalid gate predictions')
    choice=np.argmax(predicted_ap,axis=1)
    result=np.empty_like(experts[0])
    for k,score in enumerate(experts):result[choice==k]=score[choice==k]
    return result,choice


def user_gate(ctx,p,path):
    from catboost import CatBoostRegressor
    from experiments import finalize_scores
    from advanced_full_fusion import prepare_scores
    blocks=[];sources=[];feature_names=None
    for fold in p['folds']:
        train=Context(ROOT/p['fold_root']/fold)
        assert train.protocol['forecast_end_exclusive']<=ctx.protocol['train_cutoff_exclusive']
        _,_,_,labels=read_events(train.path/'validation-events.npz')
        assert not len(labels) or labels[:,6].max()<=ctx.protocol['cutoff_elapsed_seconds']
        experts,details=fixed_experts(train,p,path/'expert-sources'/fold)
        predictions=[finalize_scores(s.copy(),train)[0] for s in experts]
        x,names=gate_features(train,experts,predictions)
        assert feature_names is None or feature_names==names
        feature_names=names
        y=np.array([[ap_at_k(train.truth[u],prediction[u]) for prediction in predictions] for u in train.targets],np.float32)
        keep=np.array([bool(train.truth[u]) for u in train.targets])&~train.cold_mask
        blocks.append((x[keep],y[keep]))
        np.savez_compressed(path/f'{fold}-gate-data.npz',x=x[keep],y=y[keep],users=np.asarray(train.targets)[keep])
        sources.append(dict(fold=fold,protocol_sha256=train.protocol_hash,truth_sha256=sha256(train.path/'truth.json'),
            label_end=train.protocol['forecast_end_exclusive'],raw_hashes=train.protocol['input_hashes'],rows=int(keep.sum()),
            expert_details=details,gate_data_sha256=sha256(path/f'{fold}-gate-data.npz')))
        del train,experts,predictions,x,y,labels
    x=np.vstack([a for a,b in blocks]);y=np.vstack([b for a,b in blocks]);del blocks
    config=dict(iterations=300,depth=4,learning_rate=.05,l2_leaf_reg=10,random_seed=42,
                thread_count=1,loss_function='MultiRMSE',allow_writing_files=False,verbose=100)
    model=CatBoostRegressor(**config);model.fit(x,y);model.save_model(str(path/'gate.cbm'))
    dump(path/'gate-training-provenance.json',dict(config=config,experts=EXPERTS,features=feature_names,sources=sources,
        source_sha256=sha256(__file__),training_rows=len(y),outer_truth_used=False,
        supervised_heads_in_historical_targets=False,source_hashes={m.__name__:sha256(m.__file__) for m in (collaborative,hourly)},
        ppr_parameters=PPR,hourly_parameters=HOURLY))
    cache,_=prepare_scores(ctx)
    ppr=np.load(cache/'ppr.npy');h=np.load(cache/'hourly.npy')
    experts=[max_fusion([(ppr,1.),(h,1.)]),ppr,h]
    predictions=[finalize_scores(s.copy(),ctx)[0] for s in experts]
    features,names=gate_features(ctx,experts,predictions);assert names==feature_names
    predicted=model.predict(features);restored=CatBoostRegressor();restored.load_model(str(path/'gate.cbm'))
    np.testing.assert_array_equal(predicted,restored.predict(features))
    scores,choice=choose_rows(experts,predicted)
    np.save(path/'gate-scores.npy',scores,allow_pickle=False)
    np.savez_compressed(path/'gate-evaluation-state.npz',features=features,predicted_ap=predicted,choice=choice)
    details=dict(expert_cache=str(cache),expert_manifest_sha256=sha256(cache/'manifest.json'),
        choice_counts={k:int(np.sum(choice==j)) for j,k in enumerate(EXPERTS)},saved_model_reload_exact=True,
        evaluation_truth_used_for_gate=False,training_rows=len(y),gate_features=len(names),
        caveat='historical short-prefix/horizon shift; overlapping historical folds/users; user AP regression not guaranteed calibrated')
    dump(path/'gate-manifest.json',{**details,'protocol_sha256':ctx.protocol_hash,'scores_sha256':sha256(path/'gate-scores.npy')})
    return scores,details


def user_gate_bundle(ctx,p,path):
    source=path.parent/p['source_run'];manifest=load(source/'gate-manifest.json')
    assert manifest['protocol_sha256']==ctx.protocol_hash and sha256(source/'gate-scores.npy')==manifest['scores_sha256']
    cache=Path(manifest['expert_cache']);assert sha256(cache/'manifest.json')==manifest['expert_manifest_sha256']
    cached=load(cache/'manifest.json');assert sha256(cache/'l4.npy')==cached['payload_sha256']['l4.npy']
    scores=rrf([(np.load(source/'gate-scores.npy'),1.),(np.load(cache/'l4.npy'),.5)],np.array([int(i) for i in ctx.candidates]))
    return scores,dict(source_manifest_sha256=sha256(source/'gate-manifest.json'),composition='full RRF gate1 + frozenL4.5, no L4 historical AP used')


def register():
    from models import METHODS
    METHODS['user_gate']=user_gate
    METHODS['user_gate_bundle']=user_gate_bundle
