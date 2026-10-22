"""Empirical-Bayes Beta-Binomial future counts and greedy expected AP.

All posterior observations and priors use historical full watches/opportunities.
Future opportunity counts and airing durations are available schedule features.
Simulated relevance is a surrogate: no actual future truth is used to fit/select.
"""
import numpy as np
import advanced_schedule as coarse
from repro import ROOT,dump,load,sha256


def greedy_expected_ap(samples,candidates):
    """Exact greedy increment under the supplied finite sampled relevance sets."""
    samples=np.asarray(samples,np.int32)
    valid=samples>=0;den=np.maximum(valid.sum(1),1)
    ids=samples[valid];row=np.broadcast_to(np.arange(len(samples))[:,None],samples.shape)[valid]
    selected=[];hit=np.zeros(len(samples),np.int32)
    marginal=np.bincount(ids,weights=1/den[row],minlength=candidates)/len(samples)
    for r in range(1,6):
        utility=np.bincount(ids,weights=(1+hit[row])/den[row],minlength=candidates)/len(samples)/r
        utility[selected]=-np.inf
        # candidates are canonical ascending numeric item IDs in this caller.
        choice=int(np.argmax(utility));selected.append(choice)
        hit += np.any(samples==choice,axis=1)
    scores=marginal.astype(np.float32)
    for r,item in enumerate(selected):scores[item]=6-r
    return scores,selected


def select_sample_sets(counts,duration,numeric):
    # For equal count, simulated seconds=count*mean duration. Counts>0 share
    # the same tie order as mean duration; numeric ID breaks exact ties.
    order=np.lexsort((numeric,-duration));rank=np.empty(len(order),np.int64);rank[order]=np.arange(len(order))
    keys=counts.astype(np.int64)*(len(order)+1)+(len(order)-rank)[None,:]
    ix=np.argpartition(keys,-5,axis=1)[:,-5:]
    ix=np.take_along_axis(ix,np.argsort(-np.take_along_axis(keys,ix,axis=1),axis=1),axis=1)
    return np.where(np.take_along_axis(counts,ix,axis=1)>0,ix,-1).astype(np.int32)


def posterior_predictive(ctx,p,path):
    samples=int(p.get('samples',256));seed=int(p.get('seed',42));strength=50.
    if samples<1:raise ValueError('Positive MC count required')
    opp=load(ctx.path/'opportunities.json');m=len(ctx.candidates)
    train=np.array([opp['train'].get(i,0) for i in ctx.candidates],np.float64)
    future=np.array([opp['future'].get(i,0) for i in ctx.candidates],np.int64)
    all_opportunities=float(sum(opp['train'].values()))
    warm_hist=[h for h in ctx.history.values() if h]
    nusers=len(warm_hist);global_counts={};total=0
    for h in warm_hist:
        for item,v in h.items():
            global_counts[item]=global_counts.get(item,0)+v['airings'];total+=v['airings']
    population_rate=total/(nusers*all_opportunities)
    gc=np.array([global_counts.get(i,0) for i in ctx.candidates],np.float64)
    item_rate=(gc+50*population_rate)/(nusers*train+50)
    user_totals=np.array([sum(v['airings'] for v in ctx.history.get(u,{}).values()) for u in ctx.targets],np.float64)
    user_rate=user_totals/all_opportunities
    prior=np.clip(user_rate[:,None]*item_rate[None,:]/population_rate,1e-9,1-1e-9)
    successes=np.zeros(prior.shape,np.float64)
    for j,u in enumerate(ctx.targets):
        for i,v in ctx.history.get(u,{}).items():
            if i in ctx.candidate_map:successes[j,ctx.candidate_map[i]]=v['airings']
    assert np.all(successes<=train[None,:]) and np.all(successes>=0)
    alpha=successes+strength*prior;beta=train[None,:]-successes+strength*(1-prior)
    means=(future[None,:]*alpha/(alpha+beta)).astype(np.float32)
    # Reuse validated schedule inventory for original duration only.
    coarse.schedule_joint(ctx,dict(joint_weight=.5),path)
    cp=ROOT/load(path/'schedule-cache-reference.json')['path'];cm=load(cp/'manifest.json')
    for name,h in cm['cache_data_sha256'].items():
        assert sha256(cp/name)==h
    with np.load(cp/'future-airings.npz',allow_pickle=False) as z:
        count=np.bincount(z['item'],minlength=m)
        duration=np.bincount(z['item'],weights=z['stop']-z['start'],minlength=m)/np.maximum(count,1)
    np.testing.assert_array_equal(count,future)
    numeric=np.array([int(i) for i in ctx.candidates]);assert np.all(np.diff(numeric)>0)
    simulated=np.full((len(ctx.targets),samples,5),-1,np.int32);ranking=np.empty_like(means)
    sampled_means=np.zeros(m,np.float64);expected_means=np.zeros(m,np.float64)
    for j,u in enumerate(ctx.targets):
        rng=np.random.default_rng(np.random.SeedSequence([seed,j]))
        probabilities=rng.beta(alpha[j],beta[j],size=(samples,m))
        counts=rng.binomial(future[None,:],probabilities)
        simulated[j]=select_sample_sets(counts,duration,numeric)
        ranking[j],_=greedy_expected_ap(simulated[j],m)
        sampled_means+=counts.mean(0);expected_means+=means[j]
        if j%100==0:print(f'posterior predictive {j}/{len(ctx.targets)}',flush=True)
    np.savez_compressed(path/'posterior-state.npz',successes=successes,prior=prior,train=train,future=future,
                        duration=duration,users=np.asarray(ctx.targets),items=np.asarray(ctx.candidates))
    np.save(path/'sampled-top5.npy',simulated,allow_pickle=False)
    np.save(path/'mean-scores.npy',means,allow_pickle=False)
    np.save(path/'greedy-scores.npy',ranking,allow_pickle=False)
    details=dict(protocol_sha256=ctx.protocol_hash,source_sha256=sha256(__file__),seed=seed,samples=samples,
        alpha_beta='alpha=s+50*prior; beta=train_opportunities-s+50*(1-prior)',
        prior='user overall historical full-per-airing rate * pooled programme full-per-airing rate / population overall rate, clipped1e-9; programme pooled rate shrunk by50global pseudo-observations',
        population_rate=population_rate,training_warm_users=nusers,all_train_opportunities=all_opportunities,
        prior_strength=strength,all_candidate_count=m,actual_future_truth_used=False,
        source_events_sha256=sha256(ctx.path/'train-events.npz'),opportunities_sha256=sha256(ctx.path/'opportunities.json'),
        schedule_manifest_sha256=sha256(cp/'manifest.json'),
        sample_ties='count desc, count*mean future airing original duration desc, numeric ID asc; exclude zero counts',
        ranking='greedy finite-sample expected AP@5; not a claim of globally optimal expected AP or calibrated forecast',
        native_scores='ordinal6..2 for selected greedy positions; remaining values empirical relevance/min(5,truthsize)',
        empirical_bayes_caveat='pooled training prior includes target-user history; observations reused in plug-in prior, posterior uncertainty is approximate',
        mc_aggregate_mean_abs_error=float(np.mean(np.abs(sampled_means-expected_means))),
        payload_sha256={name:sha256(path/name) for name in ('posterior-state.npz','sampled-top5.npy','mean-scores.npy','greedy-scores.npy')})
    dump(path/'posterior-manifest.json',details)
    return means,details


def posterior_greedy(ctx,p,path):
    source=path.parent/p['source_run'];manifest=load(source/'posterior-manifest.json')
    assert manifest['protocol_sha256']==ctx.protocol_hash
    for name,h in manifest['payload_sha256'].items():assert sha256(source/name)==h
    return np.load(source/'greedy-scores.npy'),dict(source_manifest_sha256=sha256(source/'posterior-manifest.json'),
        optimization='greedy expected AP under train-only posterior-predictive simulated top-five relevance sets')


def register():
    from models import METHODS
    METHODS['posterior_predictive']=posterior_predictive
    METHODS['posterior_greedy']=posterior_greedy
