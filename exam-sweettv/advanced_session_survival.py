"""Empirical single-session completion correction for hourly airing mass.

For session length L and airing duration d, continuous start offsets permitting
80% overlap have measure (L-.6*d) I[L>=.8*d]. Divide summed measure by summed L
to convert stationary on-mass to single-session completion mass. This proxy does
not join disjoint short sessions and does not assume independent repeated trials.
"""
from collections import defaultdict
import numpy as np
from scipy import sparse
from sklearn.preprocessing import normalize
import advanced_hourly_schedule as hourly
import advanced_schedule as coarse
from advanced_weekly_sessions import union_segments
from etl import Clock
from repro import ROOT,dump,load,sha256


def completion_factor(lengths,durations):
    lengths=np.sort(np.asarray(lengths,np.float64));d=np.asarray(durations,np.float64)
    if np.any(lengths<=0) or np.any(d<=0):raise ValueError('Durations must be positive')
    if not len(lengths):return np.zeros(d.shape,np.float64)
    prefix=np.concatenate([[0.],np.cumsum(lengths)])
    k=np.searchsorted(lengths,.8*d,side='left')
    return (prefix[-1]-prefix[k]-.6*d*(len(lengths)-k))/prefix[-1]


def _profile(ctx,p,hpath,channels):
    with np.load(hpath/'train-hour-segments.npz',allow_pickle=False) as z:
        seconds=z['seconds']*np.exp2(-z['age_days']/28)
        joint=sparse.csr_matrix((seconds,(z['rows'],z['cols'])),shape=(len(ctx.targets),channels*168),dtype=np.float32)
    marginal=.9*(.7*hourly._marginal(joint,channels,'hour')+.3*hourly._marginal(joint,channels,'weekday'))+.1*hourly._marginal(joint,channels,'channel')
    profile=.5*joint+.5*marginal
    return normalize(hourly.smooth_hours(profile,channels,.25),norm='l1')


def session_survival(ctx,p,path):
    params=dict(joint_weight=.5,neighbor_smooth=.25,half_life_days=28)
    baseline,_=hourly.schedule_hourly(ctx,params,path)
    hpath=ROOT/load(path/'hourly-cache-reference.json')['path'];hm=load(hpath/'manifest.json')
    cpath=ROOT/hm['coarse_cache'];cm=load(cpath/'manifest.json')
    for name,expected in cm['cache_data_sha256'].items():
        if sha256(cpath/name)!=expected:raise ValueError('Coarse source artifact modified')
    clock=Clock(ctx.protocol['clock']);start=clock.seconds(ctx.protocol['train_start'])
    cutoff=ctx.protocol['cutoff_elapsed_seconds'];end=clock.seconds(ctx.protocol['forecast_end_exclusive'])
    with np.load(hpath/'train-hour-segments.npz',allow_pickle=False) as z:
        stops=np.rint(cutoff-z['age_days']*86400).astype(np.int64)
        assert np.max(np.abs(cutoff-z['age_days']*86400-stops),initial=0)<1e-5
        u,ch,a,b=union_segments(z['rows'],z['cols']//168,stops-z['seconds'].astype(np.int64),stops,start,cutoff)
    lengths=b-a; channels=len(hm['channels']);profile=_profile(ctx,p,hpath,channels)
    # Verify reconstructed profile before adding any duration assumption.
    reference=(profile@sparse.load_npz(hpath/'future-hour-counts.npz').T).toarray().astype(np.float32)
    reference[reference<=0]=np.nan
    np.testing.assert_array_equal(reference,baseline)
    sparse.save_npz(path/'hourly-profile.npz',profile)
    np.savez_compressed(path/'session-lengths.npz',users=u,channels=ch,lengths=lengths)
    by_pair=defaultdict(list)
    for user,channel,length in zip(u,ch,lengths):by_pair[(int(user),int(channel))].append(int(length))
    with np.load(cpath/'future-airings.npz',allow_pickle=False) as z:items,fch,fa,fb=(z[k] for k in ('item','channel','start','stop'))
    corrected=np.zeros(baseline.shape,np.float64);uncorrected=np.zeros_like(corrected);summaries=[]
    for channel in np.unique(fch):
        take=np.flatnonzero(fch==channel);duration=fb[take]-fa[take]
        unique,inverse=np.unique(duration,return_inverse=True)
        global_factor=completion_factor(lengths,unique)
        channel_lengths=lengths[ch==channel];nch=len(channel_lengths)
        channel_factor=(nch*completion_factor(channel_lengths,unique)+100*global_factor)/(nch+100)
        factors=np.empty((len(ctx.targets),len(unique)))
        for user in range(len(ctx.targets)):
            ls=by_pair.get((user,int(channel)),[]);n=len(ls)
            factors[user]=(n*completion_factor(ls,unique)+20*channel_factor)/(n+20)
        assert np.all(factors>=-1e-12) and np.all(factors<=1+1e-12)
        rr=[];cc=[];vv=[]
        for j,idx in enumerate(take):
            for wd,_,secs,segstart in coarse.segments(clock,max(fa[idx],cutoff),min(fb[idx],end)):
                rr.append(wd*24+clock.local(segstart).hour);cc.append(j);vv.append(secs/(fb[idx]-fa[idx]))
        schedule=sparse.csr_matrix((vv,(rr,cc)),shape=(168,len(take)),dtype=np.float32)
        onmass=(profile[:,int(channel)*168:(int(channel)+1)*168]@schedule).toarray()
        mapping=sparse.csr_matrix((np.ones(len(take)),(np.arange(len(take)),items[take])),shape=(len(take),len(ctx.candidates)))
        uncorrected += onmass@mapping
        corrected += (onmass*factors[:,inverse])@mapping
        summaries.append(dict(channel=hm['channels'][int(channel)],sessions=nch,airings=len(take),unique_durations=len(unique),mean_factor=float(factors.mean())))
    # Reassociation across separate airings changes float rounding only.
    diff=np.nanmax(np.abs(uncorrected-np.nan_to_num(baseline)))
    np.testing.assert_allclose(uncorrected,np.nan_to_num(baseline),rtol=3e-6,atol=2e-7)
    scores=corrected.astype(np.float32);scores[scores<=0]=np.nan
    np.save(path/'survival-scores.npy',scores,allow_pickle=False)
    details=dict(protocol_sha256=ctx.protocol_hash,source_sha256=sha256(__file__),
        hourly_manifest_sha256=sha256(hpath/'manifest.json'),coarse_manifest_sha256=sha256(cpath/'manifest.json'),
        hourly_cache=str(hpath),coarse_cache=str(cpath),session_union_count=len(lengths),
        max_training_stop=int(b.max()),min_training_start=int(a.min()),factor_one_profile_replay_bit_exact=True,
        per_airing_reassociation_maxdiff=float(diff),channels=summaries,
        formula='sum (L-0.6*d)*I(L>=0.8*d) / sum L',shrinkage_sessions_user_channel=20,shrinkage_sessions_channel=100,
        session_gap_bridge=0,future_features='ID/channel/time/original duration only',future_views_used=False,
        limitations='stationary single-session offset proxy; boundary-censored train sessions retained; multiple disjoint sessions reaching80% are not modelled')
    dump(path/'survival-provenance.json',details)
    return scores,details


def survival_bundle(ctx,p,path):
    from advanced_full_fusion import prepare_scores,max_fusion,rrf
    cache,_=prepare_scores(ctx)
    run=path.parent/p['source_run']; record=load(run/'record.json')
    assert record['protocol_sha256']==ctx.protocol_hash
    scores=np.load(run/'survival-scores.npy',allow_pickle=False)
    core=max_fusion([(np.load(cache/'ppr.npy'),1.),(scores,1.)])
    result=rrf([(core,1.),(np.load(cache/'l4.npy'),.5)],np.array([int(i) for i in ctx.candidates]))
    return result,dict(composition='exact FULL-shfl4 weights, hourly component replaced by empirical session-survival hourly',
        survival_scores_sha256=sha256(run/'survival-scores.npy'),base_manifest_sha256=sha256(cache/'manifest.json'))


def register():
    from models import METHODS
    METHODS['session_survival']=session_survival
    METHODS['session_survival_bundle']=survival_bundle
