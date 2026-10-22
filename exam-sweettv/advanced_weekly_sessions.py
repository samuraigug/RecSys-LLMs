"""Historical weekly replay of exact user/channel union viewing intervals.

Each future airing is projected to the same local weekday/start time in every
historical week; its elapsed duration is retained. Full-watch probability uses
the union overlap threshold, before averaging weeks and counting programme
airings. Future content and views are never accessed.
"""
from datetime import datetime,timedelta

import numpy as np

import advanced_hourly_schedule as hourly
import advanced_schedule as coarse
from etl import Clock
from repro import ROOT,dump,load,sha256

SOURCE_SHA256=sha256(__file__)
SEMANTIC_VERSION='exact-weekly-union-replay-v1'


def interval_cdf(starts,stops,queries):
    """Integral of a sorted, disjoint union on (-infinity,t], in exact seconds."""
    starts=np.asarray(starts,np.int64);stops=np.asarray(stops,np.int64)
    queries=np.asarray(queries,np.int64)
    if not len(starts):return np.zeros(queries.shape,np.int64)
    prefix=np.concatenate([np.zeros(1,np.int64),np.cumsum(stops-starts,dtype=np.int64)])
    ix=np.searchsorted(starts,queries,side='right')-1
    safe=np.maximum(ix,0)
    return np.where(ix>=0,prefix[safe]+np.clip(queries-starts[safe],0,stops[safe]-starts[safe]),0)


def union_segments(rows,channels,starts,stops,train_start,cutoff):
    """Recover raw unions from adjacent hourly segments without counting twice."""
    rows=np.asarray(rows,np.int32);channels=np.asarray(channels,np.int32)
    starts=np.asarray(starts,np.int64);stops=np.asarray(stops,np.int64)
    if np.any(starts<train_start) or np.any(stops>cutoff) or np.any(starts>=stops):
        raise ValueError('Training intervals cross frozen cutoff or are non-positive')
    if not len(starts):return rows,channels,starts,stops
    order=np.lexsort((starts,rows,channels));rows,channels,starts,stops=(x[order] for x in (rows,channels,starts,stops))
    # Hourly source is already a per-user/channel union; only adjacency is left.
    same=(rows[1:]==rows[:-1])&(channels[1:]==channels[:-1])
    if np.any(same&(starts[1:]<stops[:-1])):raise ValueError('Hourly source has overlapping union segments')
    new=np.concatenate([[True],~same|(starts[1:]>stops[:-1])])
    left=np.flatnonzero(new);right=np.concatenate([left[1:]-1,[len(starts)-1]])
    return rows[left],channels[left],starts[left],stops[right]


def project_templates(clock,train_start,cutoff,templates):
    """DST-aware local start replay plus original elapsed duration.

    Template columns: local week-second, elapsed duration, clipped left offset,
    clipped right offset. Nonexistent spring-transition local times are skipped;
    ambiguous local times use Clock's default fold=0. Queries outside train are
    excluded from both probability numerator and denominator.
    """
    first=clock.local(train_start).date()
    first-=timedelta(days=first.weekday())
    mondays=[];d=first
    while clock.seconds(d.isoformat())<cutoff:
        mondays.append(d);d+=timedelta(days=7)
    qs=np.zeros((len(mondays),len(templates)),np.int64);qe=np.zeros_like(qs)
    valid=np.zeros(qs.shape,bool);nonexistent=0
    for w,monday in enumerate(mondays):
        origin=datetime.combine(monday,datetime.min.time())
        # Most templates share start time but have different elapsed durations.
        projected={}
        for t,(weeksecond,duration,left,right) in enumerate(templates):
            weeksecond=int(weeksecond)
            if weeksecond not in projected:
                local=origin+timedelta(seconds=weeksecond)
                try:projected[weeksecond]=clock.seconds(local.isoformat(sep=' '))
                except ValueError:projected[weeksecond]=None
            a=projected[weeksecond]
            if a is None:
                nonexistent+=1;continue
            a,b=a+int(left),a+int(right)
            qs[w,t],qe[w,t]=a,b
            valid[w,t]=train_start<=a<b<=cutoff
    return qs,qe,valid,{'historical_mondays':[d.isoformat() for d in mondays],
        'nonexistent_local_template_trials_skipped':nonexistent,
        'outside_train_trials_excluded':int((~valid).sum())-nonexistent,
        'projection':'same local weekday/start; original elapsed duration; clipped future-window offsets',
        'dst_caveat':'Stationary weekly behaviour assumption; nonexistent local starts skipped, ambiguous fold0. Future airing elapsed duration retained.'}


def replay_probabilities(starts,stops,qs,qe,durations,valid,weights,threshold=.8,fraction=False):
    overlap=interval_cdf(starts,stops,qe)-interval_cdf(starts,stops,qs)
    if np.any(overlap<0) or np.any(overlap>np.maximum(qe-qs,0)):
        raise AssertionError('Invalid union overlap')
    if fraction:
        signal=overlap/np.maximum(durations[None,:],1)
    elif threshold==.8:
        signal=overlap*5>=durations[None,:]*4
    else:
        signal=overlap>=durations[None,:]*threshold
    mass=weights*valid
    return (signal*mass).sum(axis=0)/np.maximum(mass.sum(axis=0),1e-30)


def _build(ctx,path,cache):
    # Public hourly API validates upstream source/protocol/coarse/NPZ hashes.
    hourly.schedule_hourly(ctx,{'joint_weight':.5,'neighbor_smooth':.25,'half_life_days':28},path)
    hourly_ref=load(path/'hourly-cache-reference.json');hpath=ROOT/hourly_ref['path']
    hm=load(hpath/'manifest.json');cpath=ROOT/hm['coarse_cache']
    cm=load(cpath/'manifest.json')
    for filename,expected in cm['cache_data_sha256'].items():
        if sha256(cpath/filename)!=expected:raise ValueError('Coarse projection source artifact changed: '+filename)
    clock=Clock(ctx.protocol['clock']);start=clock.seconds(ctx.protocol['train_start'])
    cutoff=ctx.protocol['cutoff_elapsed_seconds'];end=clock.seconds(ctx.protocol['forecast_end_exclusive'])
    with np.load(hpath/'train-hour-segments.npz',allow_pickle=False) as z:
        seconds=z['seconds'].astype(np.float64)
        reconstructed_stop=cutoff-z['age_days']*86400
        if np.max(np.abs(reconstructed_stop-np.rint(reconstructed_stop)),initial=0)>1e-5:
            raise ValueError('Hourly timestamp reconstruction lost integer-second precision')
        stops=np.rint(reconstructed_stop).astype(np.int64)
        starts=stops-seconds.astype(np.int64)
        u,ch,a,b=union_segments(z['rows'],z['cols']//168,starts,stops,start,cutoff)
    np.savez_compressed(cache/'training-unions.npz',rows=u,channels=ch,starts=a,stops=b)
    with np.load(cpath/'future-airings.npz',allow_pickle=False) as z:
        item,fch,fa,fb=(z[k] for k in ('item','channel','start','stop'))
    weeksec=np.array([clock.local(a).weekday()*86400+clock.local(a).hour*3600+
                      clock.local(a).minute*60+clock.local(a).second for a in fa],np.int64)
    # Dedup repeated future weekly airings into channel/time/duration templates.
    keys=np.column_stack([fch,weeksec,fb-fa,np.maximum(fa,cutoff)-fa,np.minimum(fb,end)-fa])
    unique,inverse=np.unique(keys,axis=0,return_inverse=True)
    qs,qe,valid,projection=project_templates(clock,start,cutoff,unique[:,1:])
    np.savez_compressed(cache/'future-projections.npz',template_channels=unique[:,0].astype(np.int32),
        durations=unique[:,2],query_starts=qs,query_stops=qe,valid=valid,
        airing_template=inverse.astype(np.int32),airing_items=item,airing_channels=fch)
    if len(a)!=hm['unions']:raise AssertionError('Reconstructed union count differs from hourly source')
    dump(cache/'manifest.json',{'protocol_sha256':ctx.protocol_hash,'source_sha256':SOURCE_SHA256,
        'semantic_version':SEMANTIC_VERSION,'hourly_source_sha256':hourly.SOURCE_SHA256,
        'coarse_source_sha256':coarse.BUILDER_SOURCE_SHA256,'hourly_cache':str(hpath),
        'hourly_manifest_sha256':sha256(hpath/'manifest.json'),'coarse_cache':str(cpath),
        'coarse_manifest_sha256':sha256(cpath/'manifest.json'),'channels':hm['channels'],
        'reconstructed_training_unions':len(a),'expected_hourly_union_count':hm['unions'],
        'min_training_start':int(a.min()) if len(a) else None,'max_training_stop':int(b.max()) if len(b) else None,
        'future_airings':len(fa),'unique_channel_time_duration_templates':len(unique),
        'projection':projection,'input_hashes':ctx.protocol['input_hashes'],
        'cache_data_sha256':{p.name:sha256(p) for p in sorted(cache.glob('*.npz'))}})


def weekly_sessions(ctx,p,path):
    cache=ROOT/'artifacts/runs/weekly-replay-cache-v1'/ctx.protocol_hash/SOURCE_SHA256/hourly.SOURCE_SHA256/coarse.BUILDER_SOURCE_SHA256
    if not (cache/'manifest.json').exists():
        cache.mkdir(parents=True,exist_ok=True);_build(ctx,path,cache)
    manifest=load(cache/'manifest.json')
    if (manifest['protocol_sha256']!=ctx.protocol_hash or manifest['source_sha256']!=SOURCE_SHA256 or
        manifest['semantic_version']!=SEMANTIC_VERSION or manifest['hourly_source_sha256']!=hourly.SOURCE_SHA256 or
        manifest['coarse_source_sha256']!=coarse.BUILDER_SOURCE_SHA256 or manifest['input_hashes']!=ctx.protocol['input_hashes']):
        raise ValueError('Weekly replay cache provenance mismatch')
    for base,key in [('hourly','hourly_manifest_sha256'),('coarse','coarse_manifest_sha256')]:
        if sha256(ROOT/manifest[base+'_cache']/'manifest.json')!=manifest[key]:raise ValueError('Weekly replay upstream manifest changed')
    for name,expected in manifest['cache_data_sha256'].items():
        if sha256(cache/name)!=expected:raise ValueError('Weekly replay cache artifact changed: '+name)
    cutoff=ctx.protocol['cutoff_elapsed_seconds']
    if manifest['max_training_stop'] is not None and manifest['max_training_stop']>cutoff:raise ValueError('Training cutoff violated')
    with np.load(cache/'training-unions.npz',allow_pickle=False) as z:u,ch,a,b=(z[k] for k in ('rows','channels','starts','stops'))
    with np.load(cache/'future-projections.npz',allow_pickle=False) as z:
        tch,duration,qs,qe,valid,at,ai,ac=(z[k] for k in ('template_channels','durations','query_starts','query_stops','valid','airing_template','airing_items','airing_channels'))
    if p.get('half_life_days'):
        weights=np.exp2(-(cutoff-qe)/86400/p['half_life_days'])
    else:weights=np.ones(qs.shape,np.float64)
    scores=np.zeros((len(ctx.targets),len(ctx.candidates)),np.float32)
    group_left=np.concatenate([[0],np.flatnonzero((u[1:]!=u[:-1])|(ch[1:]!=ch[:-1]))+1]) if len(u) else np.array([],np.int64)
    group_right=np.concatenate([group_left[1:],[len(u)]]) if len(u) else np.array([],np.int64)
    group_channel=ch[group_left]
    queried_cells=active_groups=0
    future_channels=np.unique(tch)
    for c in future_channels:
        templates=np.flatnonzero(tch==c);airings=np.flatnonzero(ac==c)
        # Global template IDs -> compact channel-local index for aggregation.
        inverse=np.full(len(tch),-1,np.int32);inverse[templates]=np.arange(len(templates))
        local_airing_template=inverse[at[airings]]
        qstart,qstop,ok,w=(x[:,templates] for x in (qs,qe,valid,weights));dur=duration[templates]
        groups=np.flatnonzero(group_channel==c)
        for g in groups:
            left,right=group_left[g],group_right[g]
            probability=replay_probabilities(a[left:right],b[left:right],qstart,qstop,dur,ok,w,
                                            p.get('coverage',.8),p.get('fraction_control',False))
            scores[u[left]]+=np.bincount(ai[airings],weights=probability[local_airing_template],minlength=len(ctx.candidates)).astype(np.float32)
            queried_cells+=qstart.size;active_groups+=1
    scores[scores<=0]=np.nan
    dump(path/'weekly-cache-reference.json',{'path':str(cache),'manifest_sha256':sha256(cache/'manifest.json')})
    return scores,{'algorithm':'exact historical weekly union replay; nonlinear full-airing overlap probability before programme aggregation',
        'coverage':p.get('coverage',.8),'fraction_control':bool(p.get('fraction_control')),
        'fit_cutoff':ctx.protocol['train_cutoff_exclusive'],'future_views_used':False,'future_content_used':False,
        'cache':str(cache),'training_unions':len(a),'future_airings':manifest['future_airings'],
        'templates':len(tch),'weeks':qs.shape[0],'active_user_channels':active_groups,'queried_template_week_cells':queried_cells,
        'projection':manifest['projection'],'native_scores':'expected number of qualifying future programme airings; not normalized probabilities'}


def register(methods):methods['weekly_sessions']=weekly_sessions
