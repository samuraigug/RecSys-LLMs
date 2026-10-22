"""Pre-cutoff union viewing habits × known future airing opportunities.

This is a schedule/context hypothesis, not a learned content recommender. It
uses future ID, channel, start and duration only; no future views or content.
"""
from collections import defaultdict
import csv
from pathlib import Path

import numpy as np
from scipy import sparse

from etl import Clock, FILES, clip_union
from repro import ROOT, dump, load, sha256

BUILDER_SOURCE_SHA256 = sha256(__file__)
CACHE_SCHEMA_VERSION = 2


def segments(clock, start, stop):
    """Split elapsed intervals at local hour boundaries, including DST jumps."""
    while start < stop:
        local = clock.local(start)
        edge = min(stop, start + 3600-local.minute*60-local.second)
        yield local.weekday(), local.hour//3, edge-start, start
        start = edge


def _build(ctx, cache):
    clock = Clock(ctx.protocol['clock'])
    start = clock.seconds(ctx.protocol['train_start'])
    cutoff = ctx.protocol['cutoff_elapsed_seconds']
    end = clock.seconds(ctx.protocol['forecast_end_exclusive'])
    raw = Path(getattr(ctx, 'raw_input_dir', ROOT / 'data/raw'))
    actual_hashes = {fn:sha256(raw/fn) for fn in ctx.protocol['input_hashes']}
    if actual_hashes != ctx.protocol['input_hashes']:
        raise ValueError('Raw input changed since protocol freeze')
    users = {u:j for j,u in enumerate(ctx.targets)}
    groups, channels = defaultdict(list), set()
    raw_count = kept = 0
    # Reject rows outside the training prefix before parsing their timestamps.
    cutoff_text = ctx.protocol['train_cutoff_exclusive']
    start_text = ctx.protocol['train_start']
    with (raw/'dataset11-30.csv').open(encoding='utf-8-sig', newline='') as stream:
        for row in csv.DictReader(stream):
            raw_count += 1
            channel = row['vsetv_id']
            channels.add(channel)
            if row['user_id'] not in users or row['start_time'] >= cutoff_text or row['stop_time'] <= start_text:
                continue
            a, b = clock.seconds(row['start_time']), clock.seconds(row['stop_time'])
            a, b = max(a, start), min(b, cutoff)
            if a < b:
                groups[(row['user_id'],channel)].append((a,b))
                kept += 1
    airings = set()
    for name in FILES:
        with (raw/name).open(encoding='utf-8-sig', newline='') as stream:
            for row in csv.DictReader(stream):
                if row['tv_show_id']=='0':
                    continue
                channel = row['channel_id']
                channels.add(channel)
                # ID/channel/time/duration are the complete future feature schema.
                a = clock.seconds(row['start_time'])
                b = a+int(row['duration'])
                if a < end and b > cutoff and row['tv_show_id'] in ctx.candidate_map:
                    airings.add((row['tv_show_id'],channel,a,b))
    channels = sorted(channels, key=int)
    cm = {c:j for j,c in enumerate(channels)}
    n, m, width = len(ctx.targets), len(ctx.candidates), len(channels)*56
    rr, cc, vv, age = [], [], [], []
    duration_sum = np.zeros((n,len(channels)), np.float64)
    duration_count = np.zeros_like(duration_sum)
    max_train_stop = 0
    union_count = 0
    for (user,channel), intervals in groups.items():
        u, c = users[user], cm[channel]
        for a,b in clip_union(intervals,start,cutoff):
            assert start <= a < b <= cutoff
            max_train_stop = max(max_train_stop,b)
            union_count += 1
            duration_sum[u,c] += b-a
            duration_count[u,c] += 1
            for weekday,slot,seconds,last in segments(clock,a,b):
                rr.append(u); cc.append(c*56+weekday*8+slot); vv.append(seconds)
                age.append(max(0,(cutoff-(last+seconds))/86400))
    # Event segments retained for exact train-only recency variants.
    np.savez_compressed(cache/'train-segments.npz', rows=np.array(rr,np.int32), cols=np.array(cc,np.int32),
                        seconds=np.array(vv,np.float32), age_days=np.array(age,np.float32),
                        duration_sum=duration_sum, duration_count=duration_count)
    for byduration in (False, True):
        r, c, v = [], [], []
        for item,channel,a,b in sorted(airings):
            dur = b-a
            for weekday,slot,seconds,_ in segments(clock,max(a,cutoff),min(b,end)):
                r.append(ctx.candidate_map[item]); c.append(cm[channel]*56+weekday*8+slot)
                v.append(seconds/3600 if byduration else seconds/dur)
        sparse.save_npz(cache/('future-hours.npz' if byduration else 'future-counts.npz'),
                        sparse.csr_matrix((v,(r,c)),shape=(m,width),dtype=np.float32))
    # A duration-penalized expected count requires each airing's own duration.
    np.savez_compressed(cache/'future-airings.npz', item=np.array([ctx.candidate_map[a[0]] for a in sorted(airings)],np.int32),
                        channel=np.array([cm[a[1]] for a in sorted(airings)],np.int32),
                        start=np.array([a[2] for a in sorted(airings)],np.int64),
                        stop=np.array([a[3] for a in sorted(airings)],np.int64))
    cache_hashes={p.name:sha256(p) for p in sorted(cache.glob('*.npz'))}
    dump(cache/'manifest.json', {'protocol_sha256':ctx.protocol_hash, 'channels':channels,
        'raw_rows_scanned':raw_count,'precutoff_target_rows':kept,'precutoff_union_intervals':union_count,
        'max_training_stop':max_train_stop,'cutoff_elapsed_seconds':cutoff,'future_airings':len(airings),
        'future_information':'ID/channel/start/duration only','zero_id_airtime_in_training':True,
        'raw_hashes':actual_hashes, 'source_sha256':BUILDER_SOURCE_SHA256,
        'cache_schema_version':CACHE_SCHEMA_VERSION,'cache_data_sha256':cache_hashes})


def schedule_joint(ctx,p,path):
    # Immutable per-protocol AND per-source cache: older run references remain valid.
    cache = ROOT/'artifacts/runs/schedule-cache-v2'/ctx.protocol_hash/BUILDER_SOURCE_SHA256
    if not (cache/'manifest.json').exists():
        cache.mkdir(parents=True,exist_ok=True)
        _build(ctx,cache)
    manifest=load(cache/'manifest.json')
    if manifest['protocol_sha256']!=ctx.protocol_hash:
        raise ValueError('Schedule cache protocol mismatch')
    if manifest.get('source_sha256')!=BUILDER_SOURCE_SHA256 or manifest.get('cache_schema_version')!=CACHE_SCHEMA_VERSION:
        raise ValueError('Schedule cache builder mismatch')
    if manifest['raw_hashes']!=ctx.protocol['input_hashes']:
        raise ValueError('Schedule cache input mismatch')
    for filename,expected in manifest['cache_data_sha256'].items():
        if sha256(cache/filename)!=expected:
            raise ValueError('Schedule cache artifact changed: '+filename)
    assert manifest['max_training_stop']<=ctx.protocol['cutoff_elapsed_seconds']
    with np.load(cache/'train-segments.npz',allow_pickle=False) as z:
        rows,cols,seconds,ages=(z[k] for k in ('rows','cols','seconds','age_days'))
        dursum,durcount=z['duration_sum'],z['duration_count']
    if p.get('half_life_days'):
        seconds=seconds*np.exp2(-ages/p['half_life_days'])
    n,c=len(ctx.targets),len(manifest['channels'])
    joint=sparse.csr_matrix((seconds,(rows,cols)),shape=(n,c*56)).toarray().reshape(n,c,7,8)
    # Three-way marginal profiles shrink fine bins; all are derived from train unions.
    ct=joint.sum(axis=2,keepdims=True)/7
    cw=joint.sum(axis=3,keepdims=True)/8
    channel=joint.sum(axis=(2,3),keepdims=True)/56
    marginal=p.get('ct_weight',.7)*np.broadcast_to(ct,joint.shape)+(1-p.get('ct_weight',.7))*np.broadcast_to(cw,joint.shape)
    marginal=.9*marginal+.1*np.broadcast_to(channel,joint.shape)
    strength=p.get('shrink_seconds',3600)
    if p.get('adaptive_shrink'):
        weight=joint/(joint+strength)
        profile=weight*joint+(1-weight)*marginal
    else:
        w=p.get('joint_weight',.8)
        profile=w*joint+(1-w)*marginal
    if p.get('smooth_neighbors'):
        profile=(1-p['smooth_neighbors'])*profile+p['smooth_neighbors']*(np.roll(profile,1,axis=3)+np.roll(profile,-1,axis=3))/2
    profile=profile.reshape(n,-1).astype(np.float32)
    profile/=np.maximum(profile.sum(axis=1,keepdims=True),1)
    opportunity=sparse.load_npz(cache/('future-hours.npz' if p.get('hours') else 'future-counts.npz'))
    scores=np.asarray(profile@opportunity.T)
    if p.get('duration_scale'):
        # Stationary exponential session-survival hypothesis; ablated independently.
        # One duration correction per airing, then aggregate to programme.
        scores[:]=0
        avg=(dursum+p.get('duration_prior',10)*1800)/(durcount+p.get('duration_prior',10))
        clock=Clock(ctx.protocol['clock'])
        cutoff=ctx.protocol['cutoff_elapsed_seconds']; end=clock.seconds(ctx.protocol['forecast_end_exclusive'])
        with np.load(cache/'future-airings.npz',allow_pickle=False) as z:
            for item,ch,a,b in zip(z['item'],z['channel'],z['start'],z['stop']):
                base=np.zeros(n,np.float32)
                for wd,slot,secs,_ in segments(clock,max(a,cutoff),min(b,end)):
                    base+=profile[:,ch*56+wd*8+slot]*secs/(b-a)
                scores[:,item]+=base*np.exp(-p['duration_scale']*(b-a)/np.maximum(avg[:,ch],60))
    scores[scores<=0]=np.nan
    dump(path/'schedule-cache-reference.json', {'path':str(cache),'manifest_sha256':sha256(cache/'manifest.json')})
    return scores.astype(np.float32), {'algorithm':'raw union channel×weekday×3-hour profile × future airing opportunities',
        'train_cutoff':ctx.protocol['train_cutoff_exclusive'],'future_views_used':False,'future_content_used':False,
        'future_information':'ID/channel/start/duration only','cache':str(cache),
        'raw_target_rows':manifest['precutoff_target_rows'],'union_intervals':manifest['precutoff_union_intervals'],
        'airings':manifest['future_airings'],'duration_model':'exponential union-session survival' if p.get('duration_scale') else None}


def register(methods):
    methods['schedule_joint']=schedule_joint
