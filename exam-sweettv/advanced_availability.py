"""Repeat affinity with absence measured in missed airing opportunities.

A programme that stopped airing is not treated as an abandoned preference merely
because the last watch was long ago. Only historical ID/channel/time/duration
enter the gap calculation; evaluated future viewing is never read.
"""
from collections import defaultdict
import csv
from pathlib import Path

import numpy as np

from etl import Clock, FILES, read_events
from repro import ROOT, dump, load, sha256

SOURCE_SHA256 = sha256(__file__)


def historical_airings(ctx):
    raw = Path(getattr(ctx, 'raw_input_dir', ROOT/'data/raw'))
    if {name: sha256(raw/name) for name in ctx.protocol['input_hashes']} != ctx.protocol['input_hashes']:
        raise ValueError('Raw input changed since protocol freeze')
    clock = Clock(ctx.protocol['clock'])
    start = clock.seconds(ctx.protocol['train_start'])
    cutoff = ctx.protocol['cutoff_elapsed_seconds']
    unique = set()
    for name in FILES:
        with (raw/name).open(encoding='utf-8-sig', newline='') as stream:
            for row in csv.DictReader(stream):
                if row['tv_show_id'] == '0':
                    continue
                a = clock.seconds(row['start_time'])
                if a >= cutoff:
                    continue
                b = a + int(row['duration'])
                if b > start:
                    unique.add((row['tv_show_id'], row['channel_id'], a, b))
    by_item = defaultdict(list)
    for item, channel, a, b in unique:
        by_item[item].append((a, b))
    return {item: np.array(sorted(times), dtype=np.int64) for item, times in by_item.items()}


def gap_statistics(ctx, events, users, items, airings):
    """Pre-cutoff broadcasts beginning after a person's last weak/full watch."""
    shape = len(ctx.targets), len(ctx.candidates)
    last = np.zeros(shape, dtype=np.int64)
    counts = np.zeros(shape, dtype=np.float32)
    um = {u: j for j, u in enumerate(ctx.targets)}
    ur = np.array([um.get(u, -1) for u in users])
    ir = np.array([ctx.candidate_map.get(i, -1) for i in items])
    assert not len(events) or events[:, 6].max() <= ctx.protocol['cutoff_elapsed_seconds']
    keep = (ur[events[:, 0]] >= 0) & (ir[events[:, 1]] >= 0)
    e = events[keep]
    u, i = ur[e[:, 0]], ir[e[:, 1]]
    coverage = e[:, 5]/e[:, 7]
    weights = np.where(coverage >= .8, 1., coverage**2)
    np.add.at(counts, (u, i), weights)
    meaningful = coverage >= .3
    np.maximum.at(last, (u[meaningful], i[meaningful]), e[meaningful, 6])
    missed = np.zeros(shape, dtype=np.int32)
    total = np.zeros(shape[1], dtype=np.int32)
    for item, column in ctx.candidate_map.items():
        times = airings.get(item)
        if times is None:
            continue
        starts = np.sort(times[:, 0])
        total[column] = len(starts)
        active = last[:, column] > 0
        missed[active, column] = len(starts)-np.searchsorted(starts, last[active, column], side='left')
    return counts, last, missed, total


def availability_repeat(ctx, p, path):
    users, items, _, events = read_events(ctx.path/'train-events.npz', full_only=False)
    airings = historical_airings(ctx)
    count, last, missed, total = gap_statistics(ctx, events, users, items, airings)
    opportunities = load(ctx.path/'opportunities.json')
    frozen = np.array([opportunities['train'].get(i, 0) for i in ctx.candidates])
    np.testing.assert_array_equal(total, frozen)
    future = np.array([opportunities['future'].get(i, 0) for i in ctx.candidates])
    rate = count/(total+p.get('prior_strength', 100))
    expected_missed = rate*missed
    score = rate*future*np.exp(-p.get('gap_penalty', .5)*expected_missed)
    score[score <= 0] = np.nan
    np.savez_compressed(path/'availability-gap-state.npz', count=count, last=last,
                        missed=missed, total=total, expected_missed=expected_missed)
    dump(path/'availability-inputs.json', {'source_sha256': SOURCE_SHA256,
         'protocol_sha256': ctx.protocol_hash, 'raw_hashes': ctx.protocol['input_hashes'],
         'train_event_sha256': sha256(ctx.path/'train-events.npz'),
         'max_used_view_stop': int(events[:, 6].max()) if len(events) else 0,
         'max_used_airing_start': max((int(v[:, 0].max()) for v in airings.values()), default=0)})
    return score.astype(np.float32), {'future_information': 'frozen availability counts only',
        'fit_cutoff': ctx.protocol['train_cutoff_exclusive'], 'source_sha256': SOURCE_SHA256,
        'gap_units': 'expected missed pre-cutoff airing counts, not calendar days',
        'full_catalogue_scoring': True}


def register():
    from models import METHODS
    METHODS['availability_repeat'] = availability_repeat
