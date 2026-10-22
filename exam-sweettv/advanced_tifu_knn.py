"""TIFU-inspired TV adaptation. See preregistered research/tifu-knn-20261022/DESIGN.md.

Fresh implementation of hierarchical frequency vectors and Euclidean user KNN.
No code from upstream is executed. Native signals are not probabilities.
"""
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import numpy as np
from scipy import sparse
from advanced_collaborative import _scores
from repro import ROOT, dump, load, sha256

RESEARCH = ROOT/'research/tifu-knn-20261022'


class TrainOnlyContext:
    def __init__(self, ctx):
        object.__setattr__(self, '_ctx', ctx)
    def __getattr__(self, key):
        if key in ('truth', 'display'):
            raise AssertionError('Forbidden evaluation attribute: '+key)
        return getattr(object.__getattribute__(self, '_ctx'), key)


def daily_vectors(user_rows, item_cols, day_ages, shape, number_days, rb, rg):
    """Binary daily baskets, zero-padded seven-day groups on a shared calendar."""
    if number_days <= 0 or not (0 < rb <= 1 and 0 < rg <= 1):
        raise ValueError('Invalid calendar or decay')
    if np.any(day_ages < 0) or np.any(day_ages >= number_days):
        raise ValueError('Event outside train calendar')
    tuples = np.unique(np.column_stack((user_rows, item_cols, day_ages)), axis=0)
    ages = tuples[:, 2]
    weights = rb**(ages % 7) * rg**(ages // 7) / (7*((number_days+6)//7))
    return sparse.csr_matrix((weights, (tuples[:, 0], tuples[:, 1])), shape=shape)


def nearest_users(vectors, target_rows, donor_rows, user_ids, k):
    """Full item-space squared Euclidean distances; ties use arbitrary-size IDs."""
    canonical = [str(int(u)) for u in user_ids]
    donor_rows = np.array(sorted(donor_rows, key=lambda r: (len(canonical[r]), canonical[r])), dtype=np.int64)
    count = min(k, len(donor_rows)-int(any(r in set(donor_rows) for r in target_rows)))
    if count <= 0:
        raise ValueError('Need at least two historical users')
    norms = np.asarray(vectors.multiply(vectors).sum(axis=1)).ravel()
    neighbors = np.empty((len(target_rows), count), dtype=np.int64)
    distances = np.empty((len(target_rows), count), dtype=np.float64)
    for start in range(0, len(target_rows), 100):
        rows = target_rows[start:start+100]
        d = norms[rows, None]+norms[donor_rows][None, :]-2*(vectors[rows]@vectors[donor_rows].T).toarray()
        np.maximum(d, 0, out=d)
        for j, row in enumerate(rows):
            d[j, donor_rows == row] = np.inf
            # donor order already canonical; stable argsort handles exact ties.
            selected = np.argsort(d[j], kind='stable')[:count]
            neighbors[start+j] = donor_rows[selected]
            distances[start+j] = d[j, selected]
    return neighbors, distances


def blended_vectors(vectors, target_rows, neighbors, own_weight=.7):
    n, k = neighbors.shape
    averaging = sparse.csr_matrix((np.full(n*k, 1/k),
                                  (np.repeat(np.arange(n), k), neighbors.ravel())),
                                 shape=(n, vectors.shape[0]))
    return (own_weight*vectors[target_rows]+(1-own_weight)*(averaging@vectors)).toarray()


def _fit(ctx, temporal):
    eu, ei, events = ctx.events()
    cutoff = ctx.protocol['cutoff_elapsed_seconds']
    assert np.all(events[:, 6] <= cutoff)
    zone = ZoneInfo(ctx.protocol['clock'])
    end = date.fromisoformat(ctx.protocol['train_cutoff_exclusive'])
    start = date.fromisoformat(ctx.protocol['train_start'])
    number_days = (end-start).days
    timestamps, inverse = np.unique(events[:, 6], return_inverse=True)
    ages = np.array([(end-datetime.fromtimestamp(int(t)-1, zone).date()).days-1 for t in timestamps], dtype=np.int64)[inverse]
    users, items = eu[events[:, 0]], ei[events[:, 1]]
    assert np.all(users >= 0) and np.all(items >= 0)
    rb, rg = (.9, .7) if temporal else (1., 1.)
    vectors = daily_vectors(users, items, ages, (len(ctx.users), len(ctx.items)), number_days, rb, rg)
    donors = np.flatnonzero(vectors.getnnz(axis=1))
    assert set(donors) == {ctx.user_map[u] for u in ctx.history if ctx.history[u]}
    neighbors, distances = nearest_users(vectors, ctx.target_rows, donors, ctx.users, 300)
    return vectors, neighbors, distances, number_days


def tifu_knn(ctx, parameters, path):
    if set(parameters)-{'temporal', 'blend', 'seed'}:
        raise ValueError('Only preregistered fixed variants supported')
    ctx = TrainOnlyContext(ctx)
    temporal, blend = parameters.get('temporal', True), parameters.get('blend', False)
    if blend and not temporal:
        raise ValueError('Only temporal blend preregistered')
    if blend:
        from advanced_full_fusion import rrf, _check_native
        source = ROOT/'artifacts/runs'/ctx.path.name/'FULL-shfl4'
        local = path.parent/'TIFU-temporal'
        record = load(source/'record.json')
        own_record = load(local/'record.json')
        assert record['protocol_sha256'] == own_record['protocol_sha256'] == ctx.protocol_hash
        assert record['status'] == own_record['status'] == 'SUCCEEDED'
        assert record['parameters'] == {'bundle': 'shfl4'}
        base = np.load(source/'full-fusion-scores.npy', allow_pickle=False)
        assert sha256(source/'full-fusion-scores.npy') == load(source/'full-score-provenance.json')['full_fusion_scores_sha256']
        own = np.load(local/'full-scores.npy', allow_pickle=False)
        assert sha256(local/'full-scores.npy') == load(local/'model-manifest.json')['payload_sha256']['full-scores.npy']
        _check_native(ctx, base.copy(), source)
        _check_native(ctx, own.copy(), local)
        scores = rrf([(base, 1.), (own, .05)], np.array([int(i) for i in ctx.candidates]), 60)
        details = {'algorithm': 'fixed full-catalogue RRF', 'base_weight': 1., 'tifu_weight': .05,
                   'k0': 60, 'source_records': {str(p): sha256(p/'record.json') for p in (source, local)}}
    else:
        vectors, neighbors, distances, days = _fit(ctx, temporal)
        raw = blended_vectors(vectors, ctx.target_rows, neighbors)
        scores = _scores(ctx, raw, {'opportunity_adjustment': True, 'prior_strength': 50})
        sparse.save_npz(path/'vectors.npz', vectors)
        np.savez_compressed(path/'neighbors.npz', rows=neighbors, squared_distances=distances, target_rows=ctx.target_rows)
        replay = blended_vectors(sparse.load_npz(path/'vectors.npz'), ctx.target_rows, np.load(path/'neighbors.npz')['rows'])
        np.testing.assert_array_equal(raw, replay)
        details = {'algorithm': 'TIFU-inspired calendar user-user Euclidean KNN', 'temporal': temporal,
                   'rb': .9 if temporal else 1., 'rg': .7 if temporal else 1., 'group_days': 7,
                   'train_calendar_days': days, 'neighbors': neighbors.shape[1], 'own_weight': .7,
                   'vector_shape': list(vectors.shape), 'vector_nnz': vectors.nnz,
                   'opportunity_prior': 50, 'squared_distance': True, 'reload_exact': True}
    scores[ctx.cold_mask] = np.nan
    np.save(path/'full-scores.npy', scores, allow_pickle=False)
    dump(path/'mapping.json', {'users': ctx.users, 'items': ctx.items, 'targets': ctx.targets, 'candidates': ctx.candidates})
    details.update(truth_display_blocked=True, future_views_used=False, source_sha256=sha256(__file__),
                   protocol_sha256=ctx.protocol_hash, design_sha256=sha256(RESEARCH/'DESIGN.md'),
                   parameters=parameters, upstream_manifest_sha256=sha256(RESEARCH/'sources/manifest.json'),
                   train_events_sha256=sha256(ctx.path/'train-events.npz'),
                   opportunities_sha256=sha256(ctx.path/'opportunities.json'), native_scores_are_not_probabilities=True)
    dump(path/'model-manifest.json', {**details, 'payload_sha256': {p.name: sha256(p) for p in path.iterdir() if p.suffix in ('.npy', '.npz') or p.name == 'mapping.json'}})
    return scores, details


def register():
    from models import METHODS
    METHODS['tifu_knn'] = tifu_knn
