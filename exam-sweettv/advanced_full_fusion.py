"""Controlled full-catalogue replay of the two frozen SHFL4/SHFL45 bundles.

No refit, label access, new candidates, or weight search. Preparation must reproduce
every saved component native Top-100 and both truncated reference bundles first.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

from models import METHODS
from repro import ROOT, dump, load, sha256

SOURCE_SHA256 = sha256(__file__)
ENGINE_PATH = ROOT/'artifacts/ranker-research/advanced_ranker-v5-source.py'
ENGINE_SHA256 = '54af6b09c507c8a7579230a73fdb06f915f4a03a5535edee8e09bf35a7f4afaf'
L4_SHA256 = '9f60de112ec180ba99b3f5fac549d4ea41aebd1cb58fe9ebffdd165883acb407'
DEPENDENCIES = {
    'advanced_collaborative.py': '12c15dc86025d3efe7491ed3942f886939f8f86711df1f955c1dac24d927e282',
    'advanced_hourly_schedule.py': 'aa4ac99fcced6e2be83ab786572fbb7e07ba10539d3ffccfc2025598b459f3c5',
    'advanced_schedule.py': '8f32f3533bdee350f2c28d1cc63eb64c68e278e173de5070b6ada7828e183b84',
}
RUNS = {'ppr': 'CFC-PPRF-a02', 'hourly': 'SH-j05-sm25-hl28',
        'l4': 'L4-logloss-allusers300', 'l5': 'L5-logloss-allusers-strong300',
        'core': 'SHF-PPR-recency-max-1', 'shfl4': 'SHFL4-rrf-0.5',
        'shfl45': 'SHFL45-rrf-equal'}
PPR = {'strength': 'count', 'weak_weight': 1, 'half_life_days': 56,
       'alpha': .2, 'neighbors': 200, 'opportunity_adjustment': True}
HOURLY = {'joint_weight': .5, 'neighbor_smooth': .25, 'half_life_days': 28}
BUNDLES = {'shfl4': [('core', 1.), ('l4', .5)],
           'shfl45': [('core', 1.), ('l4', .25), ('l5', .25)]}


def ordered(row, numeric_ids, limit=None):
    """Shared finalizer tie order; finite zero remains a legitimate native score."""
    valid = np.flatnonzero(np.isfinite(row))
    result = valid[np.lexsort((numeric_ids[valid], -row[valid]))]
    return result if limit is None else result[:limit]


def truncate(scores, numeric_ids, limit=100):
    out = np.full_like(scores, np.nan)
    for j, row in enumerate(scores):
        ids = ordered(row, numeric_ids, limit)
        out[j, ids] = row[ids]
    return out


def max_fusion(components):
    out = np.zeros_like(components[0][0], dtype=np.float64)
    touched = np.zeros_like(out, dtype=bool)
    for scores, weight in components:
        for j, row in enumerate(scores):
            ids = np.flatnonzero(np.isfinite(row))
            values = np.maximum(row[ids].astype(np.float64), 0)
            values /= max(values.max(initial=0), 1e-12)
            out[j, ids] += weight*values
            if weight > 0:
                touched[j, ids] = True
    out[~touched] = np.nan
    return out.astype(np.float32)


def rrf(components, numeric_ids, k0=60):
    out = np.zeros_like(components[0][0], dtype=np.float64)
    touched = np.zeros_like(out, dtype=bool)
    for scores, weight in components:
        for j, row in enumerate(scores):
            ids = ordered(row, numeric_ids)
            out[j, ids] += weight/(k0+np.arange(1, len(ids)+1))
            if weight > 0:
                touched[j, ids] = True
    out[~touched] = np.nan
    return out.astype(np.float32)


def _check_native(ctx, scores, source):
    """All stored ranks exact; scores must agree at their saved float32 precision."""
    from experiments import finalize_scores
    pred, _, indices, values = finalize_scores(scores.copy(), ctx)
    with np.load(source/'native-top100.npz', allow_pickle=False) as z:
        assert np.array_equal(indices, z['indices']), str(source)+' ranks differ'
        assert np.allclose(values, z['scores'], rtol=1e-6, atol=2e-7, equal_nan=True), str(source)+' scores differ'
        exact = np.array_equal(values, z['scores'], equal_nan=True)
        finite = np.isfinite(values)
        error = float(np.max(np.abs(values[finite]-z['scores'][finite]), initial=0))
        checked = int(finite.sum())
    assert pred == load(source/'top5.json'), str(source)+' Top-5 differs'
    return {'native_ranks_exact': True, 'native_top5_exact': True,
            'native_float32_scores_exact': exact, 'max_absolute_score_error': error,
            'scores_checked': checked, 'native_sha256': sha256(source/'native-top100.npz')}


def _sources(ctx, source_root):
    result = {'engine_v5': sha256(ENGINE_PATH)}
    assert result['engine_v5'] == ENGINE_SHA256, 'Frozen feature engine changed'
    for name, expected in DEPENDENCIES.items():
        result[name] = sha256(ROOT/name)
        assert result[name] == expected, 'Inference dependency changed: '+name
    for key, name in RUNS.items():
        run = source_root/name
        record = load(run/'record.json')
        assert record['status'] == 'SUCCEEDED'
        assert record['protocol_sha256'] == ctx.protocol_hash
        assert record['input_hashes'] == ctx.protocol['input_hashes']
        for file in ('record.json', 'native-top100.npz', 'top5.json'):
            result[key+'/'+file] = sha256(run/file)
        if key in ('l4', 'l5'):
            for file in ('catboost-ranker.cbm', 'ranker-training-provenance.json'):
                result[key+'/'+file] = sha256(run/file)
    assert load(source_root/RUNS['ppr']/'record.json')['parameters'] == PPR
    assert load(source_root/RUNS['hourly']/'record.json')['parameters'] == HOURLY
    return result


def _load_cache(ctx, cache, sources):
    manifest = load(cache/'manifest.json')
    assert manifest['protocol_sha256'] == ctx.protocol_hash
    assert manifest['source_sha256'] == SOURCE_SHA256
    assert manifest['input_sources_sha256'] == sources
    assert load(cache/'mapping.json') == {'targets': ctx.targets, 'candidates': ctx.candidates}
    for name, expected in manifest['payload_sha256'].items():
        assert sha256(cache/name) == expected, 'Full-score cache modified: '+name
    assert manifest['truncated_reference_replay_passed']
    return manifest


def prepare_scores(ctx, source_root=None):
    """Reconstruct all scores once, and refuse fusion if any reference replay fails."""
    from catboost import CatBoostClassifier
    import advanced_collaborative
    import advanced_hourly_schedule

    source_root = Path(source_root or ROOT/'artifacts/runs'/ctx.path.name)
    sources = _sources(ctx, source_root)
    cache = ROOT/'artifacts/runs/full-fusion-cache-v1'/ctx.protocol_hash/SOURCE_SHA256
    if (cache/'manifest.json').exists():
        return cache, _load_cache(ctx, cache, sources)
    cache.mkdir(parents=True, exist_ok=True)
    if any(cache.glob('*.npy')):
        raise ValueError('Incomplete full-score cache retained; do not overwrite')
    spec = importlib.util.spec_from_file_location('_full_fusion_frozen_ranker_v5', ENGINE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    numeric = np.array([int(i) for i in ctx.candidates], dtype=np.int64)
    values, checks, details = {}, {}, {}
    for key, function, params in [('ppr', advanced_collaborative.ppr, PPR),
                                  ('hourly', advanced_hourly_schedule.schedule_hourly, HOURLY)]:
        folder = cache/('reconstruction-'+key)
        folder.mkdir(exist_ok=True)
        values[key], details[key] = function(ctx, params, folder)
        checks[key] = _check_native(ctx, values[key], source_root/RUNS[key])
        np.save(cache/(key+'.npy'), values[key], allow_pickle=False)
        print('native replay passed:', key, checks[key], flush=True)
    # Base 56 are constructed with strong_context=False. Adding the v5 strong
    # context then appends four columns; every pre-existing column remains intact.
    engine = module.FeatureEngine(ctx, {'strong_context': False})
    assert len(engine.names) == 56
    names56 = engine.names[:]
    for key in ('l4', 'l5'):
        run = source_root/RUNS[key]
        record, provenance = load(run/'record.json'), load(run/'ranker-training-provenance.json')
        expected_code = L4_SHA256 if key == 'l4' else ENGINE_SHA256
        assert provenance['source_code_sha256'] == expected_code
        assert record['model_details']['feature_code_sha256'] == expected_code
        assert provenance['evaluation_cutoff_exclusive'] == ctx.protocol['train_cutoff_exclusive']
        assert provenance['all_label_end_dates_at_or_before_evaluation_cutoff']
        assert not provenance['fitting_evaluation_truth_accessed']
        assert max(s['labels_end_exclusive'] for s in provenance['sources']) <= ctx.protocol['train_cutoff_exclusive']
        assert record['parameters']['objective'] == 'Logloss'
        if key == 'l5':
            engine._strong_context(record['parameters'], cache/'reconstruction-l5-features')
            assert engine.names[:56] == names56 and len(engine.names) == 60
        assert engine.names == provenance['feature_names']
        model = CatBoostClassifier()
        model.load_model(str(run/'catboost-ranker.cbm'))
        assert model.feature_names_ == engine.names
        scores = np.empty((len(ctx.targets), len(ctx.candidates)), dtype=np.float32)
        for j in range(len(ctx.targets)):
            scores[j] = model.predict_proba(engine.matrix(j), thread_count=1)[:, 1]
        values[key] = scores
        checks[key] = _check_native(ctx, scores, run)
        np.save(cache/(key+'.npy'), scores, allow_pickle=False)
        details[key] = {'saved_model_sha256': sha256(run/'catboost-ranker.cbm'),
                        'training_provenance_sha256': sha256(run/'ranker-training-provenance.json'),
                        'training_feature_source_sha256': expected_code,
                        'inference_feature_engine_sha256': ENGINE_SHA256,
                        'feature_names': engine.names[:], 'all_candidates_scored': True,
                        'model_refit': False}
        print('native replay passed:', key, checks[key], flush=True)
    del engine
    # Gate is applied exactly where the old finalizer applied it. Cold native
    # scores cannot affect either intermediate/reference or final fusion.
    for scores in values.values():
        scores[ctx.cold_mask] = np.nan
    old = {key: truncate(scores, numeric) for key, scores in values.items()}
    old_core = max_fusion([(old['ppr'], 1), (old['hourly'], 1)])
    checks['core-truncated'] = _check_native(ctx, old_core, source_root/RUNS['core'])
    old['core'] = truncate(old_core, numeric)
    for bundle, components in BUNDLES.items():
        replay = rrf([(old[key], weight) for key, weight in components], numeric)
        checks[bundle+'-truncated'] = _check_native(ctx, replay, source_root/RUNS[bundle])
        # Both reference algorithms operate entirely on saved float32 scores/ranks.
        assert checks[bundle+'-truncated']['native_float32_scores_exact']
    checks['truncated_scores_exact'] = checks['core-truncated']['native_float32_scores_exact']
    assert checks['truncated_scores_exact']
    core = max_fusion([(values['ppr'], 1), (values['hourly'], 1)])
    np.save(cache/'core.npy', core, allow_pickle=False)
    dump(cache/'mapping.json', {'targets': ctx.targets, 'candidates': ctx.candidates})
    manifest = {'source_sha256': SOURCE_SHA256, 'protocol_sha256': ctx.protocol_hash,
                'input_sources_sha256': sources, 'input_hashes': ctx.protocol['input_hashes'],
                'shape': list(core.shape), 'checks': checks, 'reconstruction_details': details,
                'truncated_reference_replay_passed': True,
                'fit_performed': False, 'validation_truth_used_for_scoring': False,
                'future_features': 'schedule ID/channel/start/duration availability only',
                'payload_sha256': {p.name: sha256(p) for p in sorted(cache.glob('*.npy'))}}
    manifest['payload_sha256']['mapping.json'] = sha256(cache/'mapping.json')
    dump(cache/'manifest.json', manifest)
    print('all truncated references exactly reproduced; full catalogue ready', flush=True)
    return cache, manifest


def full_fusion(ctx, parameters, path):
    bundle = parameters.get('bundle', 'shfl4')
    if bundle not in BUNDLES or set(parameters)-{'bundle'}:
        raise ValueError('Only the two fixed controlled bundles are supported')
    cache, manifest = prepare_scores(ctx)
    numeric = np.array([int(i) for i in ctx.candidates], dtype=np.int64)
    components = [(np.load(cache/(key+'.npy'), allow_pickle=False), weight)
                  for key, weight in BUNDLES[bundle]]
    scores = rrf(components, numeric, 60)
    scores[ctx.cold_mask] = np.nan
    np.save(path/'full-fusion-scores.npy', scores, allow_pickle=False)
    dump(path/'full-score-provenance.json', {'cache': str(cache),
         'manifest_sha256': sha256(cache/'manifest.json'),
         'full_fusion_scores_sha256': sha256(path/'full-fusion-scores.npy'),
         'mapping_sha256': sha256(cache/'mapping.json'),
         'protocol_sha256': ctx.protocol_hash, 'source_sha256': SOURCE_SHA256})
    return scores, {'algorithm': 'full-catalogue max-score core then weighted RRF k60',
                    'bundle': bundle, 'fixed_components': BUNDLES[bundle],
                    'catalogue_size': len(ctx.candidates), 'native_truncation': None,
                    'reference_top100_replay_passed': manifest['truncated_reference_replay_passed'],
                    'saved_models_refit': False, 'truth_used_for_scoring': False,
                    'cache_manifest_sha256': sha256(cache/'manifest.json'),
                    'reference_run': RUNS[bundle]}


def register():
    METHODS['full_catalogue_fusion'] = full_fusion
