"""Frozen selected SHFL4DS inference for weeks 31–42, with no test-label access.

Supervised trees remain the validated main-window trees. Histories, graph,
content scores and routine features are rebuilt at the final July cutoff.
The two Top-100 truncations reproduce the selected historical pipeline exactly.
"""
import argparse
import csv
import importlib.util
from pathlib import Path
import time

import numpy as np
from catboost import CatBoostClassifier
from threadpoolctl import threadpool_limits

import advanced_full_fusion as ff
from experiments import finalize_scores
from models import Context
from protocol import validate_submission
from repro import ROOT, dump, load, sha256, now, environment, peak_memory_mb

SELECTED = 'SHFL4DS-rrf-equal'
SOURCE_RUNS = ROOT/'artifacts/runs/main'
MODEL_FILES = {'l4': ('L4-logloss-allusers300', 'catboost-ranker.cbm'),
               'l5': ('L5-logloss-allusers-strong300', 'catboost-ranker.cbm'),
               'ds': ('DS-nonrepeat-binary300', 'discovery.cbm')}


class Forbidden:
    def __getitem__(self, key):
        raise AssertionError('Outcome/display access forbidden during inference')
    def __iter__(self):
        raise AssertionError('Outcome/display access forbidden during inference')
    def get(self, *args):
        raise AssertionError('Outcome/display access forbidden during inference')


def compose(ctx, ppr, hourly, l4, ds):
    numeric = np.array([int(i) for i in ctx.candidates], np.int64)
    def top100(values):
        values = values.copy()
        values[ctx.cold_mask] = np.nan
        return ff.truncate(values, numeric)
    core = ff.max_fusion([(top100(ppr), 1), (top100(hourly), 1)])
    return ff.rrf([(top100(core), 1), (top100(l4), .25), (top100(ds), .25)], numeric, 60)


def lock_selection():
    record = load(SOURCE_RUNS/SELECTED/'record.json')
    assert record['status'] == 'SUCCEEDED'
    assert record['parameters'] == {'k0': 60, 'components': [
        {'run_id': 'SHF-PPR-recency-max-1', 'weight': 1},
        {'run_id': 'L4-logloss-allusers300', 'weight': .25},
        {'run_id': 'DS-nonrepeat-binary300', 'weight': .25}]}
    assert sha256(ff.ENGINE_PATH) == ff.ENGINE_SHA256
    for file, expected in ff.DEPENDENCIES.items():
        assert sha256(ROOT/file) == expected, file
    models = {}
    for key, (run, file) in MODEL_FILES.items():
        source = SOURCE_RUNS/run
        rec = load(source/'record.json')
        assert rec['status'] == 'SUCCEEDED' and rec['protocol_sha256'] == record['protocol_sha256']
        provenance_file = 'training-provenance.json' if key == 'ds' else 'ranker-training-provenance.json'
        provenance = load(source/provenance_file)
        if key == 'ds':
            assert not provenance['outer_truth_used']
            assert max(s['label_end'] for s in provenance['sources']) <= '2020-05-18'
        else:
            assert not provenance['fitting_evaluation_truth_accessed']
            assert provenance['all_label_end_dates_at_or_before_evaluation_cutoff']
            assert provenance['evaluation_cutoff_exclusive'] == '2020-05-18'
        models[key] = {'path': str((source/file).relative_to(ROOT)), 'sha256': sha256(source/file),
                       'provenance_sha256': sha256(source/provenance_file)}
    return {'selected_run': SELECTED, 'selection_record_sha256': sha256(SOURCE_RUNS/SELECTED/'record.json'),
            'selected_parameters': record['parameters'], 'main_validation_metrics': record['metrics'],
            'ppr_parameters': ff.PPR, 'hourly_parameters': ff.HOURLY,
            'models': models, 'feature_engine_sha256': ff.ENGINE_SHA256,
            'supervised_refit': False,
            'policy': 'Freeze validated supervised trees; rebuild history-derived signals through 2020-07-27.',
            'kaggle_score': None, 'test_labels_available': False}


def replay_main(output):
    ctx = Context(ROOT/'data/processed/elapsed-v2/main')
    cache, _ = ff.prepare_scores(ctx)
    source = SOURCE_RUNS/'DS-nonrepeat-binary300'
    manifest = load(source/'score-manifest.json')
    assert manifest['protocol_sha256'] == ctx.protocol_hash
    assert sha256(source/'piecewise-scores.npy') == manifest['scores_sha256']
    arrays = [np.load(cache/(name+'.npy'), allow_pickle=False) for name in ('ppr', 'hourly', 'l4')]
    scores = compose(ctx, *arrays, np.load(source/'piecewise-scores.npy', allow_pickle=False))
    predictions, _, indices, native = finalize_scores(scores, ctx)
    assert predictions == load(SOURCE_RUNS/SELECTED/'top5.json')
    with np.load(SOURCE_RUNS/SELECTED/'native-top100.npz', allow_pickle=False) as saved:
        np.testing.assert_array_equal(indices, saved['indices'])
        np.testing.assert_array_equal(native, saved['scores'])
    dump(output/'main-replay.json', {'status': 'PASS', 'all_1260_top5_exact': True,
        'all_native_top100_ranks_and_float32_scores_exact': True,
        'verified_existing_component_caches_used': True, 'new_model_fit': False,
        'map_at_5_main': load(SOURCE_RUNS/SELECTED/'record.json')['metrics']['map_at_5_main']})
    print('Main selected pipeline replay: exact Top5 and Top100 for all 1260 users', flush=True)


def predict_final(output):
    import advanced_collaborative as ac
    import advanced_hourly_schedule as ah
    started = time.perf_counter()
    ctx = Context(ROOT/'data/processed/elapsed-v2/final')
    assert ctx.protocol['train_cutoff_exclusive'] == '2020-07-27'
    assert ctx.protocol['forecast_end_exclusive'] == '2020-10-19'
    assert not any(ctx.truth.values())
    assert all(sha256(ROOT/'data/raw'/name) == value for name, value in ctx.protocol['input_hashes'].items())
    ctx.truth = ctx.display = ctx.cohorts = Forbidden()
    values, details = {}, {}
    for name, function, params in [('ppr', ac.ppr, ff.PPR), ('hourly', ah.schedule_hourly, ff.HOURLY)]:
        folder = output/name
        folder.mkdir()
        values[name], details[name] = function(ctx, params, folder)
        np.save(output/(name+'.npy'), values[name], allow_pickle=False)
        print('Final history scored:', name, flush=True)
    spec = importlib.util.spec_from_file_location('_frozen_final_engine', ff.ENGINE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    engine = module.FeatureEngine(ctx, {'strong_context': False})
    for key in ('l4', 'l5'):
        if key == 'l5':
            engine._strong_context({}, output/'strong-context')
        run, file = MODEL_FILES[key]
        model = CatBoostClassifier()
        model.load_model(str(SOURCE_RUNS/run/file))
        assert model.feature_names_ == engine.names
        scores = np.empty((len(ctx.targets), len(ctx.candidates)), np.float32)
        for j in range(len(ctx.targets)):
            scores[j] = model.predict_proba(engine.matrix(j), thread_count=1)[:, 1]
        values[key] = scores
        print('Final saved supervised model scored:', key, flush=True)
    run, file = MODEL_FILES['ds']
    model = CatBoostClassifier()
    model.load_model(str(SOURCE_RUNS/run/file))
    assert model.feature_names_ == engine.names
    values['ds'] = values.pop('l5')
    for j in range(len(ctx.targets)):
        ids = np.flatnonzero(engine.count[j] == 0)
        if len(ids):
            values['ds'][j, ids] = model.predict_proba(engine.matrix(j, ids), thread_count=1)[:, 1]
    del engine
    for name in ('l4', 'ds'):
        np.save(output/(name+'.npy'), values[name], allow_pickle=False)
    scores = compose(ctx, values['ppr'], values['hourly'], values['l4'], values['ds'])
    predictions, diagnostics, indices, native = finalize_scores(scores, ctx)
    np.savez_compressed(output/'native-top100.npz', indices=indices, scores=native)
    dump(output/'top5.json', predictions)
    dump(output/'fallback-diagnostics.json', diagnostics)
    submission = output/'final_submission.csv'
    with submission.open('x', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['user_id', 'tv_show_id'])
        writer.writerows((u, ' '.join(predictions[u])) for u in ctx.targets)
    validation = validate_submission(ROOT/'data/raw/submission.csv', submission, ctx.candidates)
    dump(output/'submission-validation.json', validation)
    dump(output/'final-inference.json', {'status': 'FINAL_INFERENCE_COMPLETE', 'completed_at': now(),
        'protocol_sha256': ctx.protocol_hash, 'input_hashes': ctx.protocol['input_hashes'],
        'history_cutoff': ctx.protocol['train_cutoff_exclusive'],
        'forecast_end': ctx.protocol['forecast_end_exclusive'],
        'targets': len(ctx.targets), 'candidates': len(ctx.candidates), 'cold_users': int(ctx.cold_mask.sum()),
        'future_viewing_used': False, 'future_content_used': False, 'supervised_refit': False,
        'truth_display_cohorts_access_forbidden': True,
        'runtime_seconds': time.perf_counter()-started, 'peak_memory_mb': peak_memory_mb(),
        'environment': environment(), 'component_details': details, 'csv_validation': validation,
        'kaggle_score': None,
        'source_sha256': {p.name: sha256(p) for p in [Path(__file__), ff.ENGINE_PATH, ROOT/'models.py', ROOT/'experiments.py']},
        'payload_sha256': {p.name: sha256(p) for p in output.iterdir() if p.is_file()}})
    print('Final submission validated:', validation, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--replay-main', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    dump(args.output/'selection.json', lock_selection())
    with threadpool_limits(limits=1):
        if args.replay_main:
            replay_main(args.output)
        else:
            predict_final(args.output)


if __name__ == '__main__':
    main()
