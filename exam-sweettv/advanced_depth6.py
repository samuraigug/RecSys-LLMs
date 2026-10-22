"""Explicit depth-six controlled experiment on the frozen L5 pipeline.

The feature engine is loaded from our retained v5 source snapshot so experiments
in other model families cannot silently change this controlled comparison.
"""
import importlib.util
from pathlib import Path

import numpy as np

from etl import read_events
from models import Context
from repro import ROOT, dump, sha256

ENGINE_PATH = ROOT/'artifacts/ranker-research/advanced_ranker-v5-source.py'
ENGINE_SHA256 = '54af6b09c507c8a7579230a73fdb06f915f4a03a5535edee8e09bf35a7f4afaf'


def load_engine():
    # An explicit expected hash is filled from the already retained local source.
    if sha256(ENGINE_PATH) != ENGINE_SHA256:
        raise ValueError('Frozen L5 feature engine changed')
    spec = importlib.util.spec_from_file_location('frozen_ranker_v5', ENGINE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.FeatureEngine


def fit_model(x, y, weights, groups, names, p):
    import lightgbm as lgb
    objective = p.get('objective', 'binary')
    config = dict(objective=objective, learning_rate=p.get('learning_rate', .05),
        num_leaves=p.get('num_leaves', 31), max_depth=6, min_data_in_leaf=30,
        lambda_l2=10., num_threads=1, verbosity=-1, seed=p.get('seed', 42),
        deterministic=True, force_col_wise=True)
    if objective == 'binary':
        data = lgb.Dataset(x, label=y, weight=weights, feature_name=names)
    elif objective == 'lambdarank':
        # Sampled pairwise ranking is a separate hypothesis; object weights
        # are deliberately not presented as unbiased pair-inclusion weights.
        if np.any(np.diff(groups) < 0):
            raise ValueError('Ranking groups must be contiguous and ordered')
        _, sizes = np.unique(groups, return_counts=True)
        assert sizes.sum() == len(y)
        config.update(label_gain=[0, 1], lambdarank_truncation_level=5)
        data = lgb.Dataset(x, label=y, group=sizes, feature_name=names)
    else:
        raise ValueError(objective)
    return lgb.train(config, data, num_boost_round=p.get('iterations', 300)), config


def depth6_ranker(ctx, p, path):
    import lightgbm as lgb
    engine = load_engine()
    rng = np.random.default_rng(p.get('seed', 42))
    blocks = []; sources = []; names = None; group_offset = 0
    for name in p['folds']:
        folder = ROOT/Path(p['fold_root'])/name
        train = Context(folder)
        assert train.protocol['forecast_end_exclusive'] <= ctx.protocol['train_cutoff_exclusive']
        _, _, _, label_events = read_events(folder/'validation-events.npz')
        assert not len(label_events) or label_events[:, 6].max() <= ctx.protocol['cutoff_elapsed_seconds']
        features = engine(train, p, path/'feature-sources'/name)
        assert names is None or names == features.names
        names = features.names
        correction = p.get('objective', 'binary') == 'binary'
        x, y, g, w = features.sampled_training(rng, 30, correction)
        blocks.append((x, y, g+group_offset, w))
        group_offset += len(train.targets)
        sources.append(dict(fold=name, protocol_sha256=train.protocol_hash,
            train_cutoff_exclusive=train.protocol['train_cutoff_exclusive'],
            labels_end_exclusive=train.protocol['forecast_end_exclusive'],
            label_last_observation=int(label_events[:, 6].max()) if len(label_events) else 0,
            truth_sha256=sha256(folder/'truth.json'), rows=len(y), positives=int(y.sum()),
            input_hashes=train.protocol['input_hashes'], feature_sources=features.details))
        del features, train, label_events
    x, y, g, w = (np.concatenate([b[k] for b in blocks]) for k in range(4))
    del blocks
    model, config = fit_model(x, y, w, g, names, p)
    model.save_model(str(path/'lightgbm-model.txt'))
    import hashlib
    def depth(node):
        return 0 if 'leaf_index' in node else 1+max(depth(node['left_child']), depth(node['right_child']))
    depths = [depth(t['tree_structure']) for t in model.dump_model()['tree_info']]
    assert max(depths) <= 6
    dump(path/'depth-and-sample-audit.json', dict(tree_depths=depths, maximum_depth=max(depths),
        arrays={k:dict(shape=list(a.shape), dtype=str(a.dtype), sha256=hashlib.sha256(a.tobytes()).hexdigest())
                for k,a in zip(('x','y','groups','weights'),(x,y,g,w))}))
    dump(path/'leafwise-training-provenance.json', dict(
        feature_engine_path=str(ENGINE_PATH), feature_engine_sha256=ENGINE_SHA256,
        feature_names=names, fit_sources=sources, config=config, parameters=p,
        lightgbm_version=lgb.__version__, source_code_sha256=sha256(__file__),
        training_rows=len(y), training_positive_rows=int(y.sum()),
        evaluation_truth_used_for_fit=False, future_views_used_in_features=False,
        sampling='inverse-inclusion weighted sampled binary loss' if p.get('objective','binary')=='binary'
                 else 'unweighted sampled pairwise loss; not unbiased full-catalogue ranking loss',
        documentation='https://lightgbm.readthedocs.io/en/v4.6.0/Parameters.html'))
    dump(path/'feature-importance.json', dict(zip(names, model.feature_importance(importance_type='gain').tolist())))
    del x, y, g, w
    features = engine(ctx, p, path/'feature-sources'/'evaluation-history-only')
    assert features.names == names
    restored = lgb.Booster(model_file=str(path/'lightgbm-model.txt'))
    scores = np.empty((len(ctx.targets), len(ctx.candidates)), np.float32)
    for j in range(len(ctx.targets)):
        matrix = features.matrix(j)
        scores[j] = model.predict(matrix, num_threads=1)
        if j in (0, len(ctx.targets)-1):
            np.testing.assert_array_equal(model.predict(matrix, num_threads=1),
                                          restored.predict(matrix, num_threads=1))
    return scores, dict(feature_engine_sha256=ENGINE_SHA256, feature_count=len(names),
        training_folds=sources, lightgbm_version=lgb.__version__, model_saved_and_reload_verified=True,
        evaluation_truth_used_for_fit=False, all_eligible_candidates_scored=True)


def register():
    from models import METHODS
    METHODS['depth6_ranker'] = depth6_ranker
