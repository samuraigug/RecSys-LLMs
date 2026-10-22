"""Cutoff-safe CatBoost reranker; no evaluation truth is available to fitting.

Fitting examples use independently frozen historical pseudo-futures ending no later
than the evaluated history cutoff. Candidate/cohort/fallback handling remains in
experiments.run. Candidate-negative sampling is confined to training; scoring uses
all eligible candidates. Neither programme nor user ID is a model feature.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime
import importlib.metadata
from pathlib import Path
import time

import numpy as np
from scipy import sparse
from scipy.stats import rankdata
from sklearn.preprocessing import normalize

from etl import read_events
from models import Context, METHODS, content_blocks
from repro import ROOT, dump, load, sha256

FOLDS = {
    'ranker-f1': ('2020-03-09', '2020-03-23', '2020-04-06'),
    'ranker-f2': ('2020-03-09', '2020-03-30', '2020-04-20'),
    'ranker-f3': ('2020-03-09', '2020-04-20', '2020-05-18'),
}
FOLD_ROOT = ROOT / 'data/processed/ranker-pseudofuture-v1'


class FeatureEngine:
    """Feature construction accesses only history and future ID/time availability."""
    def __init__(self, ctx):
        self.ctx = ctx
        self.n, self.m = len(ctx.targets), len(ctx.candidates)
        self.u = {u: j for j, u in enumerate(ctx.targets)}
        self.c = ctx.candidate_map
        self.signals, self.names = [], []
        self.count = np.zeros((self.n, self.m), dtype=np.float32)
        self.seconds = np.zeros_like(self.count)
        self.age = np.full_like(self.count, 1000)
        cutoff = ctx.protocol['cutoff_elapsed_seconds']
        for user, j in self.u.items():
            for item, row in ctx.history.get(user, {}).items():
                if item in self.c:
                    k = self.c[item]
                    self.count[j, k] = row['airings']
                    self.seconds[j, k] = row['seconds']/3600
                    self.age[j, k] = max(0, (cutoff-row['last_stop'])/86400)
        opp = load(ctx.path/'opportunities.json')
        self.train_opp = np.array([opp['train'].get(i, 0) for i in ctx.candidates], dtype=np.float32)
        self.future_opp = np.array([opp['future'].get(i, 0) for i in ctx.candidates], dtype=np.float32)
        self.add('personal_full_count', self.count)
        self.add('personal_full_hours', self.seconds)
        self.add('personal_last_age_days', self.age)
        self.add('repeat_share', self.count/np.maximum(1, self.count.sum(axis=1, keepdims=True)))
        self.add('repeat_count_rank', self.rank(self.count))
        self.repeat_rate = self.count/(50+self.train_opp)*self.future_opp
        for prior in (1, 10, 50, 100):
            self.add(f'repeat_projected_prior{prior}', self.count/(prior+self.train_opp)*self.future_opp)
        self.add('repeat_projected_rank', self.rank(self.repeat_rate))
        self.add('train_airing_opportunities', self.train_opp)
        self.add('future_airing_opportunities', self.future_opp)
        self.add('opportunity_growth', self.future_opp/(1+self.train_opp))
        train_days = (datetime.fromisoformat(ctx.protocol['train_cutoff_exclusive'])-datetime.fromisoformat(ctx.protocol['train_start'])).days
        future_days = (datetime.fromisoformat(ctx.protocol['forecast_end_exclusive'])-datetime.fromisoformat(ctx.protocol['train_cutoff_exclusive'])).days
        self.add('history_days', np.full(self.n, train_days, dtype=np.float32), user=True)
        self.add('forecast_days', np.full(self.n, future_days, dtype=np.float32), user=True)
        self.add('future_to_train_days', np.full(self.n, future_days/max(train_days, 1), dtype=np.float32), user=True)
        self.add('user_total_full_count', np.array([sum(r['airings'] for r in ctx.history.get(u, {}).values()) for u in ctx.targets], dtype=np.float32), user=True)
        self.add('user_unique_programmes', np.array([len(ctx.history.get(u, {})) for u in ctx.targets], dtype=np.float32), user=True)
        global_count, global_users = np.zeros(self.m, np.float32), np.zeros(self.m, np.float32)
        for hist in ctx.history.values():
            for item, row in hist.items():
                if item in self.c:
                    k = self.c[item]
                    global_count[k] += row['airings']
                    global_users[k] += 1
        self.add('programme_full_count', global_count)
        self.add('programme_full_users', global_users)
        self.add('programme_airings_per_user', global_count/np.maximum(1, global_users))
        self.add('programme_per_opportunity', global_count/(1+self.train_opp))
        self.add('programme_projected', global_count/(50+self.train_opp)*self.future_opp)
        self.add('programme_is_unseen', (global_users == 0).astype(np.float32))
        self._event_features(cutoff)
        self._context_features(ctx)
        # Global and user feature types are explicit: prevents accidental dimension ambiguity.
        assert len(self.names) == len(set(self.names))
        print(f'features prepared: {self.n} users x {self.m} candidates x {len(self.names)}', flush=True)

    def add(self, name, values, user=False):
        self.names.append(name)
        self.signals.append((values, 'user' if user else ('item' if values.ndim == 1 else 'pair')))

    @staticmethod
    def rank(values):
        # Identical scores receive identical ranks: tie order must not smuggle IDs
        # into features, and numeric IDs are used only by the shared finalizer.
        return rankdata(-values, axis=1, method='min').astype(np.float32)

    def _event_features(self, cutoff):
        eu, ei, _, events = read_events(self.ctx.path/'train-events.npz', full_only=False)
        if len(events):
            assert events[:, 6].max() <= cutoff
        umap = np.array([self.u.get(u, -1) for u in eu])
        imap = np.array([self.c.get(i, -1) for i in ei])
        ur, ir = umap[events[:, 0]], imap[events[:, 1]]
        full = events[:, 5] >= .8*events[:, 7]
        age = (cutoff-events[:, 6])/86400
        legal = (ur >= 0) & (ir >= 0)
        coverage = events[:, 5]/events[:, 7]
        for threshold in (.3, .5):
            matrix = np.zeros_like(self.count)
            selected = legal & (coverage >= threshold)
            np.add.at(matrix, (ur[selected], ir[selected]), 1)
            suffix = int(threshold*100)
            self.add(f'personal_count_ge{suffix}pct', matrix)
            self.add(f'personal_projected_ge{suffix}pct', matrix/(50+self.train_opp)*self.future_opp)
            self.add(f'personal_rank_ge{suffix}pct', self.rank(matrix))
        squared = np.zeros_like(self.count)
        np.add.at(squared, (ur[legal], ir[legal]), coverage[legal]**2)
        self.add('coverage_squared_sum', squared)
        self.add('coverage_squared_projected', squared/(100+self.train_opp)*self.future_opp)
        self.add('coverage_squared_rank', self.rank(squared/(100+self.train_opp)*self.future_opp))
        user_valid = ur >= 0
        user_first = np.full(self.n, cutoff, dtype=np.int64)
        user_last = np.full(self.n, 0, dtype=np.int64)
        np.minimum.at(user_first, ur[user_valid], events[user_valid, 3])
        np.maximum.at(user_last, ur[user_valid], events[user_valid, 6])
        self.add('user_first_labelled_airing_age', ((cutoff-user_first)/86400).astype(np.float32), user=True)
        self.add('user_last_labelled_view_age', ((cutoff-user_last)/86400).astype(np.float32), user=True)
        for days in (7, 14, 28):
            selected = user_valid & (age <= days)
            hours = np.bincount(ur[selected], weights=events[selected, 5]/3600, minlength=self.n).astype(np.float32)
            self.add(f'user_labelled_hours_last{days}d', hours, user=True)
        for days in (7, 14, 28):
            matrix = np.zeros_like(self.count)
            selected = legal & full & (age <= days)
            np.add.at(matrix, (ur[selected], ir[selected]), 1)
            self.add(f'personal_full_last{days}d', matrix)
            global_recent = np.bincount(ir[full & (ir >= 0) & (age <= days)], minlength=self.m).astype(np.float32)
            self.add(f'programme_full_last{days}d', global_recent)
        for label, weights in [('weak_event_count', np.ones(len(events))), ('weak_hours', events[:, 5]/3600),
                               ('coverage_sum', events[:, 5]/events[:, 7])]:
            matrix = np.zeros_like(self.count)
            np.add.at(matrix, (ur[legal], ir[legal]), weights[legal])
            self.add(label, matrix)
        decayed = np.zeros_like(self.count)
        np.add.at(decayed, (ur[legal & full], ir[legal & full]), np.exp2(-age[legal & full]/14))
        self.add('personal_decay14', decayed)
        self.add('personal_decay14_projected', decayed/(50+self.train_opp)*self.future_opp)
        del events

    def _context_features(self, ctx):
        with np.load(ctx.path/'habits.npz', allow_pickle=False) as z:
            channel, slots = z['channel_seconds'], z['slot_seconds']
        channel = channel/np.maximum(channel.sum(axis=1, keepdims=True), 1)
        slots = slots/np.maximum(slots.sum(axis=1, keepdims=True), 1)
        opportunities = sparse.load_npz(ctx.path/'opportunity-slots.npz')
        profile = (channel[:, :, None]*slots[:, None, :]).reshape(self.n, -1)
        habit = np.asarray(profile@opportunities.T).astype(np.float32)
        self.add('habit_future_hours', habit)
        self.add('habit_rank', self.rank(habit))
        self.add('future_hours', np.asarray(opportunities.sum(axis=1)).ravel().astype(np.float32))
        items, vectors, _ = content_blocks(ctx.metadata, {'category': .25, 'genres': .5, 'director': .25, 'actors': .25, 'year': .1, 'title': 2})
        imap = {i: j for j, i in enumerate(items)}
        hist = ctx.strength({'strength': 'log_count'})
        vectors_train = vectors[[imap[i] for i in ctx.items]]
        profile = normalize(hist[ctx.target_rows]@vectors_train)
        cand = sparse.vstack([vectors[imap[i]] if i in imap else sparse.csr_matrix((1, vectors.shape[1])) for i in ctx.candidates], format='csr')
        cbf = (profile@cand.T).toarray().astype(np.float32)
        self.add('cbf_title_score', cbf)
        self.add('cbf_title_rank', self.rank(cbf))
        binary = hist.copy()
        binary.data[:] = 1
        # Shrunk item cosine; remove identity so the signal is genuinely collaborative.
        similarity = (binary.T@binary).tocsr()
        degree = np.sqrt(np.asarray(binary.sum(axis=0)).ravel())
        rr, cc = similarity.nonzero()
        similarity.data /= (degree[rr]*degree[cc]+20)
        similarity.setdiag(0)
        similarity.eliminate_zeros()
        for j in range(similarity.shape[0]):
            left, right = similarity.indptr[j:j+2]
            if right-left > 100:
                values = similarity.data[left:right]
                values[np.argpartition(values, -100)[:-100]] = 0
        similarity.eliminate_zeros()
        collaborative_train = hist[ctx.target_rows]@similarity
        cf = np.zeros_like(self.count)
        dst, src = [], []
        for item, j in ctx.candidate_map.items():
            if item in ctx.item_map:
                dst.append(j); src.append(ctx.item_map[item])
        cf[:, dst] = collaborative_train[:, src].toarray()
        self.add('itemcf_score', cf)
        self.add('itemcf_rank', self.rank(cf))
        self.retrievers = [self.repeat_rate, self.count, cf, cbf, habit]

    def matrix(self, user, indices=None):
        indices = np.arange(self.m) if indices is None else indices
        columns = []
        for values, kind in self.signals:
            if kind == 'user':
                column = np.full(len(indices), values[user], dtype=np.float32)
            elif kind == 'item':
                column = values[indices]
            else:
                column = values[user, indices]
            columns.append(column)
        return np.column_stack(columns).astype(np.float32)

    def sampled_training(self, rng, negatives=30, correct_sampling=False):
        x, y, groups, weights = [], [], [], []
        for j, user in enumerate(self.ctx.targets):
            truth = self.ctx.truth[user]
            if not truth or self.ctx.cold_mask[j]:
                continue
            ids = set(self.c[i] for i in truth if i in self.c)
            for signal, k in zip(self.retrievers, (100, 80, 60, 40, 40)):
                ids.update(np.argpartition(signal[j], -k)[-k:].tolist())
            if correct_sampling:
                # Positives and deterministic hard negatives are included with
                # certainty. Uniformly sample only the remaining negatives.
                # Horvitz-Thompson weights reproduce the full-catalogue loss in
                # expectation; otherwise unseen-item probabilities are inflated.
                outside = np.array(sorted(set(range(self.m))-ids), dtype=np.int32)
                random_ids = rng.choice(outside, min(negatives, len(outside)), replace=False)
                sampled_weight = len(outside)/len(random_ids) if len(random_ids) else 1.
                random_ids = set(random_ids.tolist())
                ids.update(random_ids)
            else:
                ids.update(rng.choice(self.m, min(negatives, self.m), replace=False).tolist())
                random_ids, sampled_weight = set(), 1.
            ids = np.array(sorted(ids), dtype=np.int32)
            target = set(truth)
            labels = np.array([float(self.ctx.candidates[i] in target) for i in ids], np.float32)
            weight = np.array([sampled_weight if i in random_ids else 1. for i in ids], np.float32)
            if correct_sampling:
                assert np.all(weight[labels > 0] == 1)
                assert np.isclose(weight.sum(), self.m, rtol=1e-6)
            x.append(self.matrix(j, ids)); y.append(labels); groups.append(np.full(len(ids), j, np.int64)); weights.append(weight)
        return np.vstack(x), np.concatenate(y), np.concatenate(groups), np.concatenate(weights)


def learned_ranker(ctx, parameters, path):
    from catboost import CatBoostClassifier, CatBoostRanker, Pool
    cutoff = ctx.protocol['train_cutoff_exclusive']
    selected_folds = parameters.get('folds', [name for name, dates in FOLDS.items() if dates[2] <= cutoff])
    if not selected_folds:
        raise ValueError('No historical labels available strictly before evaluated cutoff')
    fit_sources, xs, ys, gs, ws = [], [], [], [], []
    group_offset = 0
    feature_names = None
    rng = np.random.default_rng(parameters.get('seed', 42))
    for number, name in enumerate(selected_folds):
        folder = Path(parameters.get('fold_root', FOLD_ROOT))/name
        fold_ctx = Context(folder)
        protocol = fold_ctx.protocol
        assert protocol['forecast_end_exclusive'] <= cutoff, (name, cutoff)
        assert protocol['train_cutoff_exclusive'] < protocol['forecast_end_exclusive']
        _, _, _, label_events = read_events(folder/'validation-events.npz')
        label_last_observation = int(label_events[:, 6].max()) if len(label_events) else 0
        assert label_last_observation <= ctx.protocol['cutoff_elapsed_seconds']
        del label_events
        features = FeatureEngine(fold_ctx)
        if feature_names is not None:
            assert feature_names == features.names
        feature_names = features.names
        xx, yy, gg, ww = features.sampled_training(rng, parameters.get('random_negatives', 30), parameters.get('correct_sampling', False))
        xs.append(xx); ys.append(yy); gs.append(gg+group_offset); ws.append(ww)
        group_offset += len(fold_ctx.targets)
        fit_sources.append({'fold': name, 'protocol_sha256': fold_ctx.protocol_hash,
                            'train_cutoff_exclusive': protocol['train_cutoff_exclusive'],
                            'labels_end_exclusive': protocol['forecast_end_exclusive'],
                            'train_events_sha256': sha256(folder/'train-events.npz'),
                            'training_truth_sha256': sha256(folder/'truth.json'),
                            'training_targets': len(fold_ctx.targets),
                            'training_submission_sha256': protocol['input_hashes']['submission.csv'],
                            'label_last_observation_elapsed_seconds': label_last_observation,
                            'rows': len(yy), 'positive_rows': int(yy.sum()), 'groups': int(len(set(gg))),
                            'effective_full_catalogue_weight': float(ww.sum()), 'negative_sampling_corrected': bool(parameters.get('correct_sampling', False))})
        del features, fold_ctx
    train_x, train_y, train_g, train_w = np.vstack(xs), np.concatenate(ys), np.concatenate(gs), np.concatenate(ws)
    del xs, ys, gs, ws
    config = {'iterations': parameters.get('iterations', 500), 'depth': parameters.get('depth', 6),
              'learning_rate': parameters.get('learning_rate', .05), 'l2_leaf_reg': parameters.get('l2_leaf_reg', 10),
              'random_seed': parameters.get('seed', 42), 'thread_count': 1,
              'allow_writing_files': False, 'verbose': 100}
    objective = parameters.get('objective', 'YetiRank:mode=MAP;top=5')
    if parameters.get('correct_sampling') and objective != 'Logloss':
        raise ValueError('Object sampling weights apply to Logloss; YetiRank ignores object weights')
    config['loss_function'] = objective
    fit_start = time.perf_counter()
    if objective == 'Logloss':
        model = CatBoostClassifier(**config)
        model.fit(Pool(train_x, train_y, weight=train_w, feature_names=feature_names))
    else:
        model = CatBoostRanker(**config)
        model.fit(Pool(train_x, train_y, group_id=train_g, feature_names=feature_names))
    fit_seconds = time.perf_counter()-fit_start
    model.save_model(str(path/'catboost-ranker.cbm'))
    dump(path/'ranker-training-provenance.json', {'evaluation_cutoff_exclusive': cutoff,
         'sources': fit_sources, 'fitting_evaluation_truth_accessed': False,
         'all_label_end_dates_at_or_before_evaluation_cutoff': True,
         'no_user_or_item_id_features': True, 'feature_names': feature_names,
         'config': config, 'catboost_version': importlib.metadata.version('catboost'),
         'sources_docs': ['https://catboost.ai/docs/en/concepts/loss-functions-ranking',
                          'https://catboost.ai/docs/en/concepts/python-reference_catboostranker'],
         'training_rows': len(train_y), 'training_positives': int(train_y.sum()),
         'negative_sampling_corrected': bool(parameters.get('correct_sampling', False)),
         'sample_weights_sum': float(train_w.sum()),
         'source_code_sha256': sha256(__file__),
         'training_matrix_sha256_note': 'deterministically reproducible from frozen folds, seed, and feature code'})
    importance = model.get_feature_importance(type='PredictionValuesChange')
    dump(path/'ranker-feature-importance.json', dict(zip(feature_names, importance.tolist())))
    del train_x, train_y, train_g, train_w
    features = FeatureEngine(ctx)
    assert features.names == feature_names
    scores = np.empty((len(ctx.targets), len(ctx.candidates)), dtype=np.float32)
    for j in range(len(ctx.targets)):
        matrix = features.matrix(j)
        if objective == 'Logloss':
            scores[j] = model.predict_proba(matrix)[:, 1]
        else:
            scores[j] = model.predict(matrix)
    restored = type(model)()
    restored.load_model(str(path/'catboost-ranker.cbm'))
    check_features = features.matrix(0)
    original_check = model.predict_proba(check_features) if objective == 'Logloss' else model.predict(check_features)
    restored_check = restored.predict_proba(check_features) if objective == 'Logloss' else restored.predict(check_features)
    assert np.array_equal(original_check, restored_check)
    return scores, {'training_folds': fit_sources, 'feature_count': len(feature_names),
                    'training_seconds': fit_seconds, 'catboost_version': importlib.metadata.version('catboost'),
                    'fit_labels_end_exclusive_max': max(f['labels_end_exclusive'] for f in fit_sources),
                    'future_views_used_in_features': False, 'evaluation_truth_used_for_fit': False,
                    'all_candidates_scored': True, 'training_negative_sampling_only': True,
                    'saved_model_reload_verified': True, 'feature_code_sha256': sha256(__file__)}


def register():
    METHODS['learned_ranker'] = learned_ranker
