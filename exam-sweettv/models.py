"""Real train-only recommenders. Scores are ranking signals, not probabilities."""
from __future__ import annotations

from collections import Counter
import re
import unicodedata

import numpy as np
from scipy import sparse
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from implicit.cpu.als import AlternatingLeastSquares

from etl import read_events
from repro import dump, load, sha256


class Context:
    def __init__(self, path):
        self.path = path
        self.protocol = load(path / "protocol.json")
        for name, expected in self.protocol["frozen_hashes"].items():
            if sha256(path / name) != expected:
                raise ValueError(f"Frozen ETL artifact modified: {name}")
        self.protocol_hash = sha256(path / "protocol.json")
        self.cohorts = load(path / "cohorts.json")
        self.targets = self.cohorts["all_template"]
        self.truth = load(path / "truth.json")
        self.history = load(path / "histories.json")
        self.metadata = load(path / "metadata-train.json")
        self.display = load(path / "metadata-display-only.json")
        self.candidates = (path / "candidates.txt").read_text().splitlines()
        self.candidate_map = {item: i for i, item in enumerate(self.candidates)}
        self.fallback = load(path / "fallback.json")
        self.cold_mask = np.array([not self.history.get(u) for u in self.targets])
        self.users = sorted(set(self.targets) | {u for u in self.history if self.history[u]})
        self.user_map = {u: i for i, u in enumerate(self.users)}
        self.items = sorted({i for u in self.history.values() for i in u}, key=int)
        self.item_map = {i: j for j, i in enumerate(self.items)}
        self.target_rows = np.array([self.user_map[u] for u in self.targets])
        self._events = None

    def events(self):
        if self._events is None:
            eu, ei, _, e = read_events(self.path / "train-events.npz")
            self._events = (np.array([self.user_map[u] if u in self.user_map else -1 for u in eu]),
                            np.array([self.item_map[i] if i in self.item_map else -1 for i in ei]), e)
        return self._events

    def strength(self, parameters):
        eu, ei, e = self.events()
        u, i = eu[e[:, 0]], ei[e[:, 1]]
        assert np.all(u >= 0) and np.all(i >= 0)
        weights = e[:, 5].astype(np.float64)/3600 if parameters.get("strength") == "log_seconds" else np.ones(len(e))
        half_life = parameters.get("half_life_days")
        if half_life:
            age = (self.protocol["cutoff_elapsed_seconds"]-e[:, 6])/86400
            weights *= np.exp2(-age/half_life)
        matrix = sparse.csr_matrix((weights, (u, i)), shape=(len(self.users), len(self.items)), dtype=np.float32)
        if parameters.get("strength") in ("log_count", "log_seconds"):
            matrix.data = np.log1p(matrix.data)
        return matrix


def normalized_text(value):
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def year_tokens(values):
    result = set()
    for value in values:
        match = re.fullmatch(r"\s*(\d{4})(?:\s*[-–]\s*(\d{4})?)?\s*", value)
        if not match:
            continue
        year = int(match[1])
        result.add(f"start_decade:{year//10*10}")
        if "-" in value or "–" in value:
            if match[2]:
                result.add(f"end_decade:{int(match[2])//10*10}")
                result.add("closed_range")
            else:
                result.add("ongoing")
    return sorted(result)


def content_blocks(metadata, weights):
    """Fit each vocabulary on unique historical programme documents only."""
    items = sorted(metadata, key=int)
    docs = [metadata[i] for i in items]
    blocks, schemas = [], {}
    for field, weight in weights.items():
        if weight <= 0:
            continue
        tokens = []
        for doc in docs:
            if field == "category":
                values = doc["tv_show_category"]
            elif field == "genres":
                values = [v for f in ("tv_show_genre_1", "tv_show_genre_2", "tv_show_genre_3") for v in doc[f]]
            elif field == "year":
                values = year_tokens(doc["year_of_production"])
            elif field in ("director", "actors"):
                # Actual source examples are comma-separated names, not whitespace tokens.
                values = [name.strip() for v in doc[field] for name in re.split(r"[,;]", v)]
            elif field == "title":
                values = doc["tv_show_title"]
            else:
                raise ValueError(field)
            tokens.append(sorted({normalized_text(v) for v in values if normalized_text(v) not in ("", "-", "нет", "n/a", "null")}))
        if field == "title":
            vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=30000,
                                         sublinear_tf=True, dtype=np.float32)
            matrix = vectorizer.fit_transform([" ".join(v) for v in tokens])
        else:
            vectorizer = DictVectorizer(dtype=np.float32)
            matrix = vectorizer.fit_transform([{v: 1.0 for v in values} for values in tokens])
        matrix = normalize(matrix, norm="l2")
        blocks.append(matrix*weight)
        schemas[field] = {"weight": weight, "features": matrix.shape[1],
                          "vocabulary": sorted(vectorizer.vocabulary_)}
    if not blocks:
        raise ValueError("No content blocks")
    vectors = normalize(sparse.hstack(blocks, format="csr"), norm="l2")
    return items, vectors, schemas


def popularity(ctx, parameters, path):
    counts = Counter()
    for history in ctx.history.values():
        for item, row in history.items():
            counts[item] += row["airings"] if parameters.get("frequency") else 1
    one = np.array([counts.get(i, np.nan) for i in ctx.candidates], dtype=np.float32)
    return np.tile(one, (len(ctx.targets), 1)), {"native_items": int(np.isfinite(one).sum())}


def repeat(ctx, parameters, path):
    matrix = ctx.strength({**parameters, "strength": "count"})[ctx.target_rows]
    scores = np.full((len(ctx.targets), len(ctx.candidates)), np.nan, dtype=np.float32)
    train_opps = load(ctx.path / "opportunities.json")["train"]
    future_opps = load(ctx.path / "opportunities.json")["future"]
    for i, item in enumerate(ctx.items):
        if item not in ctx.candidate_map:
            continue
        values = matrix[:, i].toarray().ravel()
        if parameters.get("opportunity_adjustment"):
            prior = parameters.get("prior_strength", 10)
            # Native rate only for personal positive items; no popularity fallback votes.
            values = values/(train_opps.get(item, 0)+prior)*future_opps.get(item, 0)
        scores[:, ctx.candidate_map[item]] = np.where(values > 0, values, np.nan)
    return scores, {"decay": "sum of individual full-airing weights by last observed stop",
                    "opportunity_adjustment": bool(parameters.get("opportunity_adjustment"))}


def cbf(ctx, parameters, path):
    weights = parameters.get("blocks", {"category": 1, "genres": 1.5, "director": .5, "actors": .5, "year": .25})
    items, vectors, schemas = content_blocks(ctx.metadata, weights)
    im = {i: j for j, i in enumerate(items)}
    train_columns = np.array([im[i] for i in ctx.items])
    history = ctx.strength(parameters)[ctx.target_rows]
    features = vectors[train_columns]
    profiles = normalize(history @ features, norm="l2")
    candidate_vectors = sparse.vstack([vectors[im[i]] if i in im else sparse.csr_matrix((1, vectors.shape[1]))
                                      for i in ctx.candidates], format="csr")
    scores = (profiles @ candidate_vectors.T).toarray().astype(np.float32)
    scores[scores <= 0] = np.nan
    sparse.save_npz(path / "cbf-item-vectors.npz", candidate_vectors)
    sparse.save_npz(path / "cbf-user-profiles.npz", profiles)
    dump(path / "cbf-vocabulary.json", schemas)
    restored = (sparse.load_npz(path / "cbf-user-profiles.npz") @ sparse.load_npz(path / "cbf-item-vectors.npz").T).toarray()
    assert np.allclose(np.nan_to_num(scores), restored, atol=1e-6)
    return scores, {"blocks": {f: {"weight": v["weight"], "features": v["features"]} for f, v in schemas.items()},
                    "total_features": vectors.shape[1], "metadata_fit_cutoff": ctx.protocol["train_cutoff_exclusive"],
                    "future_metadata_used": False, "saved_model_reload_verified": True}


def als(ctx, parameters, path):
    strength = ctx.strength(parameters)
    confidence = strength.copy()
    confidence.data = 1 + parameters.get("confidence_scale", 10)*confidence.data
    model = AlternatingLeastSquares(factors=parameters.get("factors", 64),
                                    regularization=parameters.get("regularization", .1),
                                    iterations=parameters.get("iterations", 20), alpha=1,
                                    num_threads=1, random_state=parameters.get("seed", 42))
    model.fit(confidence, show_progress=False)
    assert model.user_factors.shape == (len(ctx.users), model.factors)
    assert model.item_factors.shape == (len(ctx.items), model.factors)
    assert np.isfinite(model.user_factors).all() and np.isfinite(model.item_factors).all()
    # Score every warm candidate. No randomly initialized vectors for cold items.
    candidate_cols = [(j, ctx.item_map[i]) for j, i in enumerate(ctx.candidates) if i in ctx.item_map]
    scores = np.full((len(ctx.targets), len(ctx.candidates)), np.nan, dtype=np.float32)
    out_cols, in_cols = map(np.array, zip(*candidate_cols))
    scores[:, out_cols] = model.user_factors[ctx.target_rows] @ model.item_factors[in_cols].T
    model.save(path / "als-model.npz")
    dump(path / "als-mappings.json", {"users": ctx.users, "items": ctx.items, "candidate_items": ctx.candidates})
    restored = AlternatingLeastSquares.load(path / "als-model.npz")
    assert np.array_equal(restored.user_factors, model.user_factors)
    assert np.array_equal(restored.item_factors, model.item_factors)
    # Cross-check library native indices against our complete dot-product scoring.
    u = next(ctx.user_map[x] for x in ctx.targets if ctx.history.get(x))
    native_ids, native_scores = model.recommend(u, confidence[u], N=min(20, len(in_cols)),
                                               items=in_cols, filter_already_liked_items=False)
    expected = model.user_factors[u] @ model.item_factors[native_ids].T
    assert np.allclose(expected, native_scores, rtol=1e-4, atol=1e-5)
    return scores, {"shape": list(confidence.shape), "nnz": confidence.nnz,
                    "confidence": "positive entries 1+scale*strength; missing preference=0/confidence=1",
                    "library_alpha": 1, "filter_already_liked_items": False,
                    "scoring": "full eligible warm candidate dot products; numeric ID tie break outside library",
                    "native_api_mapping_crosscheck": True, "saved_model_reload_verified": True,
                    "factor_norms": {"users": float(np.linalg.norm(model.user_factors)),
                                     "items": float(np.linalg.norm(model.item_factors))}}


def habits(ctx, parameters, path):
    with np.load(ctx.path / "habits.npz", allow_pickle=False) as z:
        channel, slots = z["channel_seconds"], z["slot_seconds"]
    channel = channel/np.maximum(channel.sum(axis=1, keepdims=True), 1)
    slots = slots/np.maximum(slots.sum(axis=1, keepdims=True), 1)
    if parameters.get("channel_only"):
        slots[:] = 1/8
    profile = (channel[:, :, None]*slots[:, None, :]).reshape(len(ctx.targets), -1)
    opportunities = sparse.load_npz(ctx.path / "opportunity-slots.npz")
    scores = np.asarray(profile @ opportunities.T).astype(np.float32)
    scores[scores <= 0] = np.nan
    return scores, {"context": "all pre-cutoff union channel seconds, including unlabelled airtime",
                    "time_slots": "8 three-hour local-time bins; channel and time marginals",
                    "availability": "future scheduled hours by item/channel/time; no future content or views"}


def hybrid(ctx, parameters, path):
    scores = np.zeros((len(ctx.targets), len(ctx.candidates)), dtype=np.float64)
    touched = np.zeros_like(scores, dtype=bool)
    sources = []
    for component in parameters["components"]:
        run = path.parent / component["run_id"]
        record = load(run / "record.json")
        if record["protocol_sha256"] != ctx.protocol_hash:
            raise ValueError("Cannot fuse different protocols")
        z = np.load(run / "native-top100.npz", allow_pickle=False)
        indices = z["indices"]
        for u in range(len(ctx.targets)):
            valid = indices[u][indices[u] >= 0]
            scores[u, valid] += component["weight"]/(parameters.get("k0", 60)+np.arange(1, len(valid)+1))
            if component["weight"] > 0:
                touched[u, valid] = True
        sources.append({"run_id": component["run_id"], "weight": component["weight"],
                        "native_rank_hash": sha256(run / "native-top100.npz")})
    scores[~touched] = np.nan
    return scores.astype(np.float32), {"sources": sources, "fusion": "weighted reciprocal native ranks; fallback only after fusion"}


METHODS = {"popularity": popularity, "repeat": repeat, "cbf": cbf, "als": als, "habits": habits, "hybrid": hybrid}
