"""Train-only linear and graph recommenders; shared experiment protocol applies.

Sources: https://arxiv.org/abs/1905.03375 (EASE, Steck 2019),
https://github.com/benfred/implicit/blob/main/implicit/nearest_neighbours.py
(BM25 weighting). No future views, truth, or future content is read here.
"""
from __future__ import annotations

import numpy as np
from scipy import sparse, linalg
from sklearn.preprocessing import normalize
from implicit.nearest_neighbours import bm25_weight
from etl import read_events
from repro import dump, load, sha256


def _matrix(ctx, p):
    if p.get("weak_weight") is not None:
        eu, ei, _, e = read_events(ctx.path / "train-events.npz", full_only=False)
        um = np.array([ctx.user_map.get(u, -1) for u in eu])
        im = np.array([ctx.item_map.get(i, -1) for i in ei])
        u, i = um[e[:, 0]], im[e[:, 1]]
        keep = (u >= 0) & (i >= 0)
        e, u, i = e[keep], u[keep], i[keep]
        assert np.all(e[:, 6] <= ctx.protocol["cutoff_elapsed_seconds"])
        fraction = e[:, 5] / e[:, 7]
        weights = np.where(fraction >= .8, 1., p["weak_weight"] * fraction ** p.get("fraction_power", 2))
        if p.get("half_life_days"):
            weights *= np.exp2(-(ctx.protocol["cutoff_elapsed_seconds"]-e[:, 6])/86400/p["half_life_days"])
        x = sparse.csr_matrix((weights, (u, i)), shape=(len(ctx.users), len(ctx.items)))
        if p.get("strength") == "log_count":
            x.data = np.log1p(x.data)
    else:
        x = ctx.strength(p).astype(np.float64)
    # A disabled weak event is absent, including before binary transformation.
    x.eliminate_zeros()
    if p.get("strength", "binary") == "binary":
        x.data[:] = 1
    if p.get("row_normalize"):
        x = normalize(x, norm="l2")
    return x


def _scores(ctx, raw, p):
    scores = np.full((len(ctx.targets), len(ctx.candidates)), np.nan, np.float32)
    cols = [(j, ctx.item_map[i]) for j, i in enumerate(ctx.candidates) if i in ctx.item_map]
    out, inside = map(np.array, zip(*cols))
    values = np.asarray(raw[:, inside], dtype=np.float64)
    if p.get("opportunity_adjustment"):
        opps = load(ctx.path / "opportunities.json")
        prior = p.get("prior_strength", 50)
        ratios = np.array([opps["future"].get(ctx.candidates[j], 0) /
                           (opps["train"].get(ctx.candidates[j], 0) + prior) for j in out])
        values *= ratios
    scores[:, out] = values
    if p.get("positive_only", True):
        scores[scores <= 0] = np.nan
    return scores


def _details(ctx, x):
    return {"shape": list(x.shape), "nnz": x.nnz,
            "fit_cutoff": ctx.protocol["train_cutoff_exclusive"],
            "future_views_used": False, "future_metadata_used": False,
            "repeat_filter": False, "scoring": "all eligible warm items; common fallback"}


def ease(ctx, parameters, path):
    x = _matrix(ctx, parameters)
    gram = (x.T @ x).toarray()
    regularization = parameters.get("regularization", 100)
    if regularization <= 0:
        raise ValueError("EASE regularization must be positive")
    gram.flat[::len(ctx.items)+1] += regularization
    inv = linalg.cho_solve(linalg.cho_factor(gram, lower=True, check_finite=True),
                         np.eye(len(ctx.items)), check_finite=True)
    weights = -inv / np.diag(inv)[None, :]
    np.fill_diagonal(weights, 0)
    if parameters.get("positive_weights"):
        weights = np.maximum(weights, 0)
    raw = x[ctx.target_rows] @ weights
    np.savez_compressed(path / "ease-weights.npz", weights=weights.astype(np.float32))
    restored = np.load(path / "ease-weights.npz")["weights"]
    assert np.allclose(raw[:5], x[ctx.target_rows[:5]] @ restored, atol=1e-5, rtol=1e-5)
    dump(path / "item-mapping.json", ctx.items)
    return _scores(ctx, raw, parameters), {**_details(ctx, x),
        "algorithm": "EASE constrained linear ridge, diag(B)=0",
        "regularization": regularization, "saved_model_reload_verified": True,
        "source": "https://arxiv.org/abs/1905.03375"}


def _prune(matrix, k):
    matrix = np.asarray(matrix)
    if k and k < len(matrix):
        for row in matrix:
            keep = np.argpartition(row, -k)[-k:]
            mask = np.ones(len(row), dtype=bool)
            mask[keep] = False
            row[mask] = 0
    return sparse.csr_matrix(matrix)


def itemknn(ctx, parameters, path):
    x = _matrix(ctx, parameters)
    kind = parameters.get("similarity", "cosine")
    item_users = x.T.tocsr()
    if kind == "bm25":
        item_users = bm25_weight(item_users, K1=parameters.get("k1", 100),
                                 B=parameters.get("b", .8)).tocsr()
    elif kind != "cosine":
        raise ValueError(kind)
    item_users = normalize(item_users, norm="l2")
    sim = (item_users @ item_users.T).toarray()
    shrink = parameters.get("shrink", 0)
    if shrink:
        binary = x.copy()
        binary.data[:] = 1
        cooccurrence = (binary.T @ binary).toarray()
        sim *= cooccurrence / (cooccurrence + shrink)
    np.fill_diagonal(sim, parameters.get("self_weight", 0))
    weights = _prune(sim, parameters.get("neighbors", 100))
    raw = (x[ctx.target_rows] @ weights).toarray()
    sparse.save_npz(path / "knn-weights.npz", weights)
    restored = sparse.load_npz(path / "knn-weights.npz")
    assert np.allclose(raw[:5], (x[ctx.target_rows[:5]] @ restored).toarray())
    dump(path / "item-mapping.json", ctx.items)
    return _scores(ctx, raw, parameters), {**_details(ctx, x),
        "algorithm": "itemKNN", "similarity": kind, "weights_nnz": weights.nnz,
        "saved_model_reload_verified": True,
        "source": "https://github.com/benfred/implicit/blob/main/implicit/nearest_neighbours.py"}


def ppr(ctx, parameters, path):
    x = _matrix(ctx, parameters)
    # Bipartite item-user-item graph. Degree normalization suppresses prolific users.
    gx = _matrix(ctx, {**parameters, "strength": parameters.get("graph_strength", parameters.get("strength", "binary"))})
    users = normalize(gx, norm="l1")
    graph = (gx.T @ users).toarray()
    np.fill_diagonal(graph, 0)
    graph = _prune(graph, parameters.get("neighbors", 200))
    transition = normalize(graph, norm="l1")
    dangling = np.asarray(transition.sum(axis=1)).ravel() == 0
    teleport = normalize(x[ctx.target_rows], norm="l1").toarray()
    alpha = parameters.get("alpha", .85)
    state = teleport.copy()
    tolerance = parameters.get("tolerance", 1e-8)
    residual = float("inf")
    for iteration in range(parameters.get("max_iter", 200)):
        new = alpha * (state @ transition + state[:, dangling].sum(axis=1, keepdims=True)*teleport) + (1-alpha)*teleport
        residual = float(np.max(np.sum(np.abs(new-state), axis=1)))
        state = new
        if residual < tolerance:
            break
    if residual >= tolerance:
        raise RuntimeError(f"PPR did not converge: {residual}")
    warm = teleport.sum(axis=1) > 0
    assert np.allclose(state[warm].sum(axis=1), 1)
    sparse.save_npz(path / "ppr-transition.npz", transition)
    sparse.save_npz(path / "ppr-teleport.npz", sparse.csr_matrix(teleport))
    restored = sparse.load_npz(path / "ppr-transition.npz")
    restored_teleport = sparse.load_npz(path / "ppr-teleport.npz").toarray()
    replay = alpha * (state @ restored + state[:, dangling].sum(axis=1, keepdims=True)*restored_teleport) + (1-alpha)*restored_teleport
    reload_residual = float(np.max(np.sum(np.abs(replay-state), axis=1)))
    assert reload_residual < tolerance
    dump(path / "item-mapping.json", ctx.items)
    return _scores(ctx, state, parameters), {**_details(ctx, x),
        "algorithm": "Personalized PageRank on item-user-item train graph",
        "graph_strength": parameters.get("graph_strength", parameters.get("strength", "binary")),
        "iterations": iteration+1, "l1_residual_max": residual, "alpha": alpha,
        "saved_model_fixed_point_residual_max": reload_residual,
        "warm_probability_mass_min": float(state[warm].sum(axis=1).min()),
        "warm_probability_mass_max": float(state[warm].sum(axis=1).max()),
        "dangling_items": int(dangling.sum()), "teleport": "normalized train user strength"}


def register(methods):
    methods.update(ease=ease, itemknn=itemknn, ppr=ppr, cf_score_fusion=score_fusion)


def score_fusion(ctx, parameters, path):
    """Fuse native train-only scores. Normalization never accesses validation truth."""
    total = np.zeros((len(ctx.targets), len(ctx.candidates)), np.float64)
    touched = np.zeros_like(total, bool)
    sources = []
    for component in parameters["components"]:
        run = path.parent / component["run_id"]
        record = load(run / "record.json")
        if record["protocol_sha256"] != ctx.protocol_hash:
            raise ValueError("Cannot fuse different protocols")
        z = np.load(run / "native-top100.npz", allow_pickle=False)
        for u, indices in enumerate(z["indices"]):
            valid = indices >= 0
            selected = indices[valid]
            values = np.maximum(z["scores"][u][valid].astype(np.float64), 0)
            if parameters.get("normalization", "max") == "sum":
                values /= max(values.sum(), 1e-12)
            else:
                values /= max(values.max(initial=0), 1e-12)
            total[u, selected] += component["weight"] * values
            if component["weight"] > 0:
                touched[u, selected] = True
        sources.append({"run_id": component["run_id"], "weight": component["weight"],
                        "native_rank_hash": sha256(run / "native-top100.npz")})
    total[~touched] = np.nan
    return total.astype(np.float32), {"sources": sources, "fusion": parameters.get("normalization", "max") +
                                    " normalized native positive scores, no truth used to calculate scores"}
