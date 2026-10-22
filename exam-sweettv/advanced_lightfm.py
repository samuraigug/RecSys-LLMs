"""Train-only LightFM WARP; identity factors trained from positive viewing events.

Official implementation: https://github.com/lyst/lightfm
API: https://making.lyst.com/lightfm/docs/lightfm.html
No evaluation truth, future views or content metadata are read by this module.
"""
from __future__ import annotations

import importlib.metadata
import json
import pickle
from pathlib import Path

import numpy as np
from scipy import sparse

from etl import read_events
from models import METHODS
from repro import dump, sha256


def interactions(ctx, parameters):
    """Build aligned binary positives and explicit train-only sample weights.

    Only genuinely observed positives create item identities, including weak-only
    identities in the >=50% ablation. Future schedule membership never creates
    randomly initialised item vectors.
    """
    eu, ei, _, events = read_events(ctx.path / "train-events.npz", full_only=False)
    cutoff = ctx.protocol["cutoff_elapsed_seconds"]
    if len(events):
        assert np.all(events[:, 6] <= cutoff)
        assert np.all(events[:, 5] <= events[:, 7]) and np.all(events[:, 7] > 0)
    threshold = parameters.get("watch_fraction", .8)
    if not 0 < threshold <= 1:
        raise ValueError("watch_fraction must be in (0,1]")
    fraction = events[:, 5] / events[:, 7]
    selected = fraction >= threshold
    events, fraction = events[selected], fraction[selected]
    if not len(events):
        raise ValueError("No pre-cutoff positive interactions")
    users = sorted(set(ctx.targets) | {eu[j] for j in events[:, 0]})
    items = sorted({ei[j] for j in events[:, 1]}, key=int)
    um, im = {u: j for j, u in enumerate(users)}, {i: j for j, i in enumerate(items)}
    rr = np.fromiter((um[eu[j]] for j in events[:, 0]), np.int32, len(events))
    cc = np.fromiter((im[ei[j]] for j in events[:, 1]), np.int32, len(events))
    mode = parameters.get("weighting", "uniform")
    if mode not in ("uniform", "count", "log_count", "coverage_sum"):
        raise ValueError("Unknown sample weighting: " + mode)
    values = fraction.astype(np.float32) if mode == "coverage_sum" else np.ones(len(events), np.float32)
    half_life = parameters.get("half_life_days")
    if half_life:
        values *= np.exp2(-(cutoff-events[:, 6])/86400/half_life).astype(np.float32)
    weights = sparse.csr_matrix((values, (rr, cc)), shape=(len(users), len(items)), dtype=np.float32)
    weights.eliminate_zeros()
    if mode == "uniform" and not half_life:
        weights.data[:] = 1
    elif mode == "log_count":
        weights.data = np.log1p(weights.data)
    weights = weights.tocoo()
    positives = sparse.coo_matrix((np.ones(weights.nnz, np.float32),
                                   (weights.row, weights.col)), shape=weights.shape)
    assert np.array_equal(positives.row, weights.row) and np.array_equal(positives.col, weights.col)
    assert np.isfinite(weights.data).all() and np.all(weights.data > 0)
    return users, items, positives, weights, {
        "selected_airing_events": len(events), "train_user_item_pairs": positives.nnz,
        "watch_fraction": threshold, "sample_weighting": mode,
        "sample_weight_min": float(weights.data.min()), "sample_weight_max": float(weights.data.max()),
        "sample_weights_coordinate_alignment_verified": True,
        "training_last_stop_max": int(events[:, 6].max()), "cutoff_elapsed_seconds": cutoff,
    }


def lightfm_warp(ctx, parameters, path):
    from lightfm import LightFM

    users, items, positives, weights, details = interactions(ctx, parameters)
    user_map, item_map = {u: j for j, u in enumerate(users)}, {i: j for j, i in enumerate(items)}
    config = {
        "no_components": parameters.get("factors", 32), "loss": "warp",
        "learning_schedule": "adagrad", "learning_rate": parameters.get("learning_rate", .05),
        "item_alpha": parameters.get("item_alpha", 0.), "user_alpha": parameters.get("user_alpha", 0.),
        "max_sampled": parameters.get("max_sampled", 10), "random_state": parameters.get("seed", 42),
    }
    model = LightFM(**config)
    model.fit(positives, sample_weight=weights, epochs=parameters.get("epochs", 15), num_threads=1, verbose=False)
    assert np.isfinite(model.user_embeddings).all() and np.isfinite(model.item_embeddings).all()
    ub, uf = model.get_user_representations()
    ib, factors = model.get_item_representations()
    target_rows = np.array([user_map[u] for u in ctx.targets], np.int32)
    cols = [(j, item_map[i]) for j, i in enumerate(ctx.candidates) if i in item_map]
    dst, src = map(np.array, zip(*cols))
    scores = np.full((len(ctx.targets), len(ctx.candidates)), np.nan, dtype=np.float32)
    values = uf[target_rows] @ factors[src].T + ib[src][None, :] + ub[target_rows][:, None]
    positive_users = np.asarray(positives.tocsr().getnnz(axis=1)).ravel() > 0
    values[~positive_users[target_rows]] = np.nan
    scores[:, dst] = values
    user = int(target_rows[np.flatnonzero(positive_users[target_rows])[0]])
    probe_items = src[:min(30, len(src))].astype(np.int32)
    expected = uf[user] @ factors[probe_items].T + ib[probe_items] + ub[user]
    native = model.predict(user, probe_items, num_threads=1)
    assert np.allclose(expected, native, atol=1e-5, rtol=1e-5)
    # LightFM has no native save API; persist its own object and immediately verify
    # a reload of our newly written, trusted local payload.
    model_path = path / "lightfm-model.pickle"
    with model_path.open("wb") as stream:
        pickle.dump(model, stream, protocol=pickle.HIGHEST_PROTOCOL)
    with model_path.open("rb") as stream:
        restored = pickle.load(stream)
    assert np.array_equal(native, restored.predict(user, probe_items, num_threads=1))
    dump(path / "lightfm-mappings.json", {"users": users, "items": items, "candidate_items": ctx.candidates})
    sparse.save_npz(path / "lightfm-interactions.npz", positives.tocsr())
    sparse.save_npz(path / "lightfm-sample-weights.npz", weights.tocsr())
    distribution = importlib.metadata.distribution("lightfm")
    direct_url = distribution.read_text("direct_url.json")
    return scores, {**details, "algorithm": "LightFM WARP identity-feature matrix factorization",
        "config": config, "epochs": parameters.get("epochs", 15), "num_threads": 1,
        "fit_cutoff": ctx.protocol["train_cutoff_exclusive"], "future_views_used": False,
        "future_content_used": False, "filter_already_liked_items": False,
        "cold_items": "no score without a pre-cutoff positive training event",
        "lightfm_version": distribution.version, "lightfm_direct_url": json.loads(direct_url) if direct_url else None,
        "native_predict_mapping_crosscheck": True, "saved_model_reload_verified": True,
        "model_sha256": sha256(model_path), "train_events_sha256": sha256(ctx.path / "train-events.npz"),
        "sources": ["https://making.lyst.com/lightfm/docs/lightfm.html", "https://github.com/lyst/lightfm"],
    }


def register():
    METHODS["lightfm_warp"] = lightfm_warp
