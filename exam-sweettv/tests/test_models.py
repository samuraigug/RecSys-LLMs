import tempfile
from pathlib import Path
import types
import unittest
import numpy as np
from scipy import sparse
from models import content_blocks, year_tokens, hybrid
from experiments import finalize_scores
from repro import dump


class ModelContractTests(unittest.TestCase):
    def test_open_year_is_ongoing_and_never_current_year(self):
        self.assertEqual(year_tokens(["1996-"]), ["ongoing", "start_decade:1990"])
        self.assertEqual(year_tokens([""]), [])

    def test_sparse_blocks_no_shared_missing_genre(self):
        empty = {k: [] for k in ("tv_show_category", "tv_show_genre_1", "tv_show_genre_2", "tv_show_genre_3", "year_of_production", "director", "actors", "tv_show_title")}
        a = {**empty, "tv_show_category": ["News"], "actors": ["First Actor, Second Actor"]}
        b = {**empty, "tv_show_category": ["Sports"]}
        _, x, schemas = content_blocks({"1": a, "2": b}, {"category": 1, "actors": .5})
        self.assertEqual(float((x[0] @ x[1].T).toarray()[0, 0]), 0)
        self.assertTrue(np.allclose(np.asarray(x.multiply(x).sum(axis=1)).ravel(), 1))
        self.assertEqual(schemas["actors"]["vocabulary"], ["first actor", "second actor"])

    def test_shortlist_fills_once_and_cold_gate_overrides_scores(self):
        ctx = types.SimpleNamespace(targets=["18443832178892490752", "cold"], candidates=["1", "2", "3", "4", "5", "6"],
                                    fallback=["6", "5", "4", "3", "2", "1"], cold_mask=np.array([False, True]))
        scores = np.array([[2, 2, np.nan, np.nan, np.nan, np.nan], [1, 2, 3, 4, 5, 6]])
        predictions, d, _, _ = finalize_scores(scores, ctx)
        self.assertEqual(predictions[ctx.targets[0]], ["1", "2", "6", "5", "4"])
        self.assertEqual(predictions["cold"], ["6", "5", "4", "3", "2"])
        self.assertEqual(d[ctx.targets[0]]["fallback_positions"], 3)

    def test_rrf_does_not_vote_with_component_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            for run in ("a", "b"):
                dump(path/run/"record.json", {"protocol_sha256": "hash"})
                np.savez_compressed(path/run/"native-top100.npz", indices=np.array([[0, -1, -1]]))
            ctx = types.SimpleNamespace(targets=["u"], candidates=["1", "2", "3"], protocol_hash="hash")
            scores, _ = hybrid(ctx, {"components": [{"run_id":"a","weight":1}, {"run_id":"b","weight":1}]}, path/"h")
            self.assertTrue(np.isnan(scores[0, 1:]).all())
            self.assertAlmostEqual(scores[0, 0], 2/61)


if __name__ == "__main__":
    unittest.main()
