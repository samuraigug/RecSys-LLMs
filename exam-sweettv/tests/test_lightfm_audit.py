"""LightFM contracts: labels stay outside fit, weights align, cold vectors excluded."""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import numpy as np

from advanced_lightfm import interactions, lightfm_warp
from etl import save_events


class Forbidden(dict):
    def __getitem__(self, key):
        raise AssertionError("Evaluation truth must not enter training")
    def get(self, *args):
        raise AssertionError("Evaluation truth must not enter training")


class LightFMAuditTests(unittest.TestCase):
    def fixture(self, path):
        save_events(path / "train-events.npz", [
            ("u1", "1", "10", 100, 200, 100, 200, 100),
            ("u1", "1", "10", 300, 400, 100, 400, 100),
            ("u1", "2", "10", 500, 600, 50, 550, 100),
            ("u1", "3", "10", 700, 800, 49, 749, 100),
            ("u2", "1", "10", 100, 200, 80, 180, 100),
            ("u2", "2", "10", 300, 400, 100, 400, 100),
        ])
        return SimpleNamespace(path=path, targets=["u1", "18446744073709551600"],
            candidates=["1", "2", "3", "999"],
            protocol={"cutoff_elapsed_seconds": 1000, "train_cutoff_exclusive": "2020-05-18"},
            truth=Forbidden(), metadata=Forbidden(), display=Forbidden())

    def test_full_and_partial_thresholds_and_weights_are_train_only(self):
        with TemporaryDirectory() as directory:
            ctx = self.fixture(Path(directory))
            users, items, positives, weights, _ = interactions(ctx, {})
            self.assertEqual(items, ["1", "2"])
            self.assertIn("18446744073709551600", users)
            self.assertTrue(np.all(weights.data == 1))
            users, items, positives, weights, details = interactions(ctx, {"watch_fraction": .5, "weighting": "coverage_sum"})
            dense = weights.toarray()
            np.testing.assert_allclose(dense[users.index("u1")], [2., .5])
            np.testing.assert_allclose(dense[users.index("u2")], [.8, 1.])
            self.assertEqual(items, ["1", "2"])
            self.assertTrue(details["sample_weights_coordinate_alignment_verified"])
            np.testing.assert_array_equal(positives.row, weights.row)
            np.testing.assert_array_equal(positives.col, weights.col)

    def test_future_event_is_rejected_even_if_below_threshold(self):
        with TemporaryDirectory() as directory:
            path = Path(directory)
            ctx = self.fixture(path)
            save_events(path / "train-events.npz", [("u1", "1", "10", 900, 1100, 10, 1050, 200)])
            with self.assertRaises(AssertionError):
                interactions(ctx, {})

    def test_seed_reload_and_no_random_cold_item_scores(self):
        with TemporaryDirectory() as directory:
            path = Path(directory)
            ctx = self.fixture(path)
            params = {"seed": 42, "factors": 4, "epochs": 3}
            first, second = path / "first", path / "second"
            first.mkdir(); second.mkdir()
            scores, details = lightfm_warp(ctx, params, first)
            replay, _ = lightfm_warp(ctx, params, second)
            np.testing.assert_array_equal(scores, replay)
            self.assertTrue(np.isfinite(scores[0, :2]).all())
            self.assertTrue(np.isnan(scores[:, 2:]).all())
            self.assertTrue(np.isnan(scores[1]).all())
            self.assertTrue(details["native_predict_mapping_crosscheck"])
            self.assertTrue(details["saved_model_reload_verified"])


if __name__ == "__main__":
    unittest.main()
