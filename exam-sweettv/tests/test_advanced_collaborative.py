"""Structural safeguards: graph direction/mass, novel items, cutoff and reload."""
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np
from scipy import sparse

from advanced_collaborative import _matrix, ease, ppr
from etl import save_events
from repro import dump


class Forbidden(dict):
    def __getitem__(self, key):
        raise AssertionError("Evaluation information must not be read by fitting")
    def get(self, *args):
        raise AssertionError("Evaluation information must not be read by fitting")
    def items(self):
        raise AssertionError("Evaluation information must not be read by fitting")
    def values(self):
        raise AssertionError("Evaluation information must not be read by fitting")


class CollaborativeSafetyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name)
        # u1 only watched item1. Its co-view neighbour item2 must remain eligible.
        x = sparse.csr_matrix([[1, 0, 0], [1, 1, 0], [0, 0, 0], [0, 1, 1]], dtype=float)
        self.ctx = SimpleNamespace(
            path=self.path, targets=['u1', 'u2', 'cold'], users=['u1', 'u2', 'cold', 'u3'],
            items=['1', '2', '3'], candidates=['1', '2', '3', '4'],
            item_map={'1': 0, '2': 1, '3': 2}, user_map={'u1': 0, 'u2': 1, 'cold': 2, 'u3': 3},
            target_rows=np.array([0, 1, 2]), strength=lambda _: x.copy(),
            protocol={'train_cutoff_exclusive': '2020-05-18', 'cutoff_elapsed_seconds': 2000},
            truth=Forbidden(), display=Forbidden(), metadata=Forbidden())
        dump(self.path / 'opportunities.json', {'train': {'1': 10, '2': 10, '3': 10},
                                             'future': {'1': 10, '2': 10, '3': 10}})

    def tearDown(self):
        self.temp.cleanup()

    def test_ppr_preserves_mass_and_recommends_unseen_item(self):
        scores, details = ppr(self.ctx, {'strength': 'count', 'alpha': .7}, self.path)
        np.testing.assert_allclose(np.nansum(scores[:2], axis=1), 1, atol=1e-7)
        self.assertGreater(scores[0, 1], 0)  # unseen by u1; found through co-view graph
        self.assertGreater(scores[0, 2], 0)  # transitive recommendation
        self.assertTrue(np.isnan(scores[:, 3]).all())  # cold item4 is not fabricated
        self.assertTrue(np.isnan(scores[2]).all())
        self.assertLess(details['saved_model_fixed_point_residual_max'], 1e-8)
        transition = sparse.load_npz(self.path / 'ppr-transition.npz')
        np.testing.assert_array_equal(transition.diagonal(), 0)
        np.testing.assert_allclose(np.asarray(transition.sum(axis=1)).ravel(), 1)

    def test_future_weak_event_is_rejected(self):
        save_events(self.path / 'train-events.npz', [('u1', '1', '1', 1900, 2100, 100, 2100, 200)])
        with self.assertRaises(AssertionError):
            ppr(self.ctx, {'strength': 'count', 'weak_weight': 1}, self.path)

    def test_ease_excludes_identity_shortcut_and_reloads(self):
        scores, details = ease(self.ctx, {'strength': 'binary', 'regularization': 1}, self.path)
        weights = np.load(self.path / 'ease-weights.npz')['weights']
        np.testing.assert_array_equal(np.diag(weights), 0)
        self.assertGreater(scores[0, 1], 0)
        self.assertTrue(np.isnan(scores[:, 3]).all())
        self.assertTrue(details['saved_model_reload_verified'])

    def test_disabled_partial_event_stays_absent_when_binary(self):
        save_events(self.path / 'train-events.npz', [
            ('u1', '1', '1', 100, 200, 100, 200, 100),
            ('u1', '2', '1', 300, 500, 100, 500, 200)])
        matrix = _matrix(self.ctx, {'strength':'binary', 'weak_weight':0})
        self.assertEqual(matrix.nnz, 1)
        self.assertEqual(matrix[0, 0], 1)
        self.assertEqual(matrix[0, 1], 0)


if __name__ == '__main__':
    unittest.main()
