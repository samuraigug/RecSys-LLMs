"""Meaningful small fixtures for catalogue truncation, ties, and fixed bundles."""
import sys
from pathlib import Path
import unittest
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from advanced_full_fusion import max_fusion, rrf, truncate, ordered, BUNDLES, _load_cache, SOURCE_SHA256
from repro import dump, sha256


class FullFusionAudit(unittest.TestCase):
    def test_score_tail_agreement_can_change_winner(self):
        numeric = np.array([10, 20, 30])
        a = np.array([[1, .01, .9]], np.float32)
        b = np.array([[.01, 1, .9]], np.float32)
        full = max_fusion([(a, 1), (b, 1)])
        old = max_fusion([(truncate(a, numeric, 1), 1),
                          (truncate(b, numeric, 1), 1)])
        self.assertEqual(ordered(full[0], numeric)[0], 2)
        self.assertEqual(ordered(old[0], numeric)[0], 0)
        self.assertTrue(np.isnan(old[0, 2]))

    def test_rrf_preserves_tie_order_and_nan_absence(self):
        numeric = np.array([30, 10, 20, 40])
        a = np.array([[5, 5, 3, np.nan]], np.float32)
        b = np.array([[np.nan, np.nan, 8, np.nan]], np.float32)
        out = rrf([(a, 1), (b, .5)], numeric)
        self.assertAlmostEqual(float(out[0, 1]), 1/61)
        self.assertAlmostEqual(float(out[0, 0]), 1/62)
        self.assertAlmostEqual(float(out[0, 2]), 1/63+.5/61)
        self.assertTrue(np.isnan(out[0, 3]))

    def test_full_ranks_include_tail_without_changing_top_ranks(self):
        numeric = np.arange(105)
        row = np.arange(105, 0, -1, dtype=np.float32)[None, :]
        full = rrf([(row, 1)], numeric)
        old = rrf([(truncate(row, numeric), 1)], numeric)
        np.testing.assert_array_equal(full[0, :100], old[0, :100])
        self.assertEqual(int(np.isfinite(full).sum()), 105)
        self.assertEqual(int(np.isfinite(old).sum()), 100)
        self.assertEqual(BUNDLES['shfl4'], [('core', 1.), ('l4', .5)])
        self.assertEqual(BUNDLES['shfl45'], [('core', 1.), ('l4', .25), ('l5', .25)])

    def test_cache_rejects_replaced_scores(self):
        with TemporaryDirectory() as tmp:
            cache = Path(tmp)
            ctx = SimpleNamespace(protocol_hash='fixture', targets=['big-string-id'], candidates=['10'])
            dump(cache/'mapping.json', {'targets': ctx.targets, 'candidates': ctx.candidates})
            np.save(cache/'core.npy', np.array([[1]], np.float32))
            dump(cache/'manifest.json', {'protocol_sha256': 'fixture', 'source_sha256': SOURCE_SHA256,
                 'input_sources_sha256': {'cbm': 'frozen'}, 'truncated_reference_replay_passed': True,
                 'payload_sha256': {'core.npy': sha256(cache/'core.npy'),
                                    'mapping.json': sha256(cache/'mapping.json')}})
            _load_cache(ctx, cache, {'cbm': 'frozen'})
            np.save(cache/'core.npy', np.array([[.5]], np.float32))
            with self.assertRaisesRegex(AssertionError, 'Full-score cache modified'):
                _load_cache(ctx, cache, {'cbm': 'frozen'})


if __name__ == '__main__':
    unittest.main()
