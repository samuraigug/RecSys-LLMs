"""Independent check of the all-catalogue Logloss sampling correction.

No model is fitted. A fixed catalogue probability vector permits exact summation
of its Logloss/intercept gradient, then repeated sampled estimates are checked
against that independent reference.
"""
from types import SimpleNamespace
import unittest

import numpy as np

from advanced_ranker import FeatureEngine


class RankerSamplingAuditTests(unittest.TestCase):
    def engine(self):
        engine = FeatureEngine.__new__(FeatureEngine)
        engine.n, engine.m = 1, 500
        candidates = [str(j + 1) for j in range(engine.m)]
        engine.ctx = SimpleNamespace(targets=["user"], candidates=candidates,
            truth={"user": ["451", "452"]}, cold_mask=np.array([False]))
        engine.c = {item: j for j, item in enumerate(candidates)}
        # A synthetic observation index makes exact catalogue values recoverable
        # in this test; real model features contain no user/item ID column.
        engine.signals = [(np.arange(engine.m, dtype=np.float32), "item")]
        engine.retrievers = [np.arange(engine.m, 0, -1, dtype=float)[None, :]] * 5
        engine.retriever_limits = [100, 80, 60, 40, 40]
        return engine

    def test_corrected_sampling_estimates_full_catalogue_logloss_and_gradient(self):
        engine = self.engine()
        probability = np.linspace(.01, .99, engine.m)
        actual = np.zeros(engine.m)
        actual[[450, 451]] = 1
        full_loss = -(actual*np.log(probability)+(1-actual)*np.log1p(-probability)).sum()
        full_gradient = (probability-actual).sum()
        estimated_losses, estimated_gradients, unweighted_losses = [], [], []
        for seed in range(256):
            features, labels, groups, weights = engine.sampled_training(
                np.random.default_rng(seed), negatives=30, correct_sampling=True)
            indices = features[:, 0].astype(int)
            p = probability[indices]
            loss = -(labels*np.log(p)+(1-labels)*np.log1p(-p))
            estimated_losses.append(float(weights@loss))
            estimated_gradients.append(float(weights@(p-labels)))
            unweighted_losses.append(float(loss.sum()))
            self.assertEqual(int((labels > 0).sum()), 2)
            np.testing.assert_array_equal(weights[labels > 0], 1)
            self.assertAlmostEqual(float(weights.sum()), engine.m, places=3)
        for estimates, exact in [(estimated_losses, full_loss), (estimated_gradients, full_gradient)]:
            standard_error = np.std(estimates, ddof=1)/np.sqrt(len(estimates))
            self.assertLess(abs(np.mean(estimates)-exact), 4*standard_error)
        # The same rows without correction fail to represent the target loss.
        # Passing these weights to Pool is audited separately in learned_ranker.
        self.assertGreater(abs(np.mean(unweighted_losses)-full_loss), 100)


if __name__ == "__main__":
    unittest.main()
