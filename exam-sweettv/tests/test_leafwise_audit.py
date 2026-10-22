import unittest

import numpy as np

from advanced_leafwise import fit_model, load_engine


class LeafwiseContractTests(unittest.TestCase):
    def test_imports_only_the_frozen_feature_engine(self):
        engine = load_engine()
        self.assertEqual(engine.__module__, 'frozen_ranker_v5')

    def test_unordered_query_groups_are_rejected_before_ranker_fit(self):
        # Counts alone would silently treat two unrelated users as one query.
        with self.assertRaisesRegex(ValueError, 'contiguous and ordered'):
            fit_model(np.zeros((4, 1), np.float32), np.array([1, 0, 1, 0]),
                      np.ones(4), np.array([0, 1, 0, 1]), ['signal'],
                      {'objective': 'lambdarank', 'iterations': 1})
