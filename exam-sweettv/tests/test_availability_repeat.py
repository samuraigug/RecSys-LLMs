from types import SimpleNamespace
import unittest

import numpy as np

from advanced_availability import gap_statistics


class AvailabilityRecencyTests(unittest.TestCase):
    def test_equal_calendar_age_distinguishes_off_air_from_missed_airings(self):
        # Both programmes were last watched at exactly the same time. Only one
        # continued airing, so only it supplies evidence of a lost preference.
        ctx = SimpleNamespace(targets=['u'], candidates=['a', 'b'],
                              candidate_map={'a': 0, 'b': 1},
                              protocol={'cutoff_elapsed_seconds': 1000})
        events = np.array([[0, 0, 0, 100, 200, 80, 180, 100],
                           [0, 1, 0, 100, 200, 80, 180, 100]])
        count, last, missed, total = gap_statistics(ctx, events, ['u'], ['a', 'b'],
            {'a': np.array([[100, 200], [500, 600], [800, 900]]),
             'b': np.array([[100, 200]])})
        np.testing.assert_array_equal(count, [[1, 1]])
        np.testing.assert_array_equal(last, [[180, 180]])
        np.testing.assert_array_equal(missed, [[2, 0]])
        np.testing.assert_array_equal(total, [3, 1])

    def test_post_cutoff_view_cannot_create_an_affinity(self):
        ctx = SimpleNamespace(targets=['u'], candidates=['a'], candidate_map={'a': 0},
                              protocol={'cutoff_elapsed_seconds': 1000})
        events = np.array([[0, 0, 0, 1100, 1200, 100, 1200, 100]])
        with self.assertRaises(AssertionError):
            gap_statistics(ctx, events, ['u'], ['a'], {})
