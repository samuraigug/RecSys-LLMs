import unittest
from collections import defaultdict
from protocol import Airing, View, join_interactions
from etl import Clock, aggregate, clip_union, match_events, metadata_before


class ElapsedTimeTests(unittest.TestCase):
    def test_source_zero_row_precedes_duration_validation(self):
        self.assertEqual(join_interactions([], [Airing("1", "0", 1, 1)], cutoff=10), [])
        with self.assertRaises(ValueError):
            join_interactions([], [Airing("1", "2", 1, 1)], cutoff=10)

    def test_dst_real_airing_example_and_nonexistent_hour(self):
        clock = Clock()
        start = clock.seconds("29.03.2020 01:35:00")
        self.assertEqual(clock.seconds("29.03.2020 05:20:00")-start, 9900)
        with self.assertRaises(ValueError):
            clock.seconds("29.03.2020 03:15:00")

    def test_matcher_against_reference_with_window_clipping(self):
        groups = {("u", "c"): [(0, 70), (50, 90), (90, 120), (140, 200)]}
        airings = [Airing("c", "1", 0, 100), Airing("c", "2", 100, 200)]
        events, _ = match_events(groups, airings, 0, 170)
        expected = join_interactions([View(u, c, a, b) for (u, c), spans in groups.items() for a, b in spans],
                                     airings, cutoff=170, min_seconds=0, min_fraction=.8)
        got = aggregate(events)
        self.assertEqual({(r.user, r.item): [r.airings, r.seconds, r.last_stop] for r in expected}, dict(got))

    def test_future_view_changes_cannot_change_training_events(self):
        airings = [Airing("c", "1", 0, 100), Airing("c", "2", 100, 200)]
        a, _ = match_events({("u", "c"): [(0, 90), (110, 200)]}, airings, 0, 100)
        b, _ = match_events({("u", "c"): [(0, 90), (110, 99999)], ("future", "c"): [(100, 500)]}, airings, 0, 100)
        self.assertEqual(a, b)

    def test_content_cutoff_precedes_aggregation(self):
        from etl import FIELDS
        rows = [{"item": "1", "start": t, **{f: value for f in FIELDS}} for t, value in [(0, "known"), (100, "future")]]
        self.assertEqual(metadata_before(rows, 100)["1"]["tv_show_title"], ["known"])


if __name__ == "__main__":
    unittest.main()
