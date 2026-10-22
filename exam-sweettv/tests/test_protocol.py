import csv
import random
import tempfile
import unittest
from pathlib import Path

from audit import inventory, parse_time
from protocol import Airing, View, ap_at_k, join_interactions, map_at_k, merge_intervals, overlap_seconds, rank_top_k, validate_submission


class JoinTests(unittest.TestCase):
    def test_half_open_boundary_and_invalid_interval(self):
        self.assertEqual(overlap_seconds(0, 100, 100, 200), 0)
        self.assertEqual(overlap_seconds(50, 150, 100, 200), 50)
        with self.assertRaises(ValueError):
            overlap_seconds(2, 1, 0, 1)

    def test_duplicate_view_union_and_duplicate_schedule(self):
        views = [View("u", "c", 0, 100), View("u", "c", 50, 150), View("u", "c", 0, 100)]
        a = Airing("c", "1", 0, 200)
        result = join_interactions(views, [a, a], cutoff=1000)
        self.assertEqual((result[0].seconds, result[0].airings), (150, 1))

    def test_channel_match_zero_id_and_repeat_airings(self):
        views = [View("u", "a", 0, 500)]
        airings = [Airing("b", "2", 0, 500), Airing("a", "0", 0, 500),
                   Airing("a", "1", 0, 100), Airing("a", "1", 200, 300)]
        result = join_interactions(views, airings, cutoff=500)
        self.assertEqual([(r.item, r.seconds, r.airings) for r in result], [("1", 200, 2)])

    def test_cutoff_applied_before_watch_threshold(self):
        a = Airing("c", "1", 100, 300)
        before = join_interactions([View("u", "c", 150, 300)], [a], cutoff=200)
        self.assertEqual(before, [])  # 50 train seconds, not 150 train+future seconds
        after_changed = join_interactions([View("u", "c", 150, 9000), View("future", "c", 210, 250)], [a], cutoff=200)
        self.assertEqual(before, after_changed)

    def test_fraction_uses_programme_duration(self):
        result = join_interactions([View("u", "c", 100, 200)], [Airing("c", "1", 0, 1000)],
                                   cutoff=2000, min_seconds=60, min_fraction=0.2)
        self.assertEqual(result, [])  # 100/1000, not 100/100

    def test_multiple_segments_can_reach_one_airing_threshold(self):
        result = join_interactions([View("u", "c", 0, 35), View("u", "c", 50, 85)],
                                   [Airing("c", "1", 0, 100)], cutoff=500)
        self.assertEqual(result[0].seconds, 70)

    def test_interval_union(self):
        self.assertEqual(merge_intervals([(10, 20), (0, 10), (30, 40), (7, 15)]), [(0, 20), (30, 40)])

    def test_indexed_join_against_independent_seconds_enumeration(self):
        rng = random.Random(42)
        views = [View(str(rng.randrange(4)), str(rng.randrange(2)), t, t + rng.randrange(1, 45))
                 for t in [rng.randrange(150) for _ in range(150)]]
        airings = [Airing(str(rng.randrange(2)), str(rng.randrange(1, 9)), t, t + rng.randrange(1, 30))
                   for t in [rng.randrange(150) for _ in range(60)]]
        cutoff, expected = 120, {}
        for user in ["0", "1", "2", "3"]:
            for a in set(airings):
                covered = {second for v in views if v.user == user and v.channel == a.channel
                           for second in range(max(v.start, a.start), min(v.stop, a.stop, cutoff))}
                if len(covered) < 3 or len(covered)/(a.stop-a.start) < 0.8:
                    continue
                key = (user, a.item)
                seconds, count, last = expected.get(key, (0, 0, 0))
                expected[key] = (seconds + len(covered), count + 1, max(last, max(covered) + 1))
        actual = join_interactions(views, airings, cutoff=cutoff, min_seconds=3, min_fraction=0.8)
        self.assertEqual({(r.user, r.item): (r.seconds, r.airings, r.last_stop) for r in actual}, expected)


class MetricTests(unittest.TestCase):
    def test_hand_computable_average_precision(self):
        self.assertAlmostEqual(ap_at_k(["1", "2", "3"], ["1", "9", "2", "8", "3"]), (1 + 2/3 + 3/5)/3)

    def test_duplicates_do_not_create_extra_hits(self):
        self.assertAlmostEqual(ap_at_k(["1", "2"], ["1", "1", "2"]), (1 + 2/3)/2)
        self.assertEqual(ap_at_k([], ["1"]), 0)

    def test_macro_mean_includes_cold_user(self):
        self.assertEqual(map_at_k({"warm": ["1"], "cold": ["2"]},
                                 {"warm": ["1"], "cold": ["1"]}, ["warm", "cold"]), 0.5)
        with self.assertRaises(ValueError):
            map_at_k({"u": ["1"]}, {}, ["u"])

    def test_common_fallback_deterministic_ties_and_seen_items(self):
        pool = [str(i) for i in range(1, 8)]
        self.assertEqual(rank_top_k({}, pool, ["7", "6", "5", "4", "3", "2", "1"]), ["7", "6", "5", "4", "3"])
        self.assertEqual(rank_top_k({"2": 1.0, "1": 1.0}, pool, ["7", "6", "5"]), ["1", "2", "7", "6", "5"])
        with self.assertRaises(ValueError):
            rank_top_k({}, ["0", "1", "2", "3", "4"], [])


class FileTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def write_csv(self, name, rows):
        path = self.base / name
        with path.open("w", newline="") as stream:
            csv.writer(stream).writerows([["user_id", "tv_show_id"], *rows])
        return path

    def test_submission_preserves_large_ids_and_order(self):
        user = "8646014723560466568"
        template = self.write_csv("template.csv", [[user, "0 0 0 0 0"]])
        submission = self.write_csv("submission.csv", [[user, "1 2 3 4 5"]])
        result = validate_submission(template, submission, ["1", "2", "3", "4", "5"], expected_rows=1)
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(len(result["sha256"]), 64)

    def test_rejects_invalid_predictions_and_user_mismatch(self):
        template = self.write_csv("template.csv", [["u", "0 0 0 0 0"]])
        for user, ids in [("v", "1 2 3 4 5"), ("u", "1 1 2 3 4"), ("u", "0 1 2 3 4"), ("u", "1 2 3 4 6")]:
            submission = self.write_csv("submission.csv", [[user, ids]])
            with self.assertRaises(ValueError):
                validate_submission(template, submission, ["1", "2", "3", "4", "5"], expected_rows=1)

    def test_missing_data_is_not_pass(self):
        self.assertEqual(inventory(self.base)["status"], "MISSING_DATA")

    def test_streaming_inventory_reads_semicolon_and_zero_id(self):
        path = self.base / "schedule.csv"
        path.write_text("channel_id;start_time;duration;tv_show_id;tv_show_category\n1;2020-01-01;60;0;A\n1;2020-01-01;60;10;B\n")
        result = inventory(self.base)
        self.assertEqual(result["status"], "REVIEW_REQUIRED")
        self.assertEqual(result["files"][0]["labelled_fraction"], 0.5)
        self.assertEqual(result["files"][0]["unique_items_nonzero"], 1)

    def test_actual_two_date_formats_agree(self):
        self.assertEqual(parse_time("09.03.2020 06:00:00"), parse_time("2020-03-09 06:00:00"))


if __name__ == "__main__":
    unittest.main()
