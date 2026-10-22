"""Independent schedule contracts: raw unions, cutoff, slots, counts and DST.

Fixtures live entirely in temporary directories. These tests do not fit a model
or touch the project's real schedule cache or raw data.
"""
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import csv
import shutil
import unittest
from unittest.mock import patch
from zoneinfo import ZoneInfo

import numpy as np
from scipy import sparse

import advanced_schedule as schedule
from etl import Clock, FILES
from repro import load, sha256


BIG_USER = "18446744073709551600"


class ScheduleContractAuditTests(unittest.TestCase):
    def write_csv(self, path, fields, rows):
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    def fixture(self, root, noisy_future=False):
        raw = root / "data/raw"
        raw.mkdir(parents=True, exist_ok=True)
        fields = ["user_id", "vsetv_id", "start_time", "stop_time", "duraton"]
        views = [
            [BIG_USER, "10", "2020-01-06 02:30:00", "2020-01-06 03:30:00", "3600"],
            [BIG_USER, "10", "2020-01-06 02:30:00", "2020-01-06 03:30:00", "3600"],
            [BIG_USER, "10", "2020-01-06 03:00:00", "2020-01-06 04:00:00", "3600"],
            [BIG_USER, "10", "2020-01-12 23:30:00", "2020-01-13 00:30:00", "3600"],
            # This channel has only unlabelled historical airtime.
            [BIG_USER, "20", "2020-01-07 06:00:00", "2020-01-07 06:30:00", "1800"],
            ["other-user", "10", "2020-01-06 00:00:00", "2020-01-07 00:00:00", "86400"],
            [BIG_USER, "10", "2020-01-13 00:00:00", "2020-01-13 23:00:00", "82800"],
        ]
        if noisy_future:
            # Change both the person and duration of post-cutoff behaviour;
            # retain known channels to avoid an irrelevant topology-only change.
            views[-1] = ["other-user", "20", "2020-01-14 01:00:00", "2020-01-19 23:00:00", "511200"]
        self.write_csv(raw / "dataset11-30.csv", fields, [dict(zip(fields, row)) for row in views])
        fields = ["channel_id", "start_time", "duration", "tv_show_id", "tv_show_title", "actors"]
        airings = [
            ["10", "13.01.2020 02:30:00", "5400", "100", "Future title", "Future actor"],
            ["10", "14.01.2020 06:00:00", "1800", "100", "Future title", "Future actor"],
            # Half the original airing intersects the prediction window.
            ["20", "12.01.2020 23:00:00", "7200", "200", "Boundary", "Future actor"],
            # Zero-id metadata is not an item, and need not have parseable times.
            ["20", "not parsed", "not parsed", "0", "Unlabelled", ""],
        ]
        if noisy_future:
            for row in airings:
                row[4], row[5] = "Unrelated title must not affect scores", "Different actor"
        for name in FILES:
            # Repeated rows across files must count as one airing.
            self.write_csv(raw / name, fields, [dict(zip(fields, row)) for row in airings])
        clock = Clock("Europe/Kyiv")
        return SimpleNamespace(
            targets=[BIG_USER], candidates=["100", "200"], candidate_map={"100": 0, "200": 1},
            protocol_hash="synthetic-frozen-protocol",
            protocol={"clock": "Europe/Kyiv", "train_start": "2020-01-06",
                      "train_cutoff_exclusive": "2020-01-13", "forecast_end_exclusive": "2020-01-20",
                      "cutoff_elapsed_seconds": clock.seconds("2020-01-13"),
                      "input_hashes": {name: sha256(raw / name) for name in ["dataset11-30.csv", *FILES]}})

    def build(self, root, ctx, name="cache"):
        cache = root / name
        cache.mkdir()
        with patch.object(schedule, "ROOT", root):
            schedule._build(ctx, cache)
        return cache

    def test_raw_seconds_union_cutoff_and_unlabelled_channel_are_exact(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            ctx = self.fixture(root)
            cache = self.build(root, ctx)
            manifest = load(cache / "manifest.json")
            self.assertEqual(manifest["channels"], ["10", "20"])
            self.assertEqual(manifest["precutoff_union_intervals"], 3)
            self.assertEqual(manifest["max_training_stop"], ctx.protocol["cutoff_elapsed_seconds"])
            with np.load(cache / "train-segments.npz", allow_pickle=False) as z:
                dense = sparse.csr_matrix((z["seconds"], (z["rows"], z["cols"])), shape=(1, 112)).toarray()
                # Monday 02:30–04:00: split at the 03:00 boundary, no duplicate seconds.
                self.assertEqual(dense[0, 0], 1800)
                self.assertEqual(dense[0, 1], 3600)
                # Sunday 23:30–cutoff contributes only its pre-cutoff half hour.
                self.assertEqual(dense[0, 6 * 8 + 7], 1800)
                # Actual raw seconds survive even though channel 20 has no historical item id.
                self.assertEqual(dense[0, 56 + 1 * 8 + 2], 1800)
                self.assertEqual(float(dense.sum()), 9000)
                np.testing.assert_array_equal(z["duration_sum"], [[7200, 1800]])
                np.testing.assert_array_equal(z["duration_count"], [[2, 1]])
            self.assertEqual(manifest["source_sha256"], sha256(schedule.__file__))

    def test_future_count_uses_each_original_duration_and_deduplicates(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            cache = self.build(root, self.fixture(root))
            counts = sparse.load_npz(cache / "future-counts.npz").toarray()
            hours = sparse.load_npz(cache / "future-hours.npz").toarray()
            self.assertEqual(load(cache / "manifest.json")["future_airings"], 3)
            self.assertAlmostEqual(float(counts[0, 0]), 1 / 3, places=6)
            self.assertAlmostEqual(float(counts[0, 1]), 2 / 3, places=6)
            self.assertEqual(counts[0, 1 * 8 + 2], 1)
            self.assertEqual(counts[1, 56], .5)
            np.testing.assert_allclose(counts.sum(axis=1), [2, .5])
            np.testing.assert_allclose(hours.sum(axis=1), [2, 1])

    def test_future_behaviour_and_content_do_not_change_features_or_scores(self):
        with TemporaryDirectory() as directory:
            base = Path(directory)
            arrays, scores = [], []
            for label, noisy in [("original", False), ("changed-future", True)]:
                root = base / label
                ctx = self.fixture(root, noisy_future=noisy)
                output = root / "result"
                output.mkdir()
                with patch.object(schedule, "ROOT", root):
                    score, _ = schedule.schedule_joint(ctx, {"joint_weight": .8}, output)
                cache = Path(load(output / "schedule-cache-reference.json")["path"])
                with np.load(cache / "train-segments.npz", allow_pickle=False) as z:
                    arrays.append({k: z[k].copy() for k in z.files})
                scores.append(score)
            for key in arrays[0]:
                np.testing.assert_array_equal(arrays[0][key], arrays[1][key])
            np.testing.assert_array_equal(scores[0], scores[1])

    def test_changed_raw_input_cannot_build_a_cache_for_a_frozen_protocol(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            ctx = self.fixture(root)
            with (root / "data/raw/dataset11-30.csv").open("a", encoding="utf-8") as stream:
                stream.write("\n")
            with self.assertRaisesRegex(ValueError, "Raw input changed"):
                self.build(root, ctx)

    def test_explicit_training_bundle_retains_template_hash_validation(self):
        """A training cohort can use its own template, but all hashes still bind."""
        with TemporaryDirectory() as directory:
            root = Path(directory)
            ctx = self.fixture(root)
            raw = root / "data/raw"
            (raw / "submission.csv").write_text("user_id,recommendations\noriginal,100\n")
            training_raw = root / "training-bundle"
            shutil.copytree(raw, training_raw)
            template = training_raw / "submission.csv"
            template.write_text("user_id,recommendations\n" + BIG_USER + ",100\n")
            ctx.raw_input_dir = training_raw
            ctx.protocol["input_hashes"]["submission.csv"] = sha256(template)
            cache = self.build(root, ctx, "training-cache")
            self.assertEqual(load(cache / "manifest.json")["raw_hashes"],
                             ctx.protocol["input_hashes"])
            # A changed training template cannot borrow a frozen protocol even
            # though the viewing log and all three schedule files are unchanged.
            template.write_text("user_id,recommendations\nchanged-user,100\n")
            with self.assertRaisesRegex(ValueError, "Raw input changed"):
                self.build(root, ctx, "rejected-cache")

    def test_modified_cached_features_are_rejected_before_scoring(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            ctx = self.fixture(root)
            output = root / "result"
            output.mkdir()
            with patch.object(schedule, "ROOT", root):
                schedule.schedule_joint(ctx, {}, output)
                cache = Path(load(output / "schedule-cache-reference.json")["path"])
                with (cache / "train-segments.npz").open("ab") as stream:
                    stream.write(b"modified")
                with self.assertRaisesRegex(ValueError, "Schedule cache artifact changed"):
                    schedule.schedule_joint(ctx, {}, output)

    def test_segments_preserve_elapsed_seconds_at_both_dst_transitions(self):
        clock = Clock("Europe/Kyiv")
        zone = ZoneInfo("Europe/Kyiv")
        spring_a = int(datetime(2020, 3, 29, 2, 30, tzinfo=zone).timestamp())
        spring_b = int(datetime(2020, 3, 29, 4, 30, tzinfo=zone).timestamp())
        spring = list(schedule.segments(clock, spring_a, spring_b))
        self.assertEqual(sum(x[2] for x in spring), 3600)
        self.assertEqual([(x[1], x[2]) for x in spring], [(0, 1800), (1, 1800)])
        fall_a = int(datetime(2020, 10, 25, 3, 30, tzinfo=zone, fold=0).timestamp())
        fall_b = int(datetime(2020, 10, 25, 4, 30, tzinfo=zone).timestamp())
        fall = list(schedule.segments(clock, fall_a, fall_b))
        self.assertEqual(sum(x[2] for x in fall), 7200)
        self.assertTrue(all(x[2] > 0 for x in fall))
        self.assertTrue(all(x[1] == 1 for x in fall))


if __name__ == "__main__":
    unittest.main()
