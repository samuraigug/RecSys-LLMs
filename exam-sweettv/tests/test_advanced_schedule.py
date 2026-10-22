"""Union/cutoff and future opportunity contracts of raw schedule features."""
import csv
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy import sparse

import advanced_schedule
from etl import Clock, FILES
from repro import load


class ScheduleSafetyTests(unittest.TestCase):
    def test_dst_segments_preserve_elapsed_mass(self):
        clock=Clock('Europe/Kyiv')
        a,b=clock.seconds('2020-03-29 00:00:00'),clock.seconds('2020-03-29 06:00:00')
        parts=list(advanced_schedule.segments(clock,a,b))
        self.assertEqual(sum(p[2] for p in parts),b-a)
        self.assertEqual(b-a,5*3600)
        self.assertTrue(all(p[0]==6 for p in parts))

    def test_union_cutoff_and_airing_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); raw=root/'data/raw';raw.mkdir(parents=True)
            views=[['u','1','2020-03-09 00:00:00','2020-03-09 01:00:00'],
                   ['u','1','2020-03-09 00:30:00','2020-03-09 01:30:00'],
                   ['u','1','2020-03-10 00:00:00','2020-03-20 00:00:00']]
            with (raw/'dataset11-30.csv').open('w',newline='') as f:
                w=csv.writer(f);w.writerow(['user_id','vsetv_id','start_time','stop_time']);w.writerows(views)
            for fn in FILES:
                with (raw/fn).open('w',newline='') as f:
                    w=csv.writer(f);w.writerow(['tv_show_id','channel_id','start_time','duration'])
                    w.writerow(['11','1','16.03.2020 00:00:00',3600])
                    w.writerow(['12','1','17.03.2020 00:00:00',3600])
                    w.writerow(['0','999','16.03.2020 00:00:00',0])
            clock=Clock('Europe/Kyiv')
            ctx=SimpleNamespace(targets=['u'],candidates=['11','12'],candidate_map={'11':0,'12':1},
                protocol_hash='a'*64,protocol={'clock':'Europe/Kyiv','train_start':'2020-03-09',
                    'train_cutoff_exclusive':'2020-03-10','forecast_end_exclusive':'2020-03-24',
                    'cutoff_elapsed_seconds':clock.seconds('2020-03-10'),'input_hashes':{}})
            path=root/'result';path.mkdir()
            with patch.object(advanced_schedule,'ROOT',root):
                scores,_=advanced_schedule.schedule_joint(ctx,{'joint_weight':1},path)
            cache=root/'artifacts/runs/schedule-cache-v2'/ctx.protocol_hash/advanced_schedule.BUILDER_SOURCE_SHA256
            manifest=load(cache/'manifest.json')
            self.assertEqual(manifest['precutoff_target_rows'],2)
            self.assertEqual(manifest['precutoff_union_intervals'],1)
            self.assertLessEqual(manifest['max_training_stop'],ctx.protocol['cutoff_elapsed_seconds'])
            self.assertEqual(manifest['channels'],['1'])  # zero-ID-only future channel omitted
            with np.load(cache/'train-segments.npz') as z:
                self.assertEqual(float(z['seconds'].sum()),5400)  # overlapping 30 min counted once
            opp=sparse.load_npz(cache/'future-counts.npz')
            np.testing.assert_allclose(np.asarray(opp.sum(axis=1)).ravel(),[1,1])
            self.assertGreater(scores[0,0],0)  # eligible new programme on watched weekly slot
            self.assertTrue(np.isnan(scores[0,1]))  # future raw row never becomes train habit
            # A same-shaped corrupted cache must fail before it can change scores.
            with (cache/'future-counts.npz').open('ab') as stream:
                stream.write(b'corrupted')
            with patch.object(advanced_schedule,'ROOT',root):
                with self.assertRaisesRegex(ValueError,'artifact changed'):
                    advanced_schedule.schedule_joint(ctx,{'joint_weight':1},path)


if __name__=='__main__':unittest.main()
