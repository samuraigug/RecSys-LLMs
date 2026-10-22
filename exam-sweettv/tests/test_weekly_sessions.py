"""Independent fixtures for nonlinear full-watch weekly interval replay."""
import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from etl import Clock
import advanced_weekly_sessions as weekly
from advanced_weekly_sessions import interval_cdf,union_segments,project_templates,replay_probabilities
from repro import dump,sha256


class WeeklySessionsTests(unittest.TestCase):
    def test_cdf_matches_second_grid_on_disjoint_union(self):
        starts=np.array([2,10,20]);stops=np.array([6,15,22]);queries=np.arange(30)
        expected=[sum(any(a<=s<b for a,b in zip(starts,stops)) for s in range(t)) for t in queries]
        np.testing.assert_array_equal(interval_cdf(starts,stops,queries),expected)

    def test_adjacency_merges_but_channels_and_gaps_remain(self):
        rows,ch,a,b=union_segments([0,0,0,0,1],[0,0,0,1,0],[0,10,30,0,0],[10,20,40,20,10],0,100)
        self.assertEqual(list(zip(rows,ch,a,b)),[(0,0,0,20),(0,0,30,40),(1,0,0,10),(0,1,0,20)])
        with self.assertRaisesRegex(ValueError,'cutoff'):
            union_segments([0],[0],[90],[101],0,100)
        with self.assertRaisesRegex(ValueError,'overlapping'):
            union_segments([0,0],[0,0],[0,5],[10,20],0,100)

    def test_eighty_percent_inclusive_and_nonlinear_control(self):
        qstart=np.array([[0,100],[200,300]]);qstop=qstart+100
        starts=np.array([0,100,200,300]);stops=starts+[80,79,80,0]
        # No zero-length interval enters the union; the last weekly trial is absent.
        starts,stops=starts[:-1],stops[:-1]
        full=replay_probabilities(starts,stops,qstart,qstop,np.array([100,100]),np.ones((2,2),bool),np.ones((2,2)))
        fraction=replay_probabilities(starts,stops,qstart,qstop,np.array([100,100]),np.ones((2,2),bool),np.ones((2,2)),fraction=True)
        np.testing.assert_allclose(full,[1,0])
        np.testing.assert_allclose(fraction,[.8,.395])

    def test_week_boundary_and_cutoff_trials(self):
        clock=Clock('Europe/Kyiv');start=clock.seconds('2020-03-09');cutoff=clock.seconds('2020-03-23')
        # Sunday23:45 30-minute airing crosses into the following Monday.
        template=np.array([[6*86400+23*3600+45*60,1800,0,1800]])
        qs,qe,valid,details=project_templates(clock,start,cutoff,template)
        self.assertEqual(valid[:,0].tolist(),[True,False])
        self.assertEqual(clock.local(qe[0,0]).weekday(),0)
        self.assertEqual(qe[0,0]-qs[0,0],1800)
        probability=replay_probabilities([qs[0,0]],[qe[0,0]],qs,qe,np.array([1800]),valid,np.ones(valid.shape))
        np.testing.assert_array_equal(probability,[1])  # excluded trial does not dilute probability

    def test_disjoint_chunks_can_jointly_reach_eighty_percent(self):
        result=replay_probabilities(np.array([0,60]),np.array([40,100]),
            np.array([[0]]),np.array([[100]]),np.array([100]),np.array([[True]]),np.array([[1.]]))
        np.testing.assert_array_equal(result,[1])

    def test_forecast_boundary_uses_original_airing_denominator(self):
        # A 100-second airing contributes only 50 available seconds to the window.
        # Watching all those seconds is still below80% of the original airing.
        full=replay_probabilities(np.array([0]),np.array([50]),np.array([[0]]),np.array([[50]]),
            np.array([100]),np.array([[True]]),np.array([[1.]]))
        fraction=replay_probabilities(np.array([0]),np.array([50]),np.array([[0]]),np.array([[50]]),
            np.array([100]),np.array([[True]]),np.array([[1.]]),fraction=True)
        np.testing.assert_array_equal(full,[0]);np.testing.assert_array_equal(fraction,[.5])

    def test_dst_nonexistent_start_skipped_and_elapsed_duration_kept(self):
        clock=Clock('Europe/Kyiv');start=clock.seconds('2020-03-23');cutoff=clock.seconds('2020-04-06')
        templates=np.array([[6*86400+3*3600+30*60,3600,0,3600],
                            [6*86400+2*3600+30*60,3600,0,3600]])
        qs,qe,valid,details=project_templates(clock,start,cutoff,templates)
        self.assertFalse(valid[0,0]);self.assertTrue(valid[1,0])
        self.assertEqual(details['nonexistent_local_template_trials_skipped'],1)
        self.assertEqual(qe[0,1]-qs[0,1],3600)
        self.assertEqual(clock.local(qe[0,1]).hour,4)

    def test_corrupted_cached_union_is_rejected_before_scoring(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);ctx=SimpleNamespace(protocol_hash='f'*64,protocol={'input_hashes':{}})
            cache=root/'artifacts/runs/weekly-replay-cache-v1'/ctx.protocol_hash/weekly.SOURCE_SHA256/weekly.hourly.SOURCE_SHA256/weekly.coarse.BUILDER_SOURCE_SHA256
            cache.mkdir(parents=True)
            upstream=root/'upstream';upstream.mkdir();dump(upstream/'manifest.json',{})
            payload=cache/'training-unions.npz';payload.write_bytes(b'original')
            expected=sha256(payload);payload.write_bytes(b'changed')
            dump(cache/'manifest.json',{'protocol_sha256':ctx.protocol_hash,'source_sha256':weekly.SOURCE_SHA256,
                'semantic_version':weekly.SEMANTIC_VERSION,'hourly_source_sha256':weekly.hourly.SOURCE_SHA256,
                'coarse_source_sha256':weekly.coarse.BUILDER_SOURCE_SHA256,'input_hashes':{},
                'hourly_cache':str(upstream),'coarse_cache':str(upstream),
                'hourly_manifest_sha256':sha256(upstream/'manifest.json'),'coarse_manifest_sha256':sha256(upstream/'manifest.json'),
                'cache_data_sha256':{'training-unions.npz':expected}})
            with patch.object(weekly,'ROOT',root):
                with self.assertRaisesRegex(ValueError,'artifact changed'):
                    weekly.weekly_sessions(ctx,{},root)


if __name__=='__main__':unittest.main()
