"""Faithful tests of shared channel learning and cold-programme identities."""
import csv
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
from scipy import sparse

import advanced_collective as collective
from etl import save_events,FILES
from repro import sha256
from tests import test_schedule_audit as schedule_fixture

BIG_USER=schedule_fixture.BIG_USER


class CollectiveAuditTests(unittest.TestCase):
    def fixture(self,root):
        helper=schedule_fixture.ScheduleContractAuditTests()
        ctx=helper.fixture(root)
        ctx.path=root/'processed';ctx.path.mkdir()
        ctx.items=['100'];ctx.history={BIG_USER:{'100':{'airings':1,'seconds':5400,'last_stop':1578276000}}}
        ctx.candidates=['100','200','300'];ctx.candidate_map={i:j for j,i in enumerate(ctx.candidates)}
        path=root/'data/raw'/FILES[0]
        with path.open(encoding='utf-8',newline='') as f:rows=list(csv.DictReader(f));fields=list(rows[0])
        rows.extend([
            dict(zip(fields,['10','06.01.2020 02:30:00','5400','100','Past programme','Actor'])),
            dict(zip(fields,['99','15.01.2020 01:00:00','3600','300','Unknown future channel','Future actor']))])
        helper.write_csv(path,fields,rows)
        ctx.protocol['input_hashes']={n:sha256(root/'data/raw'/n) for n in ['dataset11-30.csv',*FILES]}
        cutoff=ctx.protocol['cutoff_elapsed_seconds']
        save_events(ctx.path/'train-events.npz',[(BIG_USER,'100','10',cutoff-5400,cutoff,5400,cutoff,5400)])
        return ctx

    def test_raw_context_contains_all_prefix_users_but_no_future_seconds(self):
        with TemporaryDirectory() as directory:
            root=Path(directory);ctx=self.fixture(root)
            users,channels,seconds,past,future,last=collective.build_context(ctx,root/'data/raw')
            self.assertIn('other-user',users)
            self.assertEqual(float(seconds[users.index(BIG_USER)].sum()),9000)
            self.assertEqual(float(seconds[users.index('other-user')].sum()),86400)
            self.assertLessEqual(last,ctx.protocol['cutoff_elapsed_seconds'])
            self.assertEqual(channels,['10','20'])
            self.assertEqual(len(past),1)

    def test_cold_programme_uses_trained_channel_and_seed_replay(self):
        with TemporaryDirectory() as directory:
            root=Path(directory);ctx=self.fixture(root);one=root/'one';two=root/'two';control=root/'control'
            for p in [one,two,control]:p.mkdir()
            params={'factors':4,'epochs':3,'seed':42,'shared_channels':True,'auxiliary_weight':.1}
            with patch.object(collective,'ROOT',root):
                first,detail=collective.collective_warp(ctx,params,one)
                replay,_=collective.collective_warp(ctx,params,two)
                plain,_=collective.collective_warp(ctx,{**params,'shared_channels':False},control)
            np.testing.assert_array_equal(first,replay)
            features=sparse.load_npz(one/'candidate-features.npz')
            self.assertEqual(features[1,0],0) # item200 has no identity feature
            self.assertGreater(features[1,2],0) # observed channel20 supplies its representation
            self.assertTrue(np.isfinite(first[0,1]))
            self.assertTrue(np.isnan(first[0,2])) # unseen channel99 gets no random representation
            self.assertTrue(np.isnan(plain[0,1]))
            self.assertEqual(detail['cold_programme_identity_nnz'],0)
            self.assertTrue(detail['saved_model_reload_verified'])
            self.assertTrue(detail['library_predict_crosscheck'])

    def test_changed_raw_input_cannot_enter_frozen_context(self):
        with TemporaryDirectory() as directory:
            root=Path(directory);ctx=self.fixture(root)
            with (root/'data/raw/dataset11-30.csv').open('a') as stream:stream.write('\n')
            with patch.object(collective,'ROOT',root):
                with self.assertRaisesRegex(ValueError,'Raw data changed'):
                    collective._context(ctx)


if __name__=='__main__':unittest.main()
