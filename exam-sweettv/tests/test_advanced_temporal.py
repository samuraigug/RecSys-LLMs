"""Tests for training feature/label separation and exposure correction."""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
import numpy as np
from advanced_temporal import temporal_repeat
from etl import save_events
from repro import dump


class TemporalFeatureTests(unittest.TestCase):
    def context(self, path):
        save_events(path/'train-events.npz', [
            ('u','1','10',100,200,100,200,100),
            ('u','1','10',300,400,50,350,100),
            ('u','2','10',500,600,50,550,100)])
        dump(path/'opportunities.json', {'train':{'1':10,'2':10},'future':{'1':20,'2':20}})
        return SimpleNamespace(path=path,targets=['u'],candidates=['1','2'],candidate_map={'1':0,'2':1},
                               protocol={'cutoff_elapsed_seconds':1000,'train_cutoff_exclusive':'2000-01-01'},
                               truth={'u':['1']})

    def test_partial_training_features_preserve_truth_and_exposure(self):
        with TemporaryDirectory() as directory:
            p=Path(directory);ctx=self.context(p)
            scores,_=temporal_repeat(ctx, {'weak_weight':1,'prior_strength':10},p)
            np.testing.assert_allclose(scores,[[1.25,.25]])
            ctx.truth={'u':['2']}
            other,_=temporal_repeat(ctx, {'weak_weight':1,'prior_strength':10},p)
            np.testing.assert_array_equal(scores,other)
            features,_=temporal_repeat(ctx,{'weak_weight':0,'feature_full_threshold':.5,'prior_strength':10},p)
            np.testing.assert_allclose(features,[[2,1]])
            self.assertEqual(ctx.truth,{'u':['2']})

    def test_future_event_cannot_enter_training_features(self):
        with TemporaryDirectory() as directory:
            p=Path(directory);ctx=self.context(p)
            save_events(p/'train-events.npz',[('u','1','10',900,1100,200,1100,200)])
            with self.assertRaises(AssertionError):temporal_repeat(ctx,{},p)

if __name__=='__main__':unittest.main()
