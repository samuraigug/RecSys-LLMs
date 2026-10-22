import unittest
from unittest.mock import patch
from types import SimpleNamespace
from pathlib import Path
import tempfile
import numpy as np
from scipy.stats import rankdata
from advanced_content_discovery import add_content_features, sample_discovery

class ContentDiscoveryTests(unittest.TestCase):
    def fixture(self,path):
        class F:
            def add(self,name,value):self.names.append(name);self.signals.append(value)
            def rank(self,value):return rankdata(-value,axis=1,method='min')
        f=F();f.u={'9007199254740993':0};f.n=1;f.names=[];f.signals=[];f.details={}
        def doc(value):return dict(tv_show_category=[value],tv_show_genre_1=[value],tv_show_genre_2=[],tv_show_genre_3=[],actors=[value],director=[value])
        class Poison:
            def __getattribute__(self,name):raise AssertionError('truth/display access')
        f.ctx=SimpleNamespace(metadata={'1':doc('a'),'2':doc('b')},path=path,protocol={'cutoff_elapsed_seconds':100},candidates=['1','2','3'],truth=Poison(),display=Poison())
        (path/'train-events.npz').write_bytes(b'fixture');(path/'metadata-train.json').write_text('{}')
        return f
    def test_train_metadata_only_and_string_users(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);f=self.fixture(p)
            e=np.array([[0,0,0,0,100,90,100,100]])
            with patch('advanced_content_discovery.read_events',return_value=(['9007199254740993'],['1'],[],e)):
                add_content_features(f,p/'features')
            self.assertEqual(len(f.names),8)
            for values in f.signals[::2]:np.testing.assert_allclose(values,[[1,0,0]])
    def test_post_cutoff_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);f=self.fixture(p)
            e=np.array([[0,0,0,0,101,90,101,100]])
            with patch('advanced_content_discovery.read_events',return_value=(['9007199254740993'],['1'],[],e)):
                with self.assertRaisesRegex(ValueError,'Post-cutoff'):add_content_features(f,p/'features')
    def test_negative_only_groups_and_conditional_mass(self):
        f=SimpleNamespace(c={str(i):i for i in range(200)},count=np.zeros((2,200)))
        f.count[:,:20]=1
        f.ctx=SimpleNamespace(targets=['9007199254740993','negative-only'],truth={'9007199254740993':['30'],'negative-only':['5']},cold_mask=np.array([False,False]),candidates=list(f.c))
        f.retrievers=[np.tile(np.arange(200),(2,1)) for _ in range(5)]
        f.matrix=lambda j,ids:np.column_stack((np.full(len(ids),j),ids)).astype(np.float32)
        x,y,g,w,a=sample_discovery(f,np.random.default_rng(42))
        self.assertEqual(set(g),{0,1});self.assertEqual(a[1]['positive'],0)
        self.assertEqual(a[0]['user'],'9007199254740993')
        for j in (0,1):self.assertAlmostEqual(float(w[g==j].sum()),180,places=4)
        self.assertTrue(np.all(x[:,1]>=20));self.assertTrue(np.all(w[y>0]==1))
