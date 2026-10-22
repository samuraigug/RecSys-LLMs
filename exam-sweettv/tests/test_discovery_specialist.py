import unittest
from types import SimpleNamespace
import numpy as np
from advanced_discovery_specialist import sample_discovery

class DiscoverySamplingTests(unittest.TestCase):
    def test_conditional_mass_and_users_without_discovery_positives(self):
        class Engine:
            def matrix(self,j,ids):
                return np.column_stack((np.full(len(ids),j),ids)).astype(np.float32)
        f=Engine(); f.m=200; f.c={str(i):i for i in range(200)}
        f.ctx=SimpleNamespace(targets=['big-user-id','other'],truth={'big-user-id':['5','30'],'other':['5']},
                              cold_mask=np.array([False,False]),candidates=list(f.c))
        f.count=np.zeros((2,200));f.count[:,:20]=1
        f.retrievers=[np.tile(np.arange(200),(2,1)) for _ in range(5)]
        x,y,g,w,audit=sample_discovery(f,np.random.default_rng(42),30)
        self.assertTrue(np.all(x[:,1]>=20))
        self.assertEqual(int(y.sum()),1)
        self.assertEqual(set(g),{0,1}) # no-discovery user is negative evidence
        for j in (0,1): self.assertAlmostEqual(float(w[g==j].sum()),180,places=4)
        self.assertTrue(np.all(w[y>0]==1))
        self.assertEqual([a['eligible'] for a in audit],[180,180])
