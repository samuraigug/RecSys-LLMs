import unittest
import numpy as np
from advanced_posterior_predictive import greedy_expected_ap,select_sample_sets
class PosteriorTests(unittest.TestCase):
    def test_greedy_increments_equal_explicit_ap_enumeration(self):
        sets=np.array([[0,1,2,-1,-1],[1,3,-1,-1,-1],[2,4,5,-1,-1],[-1]*5],np.int32)
        _,selected=greedy_expected_ap(sets,6)
        def ap(order,row):
            truth=set(int(i) for i in row if i>=0)
            if not truth:return 0.
            return sum((sum(i in truth for i in order[:r])/r) for r,i in enumerate(order,1) if i in truth)/min(5,len(truth))
        order=[]
        for chosen in selected:
            values=[np.mean([ap(order+[i],row) for row in sets]) if i not in order else -1 for i in range(6)]
            self.assertEqual(chosen,int(np.argmax(values)));order.append(chosen)
    def test_sample_count_duration_and_numeric_ties(self):
        count=np.array([[2,2,2,0,1,2]])
        d=np.array([100,200,200,500,500,50]);ids=np.array([10,20,30,40,50,60])
        np.testing.assert_array_equal(select_sample_sets(count,d,ids),[[1,2,0,5,4]])
    def test_beta_binomial_expected_mean(self):
        rng=np.random.default_rng(7);a,b,n=2.,8.,20
        count=rng.binomial(n,rng.beta(a,b,100000))
        self.assertAlmostEqual(float(count.mean()),n*a/(a+b),delta=.05)
