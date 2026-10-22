import unittest
import numpy as np
from advanced_session_survival import completion_factor
class CompletionGeometryTests(unittest.TestCase):
    def test_continuous_offset_measure_by_midpoint_enumeration(self):
        # Integer endpoints permit exact midpoint integration with step1.
        for length in (10,40,80,100,200):
            for duration in (50,100,150):
                offsets=np.arange(-duration,length,dtype=float)+.5
                overlap=np.maximum(0,np.minimum(length,offsets+duration)-np.maximum(0,offsets))
                measured=np.sum(overlap>=.8*duration)
                self.assertAlmostEqual(completion_factor([length],[duration])[0],measured/length,places=12)
    def test_empirical_mixture_is_length_weighted(self):
        d=np.array([50.,100.,200.]);lengths=[10,80,200]
        expected=sum(l*completion_factor([l],d) for l in lengths)/sum(lengths)
        np.testing.assert_allclose(completion_factor(lengths,d),expected,rtol=0,atol=1e-15)
        self.assertTrue(np.all(completion_factor([],d)==0))
