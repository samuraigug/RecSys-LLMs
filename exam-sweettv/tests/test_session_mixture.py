import unittest
import numpy as np
from advanced_session_mixture import completion_factor,shrink_population
class PopulationMixtureTests(unittest.TestCase):
    def test_matches_twenty_prior_sessions_with_different_durations(self):
        own=[10,80];prior=[50,200];durations=np.array([20,50,100,200,300])
        actual=shrink_population(own,durations,20,np.mean(prior),completion_factor(prior,durations))
        expected=completion_factor(own+prior*10,durations)
        np.testing.assert_allclose(actual,expected,rtol=0,atol=1e-15)
