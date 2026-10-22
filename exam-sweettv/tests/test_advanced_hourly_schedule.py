"""Weekly time smoothing must preserve mass and day/channel boundaries."""
import unittest
import numpy as np
from scipy import sparse
from advanced_hourly_schedule import _marginal,smooth_hours


class HourlyScheduleTests(unittest.TestCase):
    def test_hour_neighbors_cross_midnight_and_week_boundary_only(self):
        profile=sparse.csr_matrix(([1.,2.],([0,1],[23,167])),shape=(2,336))
        smoothed=smooth_hours(profile,2,.5).toarray()
        self.assertEqual(set(np.flatnonzero(smoothed[0])),{22,23,24})
        self.assertEqual(set(np.flatnonzero(smoothed[1])),{166,167,0})
        self.assertEqual(smoothed[1,168],0)  # no leakage into channel2
        np.testing.assert_allclose(smoothed.sum(axis=1),[1,2])

    def test_all_marginals_preserve_viewing_mass(self):
        profile=sparse.csr_matrix(([1.,2.,3.],([0,0,1],[23,200,335])),shape=(2,336))
        for axis in ['hour','weekday','channel']:
            result=_marginal(profile,2,axis)
            np.testing.assert_allclose(np.asarray(result.sum(axis=1)).ravel(),[3,3],atol=1e-6)


if __name__=='__main__':unittest.main()
