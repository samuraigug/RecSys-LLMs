import unittest
from fractions import Fraction
from types import SimpleNamespace
import numpy as np
from scipy import sparse
from advanced_tifu_knn import daily_vectors, nearest_users, blended_vectors, TrainOnlyContext


class TifuTests(unittest.TestCase):
    def test_independent_daily_group_enumeration_and_padding(self):
        # Duplicate airing within newest day counts once; day7 resets within decay.
        rows = np.array([0, 0, 0, 0, 0, 1])
        cols = np.array([0, 0, 0, 0, 1, 0])
        ages = np.array([0, 0, 1, 7, 8, 6])
        result = daily_vectors(rows, cols, ages, (2, 2), 9, .9, .7).toarray()
        expected = np.array([[float((Fraction(1)+Fraction(9,10)+Fraction(7,10))/14),
                              float(Fraction(7,10)*Fraction(9,10)/14)],
                             [float(Fraction(9,10)**6/14), 0.]])
        np.testing.assert_allclose(result, expected, atol=1e-16)
        control = daily_vectors(rows, cols, ages, (2, 2), 9, 1., 1.).toarray()
        np.testing.assert_allclose(control, [[3/14, 1/14], [1/14, 0.]])
        with self.assertRaises(ValueError):
            daily_vectors(np.array([0]), np.array([0]), np.array([9]), (1,1), 9, .9, .7)

    def test_euclidean_off_candidate_dimensions_self_and_giant_ids(self):
        # Only column0 is a hypothetical forecast candidate. Off-candidate column1
        # makes donor2 closest; restricting distance to column0 would choose donor1.
        v = sparse.csr_matrix([[1., 10.], [1., 0.], [2., 10.], [0., 10.]])
        ids = ['99999999999999999999999999999999', '8', '100000000000000000000', '9']
        rows, distances = nearest_users(v, np.array([0]), np.arange(4), ids, 3)
        # Equidistant donors3/2 ordered numeric9 before giant100000..., self excluded.
        np.testing.assert_array_equal(rows, [[3, 2, 1]])
        direct = [sum((a-b)**2 for a,b in zip(v.toarray()[0], v.toarray()[r])) for r in rows[0]]
        np.testing.assert_array_equal(distances, [direct])
        np.testing.assert_array_equal(distances, [[1., 1., 100.]])

    def test_independent_own_neighbor_mean_and_forbidden_context(self):
        v = sparse.csr_matrix([[1., 4.], [2., 0.], [0., 2.]])
        result = blended_vectors(v, np.array([0]), np.array([[1, 2]]))
        expected = [[float(Fraction(7,10)*1+Fraction(3,10)*Fraction(2,2)),
                     float(Fraction(7,10)*4+Fraction(3,10)*Fraction(2,2))]]
        np.testing.assert_allclose(result, expected)
        ctx = TrainOnlyContext(SimpleNamespace(truth={'leak': 1}, display={'leak': 1}, safe='ok'))
        self.assertEqual(ctx.safe, 'ok')
        for attr in ('truth', 'display'):
            with self.assertRaises(AssertionError):
                getattr(ctx, attr)


if __name__ == '__main__':
    unittest.main()
