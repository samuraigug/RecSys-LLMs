import unittest
import numpy as np
from scipy import sparse
from advanced_weekly_transition import graph_from_weeks
class DirectedGraphTests(unittest.TestCase):
    def test_temporal_direction_is_not_transposed(self):
        # Sole user watched source0 in older week, destination1 in newer week.
        older=sparse.csr_matrix([[1.,0.,0.]])
        newer=sparse.csr_matrix([[0.,1.,0.]])
        graph,counts,_=graph_from_weeks([newer,older])
        self.assertEqual(counts[0,1],1)
        self.assertEqual(counts[1,0],0)
        self.assertGreater(graph[0,1],0)
        _,control,_=graph_from_weeks([newer,older],symmetric=True)
        self.assertEqual(control[0,1],.5)
        self.assertEqual(control[1,0],.5)
    def test_heavy_user_week_mass_is_normalized_and_self_edge_removed(self):
        older=sparse.csr_matrix([[1.,1.,0.]])
        newer=sparse.csr_matrix([[0.,1.,1.]])
        _,c,_=graph_from_weeks([newer,older])
        self.assertEqual(c[0,2],.25)
        self.assertEqual(c[1,1],0)
        self.assertEqual(c.sum(),.75)
