import unittest
import numpy as np
from advanced_user_gate import choose_rows
class GateRoutingTests(unittest.TestCase):
    def test_user_choice_preserves_exact_full_expert_row(self):
        experts=[np.array([[1,2,3],[4,5,6]],np.float32),np.array([[7,8,9],[10,11,12]],np.float32),np.zeros((2,3),np.float32)]
        out,choice=choose_rows(experts,[[.1,.9,.2],[.4,.4,.4]])
        np.testing.assert_array_equal(choice,[1,0])
        np.testing.assert_array_equal(out,[[7,8,9],[4,5,6]])
        with self.assertRaises(ValueError):choose_rows(experts,[[np.nan,0,0],[0,0,0]])
