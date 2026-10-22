import unittest
import numpy as np
from advanced_depth6 import fit_model

class Depth6RegressionTest(unittest.TestCase):
    def test_effective_depth_limit_with_nonlinear_signal(self):
        # A noisy threshold needs successive partitions; check actual trees,
        # including when an obsolete CatBoost depth parameter is supplied.
        rng=np.random.default_rng(42)
        x=rng.normal(size=(4000,8)).astype(np.float32)
        y=((x[:,0]*x[:,1]+x[:,2]*x[:,3])>0).astype(np.float32)
        model,config=fit_model(x,y,np.ones(len(y)),np.zeros(len(y)),
                              [f'f{i}' for i in range(8)],
                              {'iterations':8,'depth':19,'objective':'binary'})
        self.assertEqual(config['max_depth'],6)
        def depth(node):
            return 0 if 'leaf_index' in node else 1+max(depth(node['left_child']),depth(node['right_child']))
        depths=[depth(t['tree_structure']) for t in model.dump_model()['tree_info']]
        self.assertLessEqual(max(depths),6)
        self.assertGreater(max(depths),1)
