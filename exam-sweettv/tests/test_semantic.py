import copy
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
from scipy import sparse
import advanced_semantic as m

class SemanticTests(unittest.TestCase):
    def test_masked_pooling_ignores_padding(self):
        x=np.array([[[1.,3.],[3.,7.],[999.,999.]],[[9.,5.],[100.,100.],[200.,200.]]])
        np.testing.assert_array_equal(m.masked_mean(x,[[1,1,0],[1,0,0]]),[[2,5],[9,5]])
    def test_train_document_uses_stable_shortest_title(self):
        doc={'tv_show_title':['Название, 12 эп.','Название'], 'tv_show_category':['Кино'], 'actors':['Иванов']}
        self.assertEqual(m.document(doc,'title'),'Название')
        self.assertIn('Кино',m.document(doc));self.assertIn('Иванов',m.document(doc))
    def test_scoring_ignores_eval_truth_and_unknown_future_title(self):
        class Guard:
            def __getattribute__(self,key):
                if key in ('truth','display'):raise AssertionError('Evaluation content/labels touched')
                return object.__getattribute__(self,key)
        ctx=Guard();ctx.path=Path('/unused');ctx.items=['1','2'];ctx.targets=['0007'];ctx.target_rows=np.array([0]);ctx.candidates=['1','2','999'];ctx.protocol={'train_cutoff_exclusive':'2020-01-01'}
        vectors=np.array([[1.,0.,0.],[0.,1.,0.]],np.float32)
        # Real model vectors are 384d; pad this toy to expose orientation mistakes.
        vectors=np.pad(vectors,((0,0),(0,381)))
        with TemporaryDirectory() as d,patch.object(m,'train_embeddings',return_value=(ctx.items,vectors,{})),patch.object(m,'_matrix',return_value=sparse.csr_matrix([[1.,0.]])),patch.object(m,'sha256',return_value='toy'):
            scores,details=m.semantic_cbf(ctx,{'center':False},Path(d))
            self.assertEqual(scores[0,0],1.)
            self.assertTrue(np.isnan(scores[0,1]))
            self.assertTrue(np.isnan(scores[0,2]))
            self.assertEqual(details['supported_candidates'],2)
            self.assertTrue(details['saved_model_reload_verified'])
    def test_casefold_ties_have_explicit_string_tiebreak(self):
        a={'tv_show_title':["Чистоnews", "ЧистоNews"]}
        b={'tv_show_title':list(reversed(a['tv_show_title']))}
        self.assertEqual(m.document(a,'title'),'ЧистоNews')
        self.assertEqual(m.document(a,'title'),m.document(b,'title'))
    def test_power4_kernel_preserves_repeat_and_suppresses_broad_neighbor(self):
        ctx=SimpleNamespace(path=Path('/unused'),items=['1','2'],targets=['0007'],target_rows=np.array([0]),
            candidates=['1','2','3','999'],protocol={'train_cutoff_exclusive':'2020-01-01'})
        vectors=np.zeros((3,384),np.float32)
        vectors[0,0]=1;vectors[1,1]=1;vectors[2,:3]=[.5,.5,np.sqrt(.5)]
        with TemporaryDirectory() as d,patch.object(m,'train_embeddings',return_value=(['1','2','3'],vectors,{})),patch.object(m,'_matrix',return_value=sparse.csr_matrix([[2.,1.]])),patch.object(m,'sha256',return_value='toy'):
            scores,details=m.semantic_cbf(ctx,{'center':False,'kernel_power':4},Path(d))
            np.testing.assert_allclose(scores[0,:3],[2.,1.,.1875],rtol=1e-6)
            self.assertTrue(np.isnan(scores[0,3]))
            self.assertEqual(details['kernel_power'],4)
            self.assertIn('not pure discovery',details['kernel_scope'])
    def test_pinned_model_manifest_and_file_hashes(self):
        # Read-only check of the real downloaded model; no encoder fit/inference.
        self.assertEqual(m.verify_model()['revision'],m.REVISION)

if __name__=='__main__':unittest.main()
