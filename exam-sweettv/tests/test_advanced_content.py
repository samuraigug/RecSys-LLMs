"""Train-only CBF vocabulary, partial weights and unsupported future items."""
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np
from scipy import sparse

from advanced_content import content_cbf,content_vectors
from etl import save_events
from repro import dump,load


class ContentTests(unittest.TestCase):
    def test_char_wb_ngrams_do_not_bridge_words(self):
        _,_,schema,idf=content_vectors({'1':{'tv_show_title':['abc def']}},
                                      {'blocks':{'char_title':1},'char_min_df':1})
        vocab=schema['char_title']['vocabulary']
        self.assertIn('abc',vocab)
        self.assertNotIn('c d',vocab)
        self.assertTrue(all(3<=len(v)<=5 for v in vocab))
        self.assertEqual(len(idf['char_title']),len(vocab))

    def test_partial_squared_weights_and_no_future_title_features(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)
            metadata={'1':{'tv_show_title':['alphaaaa']},'2':{'tv_show_title':['omegazzz']}}
            dump(path/'metadata-train.json',metadata)
            save_events(path/'train-events.npz',[
                ('u','1','1',100,200,100,200,100),
                ('u','2','1',300,400,50,350,100)])
            class Blocked:
                def get(self,*args):raise AssertionError('Future metadata/truth accessed')
                def __getitem__(self,key):raise AssertionError('Future metadata/truth accessed')
            ctx=SimpleNamespace(path=path,metadata=metadata,display=Blocked(),truth=Blocked(),
                protocol={'train_cutoff_exclusive':'2020-05-18','cutoff_elapsed_seconds':2000},
                targets=['u'],users=['u'],user_map={'u':0},items=['1'],item_map={'1':0},
                target_rows=np.array([0]),candidates=['1','2','3'],strength=lambda _:sparse.csr_matrix([[1.]]))
            scores,details=content_cbf(ctx,{'blocks':{'word_title':1},'weak_weight':1},path)
            self.assertAlmostEqual(float(scores[0,1]/scores[0,0]),.25,places=6)
            self.assertTrue(np.isnan(scores[0,2]))
            self.assertEqual(details['unsupported_candidate_items'],1)
            self.assertTrue(details['saved_model_reload_verified'])
            self.assertNotIn('futuresecret',load(path/'content-vocabulary.json')['word_title']['vocabulary'])
            no_partial=path/'full';no_partial.mkdir()
            full,_=content_cbf(ctx,{'blocks':{'word_title':1},'weak_weight':0},no_partial)
            self.assertTrue(np.isnan(full[0,1]))


if __name__=='__main__':unittest.main()
