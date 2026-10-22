#!/usr/bin/env python3
"""Read-only fitted-state replay and independent AP checks; no encoder fitting."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from sklearn.preprocessing import normalize
from scipy import sparse
from models import Context, normalized_text
from advanced_factorization import expanded_context
from advanced_collaborative import _matrix
from advanced_semantic import document,verify_model,MANIFEST_SHA
from experiments import finalize_scores
from threadpoolctl import threadpool_limits
from repro import dump,load,sha256
from tools.verify_experiments import verify

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    verify_model();results=[]
    evidence=ROOT/'artifacts/semantic-research';isolated=evidence/'verification-runs';isolated.mkdir(exist_ok=True)
    for split in ('main','early'):
        ctx=Context(ROOT/'data/processed/elapsed-v2'/split)
        (isolated/split).mkdir(exist_ok=True)
        for run in sorted((ROOT/'artifacts/runs'/split).glob('SEM-*')):
            record=load(run/'record.json')
            if record['status']!='SUCCEEDED':continue
            link=isolated/split/run.name
            if not link.exists():link.symlink_to(run,target_is_directory=True)
            if record['method']!='semantic_cbf':continue
            p=record['parameters'];details=record['model_details'];folder=Path(details['embedding_cache']['path'])
            manifest=load(folder/'cache-manifest.json')
            assert manifest['key']['model_manifest_sha256']==MANIFEST_SHA
            assert manifest['key']['protocol_sha256']==ctx.protocol_hash
            assert manifest['key']['metadata_sha256']==sha256(ctx.path/'metadata-train.json')
            assert details['embedding_cache']['manifest_sha256']==sha256(folder/'cache-manifest.json')
            assert manifest['key']['source_sha256']==record['loaded_model_source_sha256']==sha256(run/'advanced_semantic-source.py')
            for name,expected in manifest['files'].items():assert sha256(folder/name)==expected
            docs=load(folder/'documents.json');items=sorted(ctx.metadata,key=int)
            assert docs['items']==items
            rebuilt=[document(ctx.metadata[i],p.get('document_mode','metadata')) for i in items]
            document_bytes_equal=docs['documents']==rebuilt
            assert [normalized_text(d) for d in docs['documents']]==[normalized_text(d) for d in rebuilt]
            original=np.load(folder/'embeddings.npy',allow_pickle=False);vectors=original.copy()
            saved=np.load(run/'semantic-model.npz',allow_pickle=False)
            active=np.linalg.norm(vectors,axis=1)>0
            center=vectors[active].mean(axis=0,keepdims=True) if p.get('center',True) else np.zeros((1,384),np.float32)
            np.testing.assert_array_equal(center,saved['center'])
            if p.get('center',True):vectors[active]=normalize(vectors[active]-center)
            im={i:j for j,i in enumerate(items)};history=expanded_context(ctx) if p.get('weak_weight') is not None else ctx
            x=_matrix(history,{'strength':'count',**p})[history.target_rows]
            profiles=normalize(x@vectors[np.array([im[i] for i in history.items])]).astype(np.float32)
            np.testing.assert_array_equal(profiles,saved['profiles'])
            candidates=np.zeros((len(ctx.candidates),384),np.float32)
            for j,i in enumerate(ctx.candidates):
                if i in im:candidates[j]=vectors[im[i]]
            np.testing.assert_array_equal(candidates,saved['candidates'])
            if p.get('kernel_power'):
                stored_x=sparse.load_npz(run/'semantic-kernel-history.npz')
                assert (stored_x!=x).nnz==0
                history_vectors=vectors[np.array([im[i] for i in history.items])]
                np.testing.assert_array_equal(history_vectors,np.load(run/'semantic-kernel-history-vectors.npy',allow_pickle=False))
                scores=np.asarray(x@(np.clip(history_vectors@candidates.T,0,1)**p['kernel_power']),dtype=np.float32)
                scores*=saved['exposure_factor'][None,:]
            else:
                scores=(profiles@candidates.T)*saved['exposure_factor'][None,:]
            if p.get('hourly_product'):
                hourly=np.load(run/'semantic-hourly-modulation.npy',allow_pickle=False)
                scores=np.maximum(scores,0)**p.get('content_power',.25)*hourly
            scores[scores<=0]=np.nan
            pred,_,indices,values=finalize_scores(scores,ctx)
            assert pred==load(run/'top5.json')
            native=np.load(run/'native-top100.npz',allow_pickle=False)
            np.testing.assert_array_equal(indices,native['indices']);np.testing.assert_array_equal(values,native['scores'])
            results.append(dict(split=split,run_id=run.name,map=record['metrics']['map_at_5_main'],metadata_items=len(items),
                supported_candidates=sum(i in im for i in ctx.candidates),profiles_rederived_exact=True,candidates_rederived_exact=True,
                native_top100_exact=True,top5_exact=True,cache_payload_hashes_verified=True,source_sha256=record['loaded_model_source_sha256'],
                document_bytes_match_current_builder=document_bytes_equal,document_casefold_content_matches_train=True))
        verify(split,evidence/('independent-'+split+'.json'),ROOT/'data/processed/elapsed-v2',isolated)
    dump(args.output,dict(status='PASS',fits_repeated=False,encoder_calls_repeated=False,checks=results))

if __name__=='__main__':
    with threadpool_limits(limits=1):main()
