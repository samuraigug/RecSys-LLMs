"""Train-catalogue word/character CBF and pre-cutoff partial-view profiles.

Text processing follows scikit-learn's TF-IDF implementation. No future title,
genre, person, validation label or validation view is read during fitting.
"""
import re

import numpy as np
from scipy import sparse
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from advanced_factorization import expanded_context
from advanced_collaborative import _matrix
from models import normalized_text, year_tokens
from repro import dump, load, sha256


def _values(doc,field):
    if field in ('word_title','char_title'):values=doc['tv_show_title']
    elif field=='category':values=doc['tv_show_category']
    elif field=='genres':values=[v for f in ('tv_show_genre_1','tv_show_genre_2','tv_show_genre_3') for v in doc[f]]
    elif field=='year':values=year_tokens(doc['year_of_production'])
    elif field in ('actors','director'):values=[name.strip() for value in doc[field] for name in re.split(r'[,;]',value)]
    else:raise ValueError('Unsupported content field: '+field)
    return sorted({normalized_text(v) for v in values if normalized_text(v) not in ('','-','нет','n/a','null')})


def content_vectors(metadata,p):
    items=sorted(metadata,key=int);docs=[metadata[i] for i in items]
    weights=p.get('blocks',{'char_title':2,'category':.25,'genres':.5,'director':.25,'actors':.25,'year':.1})
    blocks=[];schemas={};idf={}
    for field,weight in weights.items():
        if weight<=0:continue
        tokens=[_values(doc,field) for doc in docs]
        if field in ('word_title','char_title'):
            char=field=='char_title'
            kwargs={'analyzer':'char_wb' if char else 'word','ngram_range':(3,5) if char else (1,2),
                    'sublinear_tf':True,'max_features':p.get('char_features',40000) if char else p.get('word_features',30000),
                    'min_df':p.get('char_min_df',2) if char else 1,'dtype':np.float32}
            vectorizer=TfidfVectorizer(**kwargs)
            matrix=vectorizer.fit_transform([' '.join(v) for v in tokens])
            idf[field]=vectorizer.idf_
            configuration={k:list(v) if isinstance(v,tuple) else v for k,v in kwargs.items() if k!='dtype'}
        else:
            vectorizer=DictVectorizer(dtype=np.float32)
            matrix=vectorizer.fit_transform([{v:1. for v in values} for values in tokens])
            configuration={'kind':'normalized categorical one-hot'}
        block=normalize(matrix,norm='l2')*weight
        blocks.append(block)
        schemas[field]={'weight':weight,'features':matrix.shape[1],'configuration':configuration,
                        'vocabulary':vectorizer.get_feature_names_out().tolist()}
    if not blocks:raise ValueError('No active CBF fields')
    vectors=normalize(sparse.hstack(blocks,format='csr'),norm='l2')
    return items,vectors,schemas,idf


def content_cbf(ctx,p,path):
    items,vectors,schema,idf=content_vectors(ctx.metadata,p)
    im={item:j for j,item in enumerate(items)}
    history_ctx=expanded_context(ctx) if p.get('weak_weight') is not None else ctx
    if any(i not in im for i in history_ctx.items):
        raise ValueError('History item has no train-only metadata')
    profile_matrix=_matrix(history_ctx,{'strength':'count',**p})[history_ctx.target_rows]
    features=vectors[np.array([im[i] for i in history_ctx.items])]
    profiles=normalize(profile_matrix@features,norm='l2')
    candidate_vectors=sparse.vstack([vectors[im[i]] if i in im else sparse.csr_matrix((1,vectors.shape[1]))
                                    for i in ctx.candidates],format='csr')
    raw=(profiles@candidate_vectors.T).toarray().astype(np.float32)
    sparse.save_npz(path/'content-train-vectors.npz',vectors)
    sparse.save_npz(path/'content-candidate-vectors.npz',candidate_vectors)
    sparse.save_npz(path/'content-user-profiles.npz',profiles)
    np.savez_compressed(path/'content-idf.npz',**idf)
    dump(path/'content-vocabulary.json',schema)
    dump(path/'content-mappings.json',{'train_items':items,'profile_items':history_ctx.items,'targets':ctx.targets,'candidates':ctx.candidates})
    restored=(sparse.load_npz(path/'content-user-profiles.npz')@sparse.load_npz(path/'content-candidate-vectors.npz').T).toarray()
    assert np.allclose(raw,restored,rtol=1e-5,atol=1e-6)
    factor=np.ones(len(ctx.candidates),np.float32)
    if p.get('opportunity_adjustment'):
        opp=load(ctx.path/'opportunities.json');prior=p.get('prior_strength',50)
        factor=np.array([opp['future'].get(i,0)/(opp['train'].get(i,0)+prior) for i in ctx.candidates],np.float32)
        raw*=factor[None,:]
    np.save(path/'content-exposure-factor.npy',factor)
    assert np.allclose(raw,restored*factor[None,:],rtol=1e-5,atol=1e-6)
    context_details=None
    if p.get('hourly_product'):
        # A separate schedule hypothesis: cosine modulates expected airing counts,
        # rather than using a global opportunity ratio across unrelated programmes.
        from advanced_hourly_schedule import schedule_hourly
        schedule,context_details=schedule_hourly(ctx,{'joint_weight':.5,'neighbor_smooth':.25,'half_life_days':28},path)
        schedule=np.nan_to_num(schedule,nan=0.)
        schedule/=np.maximum(schedule.max(axis=1,keepdims=True),1e-12)
        raw=np.maximum(raw,0)**p.get('content_power',.25)*schedule
        np.savez_compressed(path/'content-hourly-modulation.npz',scores=schedule)
        restored_context=np.load(path/'content-hourly-modulation.npz')['scores']
        replay=np.maximum(restored*factor[None,:],0)**p.get('content_power',.25)*restored_context
        assert np.allclose(raw,replay,rtol=1e-5,atol=1e-6)
    raw[raw<=0]=np.nan
    return raw,{'algorithm':'normalized content-profile cosine with word/char_wb TF-IDF title blocks',
        'blocks':{k:{'weight':v['weight'],'features':v['features']} for k,v in schema.items()},
        'train_catalogue_items':len(items),'profile_items':len(history_ctx.items),'features':vectors.shape[1],
        'profile_nnz':profiles.nnz,'fit_cutoff':ctx.protocol['train_cutoff_exclusive'],
        'metadata_sha256':sha256(ctx.path/'metadata-train.json'),'future_metadata_used':False,
        'future_views_used':False,'saved_model_reload_verified':True,
        'context_modulation':context_details,
        'unsupported_candidate_items':sum(i not in im for i in ctx.candidates),
        'source':'https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html'}


def register(methods):methods['content_cbf']=content_cbf
