"""Ranking-oriented MF and expanded weak-history graph; history only."""
import copy
from pathlib import Path
import numpy as np
from etl import read_events
from advanced_collaborative import _matrix, _scores, _details, ppr
from repro import dump
from models import METHODS


def expanded_context(ctx):
    eu,ei,_,events=read_events(ctx.path/'train-events.npz',full_only=False)
    assert np.all(events[:,6]<=ctx.protocol['cutoff_elapsed_seconds'])
    other=copy.copy(ctx)
    other.users=sorted(set(ctx.targets)|set(eu));other.user_map={u:j for j,u in enumerate(other.users)}
    other.items=sorted(set(ei),key=int);other.item_map={i:j for j,i in enumerate(other.items)}
    other.target_rows=np.array([other.user_map[u] for u in ctx.targets])
    return other


def expanded_ppr(ctx,p,path):
    other=expanded_context(ctx)
    scores,details=ppr(other,{'weak_weight':1,'strength':'count',**p},path)
    details.update(universe='all pre-cutoff full or >=60s programmes/users',original_full_items=len(ctx.items),expanded_items=len(other.items))
    return scores,details


def ranking_mf(ctx,p,path):
    from implicit.cpu.bpr import BayesianPersonalizedRanking
    from implicit.cpu.lmf import LogisticMatrixFactorization
    ctx=expanded_context(ctx) if p.get('expanded') else ctx
    x=_matrix(ctx,{'strength':'binary','weak_weight':p.get('weak_weight',0)})
    cls=BayesianPersonalizedRanking if p.get('loss','bpr')=='bpr' else LogisticMatrixFactorization
    pars={'factors':p.get('factors',64),'learning_rate':p.get('learning_rate',.05 if cls is BayesianPersonalizedRanking else .5),
          'regularization':p.get('regularization',.01 if cls is BayesianPersonalizedRanking else .6),
          'iterations':p.get('iterations',100),'num_threads':1,'random_state':p.get('seed',42)}
    model=cls(**pars);model.fit(x.astype(np.float32),show_progress=False)
    raw=model.user_factors[ctx.target_rows]@model.item_factors.T
    active=np.asarray(x.sum(axis=0)).ravel()>0
    raw[:,~active]=np.nan
    model.save(path/'ranking-mf.npz');restored=cls.load(path/'ranking-mf.npz')
    assert np.array_equal(model.user_factors,restored.user_factors)
    assert np.array_equal(model.item_factors,restored.item_factors)
    dump(path/'mf-mapping.json',{'users':ctx.users,'items':ctx.items})
    scores=_scores(ctx,raw,{**p,'positive_only':False})
    return scores,{**_details(ctx,x),'loss':p.get('loss','bpr'),'factors':pars['factors'],'iterations':pars['iterations'],
                   'seed':pars['random_state'],'sample_weights':'binary positive interactions; missing are sampled negatives, not known dislikes',
                   'saved_model_reload_verified':True,'sources':['https://github.com/benfred/implicit/blob/main/implicit/cpu/bpr.pyx']}


def register():METHODS.update(expanded_ppr=expanded_ppr,ranking_mf=ranking_mf)
