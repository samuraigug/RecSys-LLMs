"""Directed adjacent-week transfer, with a matched symmetrized control.

Weeks are elapsed seven-day bins counted backward from the frozen cutoff, based
on full-event last_stop. The oldest bin can be partial. Edges use only pairs of
already observed adjacent bins. No future interactions or display metadata.
"""
import numpy as np
from scipy import sparse
from sklearn.preprocessing import normalize
from repro import dump,sha256
from advanced_collaborative import _scores

WEEK=7*86400


def matrices(ctx):
    eu,ei,e=ctx.events()
    assert np.all(e[:,6] <= ctx.protocol['cutoff_elapsed_seconds'])
    users=eu[e[:,0]];items=ei[e[:,1]]
    assert np.all(users>=0) and np.all(items>=0)
    # A view ending exactly on a bin's upper edge belongs to that bin.
    age=np.maximum(0,ctx.protocol['cutoff_elapsed_seconds']-e[:,6])
    week=(age//WEEK).astype(np.int64)
    result=[]
    for w in range(int(week.max())+1):
        keep=week==w
        x=sparse.csr_matrix((np.ones(int(keep.sum())),(users[keep],items[keep])),shape=(len(ctx.users),len(ctx.items)))
        x.data[:]=1
        result.append(x)
    return result


def graph_from_weeks(weeks,symmetric=False,shrinkage=20.):
    if len(weeks)<2:raise ValueError('Need two historical weeks')
    m=weeks[0].shape[1]; counts=sparse.csr_matrix((m,m),dtype=np.float64)
    used_pairs=0
    for older in range(1,len(weeks)):
        left=normalize(weeks[older],norm='l1',axis=1)
        right=normalize(weeks[older-1],norm='l1',axis=1)
        counts += left.T @ right
        used_pairs += int(np.sum((weeks[older].getnnz(1)>0)&(weeks[older-1].getnnz(1)>0)))
    counts.setdiag(0);counts.eliminate_zeros()
    if symmetric:counts=(counts+counts.T)*.5
    support=np.asarray(counts.sum(1)).ravel()
    pop=np.asarray(sum(weeks).sum(0)).ravel()
    graph=sparse.diags(1/(support+shrinkage)) @ counts @ sparse.diags(1/np.sqrt(1+pop))
    return graph.tocsr(),counts.tocsr(),dict(active_user_week_pairs=used_pairs,nonzero_edges=counts.nnz,
                                          row_support_sum=float(support.sum()),symmetric=symmetric,shrinkage=shrinkage)


def weekly_transition(ctx,p,path):
    weeks=matrices(ctx)
    graph,counts,details=graph_from_weeks(weeks,p.get('symmetric',False),20.)
    # Same source profile and destination-exposure multiplier for both controls.
    profile=sparse.csr_matrix(weeks[0].shape,dtype=np.float64)
    for w,x in enumerate(weeks):profile += normalize(x,norm='l1',axis=1)*np.exp2(-7*w/28)
    profile=normalize(profile,norm='l1',axis=1)
    raw=(profile[ctx.target_rows] @ graph).toarray()
    sparse.save_npz(path/'transition-graph.npz',graph)
    sparse.save_npz(path/'transition-counts.npz',counts)
    sparse.save_npz(path/'source-profile.npz',profile[ctx.target_rows])
    dump(path/'mapping.json',dict(items=ctx.items,users=ctx.targets))
    restored=sparse.load_npz(path/'transition-graph.npz')
    np.testing.assert_array_equal(raw,(profile[ctx.target_rows]@restored).toarray())
    details.update(weeks=len(weeks),events_sha256=sha256(ctx.path/'train-events.npz'),
                   protocol_sha256=ctx.protocol_hash,source_sha256=sha256(__file__),
                   no_future_views=True,no_future_metadata=True,model_reload_exact=True,
                   time_bin='elapsed 7-day bins backward from cutoff, oldest may be partial',
                   graph='sum L1(source user-week).T @ L1(next user-week); diagonal zero; row shrink20; destination weekly-popularity sqrt correction',
                   scoring='L1 profile of weekly L1 vectors, 28-day half-life; graph transfer; same future exposure correction as PPR')
    dump(path/'transition-provenance.json',details)
    return _scores(ctx,raw,dict(opportunity_adjustment=True,prior_strength=50)),details


def register():
    from models import METHODS
    METHODS['weekly_transition']=weekly_transition
