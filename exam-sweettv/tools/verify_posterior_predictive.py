"""Independent saved-state audit: integer AP utility and full-sort MC reference."""
import sys,json,hashlib,copy,csv,shutil
from pathlib import Path
from fractions import Fraction
from collections import Counter
from unittest.mock import patch
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));OUT=ROOT/'research/posterior-predictive-audit-20261022'
from models import Context
load=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
class Poison:
    def __getattribute__(self,name):raise AssertionError('Forbidden truth/display access: '+name)
def validate_output(ctx,run,scores):
    rec=load(run/'record.json');assert rec['status']=='SUCCEEDED' and rec['protocol_sha256']==ctx.protocol_hash
    z=np.load(run/'native-top100.npz');top=load(run/'top5.json');num=np.array(list(map(int,ctx.candidates)));aps={};checked=0
    assert set(top)==set(ctx.targets)
    for j,u in enumerate(ctx.targets):
        if ctx.cold_mask[j]:assert (z['indices'][j]==-1).all() and np.isnan(z['scores'][j]).all();expected=ctx.fallback[:5]
        else:
            ids=np.lexsort((num,-scores[j]))[:100];assert np.array_equal(z['indices'][j],ids);assert z['scores'][j].tobytes()==scores[j,ids].astype(np.float32).tobytes();checked+=100
            expected=[ctx.candidates[i] for i in ids[:5]]
        assert top[u]==expected and len(set(expected))==5 and all(isinstance(i,str) and i in ctx.candidate_map for i in expected)
        truth=set(ctx.truth[u]);hits=0;ap=Fraction()
        for rank,i in enumerate(expected,1):
            if i in truth:hits+=1;ap+=Fraction(hits,rank)
        aps[u]=ap/min(5,len(truth)) if truth else Fraction()
    mean=float(sum((aps[u] for u in ctx.cohorts['main']),Fraction())/len(ctx.cohorts['main']));assert abs(mean-rec['metrics']['map_at_5_main'])<1e-14
    with (run/'per-user-ap.csv').open() as fp:
        rows=list(csv.DictReader(fp));assert [r['user_id'] for r in rows]==ctx.targets
        for r in rows:assert abs(float(r['ap_at_5'])-float(aps[r['user_id']]))<1e-14
    return {'map_fraction':mean,'native_scores_ranks_exact':checked,'all_top5_and_cold_exact':True}
def audit(split):
    ctx=Context(ROOT/'data/processed/elapsed-v2'/split);run=ROOT/'artifacts/runs'/split/'BP-mean256';v=load(run/'posterior-manifest.json')
    assert load(run/'record.json')['status']=='SUCCEEDED'
    for name,h in v['payload_sha256'].items():assert sha(run/name)==h
    assert v['protocol_sha256']==ctx.protocol_hash and v['source_sha256']==sha(ROOT/'advanced_posterior_predictive.py')
    assert sha(ctx.path/'train-events.npz')==v['source_events_sha256'] and sha(ctx.path/'opportunities.json')==v['opportunities_sha256']
    s=np.load(run/'posterior-state.npz');sets=np.load(run/'sampled-top5.npy');means=np.load(run/'mean-scores.npy');greedy=np.load(run/'greedy-scores.npy');m=len(ctx.candidates)
    assert s['users'].tolist()==ctx.targets and s['items'].tolist()==ctx.candidates
    opp=load(ctx.path/'opportunities.json');train=np.array([opp['train'].get(i,0) for i in ctx.candidates],float);future=np.array([opp['future'].get(i,0) for i in ctx.candidates],np.int64)
    assert np.array_equal(s['train'],train) and np.array_equal(s['future'],future)
    success=np.array([[ctx.history.get(u,{}).get(i,{}).get('airings',0) for i in ctx.candidates] for u in ctx.targets],float);assert np.array_equal(success,s['successes'])
    globalcount=Counter();warm=0
    for u,h in ctx.history.items():
        if h:warm+=1
        globalcount.update({i:r['airings'] for i,r in h.items()})
    opportunities=sum(opp['train'].values());pop=sum(globalcount.values())/(warm*opportunities)
    itemrate=np.array([(globalcount[i]+50*pop)/(warm*train[j]+50) for j,i in enumerate(ctx.candidates)])
    rate=np.array([sum(r['airings'] for r in ctx.history.get(u,{}).values())/opportunities for u in ctx.targets])
    prior=np.clip(rate[:,None]*itemrate[None,:]/pop,1e-9,1-1e-9)
    np.testing.assert_array_equal(s['prior'],prior);assert np.all(success<=train) and np.all(success>=0)
    alpha=success+50*prior;beta=train-success+50*(1-prior)
    reconstructed=(future*alpha/(alpha+beta)).astype(np.float32);assert reconstructed.tobytes()==means.tobytes()
    cache=ROOT/load(run/'schedule-cache-reference.json')['path'];cm=load(cache/'manifest.json');assert sha(cache/'manifest.json')==v['schedule_manifest_sha256']
    for name,h in cm['cache_data_sha256'].items():assert sha(cache/name)==h
    with np.load(cache/'future-airings.npz') as z:
        n=np.bincount(z['item'],minlength=m);dur=np.bincount(z['item'],weights=z['stop']-z['start'],minlength=m)/np.maximum(n,1)
    assert np.array_equal(n,future) and np.array_equal(dur,s['duration'])
    # Every denominator divides 60: exact integer marginal utilities avoid float ties.
    tie_deviations=[];maxgap=0;validcount=0
    for j,sim in enumerate(sets):
        assert ((sim>=-1)&(sim<m)).all()
        for row in sim:assert len(set(row[row>=0]))==len(row[row>=0])
        valid=sim>=0;den=np.maximum(valid.sum(1),1);sample_rows=np.broadcast_to(np.arange(len(sim))[:,None],sim.shape)[valid];ids=sim[valid]
        units=(60//den).astype(np.int64);marg=np.zeros(m,np.int64);np.add.at(marg,ids,units[sample_rows])
        hits=np.zeros(len(sim),np.int64);selected=[];actual=np.argsort(-greedy[j])[:5]
        for rank,choice in enumerate(actual,1):
            utility=np.zeros(m,np.int64);np.add.at(utility,ids,units[sample_rows]*(hits[sample_rows]+1));utility[selected]=-1
            assert utility[choice]==utility.max(),(split,j,rank,int(utility.max()-utility[choice]))
            canonical=int(np.argmax(utility))
            if choice!=canonical:tie_deviations.append([j,rank,int(choice),canonical])
            selected.append(int(choice));hits+=np.any(sim==choice,axis=1)
        expected=(marg/(60*len(sim))).astype(np.float32)
        for rank,i in enumerate(selected):expected[i]=6-rank
        np.testing.assert_array_equal(expected,greedy[j]);validcount+=int(valid.sum())
    # Deterministic subset MC samples: direct full lexsort, not production argpartition.
    sample_users=sorted(set([0,1,2,len(ctx.targets)//2,len(ctx.targets)-1]));numeric=np.array(list(map(int,ctx.candidates)))
    for j in sample_users:
        rng=np.random.default_rng(np.random.SeedSequence([v['seed'],j]));prob=rng.beta(alpha[j],beta[j],size=(v['samples'],m));counts=rng.binomial(future[None,:],prob)
        for k,c in enumerate(counts):
            order=np.lexsort((numeric,-dur,-c))[:5];expected=np.where(c[order]>0,order,-1);assert np.array_equal(expected,sets[j,k])
    # Read-only arithmetic replay of first three rows, same original source and schedule.
    # No schedule fit: frozen schedule reference is supplied; all outputs go to audit folder.
    import advanced_posterior_predictive as source
    proxy=copy.copy(ctx);proxy.targets=ctx.targets[:3];proxy.truth=Poison();proxy.display=Poison();path=OUT/(split+'-poison-replay');path.mkdir(exist_ok=True)
    shutil.copyfile(run/'schedule-cache-reference.json',path/'schedule-cache-reference.json')
    with patch.object(source.coarse,'schedule_joint',return_value=None):source.posterior_predictive(proxy,{'samples':v['samples'],'seed':v['seed']},path)
    assert np.array_equal(np.load(path/'sampled-top5.npy'),sets[:3])
    assert np.array_equal(np.load(path/'mean-scores.npy'),means[:3]) and np.array_equal(np.load(path/'greedy-scores.npy'),greedy[:3])
    result={'split':split,'shape':list(means.shape),'posterior_arrays_all_exact':True,'all_greedy_positions_integer_exact_maximum':len(ctx.targets)*5,'numeric_tie_deviations_count':len(tie_deviations),'numeric_tie_deviations_first20':tie_deviations[:20],'all_greedy_scores_exact':True,'sampled_sets_valid_indices':validcount,'deterministic_fullsort_mc_users':sample_users,'poison_source_replay_rows':3,'mean_output':validate_output(ctx,run,means),'greedy_output':validate_output(ctx,run.parent/'BP-greedyMAP256',greedy),'manifest_sha256':sha(run/'posterior-manifest.json'),'script_sha256':sha(Path(__file__))}
    (OUT/(split+'-evidence.json')).write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
if __name__=='__main__':
    for split in sys.argv[1:] or ['main','early']:audit(split)
