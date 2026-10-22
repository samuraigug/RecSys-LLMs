"""Read-only gate audit: historical AP reference, saved model and full routing."""
import sys,json,hashlib,copy,csv,gc
from pathlib import Path
from fractions import Fraction
import numpy as np
from scipy import sparse
from catboost import CatBoostRegressor
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));OUT=ROOT/'research/user-gate-audit-20261022'
from models import Context
from etl import read_events
from advanced_user_gate import gate_features
import advanced_hourly_schedule
load=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
class Poison:
 def __getattribute__(self,n):raise AssertionError('Forbidden truth/display access '+n)
def ap(truth,pred):
 truth=set(truth);hits=0;s=Fraction()
 for r,i in enumerate(pred,1):
  if i in truth:hits+=1;s+=Fraction(hits,r)
 return s/min(5,len(truth)) if truth else Fraction()
def finalize(ctx,scores,run=None):
 nums=np.array(list(map(int,ctx.candidates)));pred={};checked=0
 if run is not None:z=np.load(run/'native-top100.npz');top=load(run/'top5.json')
 for j,u in enumerate(ctx.targets):
  valid=np.flatnonzero(np.isfinite(scores[j])) if not ctx.cold_mask[j] else np.array([],int)
  order=valid[np.lexsort((nums[valid],-scores[j,valid]))][:100];row=[ctx.candidates[i] for i in order[:5]]
  for i in ctx.fallback:
   if len(row)==5:break
   if i not in row:row.append(i)
  pred[u]=row
  if run is not None:
   ix=np.full(100,-1,np.int32);ix[:len(order)]=order;values=np.full(100,np.nan,np.float32);values[:len(order)]=scores[j,order]
   assert np.array_equal(ix,z['indices'][j]);assert np.array_equal(values,z['scores'][j],equal_nan=True);assert row==top[u];checked+=len(order)
 if run is None:return pred
 rec=load(run/'record.json');assert rec['status']=='SUCCEEDED' and rec['protocol_sha256']==ctx.protocol_hash
 aps={u:ap(ctx.truth[u],pred[u]) for u in ctx.targets};mean=float(sum((aps[u] for u in ctx.cohorts['main']),Fraction())/len(ctx.cohorts['main']));assert abs(mean-rec['metrics']['map_at_5_main'])<1e-14
 with (run/'per-user-ap.csv').open() as f:
  rows=list(csv.DictReader(f));assert [r['user_id'] for r in rows]==ctx.targets
  for r in rows:assert abs(float(r['ap_at_5'])-float(aps[r['user_id']]))<1e-14
 return {'map_fraction':mean,'native_scores_and_ranks_exact':checked,'all_top5_cold_fallback_exact':True}
def core(a,b):
 out=np.zeros(a.shape,np.float64);touch=np.zeros(a.shape,bool)
 for source in [a,b]:
  for j,row in enumerate(source):
   ids=np.flatnonzero(np.isfinite(row));vals=np.maximum(row[ids].astype(float),0);out[j,ids]+=vals/max(vals.max(initial=0),1e-12);touch[j,ids]=True
 out[~touch]=np.nan;return out.astype(np.float32)
def audit(split):
 ctx=Context(ROOT/'data/processed/elapsed-v2'/split);origin=ROOT/'artifacts/runs'/split/'UG-AP3-depth4-300';run=origin.parent/'UG-AP3-depth4-300-recovered' if split=='main' else origin;rec=load(run/'record.json');assert rec['status']=='SUCCEEDED'
 v=load(run/'gate-training-provenance.json');manifest=load(run/'gate-manifest.json');assert v['source_sha256']==sha(ROOT/'advanced_user_gate.py');assert v['experts']==['core','ppr','hourly'];p=load(origin/'record.json')['parameters']
 if split=='main':
  assert load(origin/'record.json')['status']=='FAILED' and manifest['recovered_without_refit']
  assert sha(origin/'gate.cbm')==sha(run/'gate.cbm')==manifest['origin_model_sha256']
  assert sha(origin/'gate-training-provenance.json')==manifest['origin_training_provenance_sha256']
  assert load(origin/'gate-training-provenance.json')==v and sha(origin/'record.json')==manifest['origin_record_sha256']
 assert v['config']['depth']==4 and v['config']['iterations']==300 and v['config']['loss_function']=='MultiRMSE'
 model=CatBoostRegressor();model.load_model(str(run/'gate.cbm'));config=model.get_all_params()
 for k in ['iterations','depth','loss_function','random_seed']:assert config[k]==v['config'][k]
 folds=[]
 for source in v['sources']:
  name=source['fold'];train=Context(ROOT/p['fold_root']/name);src=origin/'expert-sources'/name;datafile=origin/(name+'-gate-data.npz')
  assert sha(datafile)==source['gate_data_sha256'] and train.protocol_hash==source['protocol_sha256'] and sha(train.path/'truth.json')==source['truth_sha256']
  assert source['label_end']==train.protocol['forecast_end_exclusive']<=ctx.protocol['train_cutoff_exclusive'];_,_,_,events=read_events(train.path/'validation-events.npz');assert events[:,6].max()<=ctx.protocol['cutoff_elapsed_seconds']
  assert source['raw_hashes']==train.protocol['input_hashes']
  for fn,h in source['raw_hashes'].items():assert sha(ROOT/p['training_raw_dir']/fn)==h
  graph=sparse.load_npz(src/'ppr/ppr-transition.npz');tele=sparse.load_npz(src/'ppr/ppr-teleport.npz').toarray();assert load(src/'ppr/item-mapping.json')==train.items
  state=tele.copy();dangling=np.asarray(graph.sum(1)).ravel()==0;a=v['ppr_parameters']['alpha']
  for _ in range(source['expert_details']['ppr']['iterations']):state=a*(state@graph+state[:,dangling].sum(1,keepdims=True)*tele)+(1-a)*tele
  opp=load(train.path/'opportunities.json');ppr=np.full((len(train.targets),len(train.candidates)),np.nan,np.float32)
  for k,i in enumerate(train.candidates):
   if i in train.item_map:ppr[:,k]=state[:,train.item_map[i]]*opp['future'].get(i,0)/(opp['train'].get(i,0)+50)
  ppr[ppr<=0]=np.nan
  ref=load(src/'hourly/hourly-cache-reference.json');cache=Path(ref['path']);cm=load(cache/'manifest.json');assert sha(cache/'manifest.json')==ref['manifest_sha256'];assert cm['max_training_stop']<=train.protocol['cutoff_elapsed_seconds']
  folder=OUT/'historical-hourly-replay'/name;folder.mkdir(parents=True,exist_ok=True)
  # Existing immutable cache only; call cannot build a cache because verified path exists.
  h,_=advanced_hourly_schedule.schedule_hourly(train,v['hourly_parameters'],folder)
  experts=[core(ppr,h),ppr,h];pred=[finalize(train,s) for s in experts]
  keep=np.array([bool(train.truth[u]) and not train.cold_mask[j] for j,u in enumerate(train.targets)])
  y=np.array([[float(ap(train.truth[u],pr[u])) for pr in pred] for j,u in enumerate(train.targets) if keep[j]],np.float32)
  saved=np.load(datafile);assert saved['users'].tolist()==np.asarray(train.targets)[keep].tolist();np.testing.assert_array_equal(saved['y'],y)
  proxy=copy.copy(train);proxy.truth=Poison();proxy.display=Poison();x,names=gate_features(proxy,experts,pred)
  assert names==v['features'];np.testing.assert_array_equal(x[keep],saved['x'])
  folds.append({'fold':name,'rows':len(y),'all_saved_y_fraction_exact':True,'all_saved_x_poison_exact':True,'users_order_exact':True,'temporal_sources_valid':True})
  del train,experts,pred,x,y,saved,state,tele,h,ppr;gc.collect()
 cache=Path(manifest['expert_cache']);assert sha(cache/'manifest.json')==manifest['expert_manifest_sha256'];cm=load(cache/'manifest.json')
 for name in ['ppr.npy','hourly.npy','l4.npy']:assert sha(cache/name)==cm['payload_sha256'][name]
 ppr=np.load(cache/'ppr.npy');h=np.load(cache/'hourly.npy');experts=[core(ppr,h),ppr,h];pred=[finalize(ctx,s) for s in experts]
 proxy=copy.copy(ctx);proxy.truth=Poison();proxy.display=Poison();x,names=gate_features(proxy,experts,pred);assert names==v['features'];saved=np.load(run/'gate-evaluation-state.npz');np.testing.assert_array_equal(x,saved['features'])
 predicted=model.predict(x,thread_count=1);np.testing.assert_array_equal(predicted,saved['predicted_ap']);choices=np.argmax(predicted,axis=1);np.testing.assert_array_equal(choices,saved['choice'])
 scores=np.vstack([experts[k][j] for j,k in enumerate(choices)]);assert sha(run/'gate-scores.npy')==manifest['scores_sha256'];assert scores.tobytes()==np.load(run/'gate-scores.npy').tobytes()
 maincheck=finalize(ctx,scores,run)
 # Full-rank fusion, not truncated native Top100.
 l4=np.load(cache/'l4.npy');nums=np.array(list(map(int,ctx.candidates)));fused=np.zeros(scores.shape,float);touch=np.zeros(scores.shape,bool)
 for source,weight in [(scores,1),(l4,.5)]:
  for j,row in enumerate(source):
   ids=np.flatnonzero(np.isfinite(row));order=ids[np.lexsort((nums[ids],-row[ids]))];fused[j,order]+=weight/(60+np.arange(1,len(order)+1));touch[j,order]=True
 fused[~touch]=np.nan;bundlecheck=finalize(ctx,fused.astype(np.float32),run.parent/'UG-FULL-shfl4-replacement')
 result={'split':split,'folds':folds,'training_rows':sum(f['rows'] for f in folds),'features':len(names),'eval_feature_rows_poison_exact':len(ctx.targets),'saved_model_all_predictions_exact':list(predicted.shape),'all_choices_and_full_score_rows_bit_exact':True,'choice_counts':{key:int(np.sum(choices==k)) for k,key in enumerate(v['experts'])},'gate':maincheck,'bundle':bundlecheck,'model_sha256':sha(run/'gate.cbm'),'script_sha256':sha(Path(__file__))}
 (OUT/(split+'-evidence.json')).write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
if __name__=='__main__':
 with threadpool_limits(limits=1):
  for split in sys.argv[1:] or ['main','early']:audit(split)
