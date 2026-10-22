"""Independent saved-model/native-score reconstruction, including regression links."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from threadpoolctl import threadpool_limits
from catboost import CatBoostClassifier,CatBoostRanker,CatBoostRegressor
from advanced_ranker import FeatureEngine
from models import Context
from repro import ROOT,load,dump,sha256
limits=threadpool_limits(limits=1)

names=('L0-yetimap500','L1-logloss500','L2-logloss-long500','L3-logloss-weighted-long500',
       'L4-control-logloss300','L4-logloss-allusers300','L5-logloss-allusers-strong300',
       'L6-poisson-allusers-strong300','L7-logloss-strong-hardneg300')
results=[]
for split in ('main','early'):
 ctx=Context(ROOT/'data/processed/elapsed-v2'/split)
 for strong in (False,True):
  family=[]
  for name in names:
   run=ROOT/'artifacts/runs'/split/name
   record=load(run/'record.json')
   assert record['status']=='SUCCEEDED'
   if bool(record['parameters'].get('strong_context'))==strong:
    family.append((run,record))
  parameters=family[0][1]['parameters']
  features=FeatureEngine(ctx,parameters,ROOT/'artifacts/ranker-research/reconstruction-v7'/split/str(strong))
  for run,record in family:
   objective=record['parameters'].get('objective','YetiRank:mode=MAP;top=5')
   klass=CatBoostClassifier if objective=='Logloss' else (CatBoostRegressor if objective in ('Poisson','RMSE') else CatBoostRanker)
   model=klass();model.load_model(str(run/'catboost-ranker.cbm'))
   assert model.feature_names_==features.names
   provenance=load(run/'ranker-training-provenance.json')
   assert provenance['evaluation_cutoff_exclusive']==ctx.protocol['train_cutoff_exclusive']
   assert all(s['labels_end_exclusive']<=ctx.protocol['train_cutoff_exclusive'] and
              s['label_last_observation_elapsed_seconds']<=ctx.protocol['cutoff_elapsed_seconds']
              for s in provenance['sources'])
   with np.load(run/'native-top100.npz') as z:ids,scores=z['indices'],z['scores']
   checked,error=0,0.
   for j in range(len(ctx.targets)):
    mask=ids[j]>=0
    if not mask.any():continue
    xx=features.matrix(j,ids[j,mask])
    if objective=='Logloss':restored=model.predict_proba(xx,thread_count=1)[:,1]
    elif objective=='Poisson':restored=model.predict(xx,prediction_type='Exponent',thread_count=1)
    else:
     restored=model.predict(xx,thread_count=1)
     if record['parameters'].get('target_transform')=='log1p':restored=np.maximum(np.expm1(restored),0)
    expected=scores[j,mask]
    error=max(error,float(np.max(np.abs(restored-expected))))
    assert np.allclose(restored,expected,atol=1e-6,rtol=1e-6),(split,run.name,j,error)
    checked+=len(restored)
   results.append({'split':split,'run_id':run.name,'status':'PASS','feature_count':len(features.names),
                   'native_scores_checked':checked,'max_absolute_float32_rounding_error':error,
                   'saved_model_sha256':sha256(run/'catboost-ranker.cbm'),
                   'chronological_label_audit':'all pseudo-future labels end at or before evaluated cutoff'})
  del features
out=ROOT/'artifacts/ranker-research/saved-model-score-reconstruction-through-L7.json'
dump(out,{'status':'PASS','audit_source_sha256':sha256(__file__),'scores_checked':sum(x['native_scores_checked'] for x in results),'results':results})
print({'status':'PASS','models_checked':len(results),'scores_checked':sum(x['native_scores_checked'] for x in results)})
