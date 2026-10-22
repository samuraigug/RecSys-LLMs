"""Reconstruct inference features and compare saved models to every native top100 score."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from catboost import CatBoostClassifier,CatBoostRanker
from advanced_ranker import FeatureEngine
from models import Context
from repro import ROOT,load,dump,sha256

results=[]
for split in ('main','early'):
 ctx=Context(ROOT/'data/processed/elapsed-v2'/split)
 features=FeatureEngine(ctx)
 for name in ('L0-yetimap500','L1-logloss500','L2-logloss-long500','L3-logloss-weighted-long500'):
  run=ROOT/'artifacts/runs'/split/name
  if not (run/'record.json').exists():
   continue
  record=load(run/'record.json')
  objective=record['parameters'].get('objective','YetiRank:mode=MAP;top=5')
  model=(CatBoostClassifier() if objective=='Logloss' else CatBoostRanker())
  model.load_model(str(run/'catboost-ranker.cbm'))
  with np.load(run/'native-top100.npz') as z:
   ids,scores=z['indices'],z['scores']
  checked,error=0,0.
  for j in range(len(ctx.targets)):
   mask=ids[j]>=0
   if not mask.any():continue
   xx=features.matrix(j,ids[j,mask])
   restored=(model.predict_proba(xx)[:,1] if objective=='Logloss' else model.predict(xx))
   expected=scores[j,mask]
   error=max(error,float(np.max(np.abs(restored-expected))))
   assert np.allclose(restored,expected,atol=2e-7,rtol=1e-6),(split,name,j,error)
   checked+=len(restored)
  results.append({'split':split,'run_id':name,'status':'PASS','native_scores_checked':checked,'max_absolute_float32_rounding_error':error,'saved_model_sha256':sha256(run/'catboost-ranker.cbm')})
 del features
out=ROOT/'artifacts/ranker-research/saved-model-score-reconstruction.json'
dump(out,{'status':'PASS','model_type_correct_prediction_API':True,'results':results})
print(results)
