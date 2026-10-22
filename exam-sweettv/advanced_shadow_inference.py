"""Separate preregistered disjoint-user inference; no model fitting."""
import importlib.util
from pathlib import Path
import shutil
import numpy as np
from catboost import CatBoostClassifier
from threadpoolctl import threadpool_limits
from repro import ROOT, dump, load, sha256, now
from models import Context
import advanced_full_fusion as ff
from tools.shadow_user_audit import BASE, RAW, PROCESSED, RUNS

class Forbidden:
    def __getitem__(self,key): raise AssertionError('Evaluation outcome access forbidden')
    def __iter__(self): raise AssertionError('Evaluation outcome access forbidden')
    def get(self,*args): raise AssertionError('Evaluation outcome access forbidden')

def validate(ctx, reg):
    old=load(ROOT/'data/processed/elapsed-v2/main/protocol.json')
    assert ctx.targets==reg['cohort'] and len(set(ctx.targets))==3620
    for key in ('train_start','train_cutoff_exclusive','forecast_end_exclusive','cutoff_elapsed_seconds','clock','labels','repeats_allowed','cold_gate','fallback'):
        assert ctx.protocol[key]==old[key],key
    assert {p.name:sha256(p) for p in RAW.glob('*.csv')}==reg['input_hashes']==ctx.protocol['input_hashes']
    for name,value in old['input_hashes'].items():
        if name!='submission.csv': assert value==reg['input_hashes'][name]
    checks={}
    original=ROOT/'data/processed/elapsed-v2/main'
    for name in ('candidates.txt','fallback.json','metadata-train.json','opportunities.json','opportunity-slots.npz'):
        assert sha256(ctx.path/name)==sha256(original/name),name
        checks[name]=sha256(ctx.path/name)
    original_history=load(original/'histories.json')
    assert {u:h for u,h in ctx.history.items() if h}=={u:h for u,h in original_history.items() if h}
    with np.load(ctx.path/'train-events.npz') as a,np.load(original/'train-events.npz') as b:
        assert a.files==b.files
        for key in a.files: np.testing.assert_array_equal(a[key],b[key])
    assert len(ctx.candidates)==6384
    assert sha256(ff.ENGINE_PATH)==reg['feature_engine_sha256']==ff.ENGINE_SHA256
    for name,expected in ff.DEPENDENCIES.items(): assert sha256(ROOT/name)==expected,name
    for run,files in reg['inference_models_sha256'].items():
        for name,value in files.items(): assert sha256(RUNS/run/name)==value
        record=load(RUNS/run/'record.json')
        assert record['input_hashes']==old['input_hashes']
        assert record['protocol_sha256']==sha256(original/'protocol.json')
    for run in ('L4-logloss-allusers300','L5-logloss-allusers-strong300'):
        p=load(RUNS/run/'ranker-training-provenance.json')
        assert p['evaluation_cutoff_exclusive']=='2020-05-18'
        assert not p['fitting_evaluation_truth_accessed']
        assert p['all_label_end_dates_at_or_before_evaluation_cutoff']
    ds=load(RUNS/'DS-nonrepeat-binary300/training-provenance.json')
    assert not ds['outer_truth_used'] and max(s['label_end'] for s in ds['sources'])<='2020-05-18'
    return dict(global_nonempty_history_exact=True,all_train_event_arrays_exact=True,identical_artifacts=checks,all_five_bundle_hashes_verified=True,model_training_cutoffs_verified=True)

def main():
    reg=load(BASE/'preregistration.json'); out=BASE/'inference';out.mkdir(exist_ok=False)
    ctx=Context(PROCESSED/'main');ctx.raw_input_dir=RAW
    checks=validate(ctx,reg)
    ctx.truth=Forbidden();ctx.display=Forbidden();ctx.cohorts=Forbidden()
    snapshots=BASE/'source_snapshots';snapshots.mkdir(exist_ok=True)
    sources={}
    for p in [ROOT/n for n in ('advanced_shadow_inference.py','tools/shadow_user_audit.py','etl.py','models.py','experiments.py','advanced_full_fusion.py','advanced_collaborative.py','advanced_hourly_schedule.py','advanced_schedule.py')]+[ff.ENGINE_PATH]:
        shutil.copy2(p,snapshots/p.name);sources[p.name]=sha256(p)
    dump(out/'validation-before-scoring.json',checks)
    import advanced_collaborative as ac
    import advanced_hourly_schedule as ah
    values={};details={}
    with threadpool_limits(limits=1):
        for key,fn,params in [('ppr',ac.ppr,ff.PPR),('hourly',ah.schedule_hourly,ff.HOURLY)]:
            folder=out/key;folder.mkdir()
            values[key],details[key]=fn(ctx,params,folder)
            values[key][ctx.cold_mask]=np.nan
            np.save(out/(key+'.npy'),values[key]);print('scored',key,flush=True)
        values['core']=ff.max_fusion([(values.pop('ppr'),1),(values.pop('hourly'),1)])
        np.save(out/'core.npy',values['core'])
        spec=importlib.util.spec_from_file_location('frozen_shadow_engine',ff.ENGINE_PATH)
        m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
        engine=m.FeatureEngine(ctx,{'strong_context':False})
        for key,run in [('l4','L4-logloss-allusers300'),('l5','L5-logloss-allusers-strong300')]:
            p=load(RUNS/run/'record.json')['parameters']
            if key=='l5':
                p={**p,'training_raw_dir':str(RAW)}
                engine._strong_context(p,out/'strong-context')
            model=CatBoostClassifier();model.load_model(str(RUNS/run/'catboost-ranker.cbm'))
            assert model.feature_names_==engine.names
            values[key]=np.empty((len(ctx.targets),len(ctx.candidates)),np.float32)
            for j in range(len(ctx.targets)):
                values[key][j]=model.predict_proba(engine.matrix(j),thread_count=1)[:,1]
            np.save(out/(key+'.npy'),values[key]);print('scored',key,flush=True)
        model=CatBoostClassifier();model.load_model(str(RUNS/'DS-nonrepeat-binary300/discovery.cbm'))
        assert model.feature_names_==engine.names
        ds=values.pop('l5');mask=engine.count==0
        for j in range(len(ctx.targets)):
            ids=np.flatnonzero(mask[j])
            if len(ids):ds[j,ids]=model.predict_proba(engine.matrix(j,ids),thread_count=1)[:,1]
        values['ds']=ds
        np.save(out/'ds.npy',ds);np.save(out/'nonrepeat-mask.npy',mask)
        del engine,mask
        numeric=np.array([int(x) for x in ctx.candidates],np.int64)
        from experiments import finalize_scores
        for name in reg['models']:
            for v in values.values():v[ctx.cold_mask]=np.nan
            scores=ff.rrf([(values[key],weight) for key,weight in reg['configs'][name]],numeric,60)
            folder=BASE/name;folder.mkdir()
            pred,diagnostics,indices,native=finalize_scores(scores,ctx)
            np.save(folder/'scores.npy',scores)
            np.savez_compressed(folder/'native-top100.npz',indices=indices,scores=native)
            dump(folder/'top5.json',pred);dump(folder/'fallback-diagnostics.json',diagnostics)
    dump(out/'mapping.json',{'targets':ctx.targets,'candidates':ctx.candidates})
    dump(out/'provenance.json',dict(completed_at=now(),preregistration_sha256=sha256(BASE/'preregistration.json'),
        protocol_sha256=ctx.protocol_hash,checks=checks,source_sha256=sources,models=reg['inference_models_sha256'],
        poisoned_truth_display_cohorts=True,fit_performed=False,full_fusion_prepare_scores_called=False,
        all_five_raw_hashes=ctx.protocol['input_hashes'],component_details=details,
        payload_sha256={str(p.relative_to(BASE)):sha256(p) for p in BASE.rglob('*') if p.is_file() and p.suffix in ('.npy','.npz') and 'processed' not in p.parts}))
    print('FROZEN INFERENCE COMPLETE',flush=True)
if __name__=='__main__':main()
