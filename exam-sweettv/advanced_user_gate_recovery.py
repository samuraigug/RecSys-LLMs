"""Recover completed gate fit after a failed score write, without refitting."""
from pathlib import Path
import shutil
import numpy as np
from catboost import CatBoostRegressor
import advanced_user_gate as gate
from advanced_full_fusion import prepare_scores,max_fusion
from experiments import finalize_scores
from repro import load,dump,sha256


def recover(ctx,p,path):
    origin=path.parent/p['source_run'];record=load(origin/'record.json')
    assert record['status']=='FAILED' and record['protocol_sha256']==ctx.protocol_hash
    provenance=load(origin/'gate-training-provenance.json')
    assert provenance['source_sha256']==sha256(gate.__file__)
    assert not provenance['outer_truth_used'] and not provenance['supervised_heads_in_historical_targets']
    for source in provenance['sources']:
        assert source['label_end']<=ctx.protocol['train_cutoff_exclusive']
        assert sha256(origin/f"{source['fold']}-gate-data.npz")==source['gate_data_sha256']
    model=CatBoostRegressor();model.load_model(str(origin/'gate.cbm'))
    cache,_=prepare_scores(ctx)
    ppr=np.load(cache/'ppr.npy');hourly=np.load(cache/'hourly.npy')
    experts=[max_fusion([(ppr,1.),(hourly,1.)]),ppr,hourly]
    predictions=[finalize_scores(s.copy(),ctx)[0] for s in experts]
    features,names=gate.gate_features(ctx,experts,predictions)
    assert names==provenance['features']
    predicted=model.predict(features);scores,choice=gate.choose_rows(experts,predicted)
    np.save(path/'gate-scores.npy',scores,allow_pickle=False)
    np.savez_compressed(path/'gate-evaluation-state.npz',features=features,predicted_ap=predicted,choice=choice)
    shutil.copy2(origin/'gate.cbm',path/'gate.cbm')
    dump(path/'gate-training-provenance.json',provenance)
    details=dict(expert_cache=str(cache),expert_manifest_sha256=sha256(cache/'manifest.json'),
        choice_counts={k:int(np.sum(choice==j)) for j,k in enumerate(gate.EXPERTS)},
        evaluation_truth_used_for_gate=False,recovered_without_refit=True,origin_run=p['source_run'],
        origin_record_sha256=sha256(origin/'record.json'),origin_model_sha256=sha256(origin/'gate.cbm'),
        origin_training_provenance_sha256=sha256(origin/'gate-training-provenance.json'),
        source_sha256=sha256(__file__),training_rows=sum(s['rows'] for s in provenance['sources']),gate_features=len(names))
    dump(path/'gate-manifest.json',{**details,'protocol_sha256':ctx.protocol_hash,'scores_sha256':sha256(path/'gate-scores.npy')})
    return scores,details


def register():
    from models import METHODS
    METHODS['user_gate_recovery']=recover
