#!/usr/bin/env python3
"""Summarise retained measurements; evaluation diagnostics never feed models."""
import csv,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from repro import dump,load,sha256


def user_aps(path):return {r['user_id']:float(r['ap_at_5']) for r in csv.DictReader((path/'per-user-ap.csv').open())}

def paired(left,right,users):
    a=user_aps(left);b=user_aps(right);d=np.array([a[u]-b[u] for u in users]);rng=np.random.default_rng(42)
    values=[]
    for _ in range(50):values.extend(d[rng.integers(len(d),size=(200,len(d)))].mean(axis=1))
    return dict(delta=float(d.mean()),ci95=[float(v) for v in np.quantile(values,[.025,.975])],bootstrap=10000,seed=42)


def main():
    result={}
    for split in ('main','early'):
        root=ROOT/'artifacts/runs'/split;processed=ROOT/'data/processed/elapsed-v2'/split
        cohorts=load(processed/'cohorts.json');truth=load(processed/'truth.json');history=load(processed/'histories.json');users=cohorts['main']
        rows=[]
        for folder in sorted(root.glob('SEM-*')):
            r=load(folder/'record.json')
            if r['status']!='SUCCEEDED':continue
            rows.append(dict(run_id=folder.name,method=r['method'],parameters=r['parameters'],map_at_5=r['metrics']['map_at_5_main'],
                repeat_share_top5=r['metrics']['repeat_share_top5'],record_sha256=sha256(folder/'record.json')))
        baseline=load(root/'SHFL4-rrf-0.5'/'record.json')['metrics']['map_at_5_main']
        diagnostics={}
        for name in ('SEM-v2-weak-meta-mean','SEM-v2-weak-meta-kernel4'):
            predictions=load(root/name/'top5.json')
            hits=[(u,i) for u in users for i in predictions[u] if i in truth[u]]
            unseen=sum(i not in history.get(u,{}) for u,i in hits)
            diagnostics[name]=dict(correct_top5_hits=len(hits),correct_hits_absent_from_full_train_history=unseen,
                correct_hits_present_in_full_train_history=len(hits)-unseen,
                note='Evaluation-only decomposition; full-view history, not external-model discovery proof and not a different selection metric.')
        result[split]=dict(baseline_run='SHFL4-rrf-0.5',baseline_map=baseline,runs=rows,
            kernel_minus_matched_mean=paired(root/'SEM-v2-weak-meta-kernel4',root/'SEM-v2-weak-meta-mean',users),
            kernel_blend_minus_baseline=paired(root/'SEM-v2-core-kernel4-rrf-0.15',root/'SHFL4-rrf-0.5',users),diagnostics=diagnostics)
    dump(ROOT/'artifacts/semantic-research/summary.json',dict(status='MEASURED',target_095_achieved=False,
        source_snapshots=['research/source-snapshots/semantic-v1','research/source-snapshots/semantic-v2'],
        v1_reproducibility_caveat='Casefold ties had process-dependent capitalization; retained cache/model states reproduce native ranks exactly. New v2 has explicit string tie-break and matched mean control.',
        chosen_for_core=False,windows=result))

if __name__=='__main__':main()
