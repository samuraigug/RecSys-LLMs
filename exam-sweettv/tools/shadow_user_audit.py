#!/usr/bin/env python3
"""Preregister and evaluate disjoint-user audit. AUDIT_ONLY_NOT_FOR_KAGGLE."""
import sys, csv, json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from repro import ROOT, dump, load, sha256, now
BASE=ROOT/'research/shadow-user-audit-20261022'
RAW=BASE/'raw'; PROCESSED=BASE/'processed'
RUNS=ROOT/'artifacts/runs/main'

def preregister():
    assert not (BASE/'preregistration.json').exists()
    original=[r['user_id'] for r in csv.DictReader((ROOT/'data/raw/submission.csv').open())]
    # Filter on start alone before reading any outcome-related values. The file
    # is scanned physically, but post-cutoff stop/duration values are never used.
    prefix=set()
    for r in csv.DictReader((ROOT/'data/raw/dataset11-30.csv').open()):
        if '2020-03-09' <= r['start_time'] < '2020-05-18': prefix.add(r['user_id'])
    cohort=sorted(prefix-set(original),key=int)
    assert len(original)==1260 and not set(cohort)&set(original)
    RAW.mkdir(parents=True)
    for p in (ROOT/'data/raw').glob('*.csv'):
        if p.name!='submission.csv': (RAW/p.name).symlink_to(p.resolve())
    with (RAW/'submission.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['user_id','tv_show_id']);w.writerows((u,'0 0 0 0 0') for u in cohort)
    (BASE/'AUDIT_ONLY_NOT_FOR_KAGGLE').write_text('Separate user audit; not official holdout and not a Kaggle submission.\n')
    models={}
    for name, files in {
       'L4-logloss-allusers300':['catboost-ranker.cbm','ranker-training-provenance.json','record.json'],
       'L5-logloss-allusers-strong300':['catboost-ranker.cbm','ranker-training-provenance.json','record.json'],
       'DS-nonrepeat-binary300':['discovery.cbm','training-provenance.json','record.json']}.items():
        models[name]={f:sha256(RUNS/name/f) for f in files}
    import advanced_full_fusion as ff
    registration=dict(status='PREREGISTERED_BEFORE_SHADOW_LABELS',created_at=now(),cohort_rule='raw users with start in [2020-03-09,2020-05-18) minus original 1260 template IDs; ordered numeric string IDs',
      prefix_users=len(prefix),original_users=1260,cohort_users=len(cohort),cohort=cohort,
      models=['FULL-shfl4','SHFL4DS-rrf-equal'],
      omitted_F14='Frozen C4 has only original target profiles and candidate vectors; it lacks persisted fitted title IDF/transformer and all historical item vectors. Frozen-only inference would require reconstruction/refitting of content transforms. Narrowing to two strong fixed models before labels, avoiding a new baseline implementation with different semantics.',
      configs={'ppr':ff.PPR,'hourly':ff.HOURLY,'core':'sum of per-user max-normalized PPR and hourly, float32','rrf_k':60,'FULL-shfl4':[['core',1],['l4',.5]],'SHFL4DS-rrf-equal':[['core',1],['l4',.25],['ds',.25]]},
      train_cutoff='2020-05-18',forecast_end='2020-07-27',candidates=6384,truth='union clipping; >=80% original airing elapsed duration; top5 count/seconds/ID',clock='Europe/Kyiv',repeats_allowed=True,
      cold_gate='no full training history -> shared train distinct-user popularity fallback',primary='MAP@5 nonempty future truth',secondary='all preregistered users, empty truth AP=0',
      inference_models_sha256=models,feature_engine_sha256=ff.ENGINE_SHA256,input_hashes={p.name:sha256(p) for p in RAW.glob('*.csv')},
      limitations='Disjoint evaluation users, NOT an independent dataset: their prefix data already entered global training and temporal outcomes are correlated. Not official holdout, not evidence for 0.95. No tuning, no future-label fitting, no outcomes passed to lead for choosing Bayesian configs.')
    dump(BASE/'preregistration.json',registration)
    print(json.dumps({k:registration[k] for k in ('created_at','prefix_users','cohort_users','models','omitted_F14')},ensure_ascii=False))
    print('preregistration_sha256',sha256(BASE/'preregistration.json'))

def evaluate():
    import numpy as np
    from fractions import Fraction
    from models import Context
    from protocol import ap_at_k
    reg=load(BASE/'preregistration.json')
    provenance=load(BASE/'inference/provenance.json')
    assert provenance['preregistration_sha256']==sha256(BASE/'preregistration.json')
    ctx=Context(PROCESSED/'main')
    assert ctx.targets==reg['cohort']
    candidates=set(ctx.candidates);numeric=np.array([int(i) for i in ctx.candidates])
    results={}
    for name in reg['models']:
        folder=BASE/name;pred=load(folder/'top5.json');diag=load(folder/'fallback-diagnostics.json')
        assert list(pred)==ctx.targets and len(pred)==len(ctx.targets)
        scores=np.load(folder/'scores.npy',mmap_mode='r')
        native=np.load(folder/'native-top100.npz');aps=[];nonempty=[];rows=[]
        for j,u in enumerate(ctx.targets):
            row=pred[u];truth=ctx.truth[u]
            assert isinstance(u,str) and all(isinstance(i,str) for i in row)
            assert len(row)==len(set(row))==5 and set(row)<=candidates and '0' not in row
            assert len(truth)==len(set(truth))<=5 and set(truth)<=candidates
            hits=0;total=Fraction(0)
            for rank,i in enumerate(row,1):
                if i in truth:hits+=1;total+=Fraction(hits,rank)
            exact=total/min(5,len(truth)) if truth else Fraction(0)
            assert abs(float(exact)-ap_at_k(truth,row))<1e-14
            aps.append(float(exact))
            if truth:nonempty.append(float(exact))
            finite=np.flatnonzero(np.isfinite(scores[j]))
            order=finite[np.lexsort((numeric[finite],-scores[j,finite]))][:100]
            expected=np.full(100,-1,dtype=np.int32);expected[:len(order)]=order
            assert np.array_equal(expected,native['indices'][j])
            assert np.array_equal(scores[j,order],native['scores'][j,:len(order)])
            expected_top=[ctx.candidates[k] for k in order[:5]]
            for i in ctx.fallback:
                if len(expected_top)==5:break
                if i not in expected_top:expected_top.append(i)
            assert row==expected_top
            if ctx.cold_mask[j]:
                assert row==ctx.fallback[:5] and not len(finite) and diag[u]['cold_gate']
            rows.append([u,str(exact),float(exact),len(truth),bool(ctx.cold_mask[j])])
        with (folder/'per-user-ap.csv').open('w',newline='') as f:
            w=csv.writer(f);w.writerow(['user_id','ap_fraction','ap_at_5','truth_items','cold']);w.writerows(rows)
        results[name]=dict(primary_nonempty_truth_users=len(nonempty),primary_map_at_5=float(np.mean(nonempty)),
            all_cohort_users=len(aps),all_cohort_map_at_5=float(np.mean(aps)),cold_users=int(ctx.cold_mask.sum()),
            native_schema_candidate_fraction_cold_checks='PASS',top5_sha256=sha256(folder/'top5.json'))
    (BASE/'cohort-ids.txt').write_text('\n'.join(ctx.targets)+'\n')
    summary=dict(status='PASS',completed_at=now(),results=results,cohort_ids_sha256=sha256(BASE/'cohort-ids.txt'),
       preregistration_sha256=sha256(BASE/'preregistration.json'),inference_provenance_sha256=sha256(BASE/'inference/provenance.json'),
       evaluation_truth_first_scoring_access='only after all frozen predictions and provenance saved',
       original_protocol_sha256=sha256(ROOT/'data/processed/elapsed-v2/main/protocol.json'))
    dump(BASE/'verification.json',summary)
    lines=['# Дополнительный аудит на непересекающихся пользователях','',
      'Аудит выполнен после широкого выбора примерно из 490 прогонов на исходных 1260 пользователях. Это дополнительная проверка обобщения по пользователям, а не независимый dataset, официальный holdout или достижение цели 0.95. Исходы не используются для новых настроек моделей.', '',
      'Когорта заранее зафиксирована: 3620 строковых user ID из 4876 пользователей raw trainprefix с началом сессии в [09.03.2020, 18.05.2020), за исключением исходных 1260 template ID. Ни наличие будущего просмотра, ни будущие метки не использовались при выборе. Четыре original CSV подключены symlink; отдельный submission.csv содержит только эту когорту. AUDIT_ONLY_NOT_FOR_KAGGLE.', '',
      'Пререгистрация: '+summary['preregistration_sha256']+'. SHA списка ID: '+summary['cohort_ids_sha256']+'.', '',
      'До открытия будущих меток закреплены FULL-shfl4 и SHFL4DS-rrf-equal. F14 заранее исключён: замороженный C4 не содержит полного fitted IDF/transformer и всех historical item vectors, необходимых для нового user profile; новый fit не выполнялся. Причина зафиксирована в preregistration.json до ETL.', '',
      '| Модель | MAP@5, непустая truth | MAP@5, вся когорта | Непустая truth / все |','|---|---:|---:|---:|']
    for name,r in results.items():lines.append(f"| {name} | {r['primary_map_at_5']:.10f} | {r['all_cohort_map_at_5']:.10f} | {r['primary_nonempty_truth_users']} / {r['all_cohort_users']} |")
    lines += ['',
      'Основная метрика исключает пустую будущую truth; дополнительная назначает ей AP=0. Обе объявлены заранее. Сравнивать абсолютное значение с исходными 1260 напрямую нельзя: это другой состав пользователей.', '',
      'Протокол: train до 18 мая; future 18 мая–27 июля; Europe/Kyiv elapsed; объединение перекрытий и усечение сессий до overlap; полный просмотр ≥80% исходной длительности эфира. Те же 6384 кандидата, повторы разрешены, отсутствие полного train history включает общий train-popularity fallback.', '',
      'Frozen inference: L4 и L5 .cbm загружены без fit; DS .cbm заменяет L5 только на nonrepeat candidates. FeatureEngine v5 SHA '+reg['feature_engine_sha256']+'. PPR и hourly реконструированы детерминированно с исходными параметрами. Core — сумма max-normalized PPR/hourly; FULL-shfl4 RRF k60: core1+L4.5; SHFL4DS: core1+L4.25+DS.25. Полный каталог, без Top100 truncation при fusion.', '',
      'Отдельный advanced_shadow_inference.py не вызывает prepare_scores и не ослабляет его protocol guard. Проверены все пять raw hashes нового bundle, четыре original data hashes, cutoff, кандидаты, train metadata/opportunities/fallback, все train event arrays и вся непустая global history. В inference truth, display и cohorts заменены объектами, вызывающими исключение при чтении. Model hashes/provenance и source snapshots сохранены.', '',
      'Независимая проверка сохранённых результатов: AP через fractions.Fraction для каждого пользователя, schema, string IDs, допустимость и уникальность Top5, cold fallback, полный native порядок и scores — PASS. Это программная проверка агента, не личный расчёт студента.', '',
      'Ограничения: trainprefix этих пользователей уже входил в обучение глобальных моделей и исторических ranker folds; временные исходы коррелируют. Когорта не является случайной выборкой населения. Этот аудит снижает риск объяснения результата исключительно подбором по тем же 1260 users, но не устраняет reuse данных и не даёт независимой оценки после выбора. Никаких параметров по этим меткам не подбирали. Исходы не передавались lead для выбора Bayesian конфигураций.', '',
      'Артефакты: preregistration.json; raw/; processed/main/ с сохранёнными ETL hashes; inference/provenance.json; source_snapshots/; отдельные Top5/native/full scores/per-user AP в каталогах двух моделей; verification.json. Original raw/template/main protocol не изменялись.']
    (BASE/'REPORT_RU.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__':
    if sys.argv[1]=='preregister': preregister()
    elif sys.argv[1]=='evaluate': evaluate()
    elif sys.argv[1]=='prepare':
        assert (BASE/'preregistration.json').exists()
        from etl import prepare
        prepare(RAW,PROCESSED,('main',),'Europe/Kyiv')
