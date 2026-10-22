# Полный каталог вместо усечения native Top-100 — 22.10.2026

## Результат

Усечение снято на всех уровнях двух заранее заданных смесей. На main MAP@5 не изменился. На early SHFL4 улучшился на 0.00010620915; SHFL45 не изменился. Это небольшой локальный результат controlled ablation, а не доказательство устойчивого прироста или достижения 0.95.

| Фиксированная смесь | Старый main | Полный main | Старый early | Полный early |
|---|---:|---:|---:|---:|
| SHFL4: core1 + L4.5 | 0.3852638664512655 | **0.3852638664512655** | 0.4056107026143790 | **0.4057169117647059** |
| SHFL45: core1 + L4.25 + L5.25 | 0.3856616855142703 | **0.3856616855142703** | 0.4046840958605665 | **0.4046840958605665** |

Для каждой смеси ровно три пользователя из 1260 получили другой Top-5. На main AP этих изменений в сумме не изменился. На early SHFL4 суммарный AP изменился на +0.13 по main cohort из 1224 пользователей. В обоих окнах native Top-100 изменился у **каждого warm пользователя**: 1242 main / 1233 early. Следовательно, полный хвост действительно участвовал в расчёте; прежний MAP не является случайным повторным чтением старой выдачи.

Изначальный ограниченный oracle из более раннего исследования был оценкой другого набора retrievers. Он не переносится автоматически на эти смеси. Нынешняя проверка показывает, что снятие Top-100 ограничения само по себе почти не изменяет верхние пять рангов.

## Зафиксированный эксперимент

Ни одна модель не переобучалась. Сетка весов не запускалась. Сохранены исходные смеси:

- core = PPR и hourly, оба с весом 1, каждый положительный native score делится на максимум своего пользователя;
- SHFL4 = weighted RRF, k0=60, core1 + L4.5;
- SHFL45 = weighted RRF, k0=60, core1 + L4.25 + L5.25.

PPR: count, weak_weight1, half_life56, alpha0.2, neighbors200, opportunity adjustment. Hourly: joint_weight0.5, neighbor_smooth0.25, half_life28. Эти параметры проверяются против исходных records.

Каталог не менялся: **6384 main / 6449 early**. Cohort main1238 / early1224; в обоих окнах всего1260. Полный native score у каждого warm пользователя содержит весь каталог благодаря L4/L5. Общий cold gate и исходный fallback сохранены:18/27 cold пользователей среди всех1260. Повторы разрешены. Порядок ties — numeric programme ID как в общем finalizer; user IDs остаются строками.

## Восстановление до full fusion

Сначала воспроизведены все четыре компонента PPR/hourly/L4/L5 в каждом окне: **990000 сохранённых native float32 scores**, все indices и Top-5 совпали **бит-в-бит**, максимальная абсолютная ошибка0.

L4/L5 загружаются только из собственных сохранённых `catboost-ranker.cbm`. Проверены имена и порядок признаков модели против training provenance. Использован замороженный v5 feature engine:

`54af6b09c507c8a7579230a73fdb06f915f4a03a5535edee8e09bf35a7f4afaf`.

L4 сначала получает ровно56 признаков с strong_context=False. Затем тот же engine дополняется четырьмя strong признаками для L5; первые56 остаются прежними. Проверены исходные training source SHA: L4 `9f60de112ec180ba99b3f5fac549d4ea41aebd1cb58fe9ebffdd165883acb407`, L5 SHAv5 выше. Для provenance всех historical label windows проверено labels_end<=evaluation cutoff. Для L5 зависимости PPR/hourly/coarse совпали с зафиксированными SHA.

До построения полной смеси эти восстановленные matrices усечены до100 точно по прежнему правилу, затем заново вычислены старый core и обе старые RRF mixtures. Все шесть reference bundles (три на окно), их native100 scores, indices и Top-5 совпали **точно**. Только после этой проверки сохранён full core и выполнен full RRF.

Inference читает pre-cutoff history/events/metadata. Hourly и opportunity features будущего используют только ID/channel/start/duration; будущих просмотров или content metadata в scoring нет. Основная evaluation truth используется лишь общим evaluator и отдельными diagnostic/verification scripts, без подбора признаков или персональных весов.

## Артефакты и воспроизведение

- `advanced_full_fusion.py`, register() добавляет `METHODS['full_catalogue_fusion']`; разрешены только bundle shfl4/shfl45 без произвольных дополнительных параметров.
- `tools/run_full_fusion.py`; две команды: `.venv/bin/python tools/run_full_fusion.py --split main` и аналогично `--split early`.
- Запуски `artifacts/runs/{main,early}/FULL-{shfl4,shfl45}/`: полный итоговый score NPY, native100, Top-5, metrics, source snapshots, provenance.
- `artifacts/runs/full-fusion-cache-v1/<protocol SHA>/<module SHA>/`: ppr/hourly/l4/l5/core `.npy`, mapping, reconstruction details и manifest с SHA всех payloads, source records, cbm/provenance, frozen engine и зависимостей. Cache read проверяет целостность.
- `artifacts/runs/full-fusion-research/independent-validation.json`: независимое восстановление **всех full max-normalized core и RRF scores**, exact native ranks/scores, Fraction AP, schema/cohort/cold/fallback. Все4 runs PASS.
- `artifacts/runs/full-fusion-research/verify.py`: сохраняет текущий независимый verifier. Более ранняя версия RRF-only проверки сохранена отдельно, не подменена новой.
- `artifacts/runs/full-fusion-research/changed-top5-diagnostic.json`: точные различия выдач и AP; это diagnostic после фиксированного эксперимента, не input модели.
- `tests/test_full_fusion_audit.py`:4/4 PASS. Fixtures показывают реальную потерю согласованных tail signals при усечении; проверяют полный RRF/ties/NaN, исходные фиксированные веса и отказ при подмене cache scores.

Source SHA `advanced_full_fusion.py`: `b4771f7c62296d62e763fd80c119ac928d5e91ccc221e6782bd9a947001ba786`.

Runner SHA: `9b58cd9f41454270012aebf78264b3952b6f72e925563c9a36f8143b0eabcfb2`.

Tests SHA: `18d6da608da279a8b843dc474dde8a4797d54378fb59eef3c139f9afff158d95`.

Полное восстановление и первый main run заняли25.99s; early23.35s. Вторые смеси использовали проверенный cache и заняли1.94s/1.77s. Peak memory процесса2.55GB/2.30GB. Новое обучение, новая seed variation, Kaggle submission и изменение общих файлов этой веткой не выполнялись.
