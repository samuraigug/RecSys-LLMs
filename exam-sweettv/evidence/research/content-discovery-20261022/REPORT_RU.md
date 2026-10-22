# Раздельные content signals в discovery expert

Matched bundle оказался ниже прежнего DS bundle на обоих окнах; гипотеза не улучшила текущий максимум.

Проверена одна фиксированная конфигурация: к frozen 60 features добавлены 8 признаков — cosine affinity и tie-aware rank отдельно для category, genres, actors, director. Заголовочный CBF остался прежним. Словари и smooth IDF обучены только на metadata-train; профиль пользователя строится из pre-cutoff events: full airing вес1, partial coverage², decay56days, log1p по программе, L2-нормализация. Выбор весов сделан до результатов: полные просмотры дают единичное свидетельство, короткие partial ослаблены, decay/log ограничивают доминирование старых сериалов.

CatBoost300/depth6/seed42 и sampling прежнего DS сохранены. Retrievers не менялись; historical groups, sampled rows, labels, inverse-inclusion weights совпали с DS (audit equality и y/w hashes). Все кандидаты возвращаются общему evaluator; repeats побитно сохранены из L5. Нет ID features или квоты discovery. Calibration двух heads не гарантируется.

| Вариант | Main MAP@5 | Early MAP@5 |
|---|---:|---:|
| DS-nonrepeat-binary300 | 0.3841576916 | 0.4045338598 |
| CDS-blocks-binary300 | 0.3833930174 | 0.4048420479 |
| CDS-blocks-logit05 | 0.3851317088 | 0.4051929012 |
| SHFL4DS-rrf-equal | 0.3858360258 | 0.4049993192 |
| SHFL4CDS-rrf-equal | 0.3857081314 | 0.4049843410 |
| FULL-shfl4 | 0.3852638665 | 0.4057169118 |

После evaluation отдельно посчитаны discovery hits: matched bundle128→131 main и191→194 early, при росте новых слотов710→725 и1000→1008. При этом общий MAP снизился: небольшое увеличение hits не компенсировало потери в рангах/повторах. Диагностика не использована для fit или нового подбора. Суммарная CatBoost feature importance добавленных блоков4.29% main/2.45% early показывает, что признаки использованы, но не доказывает полезности.

В matched bundle core1 + L4.25 + expert.25 заменён только DS; изменение main -0.0001278945, early -0.0000149782. Дополнительный заранее фиксированный logit0.5 не требует нового fit. Grid не проводился.

Проверки: `verification.json`, `independent-main.json`, `independent-early.json` — Fraction AP с denominator min(5,nrelevant), schema, cohort, native ranks, common cold fallback; маска nonrepeat восстановлена из history, repeat scores bit-exact. Независимый Astra аудит `../content-discovery-audit-20261022/REPORT_RU.md` завершён PASS:16 119 211 nonrepeat probabilities побитно восстановлены из cbm после float32;50 369 repeats exactL5;742 500 native scores/ranks и все Top5/cold проверены. Historical X68/y/w hashes восстановлены по всем3 576 176 rows; DS X60 prefix/y/w hashes совпали. Все24 vocabulary/DF/IDF восстановлены независимо. Evaltruth/display poisoned; schema hashes совпали. Original group hash отсутствует; group equivalence подтверждается одинаковым sampler AST и точными ordered sampling audits, это ограничение сохранено в независимом отчёте. `tests.log` — 3 tests: string IDs/negative-only groups/IP mass, metadata-only affinity, post-cutoff rejection. Feature names, block vocabulary/IDF, metadata/events/source hashes сохранены в каждом content-schema.json и сведены в verification.json. Исходник и runner snapshot: source.py/runner.py; модели и training-provenance в runs.

Основной protocol SHA630e87aa0dd5f0b120083c3cc217c1184e9d3e1dabe31c61089d9220ad2df231, cohorts и candidates сохранены. Main и early пересекаются, результаты post-selection и не являются независимой репликацией. Значимость не заявляется. MAP@5≈0.95 не достигнут; Kaggle/внешней публикации нет.
