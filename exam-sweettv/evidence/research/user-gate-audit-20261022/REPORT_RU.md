# Независимый аудит chronological user gate

**Итог: PASS.** Проверены завершённый early и восстановленный без refit main, все historical AP targets/features, saved-model routing и обе full-RRF выдачи. Первоначальный main FAILED из-за диска сохранён; подробности ниже.

## Проверка дизайна до fit

Прочитан `research/user-gate-20261022/DESIGN_RU.md` до появления реализации и результатов. В заявленном протоколе leakage не обнаружено: training AP targets относятся только к трём unsupervised экспертам, заново построенным на своём historical prefix. L4/DS не используются в historical labels или gate features. Frozen L4 участвует только в outer inference fusion с заранее заданным weight0.5. Fixed folds и temporal boundaries сохранены; обучение meta-regressor на historical AP labels допустимо при выполнении этих границ.

Это **предварительное заключение по дизайну**, не подтверждение ещё отсутствующего кода, запуска или артефактов. Проверки кода: одинаковая prefix/eval нормировка экспертов и cold/fallback policy для AP targets; feature-only prefix источники без IDs/labels; отсутствие hidden L4 participation; выбор полных expert score rows; cutoff и источник hashes; predictor replay и truth poisoning. Argmax regressed AP — отдельная эвристика выбора эксперта, не оптимизатор глобального MAP.

На момент предварительного review код и результаты ожидались. Последующая завершённая проверка приведена ниже. Аудитор не запускает fit и не меняет source/PLAN/registry.

## Итоговая проверка кода и завершённых outputs

**PASS на обоих окнах.** Main первоначально завершился FAILED при записи31MB `gate-scores.npy` из-за заполненного диска, после успешного fit. Этот failed run и его128-byte header сохранены. После освобождения диска владелец создал отдельный `UG-AP3-depth4-300-recovered` без refit. Аудитор сравнил byte hashes recovered/original cbm и training provenance, original record hash и recovery manifest; всё совпало. Recovery code не содержит fit. Проверены именно восстановленный main run, успешный early run и оба `UG-FULL-shfl4-replacement`.

### Historical targets и признаки

Для всех четырёх folds самостоятельно восстановлены PPR scores из сохранённых transition/teleport и зафиксированного числа итераций. Hourly scores прочитаны через существующий hash-validated cache, без создания или переобучения эксперта. Core max-normalization и tie-sorted Top5 восстановлены собственным reference. Затем **Fraction AP по actual historical truth** вычислен для всех трёх экспертов: saved y совпал точно после float32 cast для main8607 и early7812 rows, всего49 257 target values. Ни L4, ни DS в этих labels не участвуют.

У всех16 419 historical rows порядок string user IDs и keep mask `truth nonempty && !cold` совпали. Fold protocol/truth/raw/gate-data hashes проверены; actual label stop timestamps и label-window ends не превышают outercutoff. Saved X28 совпал точно с повторно построенными features при poisoned `ctx.truth` и `ctx.display` во всех historical contexts. Таким же образом проверены все1260 evaluation feature rows каждого окна. Feature columns одинаковые, уникальные28; в source feature formulas нет user/item ID признаков. Обучающие признаки строятся до чтения historical labels.

Независимая часть проверки historical y использует Fraction, собственную сортировку и сохранённый PPR state. Hourly преобразование и gate_features вызваны из исходного кода на frozen данных; каждая формула28features вручную отдельно не реализовывалась. Это ограничение reference diversity, а не утверждение о независимом переписывании всей системы.

### Saved model, routing и bundle

Loaded CatBoost config соответствует actual depth4, MultiRMSE,300iterations, seed42,lr.05,l2=10. Загрузка saved cbm без fit воспроизвела все1260×3 predicted AP обоих окон точно. `argmax` choice и полные выбранные score rows совпали побитно с сохранённым gate-scores; при exact float tie приоритет core. Количество выборов main:core1124/PPR63/hourly73; early:928/27/305 (доcold gate).

Outer bundle независимо восстановлен как full-rank RRF gate1+L4.5,k0=60, **по всему available catalogue каждого компонента**, без Top100 truncation. L4 hash/protocol проверены. Статический review подтверждает: L4 используется только здесь и не участвует в gate training targets/features. `prepare_scores` технически обслуживает общий cache нескольких моделей; это не передаёт их scores в gate feature matrix.

| Проверенный output | Main Fraction MAP@5 | Early Fraction MAP@5 |
|---|---:|---:|
| Gate alone (main recovered) | 0.38109114162627894 | 0.40068241648511255 |
| Gate1+L4.5 full RRF | 0.3845705438879914 | 0.4048763162672476 |

Все4 outputs прошли независимые Fraction AP, per-user AP CSV/order, native scores/ranks, Top5 и common cold/fallback: **495 000 native values/ranks и5040 user predictions**. Existing routing/nonfinite test1/1PASS. Никакого shadow outcome анализа или настройки весов не проводилось. Корректность не означает прироста качества; модель регрессирует исторический user AP с overlapping folds/users и horizon shift, калибровка AP не гарантируется.

### Артефакты и завершение

Скрипт `tools/verify_user_gate.py`; рядом `main-evidence.json`, `early-evidence.json`, `config-evidence.json`, `tests.log`, логи и небольшие hourly-cache references. Early evidence создан до добавления обработки main recovery; точный текст той версии сохранён как `early-audit-source.txt` и совпадает с записанным SHA. Active код/PLAN/registry и исходные artifacts аудитор не менял. Новые fullscore arrays на диск не записывались, saved model не переобучался. Изначальная остановка main audit на statusFAILED была корректной защитой от чтения незавершённого payload, а не математической ошибкой модели.
