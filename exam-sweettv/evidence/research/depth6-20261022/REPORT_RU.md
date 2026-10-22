# Явное ограничение глубины LightGBM

Проверена конкретная регуляризационная гипотеза: прежний параметр `depth=6` относился к CatBoost и не ограничивал leaf-wise LightGBM. Новый `advanced_depth6.py` задаёт настоящий `max_depth=6`. Все остальные effective config, параметры, historical folds, frozen L5 feature engine (60 признаков) и sampling protocol сохранены. Параметры не подбирались по outer truth.

| Метод | Main MAP@5 | Early MAP@5 |
|---|---:|---:|
| Старый LW binary, unlimited depth | 0.3583649704 | 0.4014596950 |
| LW binary, max_depth=6 | 0.3616287471 | 0.3991081155 |
| Core1 + depth6 RRF0.5 | 0.3751642434 | 0.4029908769 |
| FULL-shfl4 1 + depth6 RRF0.15 | 0.3801655897 | 0.4047061093 |
| Контроль FULL-shfl4 | 0.3852638665 | 0.4057169118 |

**Результат отрицательный для продвижения модели:** самостоятельный depth6 немного улучшает main против unlimited depth, ухудшая early; оба заранее объявленных blend хуже устойчивого bundle. Это не опровержение всего семейства LightGBM. Новых гиперпараметров в этой ветке не перебирали. Окна пересекаются; числа локальные, не Kaggle.

`verify.py` независимо проверил Fraction AP, schema, порядок cohort, допустимость кандидатов, native ranks, общий cold fallback для 6 outputs×1260users. Из сохранённых текстовых моделей заново восстановлены 300 деревьев на каждом окне: максимальная глубина ровно6. Сравнение provenance старого и нового run подтвердило identical fit_sources/parameters, единственная разница effective config — max_depth6. Новый SHA256 каждой X/y/group/weight training array сохранён; старый run этих хешей не сохранял, поэтому bit-exact сравнение arrays с ним не заявляется. Synthetic regression test проверяет реальную глубину при конфликтующем obsolete depth19 — PASS.

Исходник сохранён в `advanced_depth6-source.py`; runner `tools/run_depth6.py`; фактические fit+score времена main203.49s/early184.55s. Validation truth, cohort, candidate pool и fallback не менялись; training labels заканчиваются до соответствующего outer cutoff. Никаких внешних действий.
