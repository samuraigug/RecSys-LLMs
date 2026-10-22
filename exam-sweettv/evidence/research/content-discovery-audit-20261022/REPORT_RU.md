# Независимый аудит content discovery — 22.10.2026

**PASS:** отличие CDS от DS ограничено восемью дополнительными content score/rank признаками; исторические обучающие строки и базовые60 признаков воспроизведены. Все saved-model predictions, interpolation и inherited-weight bundle проверены на обоих фиксированных окнах. Обучение не запускалось, активный код и исходные runs не менялись.

## Контролируемое отличие от DS

`training_replay.py` независимо проходит historical candidates/users, восстанавливает deterministic+random selection с seed42 и последовательно хеширует небольшие пользовательские матрицы. Полная training X не хранится одновременно. Получены точные сохранённые SHA256 для **CDS X68, y, weights** на main1 851 354 и early1 724 822 строках. SHA первых60 столбцов X совпал с сохранённым **DS X60**, а y/weights — с DS y/weights. Это проверка всех3 576 176 строк, а не нескольких примеров.

Sampling audit каждого fold совпал с DS побитно по значениям после JSON parsing и с независимо восстановленным audit. Совпали порядок пользователей, число/положительность строк, conditional eligible domain и веса. Самплер, fit/scoring и interpolation AST двух реализаций идентичны; добавление8 признаков не меняет retrievers. Все actual CatBoost config и первые60 имён признаков совпали; имена68 признаков и config загруженной cbm соответствуют provenance.

Исходный DS не сохранял groups-array SHA. Поэтому сравнение groups основано на идентичном sampler AST, exact user-ordered sampling audits и восстановленном порядке строк; собственный groups SHA сохранён, но сравнение с отсутствующим original hash не заявляется. Groups вообще не передаются binary CatBoost fit.

## Preprocessing и leakage

Четыре словаря и smoothed IDF отдельно независимо восстановлены из `metadata-train.json` для всех4 historical folds и2 evaluation contexts: **24 словаря/IDF**. Собственный reference разбирает category/genres/actors/director, нормализует строки, считает document frequency по уникальным историческим программам и проверяет IDF=log((1+n)/(1+df))+1. Словари совпали точно, IDF — с допуском2e-7 на float32 вычисления. Candidate metadata не добавляется в fit vocabulary. Неизвестные программы получают zero vector.

Во время полного evaluation replay `ctx.truth` и `ctx.display` заменены объектами, бросающими исключение при доступе; обе модели полностью восстановили predictions. Historical schema hashes и evaluation schema hashes совпали с исходными artifacts. Frozen engine SHA=`54af6b09c507c8a7579230a73fdb06f915f4a03a5535edee8e09bf35a7f4afaf`; проверены source/dependency/protocol/frozen ETL, savedscore/mask/baseline hashes. Исторические labels end<=evaluation cutoff, реальные event stop также<=cutoff. Дополнительный feature builder проверяет train-event start/stop; использует coverage² для partial, full1, decay56days и log1p user-programme evidence.

IDs остаются строками в mappings; unit case с `9007199254740993` проходит. Выполнены3/3 content unit tests, включая post-cutoff rejection. Отдельный synthetic case аудитора подтверждает negative-only warm user, cold exclusion, веса400/30 и399/30 и positive weight1. Во всех восстановленных historical строках positiveweight1 и conditional IP mass проверены непосредственно. Negative-only warm groups сохранены: main f3=493/f4=261; early f2=398/f5=82.

## Полный inference replay

`replay.py` самостоятельно импортирует проверенный frozen engine, добавляет content features, загружает cbm через CatBoost и вызывает predict_proba последовательно по пользователям. specialist/fit и общий finalize_scores не вызываются. Независимо проверены mask, сортировка score+numeric-ID, user/candidate ordering, Top5 и cold/fallback.

| Объём точного сравнения | main | early |
|---|---:|---:|
| Пользователи / primary cohort / candidates | 1260 /1238 /6384 | 1260 /1224 /6449 |
| Nonrepeat scores из saved cbm | 8 012 703 | 8 106 508 |
| Repeat scores bit-exact baseline L5 | 31 137 | 19 232 |
| Binary native Top100 scores+ranks | 124 200 | 123 300 |
| Interpolation native Top100 scores+ranks | 124 200 | 123 300 |
| RRF bundle native Top100 scores+ranks | 124 200 | 123 300 |

Все16 119 211 nonrepeat values совпали **побитно после исходного float32 cast**. Replay включает также cold users до общего gate. Repeat values неизменны. Interpolation логитов с weight0.5 независимо восстановлена; repeated entries untouched. Bundle отдельно восстановлен из native ranks core1+L4.25+CDS.25, k0=60, с проверкой actual config и одинакового protocol. Во всех6 outputs совпали все Top5 и fallback; суммарно742 500 native values/ranks.

## Ограничения вывода

Использован исходный feature builder для полного model replay; независимым reference дополнительно покрыты actual training hashes, словари/DF/IDF, sampler, ranking/interpolation/RRF. Математика каждой content cosine вручную отдельно не реализовывалась. Baseline L5 cbm в этом аудите не переигрывалась: использован verified immutable l5.npy. Fraction MAP/CI выполняет владелец эксперимента, здесь они не выдаются за собственные проверки. Current config фиксируется кодом42/300; текущие runs ему соответствуют. Common target и IP weights не доказывают межмодельную калибровку probabilities. Аудит корректности не означает улучшения качества.

Все scripts/evidence/logs находятся рядом: `training_replay.py`, `replay.py`, `bundles.py`, `*-training-evidence.json`, `main-evidence.json`, `early-evidence.json`, `bundles-idf-evidence.json`, `control-evidence.json`, `sampling-evidence.json`, `tests.log`. Единственный отчёт этой ветки аудита — этот файл; shared checkpoint не редактировался.
