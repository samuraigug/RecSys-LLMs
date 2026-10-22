# Где теряется качество за пределами повторов

22.10.2026. Исследование сформулировано до новых fit; оценивались фиксированные сохранённые PPR и `CCBF-core-CCB-charword-hourly-p0.25-w0.15`. Truth используется только для итоговых диагностических таблиц. Ни пользовательские правила, ни features, ни персональные пороги по main truth не подбирались. Исходники активных чужих моделей не изменялись.

**Основная проблема — качество discovery.** У текущего сохранённого гибрида повторы составляют 88,3% выдачи main, а в truth только 55,5%. Простое добавление новых слотов это не лечит: обученный channel-feature WARP и явное interleaving проверены ниже, все варианты уступили контролю/базовой стратегии.

## Фиксированная диагностика ошибок

`diagnose.py` читает только frozen train history/events/raw-union channel profile, разрешённые future airing channel/time opportunities и сохранённые Top-5. Файл `diagnostics.json` содержит полный результат; frozen hashes обоих split проверены. Все доли ниже относятся к платформенным логам/этому временному окну: «новый» не означает «никогда в жизни не просмотренный».

| Взаимоисключающий класс truth, main | Пар user/programme | Recall@5 фиксированного гибрида |
|---|---:|---:|
| Был personal full>=80% | 3335 | 77,09% |
| Был personal partial, но не full | 867 | 12,11% |
| Не было личного full/partial; у других был full | 1228 | 1,63% |
| Нет личной истории; нет global full, но старые metadata известны | 196 | 0% |
| Metadata программы до cutoff отсутствуют | 379 | 0% |

Итого 6005 truth pairs. Выдача 6190 slots: 5465 personal full repeats, 569 partial-only, 156 personal-new/global-known. Отдельно global-full-unseen truth — 577 пар: две попали в personal-partial строку выше, поэтому 196+379=575 не заменяет полное число 577.

На early картина повторяется: full recall 80,21%, partial 19,22%, personal-new/global-known 2,05%, два последних класса 0%. Это описательное подтверждение структуры ошибок на перекрывающихся окнах, не два независимых исследования.

Пользовательская repeatability связана с размером **прошлой** full-history:

| Train-only группа | Main users | Доля truth, бывшая personal full | Early аналог |
|---|---:|---:|---:|
| 20+ известных full программ | 771 | 66,89% | 74,17% |
| 5–19 | 416 | 39,00% | 47,38% |
| 1–4 | 37 | 9,70% | 11,88% |

Таким образом, группы различаются по наблюдаемому потенциалу повторов. Но эти диагностические доли нельзя превращать в персональные main-подсказки. Чтобы применять learned gate, зависимость нужно заново оценить на полностью исторических pseudo-futures.

Группировка по **фактическому будущему** числу новых программ особенно показательна, но непригодна как доступный признак: 219 main пользователей с нулём новых truth items имеют MAP 0,6400, а 101 с пятью новыми — 0,0339. Модель выдаёт последним в среднем 2,21 новых items, хотя правильных пять. Это одновременно нехватка новых слотов и неспособность выбрать нужные новые программы; первой причиной нельзя подменять вторую.

## Что дают каналы и чего не дают

У 4032 из 6384 main кандидатов ровно один future channel; максимум 12. Доступны каналы будущего эфира, а не поведение пользователя в нём.

- Любой канал, который пользователь смотрел до cutoff, покрывает **96,45%** main truth pairs и **96,31%** новых metadata items. На early — 96,83% и 96,84%.
- Но top-3 каналов по реальным прошлым секундам покрывают только **47,26%** всей truth и **18,97%** personal-new/global-known truth. Top-5 — 59,75% и 28,50% соответственно.
- Средний по порядку пользователь имеет **2652** допустимых программы на всех прошлых просмотренных каналах; top-3 дают медиану 316. Каналы дают широкий источник discovery, а не достаточный ответ, какие пять программ выбрать.

Следствие: нельзя жёстко ограничить рекомендации несколькими любимыми каналами. С другой стороны, канал и прошлые просмотры не доказывают, что пользователь выберет любую передачу этого канала. Это мотивировало отдельный learned transfer эксперимент.

## Реализованная новая архитектура: collective channel-feature WARP

Код: `advanced_collective.py`, SHA-256 **`2bbd528db65ae8fd5595e785e46a015b412536dc6f49225fc0d3527fa11a3715`**. Runner: `tools/run_collective.py`; configs `collective-initial.json` и `collective-mixture.json`. LightFM использует ранее проверенный pinned official build; зависимости дополнительно не менялись.

Train representation объединяет:

1. Реальные full user-programme positives с log-count weights.
2. Все pre-cutoff user-channel union seconds, включая unlabelled airtime и пользователей вне exam template. Auxiliary positives имеют вес `0.1*log1p(hours)`.
3. Общий channel embedding в features программы и соответствующего auxiliary channel item. Item identity есть только у программы с положительным историческим примером. **У cold programme identity ровно ноль**; его representation строится из реально используемых обученных shared features.

Второй shared вариант добавляет две learned numeric availability features: `log1p(airings/day)` и `log1p(mean_duration_hours)`. Формулы одинаковы для training и будущего расписания. Future schema ограничена ID/channel/start/duration; title/actors/category не используются.

Префикс raw лога обрезается до cutoff до накопления интервалов. Контекст содержит 4880 user identities main / 4859 early, включая заранее известные exam users; отсутствие истории всё равно приводит к общему cold fallback. Это не изменение оценочной cohort: main 1238/1260, early 1224/1260 и их protocol SHA сохранены. Собственный immutable cache включает protocol+source SHA, input hashes и payload hashes.

Shared-channel модель даёт ненулевые обученные features **4226 из 4510** globally cold main кандидатов; у identity control их ноль. На early — 4630 из 5261. Это реальное устранение отсутствия представления, а не автоматически улучшенное ранжирование.

| Фиксированный вариант, seed 42, 32 factors, 15 epochs | Main MAP@5 | Early MAP@5 |
|---|---:|---:|
| Identity-only, тот же all-prefix user registry | 0,2803628164 | 0,3190298203 |
| Shared channels + auxiliary raw context | 0,2628879465 | 0,1761846405 |
| Shared channels + learned availability | 0,2482388709 | 0,2988187636 |
| Reference `SHFL4-rrf-0.5` | 0,3852639 | 0,4056107 |
| Separate repeat/discovery interleave, discovery weight 0.5 | 0,3247471280 | 0,3403115922 |
| Separate repeat/discovery interleave, discovery weight 0.9 | 0,2690288997 | 0,2690867829 |
| Обычный RRF: reference + shared-channel, weight 0.05 | 0,3814095315 | 0,4008866649 |

Interleave — отдельная фиксированная гипотеза: repeat head сохраняет ranking reference только по personal-full items; discovery head ранжирует все остальные learned collective scores. Scores равны weight/rank каждого head, не вероятностям. Ни принудительные quotas, ни learned gate на main truth не применялись. Два веса объявлены заранее и одинаковы между split.

**Результат отрицательный:** ни один новый вариант не подходит для финального выбора вместо reference. На main shared модели не попали ни в одну из 577 global-full-unseen truth pairs; на early shared-channel попал в одну, shared-availability в две из 946. Доступность features не превратилась в качество выбора. Сдвиг к новизне без сильного discovery head ухудшает MAP. Обычное небольшое дополнение shared-channel к reference также ухудшило оба окна, поэтому отрицательный результат не объясняется только жёстким разделением heads. Это результат данных трёх конфигураций и seed 42, а не опровержение всех collective моделей.

Проверки: `tests/test_collective_audit.py` **3/3 PASS**, 0,026 s. Проверены реальный raw union всех прошлых пользователей, отсечение future seconds, большие строковые user IDs, холодная identity=0 и ненулевой shared-channel signal, unknown future channel без представления→NaN, отказ при изменённом raw hash, seed replay, native predict и model reload. Полный native-score→Top-5 пересчёт из сохранённых моделей совпал **для всех шести WARP runs**. Независимый Fraction AP/protocol/user order/candidates/cold проверен для первых **10 runs × 1260 users**: `artifacts/runs/collective-research/independent-validation.json`. Ещё два обычных RRF blend проверены отдельным Fraction AP и независимым пересчётом native reciprocal-rank sums для всех пользователей: `independent-blend-validation.json`. Всего **12 проверенных runs**.

## Следующий конкретный эксперимент: обучаемый repeat/discovery gate

Вторая гипотеза — **два специализированных supervised head вместо единого positive classifier**. Discovery head обучается только на кандидатах, которых пользователь ещё не смотрел full, и historical future positives внутри этой области. Repeat head остаётся текущим сильным reference. Отдельный режимный classifier прогнозирует вероятность repeat/discovery по pre-cutoff истории: число программ, active days, recent new-programme rate, channel entropy, recency. Все labels и calibration — только из исторических all-user folds, ending≤outer cutoff; negatives корректируются до полного каталога. Main truth не входит в fit или индивидуальное переключение.

Мотивация — разница repeat/discovery ошибок и monotonic train-history групп выше, **а не обещание выигрыша**. Проверка должна отделить: (a) способен ли discovery head выбирать новые правильные программы при фиксированном candidate set; (b) даёт ли gate прибавку к reference без принудительного ущерба repeat users. Простые frozen weights уже дали отрицательный контроль; повторять их более густой сеткой не нужно.

Архитектурный первоисточник идеи разделения режимов — [Ren et al., RepeatNet, AAAI 2019](https://arxiv.org/abs/1812.02646): отдельные repeat/explore decoders и learned mode probabilities. Это session-based работа; наш 10-недельный TV hold-out отличается, поэтому результаты статьи не переносятся на SWEET.TV и буквальное воспроизведение нейросети не заявляется. Общие item features и их embeddings — [официальная LightFM API](https://making.lyst.com/lightfm/docs/lightfm.html).

Ни одно измерение здесь не доказывает универсальную невозможность MAP 0,95. Подтверждены конкретный характер ошибок, отрицательный результат одной новой архитектуры и следующий проверяемый эксперимент. Внешние score, публикации и скрытые test labels в исследовании не получались.
