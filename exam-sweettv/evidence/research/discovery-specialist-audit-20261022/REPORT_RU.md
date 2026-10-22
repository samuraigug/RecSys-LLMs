# Независимый аудит discovery expert — 22.10.2026

**PASS в проверенном объёме:** saved CatBoost самостоятельно воспроизвёл все nonrepeat scores обоих завершённых DS-nonrepeat-binary300. Обучение не запускалось; код модели, исходные run directories и общий PLAN не изменялись. Проверка не утверждает улучшения качества или достижения цели 0.95.

| Проверка | main | early |
|---|---:|---:|
| Фиксированная основная когорта | 1238 | 1224 |
| Все пользователи | 1260 | 1260 |
| Полный каталог | 6384 | 6449 |
| Nonrepeat scores, точное совпадение bytes после сохранённого float32 cast | 8 012 703 | 8 106 508 |
| Repeat scores, bit-exact с baseline l5.npy | 31 137 | 19 232 |
| Native Top100 scores и порядок, точное совпадение | 124 200 | 123 300 |

## Независимость replay и границы

`audit.py` загружает сохранённую `discovery.cbm` напрямую через CatBoost, независимо импортирует frozen FeatureEngine v5 после проверки полного SHA256 `54af6b09c507c8a7579230a73fdb06f915f4a03a5535edee8e09bf35a7f4afaf` и строит признаки из frozen Context. Он не вызывает specialist(), prepare_scores(), fit() или общий finalize_scores(). Матрицы 60 признаков создаются последовательно для одного пользователя; полный массив 1260×catalogue×60 не материализуется. Сам движок хранит свои матрицы отдельных признаков.

Перед построением evaluation features `ctx.truth` и `ctx.display` заменены объектами, выбрасывающими исключение при доступе. Оба полных inference прошли. Replay охватывает также cold users до общего gate. Mask независимо совпадает с count==0. Top100 самостоятельно отсортирован по score и числовому ID, сверены сохранённые scores, indices, Top5 и fallback для каждого cold user. Порядок CSV user_id совпадает с frozen all_template; порядок candidates закреплён protocol frozen hashes. Обе протокольные SHA, candidate-order SHA и user-order SHA сохранены в evidence JSON.

Проверены сохранённые SHA scores, mask, baseline, baseline cache manifest, feature engine, исходника specialist, historical protocol/truth/sampling audits и strong-feature dependencies. `Context` проверил все frozen ETL hashes. Дополнительно проверены фактические historical raw inputs относительно input_hashes, параметры сохранённого CatBoost и имена признаков. Хеши самих cbm зафиксированы этим аудитом: исходный score-manifest отдельного model hash не содержит. Это ограничение первоначального manifest, не расхождение replay.

## Временные границы и отрицательная выборка

Исторические label windows main f3/f4 заканчиваются 18.05.2020, early f2/f5 — 20.04.2020; каждый конец <= соответствующего evaluation cutoff. Проверены реальные max stop timestamps historical validation-events. Статический review пути fit подтверждает использование historical Context labels; outer ctx используется для baseline и последующего inference. FeatureEngine и strong-feature dependencies не используют validation truth/display как признаки. Это проверка кода и артефактов, не новый инструментированный fit.

Для каждого historical sampling audit независимо восстановлены eligible domain из candidates минус personal full history, положительные labels и полный список ожидаемых warm пользователей с непустой truth. Число пользователей с **нулём discovery positives, но непустой repeat truth**, реально включённых в обучение: main f3 —493, f4 —261; early f2 —398, f5 —82. Удаления этой важной отрицательной группы не обнаружено. В каждом audit sum(weights) совпадает с размером conditional nonrepeat domain.

Независимый synthetic case создаёт 400 nonrepeat candidates и retrievers, выбирающие только repeats. Для repeat-only пользователя все 30 sampled negatives имеют weight400/30; для пользователя с одним discovery positive этот positive имеет weight1, остальные sampled negatives —399/30. Проверены сохранение negative-only user, conditional mass400 и исключение cold user. Этот case отличается от unit fixture ведущего исследователя. Его существующий `tests.test_discovery_specialist` также выполнен: 1/1 PASS.

Математически deterministic retrieved candidates и все positives имеют inclusion probability1; случайная равномерная выборка n из остальных N получает N/n. Это исправляет массу conditional pointwise binary loss при текущем n=30. Не доказывает калибровку между L5 и discovery head. Параметр negatives=0 при непустом outside нарушил бы mass assertion; текущие запуски используют30. API фиксирует CatBoost42/300 и ряд hyperparameters внутри функции; текущие параметры запусков совпадают, но произвольный переданный config мог бы вводить в заблуждение.

## Что этот аудит не утверждает

Training X/y/weights arrays заново целиком не строились, их array_sha256 из provenance не воспроизведены; фактические training weights проверены через сохранённый sampling audit, статический алгоритм и независимый synthetic case. Случайная выборка каждого historical пользователя не переигрывалась. Baseline L5 model здесь заново не вычислялся: проверены immutable baseline hash и bit-exact repeats. Interpolation, blends, Fraction MAP и доверительные интервалы проверяет ведущий исследователь отдельно. Заголовок PASS относится к указанным проверкам, а не к этим отдельным задачам.

Артефакты: `audit.py`, `sampling-evidence.json`, `main-evidence.json`, `early-evidence.json`, `additional-evidence.json`, логи и отдельно построенные inference-feature provenance. Исходные данные и модели читались; новые артефакты записаны только в этой папке аудита.
