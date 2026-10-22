# Дополнительный аудит на непересекающихся пользователях

Аудит выполнен после широкого выбора примерно из 490 прогонов на исходных 1260 пользователях. Это дополнительная проверка обобщения по пользователям, а не независимый dataset, официальный holdout или достижение цели 0.95. Исходы не используются для новых настроек моделей.

Когорта заранее зафиксирована: 3620 строковых user ID из 4876 пользователей raw trainprefix с началом сессии в [09.03.2020, 18.05.2020), за исключением исходных 1260 template ID. Ни наличие будущего просмотра, ни будущие метки не использовались при выборе. Четыре original CSV подключены symlink; отдельный submission.csv содержит только эту когорту. AUDIT_ONLY_NOT_FOR_KAGGLE.

Пререгистрация: b63a134b96d635320ca21563137538dec3fe7b13f0654050f24d5afc3d724255. SHA списка ID: ccecd2d2ca1060c7601aa8d23851c7ba9565280061b918595fdbf4ed5d670daf.

До открытия будущих меток закреплены FULL-shfl4 и SHFL4DS-rrf-equal. F14 заранее исключён: замороженный C4 не содержит полного fitted IDF/transformer и всех historical item vectors, необходимых для нового user profile; новый fit не выполнялся. Причина зафиксирована в preregistration.json до ETL.

| Модель | MAP@5, непустая truth | MAP@5, вся когорта | Непустая truth / все |
|---|---:|---:|---:|
| FULL-shfl4 | 0.3307875304 | 0.3086741099 | 3378 / 3620 |
| SHFL4DS-rrf-equal | 0.3312916091 | 0.3091444905 | 3378 / 3620 |

Обе зафиксированные модели дают здесь около 0.331 MAP@5; разница SHFL4DS относительно FULL-shfl4 составляет +0.00050408. Это описательное сравнение, без заявления о статистически устойчивом превосходстве и без последующего подбора. Качество существенно ниже 0.95; меньший уровень относительно исходной когорты может отражать различие состава пользователей и не изолирует эффект переобучения на валидации. Среди всех 3620 пользователей 125 проходят cold gate, у 242 пустая future truth.

Основная метрика исключает пустую будущую truth; дополнительная назначает ей AP=0. Обе объявлены заранее. Сравнивать абсолютное значение с исходными 1260 напрямую нельзя: это другой состав пользователей.

Протокол: train до 18 мая; future 18 мая–27 июля; Europe/Kyiv elapsed; объединение перекрытий и усечение сессий до overlap; полный просмотр ≥80% исходной длительности эфира. Те же 6384 кандидата, повторы разрешены, отсутствие полного train history включает общий train-popularity fallback.

Frozen inference: L4 и L5 .cbm загружены без fit; DS .cbm заменяет L5 только на nonrepeat candidates. FeatureEngine v5 SHA 54af6b09c507c8a7579230a73fdb06f915f4a03a5535edee8e09bf35a7f4afaf. PPR и hourly реконструированы детерминированно с исходными параметрами. Core — сумма max-normalized PPR/hourly; FULL-shfl4 RRF k60: core1+L4.5; SHFL4DS: core1+L4.25+DS.25. Полный каталог, без Top100 truncation при fusion.

Отдельный advanced_shadow_inference.py не вызывает prepare_scores и не ослабляет его protocol guard. Проверены все пять raw hashes нового bundle, четыре original data hashes, cutoff, кандидаты, train metadata/opportunities/fallback, все train event arrays и вся непустая global history. В inference truth, display и cohorts заменены объектами, вызывающими исключение при чтении. Model hashes/provenance и source snapshots сохранены.

Независимая проверка сохранённых результатов: AP через fractions.Fraction для каждого пользователя, schema, string IDs, допустимость и уникальность Top5, cold fallback, полный native порядок и scores — PASS. Это программная проверка агента, не личный расчёт студента.

Ограничения: trainprefix этих пользователей уже входил в обучение глобальных моделей и исторических ranker folds; временные исходы коррелируют. Когорта не является случайной выборкой населения. Этот аудит снижает риск объяснения результата исключительно подбором по тем же 1260 users, но не устраняет reuse данных и не даёт независимой оценки после выбора. Никаких параметров по этим меткам не подбирали. Исходы не передавались lead для выбора Bayesian конфигураций.

Артефакты: preregistration.json; raw/; processed/main/ с сохранёнными ETL hashes; inference/provenance.json; source_snapshots/; отдельные Top5/native/full scores/per-user AP в каталогах двух моделей; verification.json. Original raw/template/main protocol не изменялись.

Дополнительно: original-integrity-and-timing.json подтверждает неизменность всех пяти исходных CSV и frozen main artifacts, а также сохранение preregistration до shadow truth. post-scoring-integrity.json проверяет hashes всех score payloads/source snapshots, неизменность моделей, точное сохранение L5 на repeat positions DS и native padding.
