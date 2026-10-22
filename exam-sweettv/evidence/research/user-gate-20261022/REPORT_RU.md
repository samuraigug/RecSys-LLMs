# Пользовательский выбор unsupervised эксперта по ожидаемому AP

**Gate не улучшил контроль ни на одном окне.** После обучения на исторических AP он чаще выбирает исходный core; редкие переключения на PPR или hourly не окупили ошибки выбора. Сетка параметров не запускалась.

Дизайн сохранён до fit в DESIGN_RU.md. Три фиксированных эксперта — core=max-normalized(PPR+hourly), PPR и hourly — заново строятся на каждом historical prefix. Gate предсказывает их три per-user AP через CatBoost MultiRMSE300/depth4/seed42. Используются 28 признаков прошлой истории, активности, каналов, horizon и концентрации/пересечения прогнозов экспертов. User/item IDs не являются признаками. Main:8607 обучающих строк изf3/f4; early:7812 изf2/f5.

Критичная граница утечки соблюдена: L4/DS не входят в historical AP targets или признаки. После выбора эксперта допускается только фиксированная outer смесь gate1+L4.5 с прежними весами FULL-shfl4. Ни main/early truth, ни shadow-audit labels не участвуют в обучении gate. Empty-truth/cold historical users исключены из fit; в оценке общий cold fallback сохранён.

| Вариант | Main MAP@5 | Early MAP@5 |
|---|---:|---:|
| Fixed core control | 0.3818517771 | 0.4023481754 |
| Learned expert gate | 0.3810911416 | 0.4006824165 |
| FULL-shfl4 control | 0.3852638665 | 0.4057169118 |
| Gate1+L4.5 | 0.3845705439 | 0.4048763163 |

Выборы на всех1260 строках: main core1124/PPR63/hourly73; early core928/PPR27/hourly305. Cold rows всё равно получают общий fallback. Разница между окнами согласуется с ограничением коротких historical prefixes и horizon shift; она не доказывает причинное объяснение ошибок.

Main исходный run `UG-AP3-depth4-300` действительно завершился **FAILED** после fit: диск был заполнен, запись gate-scores.npy остановилась после128-byte header. Сохранены model, training arrays, provenance, failure record и повреждённый файл. Root освободил место объединением byte-identical immutable payloads через hardlinks, не меняя пути/байты/SHA. В новом run `UG-AP3-depth4-300-recovered` модель загружена из failed bundle **без refit**; отдельный модуль advanced_user_gate_recovery.py проверяет происхождение и hashes. Результаты не записаны поверх failed directory. Recovery занял1.56s, новый main bundle1.20s; early fit+score11.91s.

Routing unit test PASS: выбранная строка эксперта сохраняется полностью, при равных predicted AP приоритет core, nonfinite predictions отвергаются. Saved-model reload exact. Независимый Astra audit PASS: все16419historical rows X28 совпали при poisoned truth/display;49257AP targets восстановлены через Fraction из saved PPR states и verified hourly cache. Loadedmodel2520×3 predictions/choices/full score rows exact;4outputs495000native scores/Top5/Fraction/cold PASS. Recovered main model/provenance byte-identical failed origin, без refit. Отчёт: `research/user-gate-audit-20261022/REPORT_RU.md`.

Источник advanced_user_gate.py и source.py snapshot не менялся во время fit/audit. Recovery имеет отдельный source snapshot. Run parameters отражают реальные depth4/MultiRMSE, а не старые параметры L5. Проектные артефакты после storage dedup остаются immutable. Цель0.95 и финальный selection не достигнуты; результаты локальные, внешних действий не было.
