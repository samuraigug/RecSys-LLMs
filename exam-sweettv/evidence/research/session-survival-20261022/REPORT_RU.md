# Empirical session-survival correction

**Устойчивого улучшения нет.** Duration-mixture bundle дал early point score0.4063210330, но main0.3823301472 ниже FULL-shfl4. Main paired95% разницы целиком отрицателен, early CI включает0. Прежний основной bundle не заменяется этим результатом.

Для одного train-session длины L и эфира оригинальной длительности d условие overlap≥0.8d допускает continuous start-offset measure `(L−0.6d)*I(L≥0.8d)`. По empirical population фактор равен сумме этих мер, делённой на суммарную длину сессий. Он переводит stationary on-mass в single-session completion mass. Raw union sessions восстановлены из verified hourly intervals, ограничены traincutoff, gap bridging0. Length-tail оценивается prefix sums sorted lengths. Future используются только channel/time/ID/original duration.

Сначала v1 использует confidence shrink факторов по числу наблюдаемых sessions: local user×channel n/(n+20), channel n/(n+100)→global. Это **эвристическая регуляризация факторов**, не literal population mixture20/100prior sessions. После независимого review root выполнена отдельная математическая ablation v2: numerator и denominator смешиваются по ожидаемой длительности pseudo-sessions. V2 equivalent добавлению20channel-prior sessions и100global-prior sessions с соответствующими population mean lengths; тест воспроизвёл это на literally expanded prior population. Числа20/100 иgap0 одинаковы; сетки параметров не было, v1source сохранён.

Поправка применяется отдельно к каждому future airing внутри hourly projection. Один matched bundle заменяет только hourly scores в FULL-shfl4, сохраняя PPR, L4, max normalization и прежние RRF weights. Это не добавление нового параметра blend.

| Вариант | Main MAP@5 | Early MAP@5 |
|---|---:|---:|
| Hourly control | 0.3697549812 | 0.3922467320 |
| V1 empirical factor confidence shrink | 0.3645187130 | 0.3856921750 |
| V2 duration-weighted population mixture | 0.3643499820 | 0.3876801924 |
| FULL-shfl4 control | 0.3852638665 | 0.4057169118 |
| FULL-shfl4 с V1 | 0.3819101149 | 0.4060700345 |
| FULL-shfl4 с V2 | 0.3823301472 | 0.4063210330 |

V2 bundle−control: main−0.00293372, post-selection paired95%[-0.00569337,−0.00027601]; early+0.00060412, CI[-0.00246777,0.00363228]. Окна пересекаются; early point maximum не считается доказанным улучшением.

Проверено:

- Геометрия offset measure на15(L,d)cases через независимое midpoint enumeration, length-weighted mixture и буквальное20prior-sessions expansion —3tests PASS.
- Factor1 profile×aggregate hourly counts воспроизводит исходный hourly score bit-exact на обоихокнах. Перегруппировка по отдельным airings отличается только float arithmetic: максимальная абсолютная разница2.31e−6 main/1.72e−6 early, проходит rtol3e−6/atol2e−7.
- Реальный empirical factor совпал с прямым sum по всем session lengths на7duration values, tol1e−14.
- Все8outputs×1260users прошли independent Fraction/schema/cohort/native/cold. Сохранённые4standalone full-score arrays воспроизводят nativeTop100 scores/order и Top5 bit-exact. Четыре bundle replay через независимое восстановление max-normalized core и full RRF также bit-exact PASS: `verify_bundles.py`, `bundle-replay.json`.

Ограничения: stationarity, user-channel length distribution без условности на hour, boundary-censored training sessions и отсутствие объединения нескольких коротких disjoint sessions, которые совместно могут дать80%. Фактор не объявляется полной calibrated probability. Содержательное следующее использование этого signal возможно как train-only feature в learned head; такого fit здесь не было.

Источники: advanced_session_survival.py/source.py v1, advanced_session_mixture.py/mixture-source.py v2, runners tools/run_session_{survival,mixture}.py. Effective constants записаны в provenance, API не предлагает произвольную grid. Runtime24–29s/standalone run,1.4–1.6s/bundle. Source, models, scores, manifests и отрицательные результаты сохранены; внешних действий не было.
