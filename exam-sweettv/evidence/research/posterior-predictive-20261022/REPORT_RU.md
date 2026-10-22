# Bayesian posterior predictive counts и greedy expected AP

**Прямое учёты uncertainty и AP в этой модели не улучшило реальный MAP.** На обоихокнах posterior mean counts лучше greedy expectedAP, оба уступают FULL-shfl4. Новых MC counts/seed/prior grids и blends не запускали.

| Вариант | Main MAP@5 | Early MAP@5 |
|---|---:|---:|
| Posterior mean counts | 0.3578888440 | 0.3780401235 |
| Greedy expectedAP,256samples | 0.3534176988 | 0.3766124274 |
| FULL-shfl4 control | 0.3852638665 | 0.4057169118 |

Train-only prior: personal overall full-watches-per-airing rate×pooled programme full-per-airing rate/population overall rate, сnumeric clipping1e−9. Programme pooled rate shrunk50global pseudo-observations; ни hourlyscores, ни cosine не интерпретируются какprobabilities. Далее Beta alpha=s+50p, beta=n−s+50(1−p), s=личныеfullairings,n=trainopportunities. Из256Beta draws получаются Binomial counts поизвестнымfutureopportunities. Успехи проверены0≤s≤n.

Каждыйsimulated relevantset — top5 positivecount programmes; ties: count×meanfutureoriginalduration, numericID. Это proxy futureseconds, не настоящиебудущиепросмотры. Greedy выбирает позициюr по finite-sample expected increment `E[I_i*(1+selected_hits)/(r*min(5,|T|))]`, emptysetsдают0. Он оптимизирует эту суррогатную модель распределения, не объявляется глобально оптимальным ранжированием и не заменяет реальнуюvalidationtruth. Posterior также приближён: programme outcomes условно независимы, stationarity, pooled empirical-Bayes prior повторно использует user history.

Всеposterior parameters, sample top5 sets иscores сохранены, seed42сdeterministic user-position stream. Runtime123.74s main/122.72s early; greedy replay<0.3s. Native greedy scores кодируютorderrank6..2 первыхпяти, остальные — empirical normalizedrelevance; это не calibratedprobabilities.

Проверки:

-3unit tests PASS: Beta-Binomial mean100k draws, exact count/duration/numeric sample sorting, greedy increments против explicit AP enumeration, включаяemptytruth.
-Own verifier:4outputs×1260Fraction/schema/native/cold, всеposterior mean иgreedy scores через независимую sparse-incidence формулу exact;3users/window×256draws воспроизведены полнымlexsort, а не model packed-key argpartition.
-Независимый Astra reviewer восстановил всеposterior arrays/means изhistories/opportunities,316413main+182536earlyfullcounts иtrainingtimestamps;12,600greedychoices являются exact integer-utility maxima;495000native scores/5040Top5+fallbackPASS. Independent MC fullsort5users/window иoriginal-source poisonedtruth/display3rows/windowPASS. Отчёт `research/posterior-predictive-audit-20261022/REPORT_RU.md`.

Численнаяоговорка сохранена: floatbincount выбрал8main+3early greedypositions не поminimalnumericID среди *математическиравных* rationalutility maxima. Всеони остаются точнымиmaximizers; protocolnative tie-break не нарушен, посколькуordinal scores различны. Это не объявляется exact rational tie-breaking; sourcev1/outputнепереписаны радиполучениядругогоscore.

Исходникadvanced_posterior_predictive.py иsource.py snapshot; runner tools/run_posterior_predictive.py. Цель0.95недостигнута, оценочнаяcohort/candidates/truthнеизменны. Этичислалокальные, непубликация/Kaggle.
