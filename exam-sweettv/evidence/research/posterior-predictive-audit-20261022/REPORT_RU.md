# Независимый аудит posterior predictive — 22.10.2026

**Корректность проверенных posterior arrays, saved-score decoding, AP и replay — PASS. Обнаружено ограничение численного tie-break:11 из12 600 greedy позиций выбрали не минимальный numeric ID среди математически равных marginal utilities.** Все11 выбора всё равно максимизируют точный текущий greedy increment; это не нарушение заявленной жадной цели. Сохранённые runs не изменялись. Никаких новых моделей не обучалось.

## Математика и проверенные данные

Из всех raw train-events независимо восстановлены full-airing counts для каждой user/programme пары; они совпали со всеми `histories.json`. Все реальные stop timestamps<=cutoff. Затем по histories и opportunities независимо восстановлены полные successes, train/future opportunities, pooled item rate, user rate, population rate и clipped prior. **Все arrays совпали точно** для main1260×6384 и early1260×6449. Проверены 0<=successes<=train, positivity alpha/beta через формулы, items/users ordering и protocol/payload/source/event/opportunity hashes.

Используется alpha=s+50p и beta=n-s+50(1-p). Тогда E[count_future]=future×alpha/(alpha+beta): оба полных `mean-scores.npy` совпали побитно после сохранённого float32 cast. Pooled item rate shrinks по50 pseudo-observations к population rate; индивидуальный prior масштабируется user_rate/population_rate и обрезается в(1e-9,1-1e-9). Это plug-in empirical Bayes: user history одновременно участвует в pooled prior и индивидуальном posterior. Формальная Bayesian calibration и независимость наблюдений не установлены. Все эфиры трактуются как возможные Bernoulli trials, доступность просмотра конкретному пользователю отдельно не моделируется.

Future airing count и mean original duration независимо восстановлены из hash-validated schedule inventory; counts совпали с future opportunities. Schedule builder статически использует будущие ID/channel/time/duration, исторические viewing intervals обрезает по cutoff, truth/display не читает. Метаданные содержания будущих программ не участвуют. Mean duration ties — суррогат будущих просмотренных секунд; при разных длительностях конкретных просмотренных эфиров он не воспроизводит точный seconds tie-break исходной truth.

## Greedy expected AP

Для каждого sampled set S размером<=5 вклад следующей позиции r равен

`I(candidate ∈ S) × (1 + hits_so_far) / (r × max(1, |S|))`.

Знаменатель соответствует min(5,truthsize), потому что симулятор сохраняет только максимум5 релевантных IDs; пустой set даёт0. Независимый reference использует **целочисленные marginal utilities**, умноженные на60 — общий кратный всех возможных знаменателей1…5. Ни greedy_expected_ap(), ни его float utility accumulator для этой проверки не вызываются. Проверены все1260×5 выборов каждого окна: каждый сохранённый выбор — максимум точной целочисленной utility среди ещё не выбранных кандидатов. Все sampled sets имеют допустимые индексы и уникальные nonnegative IDs. Полные marginal+ordinal greedy scores независимо восстановлены и совпали побитно.

В main8 и early3 позиций numerical float summation нарушает ожидаемый min-numeric-ID tie-break среди exact utility ties. Примеры main user_index14/rank3: выбранcandidate_index714 вместо186; early user_index239/rank2:187 вместо101. Полный список сохранён в evidence JSON. Это может изменить последующие greedy шаги между одинаково допустимыми текущими максимумами. Поэтому утверждать точную каноническую рациональную tie-политику нельзя. Это небольшой дефект воспроизводимости математического tie-break, а не leakage или неверная AP формула. Lead/root уведомлены; active source и существующие результаты не исправлялись постфактум.

Глобальный максимум ожидаемого AP не доказан: greedy оптимизирует текущий шаг. 256MC samples создают шум оценки, независимость beta/binomial draws поitems — модельное предположение. Numeric scores6…2 для выбранных позиций являются ordinal ranking markers, остальные — empirical relevance/denominator; вероятностями calibrated relevance эти native scores не являются.

## Replay и evaluation scope

Для user indices **0,1,2,630,1259** в каждом окне заново созданы RNG SeedSequence([42,j]), beta probabilities и binomial counts. Sample sets независимо получены **полной lexsort**, без production argpartition/select_sample_sets; все256 draws×5users×2окна совпали точно. Это subset MC replay, не повторная генерация всех1260users.

Дополнительно оригинальная posterior arithmetic выполнена для первых3 target rows каждого окна с poisoned truth/display. Историческая population осталась полной; schedule computation подменён ссылкой на уже проверенный frozen inventory. Все simulated sets/mean/greedy rows совпали. Это replay6rows, без нового обучения или нового экспериментального run; результаты сохранены только в папке аудита. Полный schedule builder с poisoned objects повторно не исполнялся.

Для всех4 завершённых outputs собственный Fraction AP, user/candidate schema, numeric tie sorting сохранённых score arrays, native Top100 и Top5/cold fallback совпали. Проверены **495 000 native scores/ranks** и5040 user predictions.

| Output | Main Fraction MAP@5 | Early Fraction MAP@5 |
|---|---:|---:|
| BP-mean256 | 0.35788884401364207 | 0.3780401234567901 |
| BP-greedyMAP256 | 0.35341769879734336 | 0.37661242737835876 |

3/3 existing unit tests PASS; они проверяют explicit AP increment enumeration, count/duration/numeric sample ordering и beta-binomial mean на synthetic draws. Результаты ниже текущих controls не делают неверной саму проверенную арифметику. Shadow cohort и любые его результаты не использовались.

Основной собственный скрипт: `tools/verify_posterior_predictive.py`. Рядом с этим отчётом: `main-evidence.json`, `early-evidence.json`, `raw-history-evidence.json`, `audit.log`, `tests.log`, отдельные3-row poison-replay artifacts. Shared PLAN/source/registry и исходные runs не изменялись.
