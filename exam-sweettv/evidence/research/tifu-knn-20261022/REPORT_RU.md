# TIFU-inspired user-user KNN: завершённая отрицательная проверка

В заранее ограниченных шести прогонах temporal weighting ухудшил matched control на обоих окнах. Единственная фиксированная смесь не улучшила main. Цель0.95 не достигнута; новую сетку параметров не запускали.

| Вариант | Main MAP@5 | Early MAP@5 |
|---|---:|---:|
| TIFU-control, без временного затухания | 0.3586171693 | 0.3837134622 |
| TIFU-temporal, rb.9/rg.7 | 0.3412679501 | 0.3736047567 |
| FULL-shfl4 + temporal RRF weight.05 | 0.3849771136 | 0.4058026961 |

Temporal−control: main −0.01734922, paired bootstrap95% [−0.02364237,−0.01136867]; early −0.01010871 [−0.01427039,−0.00563871]. Blend−FULL-shfl4: main −0.00028675 [−0.00102316,+0.00037561]; early +0.00008578 [−0.00088645,+0.00114728]. Bootstrap2000/seed42 описательный, условный после широкого поиска; раннее окно пересекается с основным и не является независимой репликацией. Малый early плюс не доказывает улучшение.

Это адаптация математической идеи TIFU-KNN (Hu et al., SIGIR2020), а не точное воспроизведение статьи или авторского кода. Полные просмотры свёрнуты в бинарные daily baskets; общая локальная календарная ось с пустыми днями, семидневные группы, старейшая неполная группа дополнена нулями. Показатель внутри группы0 у новейшего дня. Все train programmes, включая отсутствующие среди будущих candidates, участвуют в Euclidean user distance. Donors — все warm historical users, self исключён; пользовательские IDs остаются строками и могут превышать int64. Uniform300-neighbor mean смешан с own frequency весами.3/.7. Native candidate score скорректирован одинаковым known-schedule future/(train+50); вероятностью он не называется. Common cold/fallback, cohort, labels и candidates неизменны.

В upstream параметр group_size — число близких по длине групп, а здесь7 — число календарных дней в группе. Upstream within-decay применяется ко всей последовательности; здесь он сбрасывается внутри группы. Эти отличия и padded denominator объявлены до fit в DESIGN.md. Upstream Python не исполнялся; helper с будущими labels не переносился. Все четыре captured source SHA подтверждены.

Проверки:

- Три unit tests PASS: независимый Fraction расчёт grouping/duplicate daily baskets/decay, полный Euclidean с off-candidate dimension и giant-ID tie/self exclusion, own-neighbor mean и отказ доступа к truth/display.
- Все fit/score вызовы проходят через TrainOnlyContext, запрещающий ctx.truth и ctx.display. Raw event cutoff проверен, календарные даты вне train отвергаются.
- Все четыре сохранённые CSR vectors независимо восстановлены из raw frozen full events: maxdiff0. Все32,339,160 standalone full-score cells независимо пересчитаны: maxdiff0.
- Прямой расчёт squared Euclidean через разности sparse dictionaries для четырёх target users каждого standalone run проверяет все donor distances и выбранную границуk; self исключён во всех сохранённых neighbor arrays. Exact distance ties отдельно проверены unit test.
- Оба full-catalogue RRF blend независимо восстановлены через Python sort/rank: bit-exact все16,169,580cells. Ни top100 truncation, ни новый fit base не использовались.
- Все шесть full score arrays дают побитно те же saved native scores/indices; Top5/fallback diagnostics совпадают. Общий независимый Fraction/schema/protocol/cohort/native/cold verifier: 3main +3early ×1260 PASS.

Артефакты находятся в runs/{main,early}/{TIFU-control,TIFU-temporal,FULL-shfl4-TIFU005}. В каждомrun сохранены model-manifest.json, mapping.json, full-scores.npy; в standalone дополнительно vectors.npz и neighbors.npz. Source snapshot сохранён отдельно; SHA кода/design/events/opportunities/параметры и upstream manifest включены в provenance. Суммарный объёмruns ~202MiB; новых моделей/копий base не создано.

API: advanced_tifu_knn.register() регистрирует единственный метод tifu_knn. Фиксированные параметры temporal=False/True; blend=True допускается только с temporal=True. Runner: `.venv/bin/python tools/run_tifu_knn.py main` (аналогичноearly), повторный запуск запрещён existing-run guard. Проверки: `audit.py`, `audit_blend.py`, tests/test_tifu_knn.py. Все отрицательные результаты сохранены. PLAN/общий registry этим агентом не редактировались. Shadow-cohort outcomes не читались.
