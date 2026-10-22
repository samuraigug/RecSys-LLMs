# Направленные переходы между историческими неделями

**Восемь проверенных outputs не улучшили контроль.** Новая гипотеза была проверена после specialist: направленная структура item→next-week item могла помочь personal-new/global-known discovery, которую обычная co-view PPR почти не находит.

Train-only full events распределяются по семидневным elapsed bins назад от outercutoff, на основе last_stop. Самая старая неделя может быть неполной (DST не превращается в наивные timestamps). Для каждой пары соседних исторических недель считаем L1(source user-week).T×L1(next user-week), что ограничивает вклад активных пользователей. Selfedges обнуляются. Directed graph сравнивается с matched control (C+C.T)/2; затем обе матрицы получают одинаковые row shrinkage20, correction по sqrt(1+global weekly item popularity), одинаковый source profile half-life28days и future/train exposure correction prior50. Эти scores — сигналы ранжирования, не вероятности. Future views/metadata не используются.

Main:10bins,29,847active user-week pairs,317,611directed edges. Early:6bins,16,844pairs,126,720edges. Пустые user-week gaps не заполняются переходами через несколько недель.

| Вариант | Main MAP@5 | Early MAP@5 |
|---|---:|---:|
| Directed weekly | 0.2849499641 | 0.3009311456 |
| Matched symmetric | 0.2885835128 | 0.3063975127 |
| FULL-shfl4 + directed RRF0.05 | 0.3844668821 | 0.4049015069 |
| FULL-shfl4 + symmetric RRF0.05 | 0.3837156704 | 0.4045440723 |
| FULL-shfl4 control | 0.3852638665 | 0.4057169118 |

Направленность не улучшила standalone ranking, оба заранее заданных blends уступают контролю на обоих окнах. Weight/lag/shrinkage grid не выполнялся. Это отрицательный результат конкретного weekly-transition варианта, не доказательство бесполезности всех sequential models. Цель0.95 и финальный selection остаются открытыми.

`verify.py`:8outputs×1260users Fraction/schema/native/cold PASS. Четыре saved graph/profile states независимо восстановили все native Top100 scores/order и Top5 bit-exact. Synthetic tests проверили source→destination orientation, symmetric control, L1 normalization heavy users и отсутствие selfedge —2/2PASS. Независимый astra_verify: raw train-events→Python sets, без импорта реализации, восстановил все row supports, symmetric counts allcells, source profiles2520rows и12 полных directed count/graph rows на окно (rtol/atol1e-12), temporal/source hashes PASS. Это bounded review, не полный rebuild всех directed counts. Отчёт `research/weekly-transition-audit-20261022/REPORT_RU.md`.

Код advanced_weekly_transition.py (source.py snapshot), runner tools/run_weekly_transition.py; method weekly_transition, register() без аргументов. Каждый fit+score занял меньше секунды. Параметры фиксированы в реализации, кроме symmetric; все фактические значения записаны в transition-provenance. Никаких evaltruth-driven features, изменения cohort/candidates/fallback или внешних действий.
