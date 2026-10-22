# Независимый bounded аудит weekly transition

**PASS:** направление older→next week, нормировки и временные ограничения соответствуют заявленной модели. `audit.py` не импортирует advanced_weekly_transition и не вызывает matrices/graph_from_weeks. Источник — сохранённые raw train-events NPZ, full coverage>=80%, Python sets уникальных items на user/week.

Для main и early проверены все event stop<=cutoff, protocol/source/events SHA, число активных adjacent user-week pairs (29 847/16 844), суммы каждой строки directed counts. Независимо вычислены **12 полных строк counts и graph каждого окна**, включая самые поддержанные строки и граничные индексы. Каждая пара source i/destination j получает сумму 1/(|older items|×|next items|) по пользователям, у которых i наблюдался в более старой неделе, а j в следующей. Selfedges исключены. Совпадение с сохранёнными матрицами rtol/atol1e-12.

Все ячейки symmetric counts совпали с (C+C.T)/2 точно. Для тех же12 строк проверены row support shrink20 и destination sqrt(1+weekly user-item population) correction обеих моделей. Финальная graph симметричной модели сама необязательно симметрична: последующие row/destination normalizations различны. Это корректный symmetrized-counts control.

Независимо воспроизведены все1260 source-profile rows каждого окна: бинарные user-week sets, L1 внутри недели, вес2^(-week/4), итоговая L1. Пустые недели не создают переходов через gaps. Пары полностью исторические, forecast events/metadata/truth не читаются. _scores применяет одинаковый future opportunities/(train opportunities+50) из frozen opportunities, positive-only mask и внешний общий cold/fallback. Полный score/native replay уже выполнен ведущим исследователем; здесь его проверка не дублировалась.

Точная семантика bin: age//604800, поэтому события на age=604800 относятся к bin1. Описательная фраза исходника об upper edge неоднозначна; фактическая граница однозначна. Последний stop — proxy недели завершения события, не начало просмотра и не распределение просмотра по неделям. Старейший bin может быть неполным. Эти ограничения задокументированы и не являются leakage.

Scope: все row supports, все symmetric counts, все source profiles, **12 полных directed count/graph rows на окно**, не полный независимый rebuild каждой directed ячейки. Synthetic и MAP проверки ведущего здесь не выдаются за собственные. Fit не запускался, исходники и runs не менялись. В `evidence.json` сохранены asymmetric forward/reverse примеры из независимого расчёта, scope и SHA собственного скрипта.
