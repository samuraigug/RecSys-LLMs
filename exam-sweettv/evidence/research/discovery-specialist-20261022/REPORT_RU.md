# Специализированный supervised discovery head

Получен небольшой новый максимум main **0.3858360258**, но устойчивое превосходство над FULL-shfl4 не установлено. Цель0.95 остаётся недостигнутой.

Гипотеза: общий binary ranker слишком много learning capacity тратит на повторные программы. Отдельный CatBoost300/depth6 обучен только на historical кандидатах без личного full просмотра, сохраняя ту же top-five inclusion label и frozen60features L5. Historical user-fold groups с нулём discovery positives НЕ исключены: main754, early480 таких групп. Это важно для отрицательных примеров и вероятностного масштаба.

Sampling: все discovery positives, deterministic retrieved negatives внутри nonrepeat domain и30 uniform remaining negatives; последние получают inverse-inclusion weight. Weighted mass каждого user-fold равна размеру nonrepeat каталога. Обучение main1,851,354rows/22,535positives, early1,724,822rows/21,563positives. Labels заканчиваются до outercutoff; evaltruth не участвует. Main/early evaluation cohort неизменны.

Piecewise score сохраняет исходный L5 на full repeats и заменяет его discovery probability. Второй заранее заданный вариант смешивает только nonrepeat logits с весом0.5. Это общий binary target и conditional sampling domain, а не normalized mode decoder; calibration между heads не гарантируется. Никаких принудительных квот или gate по evaltruth. Третий bundle заменяет только L5 на specialist в старой смеси SHFL45, сохраняя веса core1+L4.25+specialist.25; он добавлен после малого early standalone gain, без weight search.

| Метод | Main MAP@5 | Early MAP@5 |
|---|---:|---:|
| L5 control | 0.3852286394 | 0.4042765069 |
| Specialist piecewise | 0.3841576916 | 0.4045338598 |
| Nonrepeat logit0.5 | 0.3848792856 | 0.4043125908 |
| SHFL45 прежний | 0.3856616855 | 0.4046840959 |
| SHFL4DS-rrf-equal | **0.3858360258** | **0.4049993192** |
| FULL-shfl4 | 0.3852638665 | 0.4057169118 |

Замена L5 в смеси улучшает point estimates против SHFL45 на0.00017434/0.00031522. Против FULL-shfl4 новая смесь даёт main+0.00057216, paired bootstrap95%[-0.00078445,0.00195387]; early−0.00071759, CI[-0.00211469,0.00069312]. Оба CI включают0; окна пересекаются, сравнения post-selection. Это не основание объявлять новый финальный lock.

Discovery bottleneck существенно не решён. Standalone L5 на main попадает в152/2670 nonrepeat truth pairs, specialist147/2670; early217/2691→228/2691. В смеси против FULL-shfl4:125→128 main,189→191 early. Разница мала; изменение общего числа новых слотов не означает качественный discovery.

Проверки `verify.py` PASS:6outputs×1260users независимый Fraction/schema/cohort/native/cold; сохранённые piecewise scores воспроизводят Top5, nonrepeat mask независимо совпадает с frozen history, repeat scores bit-exact с verified baseline L5 cache; все conditional sampling masses сверены. Unit test проверяет negative-only user, nonrepeat domain и weight1 positives. Независимый astra_verify завершил saved-model replay PASS: 8 012 703 main + 8 106 508 early nonrepeat scores побитно совпали после float32 cast, evaltruth/display были poisoned; повторные scores, native100, Top5, cold, temporal/source/IP checks PASS. Полный отдельный audit: `research/discovery-specialist-audit-20261022/REPORT_RU.md`. Training X/y/w hashes не пересоздавались независимым аудитором; baseline L5 model в этом аудите повторно не восстанавливался.

Исходник `advanced_discovery_specialist.py`, snapshot `source.py`; runner `tools/run_discovery_specialist.py`. Для текущего фиксированного эксперимента config seed42/iterations300 hardcoded и совпадает с параметрами runs; произвольное изменение этих API parameters не поддержано и не проверено. Реальный effective config и dependency/cache hashes в training-provenance. Fit+score191.38s main/168.39s early. Внешних публикаций/submit не было.
