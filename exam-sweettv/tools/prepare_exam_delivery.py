#!/usr/bin/env python3
"""Prepare local exam files and a public-code allowlist; never publish or submit."""
import argparse
import csv
from fractions import Fraction
import importlib.metadata
import json
from pathlib import Path
import shutil
import sys
import zipfile
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from repro import ROOT, dump, load, sha256

SELECTED = 'SHFL4DS-rrf-equal'
RUNS = ['B1-frequency', 'C4-title-dominant', 'A6-als64-reg001', 'CFC-PPR-best-replay',
        'CFC-PPRF-a02', 'SH-j05-sm25-hl28', 'SHF-PPR-recency-max-1',
        'L4-logloss-allusers300', 'L5-logloss-allusers-strong300', 'DS-nonrepeat-binary300', SELECTED]
NAMES = {'B1-frequency': 'Popularity: число полных просмотров', 'C4-title-dominant': 'CBF: метаданные и названия',
         'A6-als64-reg001': 'MF: implicit ALS', 'CFC-PPR-best-replay': 'PPR: отдельный лучший вариант',
         'CFC-PPRF-a02': 'PPR: компонент финального гибрида', 'SH-j05-sm25-hl28': 'Расписание и привычки просмотра',
         'SHF-PPR-recency-max-1': 'PPR + расписание', 'L4-logloss-allusers300': 'CatBoost L4',
         'L5-logloss-allusers-strong300': 'CatBoost L5', 'DS-nonrepeat-binary300': 'L5 + эксперт новых программ',
         SELECTED: 'Выбранный гибрид SHFL4DS'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--delivery', type=Path, required=True)
    p.add_argument('--repository-folder', type=Path, required=True)
    a = p.parse_args()
    repo, delivery = a.repository_folder, a.delivery
    repo.mkdir(parents=True, exist_ok=False)
    materials = delivery/'report-materials'
    materials.mkdir(parents=True, exist_ok=False)
    run = ROOT/'artifacts/delivery-20261022/final'
    audit = ROOT/'artifacts/delivery-20261022/final-independent-check-result.json'
    assert load(audit)['status'] == 'PASS'
    assert load(ROOT/'artifacts/delivery-20261022/tests.json')['status'] == 'PASS'
    def copy(source, target):
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        assert sha256(source) == sha256(target)
    for source in ROOT.glob('*.py'):
        copy(source, repo/source.name)
    for directory in ('tests', 'tools'):
        for source in (ROOT/directory).rglob('*.py'):
            copy(source, repo/source.relative_to(ROOT))
    for source in (ROOT/'artifacts/ranker-research').glob('*.py'):
        copy(source, repo/source.relative_to(ROOT))
    allowed = {'record.json', 'top5.json', 'native-top100.npz', 'per-user-ap.csv', 'two-user-examples.json',
               'fallback-diagnostics.json', 'catboost-ranker.cbm', 'discovery.cbm',
               'ranker-training-provenance.json', 'training-provenance.json', 'ranker-feature-importance.json'}
    for split in ('main', 'early'):
        for name in RUNS:
            for source in (ROOT/'artifacts/runs'/split/name).glob('*'):
                if source.is_file() and source.name in allowed:
                    copy(source, repo/source.relative_to(ROOT))
    for name in ('search-summary-20261022-v4.csv', 'search-summary-20261022-v4.json'):
        copy(ROOT/'artifacts/runs'/name, repo/'evidence'/name)
    for source in (ROOT/'artifacts/delivery-20261022').glob('*'):
        if source.is_file():
            copy(source, repo/'evidence/delivery'/source.name)
    for source in run.iterdir():
        if source.is_file() and source.suffix == '.json':
            copy(source, repo/'evidence/final'/source.name)
    copy(ROOT/'artifacts/delivery-20261022/main-replay/main-replay.json', repo/'evidence/main-replay.json')
    for split in ('main', 'early', 'final'):
        copy(ROOT/'data/processed/elapsed-v2'/split/'protocol.json', repo/'evidence/protocols'/f'{split}.json')
    for source in (ROOT/'research').glob('*-20261022/REPORT_RU.md'):
        copy(source, repo/'evidence/research'/source.parent.name/source.name)
    copy(ROOT/'docs/SOURCES.json', materials/'source-index-preparation.json')
    copy(ROOT/'research/requirements-advanced-lock-20261022-v3.txt', repo/'evidence/original-research-environment.txt')
    copy(run/'final_submission.csv', repo/'submissions/final_submission.csv')
    copy(run/'final_submission.csv', delivery/'kaggle/final_submission.csv')
    copy(audit, delivery/'kaggle/verification.json')
    packages = ('numpy', 'scipy', 'scikit-learn', 'implicit', 'catboost', 'threadpoolctl', 'duckdb')
    (repo/'requirements.txt').write_text(''.join(f'{name}=={importlib.metadata.version(name)}\n' for name in packages))
    (repo/'.gitignore').write_text('.venv/\n__pycache__/\n*.pyc\ndata/\nprivate/\n*session.json\n*session.jsonl\nreproduced-final/\nartifacts/runs/*cache*/\n')
    rows=[]
    for name in RUNS:
        main_record=load(ROOT/'artifacts/runs/main'/name/'record.json')
        m=main_record['metrics']
        early_path=ROOT/'artifacts/runs/early'/name/'record.json'
        early=load(early_path)['metrics']['map_at_5_main'] if early_path.exists() else None
        rows.append({'run_id':name,'method':NAMES[name], 'main_MAP_at_5_1238_users':m['map_at_5_main'],
                     'all_template_MAP_at_5_1260_users':m['map_at_5_all_template'],
                     'early_MAP_at_5_1224_users':early, 'runtime_seconds':main_record['runtime_seconds']})
    with (materials/'validation-comparison.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    copy(materials/'validation-comparison.csv', repo/'results/validation-comparison.csv')
    examples=[]
    for name in RUNS:
        for example in load(ROOT/'artifacts/runs/main'/name/'two-user-examples.json'):
            truth={i['id'] for i in example['truth']}
            for rank,item in enumerate(example['top5'],1):
                examples.append({'run_id':name,'user_id':example['user_id'],'user_type':example['selection'],
                    'rank':rank,'programme_id':item['id'],'title':item['title'],'category':item['category'],
                    'relevant':item['id'] in truth,'AP_at_5':example['ap_at_5']})
    with (materials/'two-user-examples.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(examples[0]));w.writeheader();w.writerows(examples)
    copy(materials/'two-user-examples.csv', repo/'results/two-user-examples.csv')
    chosen=load(ROOT/'artifacts/runs/main'/SELECTED/'two-user-examples.json')
    copy(ROOT/'artifacts/runs/main'/SELECTED/'two-user-examples.json', materials/'selected-two-users-with-truth.json')
    e=chosen[0]; truth={i['id'] for i in e['truth']};hits=0; terms=[]; table=[]
    for rank,item in enumerate(e['top5'],1):
        relevant=item['id'] in truth
        hits+=relevant
        term=Fraction(hits,rank) if relevant else Fraction(0)
        terms.append(term)
        table.append(f"| {rank} | {item['id']} | {'да' if relevant else 'нет'} | {term} |")
    ap=sum(terms)/min(5,len(truth))
    (materials/'personal-check-example.md').write_text(
        '# Пример для личной проверки AP@5\n\nЭто автоматически подготовленный разбор реальной выдачи. Он не означает, что студент уже выполнил личную проверку.\n\n'
        f"Пользователь `{e['user_id']}`, стратегия `{SELECTED}`. Релевантные ID: {', '.join(i['id'] for i in e['truth'])}.\n\n"
        '| Ранг | ID | Попадание | Слагаемое precision |\n|---|---|---|---|\n'+'\n'.join(table)+
        f'\n\nAP@5 = сумма слагаемых / min(5, число релевантных) = {ap} = {float(ap):.10f}. '
        'Это AP одного пользователя, не средний MAP всей выборки.\n\n'
        'Пересчитай попадания и дробь самостоятельно, сверь с JSON и таблицей и опиши своими словами, что проверил. '
        'Если проверка не выполнена, в AI Usage Disclosure нельзя писать, что ты её выполнил.\n',encoding='utf-8')
    table='\n'.join(f"| {r['method']} | {r['main_MAP_at_5_1238_users']:.8f} | {r['all_template_MAP_at_5_1260_users']:.8f} |" for r in rows)
    (materials/'facts-for-report.md').write_text('''# Факты и материалы для самостоятельного PDF

Студент: Аветисян Гурген Мартикович. Группа AVITO. Дедлайн 25.10.2026, 19:00 MSK.
PDF готовит студент. Этот файл — источник проверенных чисел и ограничений, не готовый отчёт.

## Результаты

Все числа ниже локальные. Официальных Kaggle public/private scores пока нет.
Основная валидация: история 09.03–18.05.2020, прогноз 18.05–27.07.2020; границы справа исключающие.
Основной MAP: 1238 пользователей с непустой truth, включая 14 cold users.
По всем 1260 строкам шаблона 22 пустые truth учитываются как AP=0. Когорты не смешивать.

| Метод | MAP@5, основная когорта | MAP@5, все 1260 |
|---|---|---|
'''+table+'''

Для того же выбранного SHFL4DS раннее окно даёт 0.4049993192 (1224 пользователя).
Окна пересекаются и не являются независимыми. На main проверено много конфигураций;
оценка победителя после подбора может быть оптимистичной. Цель 0.95 не достигнута.
Полная сводка содержит 506 записей, 504 успешных запуска, включая отрицательные эксперименты и две реальные ошибки.
Малый выигрыш SHFL4DS над FULL-shfl4 не доказывает статистически устойчивое превосходство.

## Протокол и итоговый прогноз

Join: один канал и пересечение полуоткрытых интервалов; просмотры объединяются и обрезаются до cutoff до подсчёта секунд.
Полный просмотр: суммарное покрытие не менее 80% исходной длительности эфира.
Truth: Top-5 программ по числу полных эфиров, затем секундам, затем числовому ID.
Это реконструкция правила организатора, точный скрытый код разметки недоступен.
ID=0 исключён; user IDs сохраняются строками. Europe/Kyiv выведен из наблюдаемых длительностей, а не указан организатором.
Повторные рекомендации разрешены для всех методов. Cold fallback одинаков: популярность по числу уникальных зрителей в train.
Будущее расписание используется только для ID, канала, времени и длительности. Будущие просмотры и текстовые метаданные не являются признаками.

Состав SHFL4DS: PPR + почасовые привычки просмотра, CatBoost L4, L5 с отдельным экспертом ранее не просмотренных программ.
Внешняя RRF: веса 1 / 0.25 / 0.25, k0=60; сохранены Top-100 ограничения исходной реализации.
PPR: alpha=.2, neighbors=200, weak_weight=1, half_life_days=56, коррекция эфирных возможностей.
Почасовые привычки: joint_weight=.5, neighbor_smooth=.25, half_life_days=28.
Деревья: 300 iterations, depth=6, learning_rate=.05, l2_leaf_reg=10, seed=42, один поток.
Точные параметры, даты исторических обучающих окон и хеши — в records/provenance репозитория.

Финальный CSV рассчитан на истории до 27.07.2020 для окна 27.07–19.10.2020: 1260 пользователей, 6649 допустимых кандидатов, 4 cold users.
Граф и признаки пересчитаны по полной истории. Supervised деревья оставлены из основной валидации;
нового обучения этих деревьев при финальной сборке не было. Не описывать это как полный supervised refit на неделях 11–30.
Все 6300 рекомендаций присутствуют и в фактическом файле расписания 31–42. Kaggle score заранее неизвестен.

## Примеры, анализ и проверки

`two-user-examples.csv` содержит упорядоченные Top-5 всех сравниваемых методов для одной и той же пары: активный зритель и cold user.
Названия и категории предназначены для описания; разные эфиры одного ID могут иметь разные названия.
В JSON выбранного метода сохранены история и truth. У активного пользователя итоговый AP=1/15: это реальный неудачный пример,
показывающий ограничение склонности к повторениям. Для cold user общий fallback даёт одинаковую выдачу.
Нельзя делать вывод о понравившейся программе или о том, что человек никогда её не видел, только по отсутствию просмотра в этом логе.

Проверки агента: 91/91 unittest PASS; точное воспроизведение выбранного main Top-5/Top-100;
независимая проверка всего финального CSV, каталога по raw schedule, 125600 native scores и cold fallback.
Это автоматические проверки, не личная работа студента. Для личного расчёта см. `personal-check-example.md`.
Пример реального инженерного сбоя: user-gate main run исчерпал диск при записи массива;
сохранённую модель восстановили без refit в новом run, ошибочный оригинальный run сохранён. Вариант не выбран из-за меньшего качества.

## AI disclosure и незакрытые пункты

Использовался Codex для анализа данных, реализации, экспериментов, проверок и подготовки материалов;
часть исследовательских и проверочных подзадач выполнялась агентами gpt-6-astra.
GitHub skills/Arena и научные источники использовались как подходы к организации поиска; MAP измерялся кодом на данных.
Экспорт относится к текущему чату Codex. Не приписывать ему автоматически отдельные старые чаты или полные внутренние журналы дочерних агентов.
В PDF отдельно укажи, что действительно проверил лично, а что принял без собственной проверки.

Нужно добавить: доступную GitHub-ссылку после публикации, настоящий скриншот выбранной Kaggle отправки с Public и Private,
ФИО и Kaggle-имя, собственный анализ выбора стратегии и личную проверку. В исходных guidelines противоречие 5/6 страниц;
отчёт до 5 страниц без списка литературы укладывается в оба ограничения. Источники нужно проверить перед цитированием.
В LMS отправляются ровно два файла: твой PDF и неизменённый нативный экспорт сессии. CSV и код остаются в GitHub/Kaggle.
''',encoding='utf-8')
    expected_hash=sha256(run/'final_submission.csv')
    (repo/'README.md').write_text(f'''# SWEET.TV — экзамен LLM4Rec, AVITO

Аветисян Гурген Мартикович. Текущая локальная стратегия: `{SELECTED}`.
MAP@5 на основной валидации: **0.3858360258** (1238 пользователей); по всем 1260: **0.3790992063**.
Это не Kaggle score. Файл пока не отправлен; публичный и приватный результаты неизвестны.

## Файлы

- `submissions/final_submission.csv` — единственный подготовленный финальный прогноз для Kaggle.
- `results/validation-comparison.csv` — popularity, CBF, MF, PPR, компоненты и гибрид на одном holdout.
- `results/two-user-examples.csv` — одна и та же пара пользователей для всех методов.
- `artifacts/runs/main/` — конфигурации, веса моделей, provenance и реальные результаты.
- `evidence/` — протоколы, независимые проверки, полный реестр поиска и отрицательные ветки.
- `final_submission.py` — применение сохранённых моделей и пересчёт истории до финального cutoff.

SHA-256 финального CSV: `{expected_hash}`.
Проверены 1260 строк, порядок и строковые ID шаблона, по 5 уникальных ненулевых программ из расписания 31–42.
Сырые данные соревнования и приватный AI export не включены в репозиторий.

## Воспроизведение финального прогноза

Проверенное окружение: Python 3.12, версии зависимостей в `requirements.txt`.
Сначала принять Rules своим аккаунтом Kaggle и поместить пять оригинальных CSV в `data/raw/`:
`dataset11-30.csv`, три `export_arh_*-final.csv`, исходный шаблон `submission.csv`.
Ожидаемые SHA каждого файла записаны в `evidence/protocols/final.json`.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python etl.py --splits final
.venv/bin/python final_submission.py --output reproduced-final
.venv/bin/python tools/verify_final_submission.py --run reproduced-final --output reproduced-final-check.json
```

На исходном окружении результирующий CSV должен совпасть с сохранённым SHA. На иной платформе сначала проверить фактическое совпадение.
Команда применяет сохранённые supervised деревья; граф, привычки и признаки пересчитываются по истории до 27.07.2020.
Это **не** повторное обучение CatBoost на всей истории. Историческое обучение и параметры сохранены в provenance;
исходники версий feature engine и генератор обучающих окон находятся в `artifacts/ranker-research/`.
Полное исследовательское окружение сохранено как свидетельство в `evidence/original-research-environment.txt`;
его LightFM file-URL относится к исходной машине и не нужен для финального прогнозирования.

## Протокол

Основной train: 09.03–18.05.2020; validation: 18.05–27.07.2020, правая граница исключающая.
Join по каналу и пересечению интервалов; перекрывающиеся просмотры объединены.
Полный просмотр — >=80% эфирной длительности; truth — пять наиболее часто полностью просмотренных программ.
Знаменатель AP@5: min(5, число релевантных ID). Повторы разрешены; cold fallback общий для всех методов.
Будущая информация ограничена расписанием ID/channel/time/duration; нет future viewing или future content features.
Основной holdout использовался для выбора среди многих вариантов, поэтому winner score может быть оптимистичным.
Ранний holdout пересекается с основным. Никаких online A/B или revenue результатов не заявляется.

## Проверки и сдача

На рабочей машине 91/91 tests PASS. `evidence/main-replay.json` подтверждает точное совпадение выбранного исторического pipeline.
Независимый финальный аудит подтверждает каталог, все выдачи и 125600 native scores; журнал — `evidence/delivery/`.
Личные проверки студента не заменяются этими автоматическими результатами.
Официальный Kaggle score и скриншот добавляются после реальной Late Submission. Для оценки по заданию важен returned public score внутри группы;
старый leaderboard заморожен. В PDF следует сохранить обе колонки Public/Private.
''',encoding='utf-8')
    (delivery/'kaggle/submission-description.txt').write_text('Аветисян Гурген Мартикович — <вставить точное Kaggle-имя> — FINAL SHFL4DS\n',encoding='utf-8')
    (delivery/'README_СДАЧА.md').write_text(f'''# Комплект экзамена AVITO

Студент: Аветисян Гурген Мартикович. Дедлайн: 25 октября 2026, 19:00 МСК.

1. **Kaggle:** загрузить `kaggle/final_submission.csv` через Late Submission. В описании указать ФИО и собственное Kaggle-имя.
   Дождаться COMPLETE, сохранить ID попытки и Public/Private, назначить лучшую Selected и сделать настоящий скриншот строки.
   По рекомендации преподавателя планировать не более примерно пяти попыток в день; фактический лимит показывает интерфейс.
2. **GitHub:** локальная папка подготовлена в `{repo}` для репозитория samuraigug/RecSys-LLMs.
   Внешняя публикация ещё не выполнена. Опубликовать только `exam-sweettv/`; существующий чужой/старый `session.json` не добавлять.
   В репозитории должен остаться байт-в-байт тот CSV, который получил показанный score.
3. **PDF готовишь ты:** использовать `report-materials/`, добавить ссылку на опубликованный код, скриншот Kaggle,
   свой бизнес-анализ и реально выполненную личную проверку. Итоговая локальная метрика 0.3858360258, не 0.95.
4. **Smart LMS:** ровно твой PDF и `lms/avetisyan_gurgen_exam_session.json`. Архив с кодом, CSV и эти инструкции в LMS не добавлять.
   Экспорт нативный и не редактируется. Если после него будет существенная новая работа, нужно получить свежий экспорт.

SHA-256 CSV: `{expected_hash}`.
На момент сборки Kaggle-отправки, официального score, скриншота, GitHub-публикации и загрузки в LMS нет.
Экспорт текущего чата не заменяет журналы отдельных старых чатов; раскрытие AI должно соответствовать фактическому содержимому.
''',encoding='utf-8')
    files={str(f.relative_to(repo)):sha256(f) for f in repo.rglob('*') if f.is_file()}
    dump(repo/'FILE_MANIFEST.json', {'scope':'public code package; excludes raw data and private session', 'files':files})
    archive=delivery/'github_exam_sweettv.zip'
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
        for f in sorted(repo.rglob('*')):
            if f.is_file(): z.write(f,Path('exam-sweettv')/f.relative_to(repo))
    dump(delivery/'package-provenance.json',{'repository_folder':str(repo),'repository_url':'https://github.com/samuraigug/RecSys-LLMs',
         'repository_published':False,'kaggle_submitted':False,'csv_sha256':expected_hash,
         'github_archive_sha256':sha256(archive),'public_files':len(files)+1,
         'private_session_in_public_package':False,'pdf_prepared_by':'student'})
    print(json.dumps({'status':'LOCAL_PACKAGE_PREPARED','files':len(files)+1,'archive':str(archive),'csv_sha256':expected_hash}))


if __name__ == '__main__':
    main()
