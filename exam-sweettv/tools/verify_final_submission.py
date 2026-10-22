#!/usr/bin/env python3
"""Independent CSV/catalogue/fusion audit; does not call the inference finalizer."""
import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from repro import ROOT, dump, load, sha256


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    processed = ROOT/'data/processed/elapsed-v2/final'
    protocol = load(processed/'protocol.json')
    assert all(sha256(processed/n) == h for n, h in protocol['frozen_hashes'].items())
    assert all(sha256(ROOT/'data/raw'/n) == h for n, h in protocol['input_hashes'].items())
    zone = ZoneInfo('Europe/Kyiv')
    start = int(datetime(2020, 7, 27, tzinfo=zone).timestamp())
    stop = int(datetime(2020, 10, 19, tzinfo=zone).timestamp())
    eligible, future_file_ids = set(), set()
    for name in ('export_arh_11-20-final.csv', 'export_arh_21-30-final.csv', 'export_arh_31-42-final.csv'):
        with (ROOT/'data/raw'/name).open(encoding='utf-8-sig', newline='') as f:
            for row in csv.DictReader(f):
                item = row['tv_show_id']
                if item == '0':
                    continue
                begin = int(datetime.strptime(row['start_time'], '%d.%m.%Y %H:%M:%S').replace(tzinfo=zone).timestamp())
                if begin < stop and begin+int(row['duration']) > start:
                    eligible.add(item)
                    if name == 'export_arh_31-42-final.csv':
                        future_file_ids.add(item)
    candidates = (processed/'candidates.txt').read_text().splitlines()
    assert eligible == set(candidates)
    with (ROOT/'data/raw/submission.csv').open(newline='') as f:
        template = list(csv.reader(f))
    with (a.run/'final_submission.csv').open(newline='') as f:
        rows = list(csv.reader(f))
    assert len(rows) == len(template) == 1261
    assert rows[0] == template[0] == ['user_id', 'tv_show_id']
    assert [r[0] for r in rows] == [r[0] for r in template]
    assert len({r[0] for r in rows[1:]}) == 1260
    for user, text in rows[1:]:
        items = text.split()
        assert len(items) == len(set(items)) == 5
        assert all(item != '0' and item in eligible for item in items)
    history, fallback = load(processed/'histories.json'), load(processed/'fallback.json')[:5]
    arrays = {name: np.load(a.run/(name+'.npy'), allow_pickle=False) for name in ('ppr', 'hourly', 'l4', 'ds')}
    native = np.load(a.run/'native-top100.npz', allow_pickle=False)
    numeric = [int(i) for i in candidates]
    def order(values):
        return sorted((k for k in range(len(values)) if np.isfinite(values[k])), key=lambda k: (-float(values[k]), numeric[k]))[:100]
    cold = 0
    checked = 0
    for j, (user, predicted) in enumerate(rows[1:]):
        if not history.get(user):
            cold += 1
            assert predicted.split() == fallback
            assert (native['indices'][j] == -1).all()
            continue
        core = {}
        for name in ('ppr', 'hourly'):
            row = arrays[name][j]
            ids = order(row)
            largest = max([max(float(row[k]), 0.) for k in ids]+[1e-12])
            for k in ids:
                core[k] = core.get(k, 0.)+max(float(row[k]), 0.)/largest
        core = {k: np.float32(v) for k, v in core.items()}
        core_order = sorted(core, key=lambda k: (-float(core[k]), numeric[k]))[:100]
        total = {}
        for ids, weight in [(core_order, 1.), (order(arrays['l4'][j]), .25), (order(arrays['ds'][j]), .25)]:
            for rank, k in enumerate(ids, 1):
                total[k] = total.get(k, 0.)+weight/(60+rank)
        total = {k: np.float32(v) for k, v in total.items()}
        ranked = sorted(total, key=lambda k: (-float(total[k]), numeric[k]))[:100]
        expected = [candidates[k] for k in ranked[:5]]
        expected += [i for i in fallback if i not in expected][:5-len(expected)]
        assert predicted.split() == expected
        assert ranked == native['indices'][j].tolist()
        np.testing.assert_array_equal(np.array([total[k] for k in ranked], np.float32), native['scores'][j])
        checked += len(ranked)
    chosen = {item for _, text in rows[1:] for item in text.split()}
    result = {'status': 'PASS', 'rows': 1260, 'recommendations': 6300,
        'user_ids_and_order_exact': True, 'five_unique_nonzero_eligible_ids_each': True,
        'eligible_catalogue_reconstructed_from_raw_schedule': len(eligible),
        'chosen_items_in_actual_31_42_file': chosen <= future_file_ids,
        'chosen_items_outside_31_42_file': sorted(chosen-future_file_ids, key=int),
        'independent_all_user_fusion_and_top5_exact': True,
        'native_scores_checked': checked, 'cold_users_with_identical_fallback': cold,
        'csv_sha256': sha256(a.run/'final_submission.csv'),
        'protocol_sha256': sha256(processed/'protocol.json'), 'kaggle_score': None}
    dump(a.output, result)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
