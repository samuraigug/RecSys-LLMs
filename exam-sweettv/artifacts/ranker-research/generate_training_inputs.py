"""Reproducible equivalent of the inline generator actually executed 2026-10-22.

Original execution was an inline Python command in this Codex session. This saved
script archives the same deterministic algorithm; it was not the original launcher.
Existing training inputs are never overwritten. No exam template is modified.
"""
import argparse
import csv
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from repro import ROOT,dump,sha256

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
raw=ROOT/'data/raw'
sets={'main':set(),'early':set()}
with (raw/'dataset11-30.csv').open(encoding='utf-8-sig',newline='') as f:
 for row in csv.DictReader(f):
  if row['start_time']<'2020-05-18':sets['main'].add(row['user_id'])
  if row['start_time']<'2020-04-20':sets['early'].add(row['user_id'])
for split,users in sets.items():
 folder=a.output/split
 folder.mkdir(parents=True,exist_ok=False)
 for name in ('dataset11-30.csv','export_arh_11-20-final.csv','export_arh_21-30-final.csv','export_arh_31-42-final.csv'):
  (folder/name).symlink_to(raw/name)
 with (folder/'submission.csv').open('w',encoding='utf-8',newline='') as f:
  w=csv.writer(f);w.writerow(['user_id','tv_show_id']);w.writerows((u,'0 0 0 0 0') for u in sorted(users))
 dump(folder/'TRAINING_ONLY_NOT_FOR_KAGGLE.json',{'purpose':'Additional training users only; never changes exam main/early target cohort or submission','ids_as_strings':True,'user_inventory_start_before_exclusive':'2020-05-18' if split=='main' else '2020-04-20','users':len(users),'exam_submission_source_sha256':sha256(raw/'submission.csv'),'training_only_submission_sha256':sha256(folder/'submission.csv'),'immutable_original_csv_hashes':{p.name:sha256(p) for p in folder.glob('*.csv') if p.name!='submission.csv'}})
 print(split,len(users))
