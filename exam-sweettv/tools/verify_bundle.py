#!/usr/bin/env python3
"""Check the frozen handoff manifest. Run before changing the packaged project."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    root=Path(__file__).resolve().parents[1]
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--profile',choices=['auto','full','core'],default='auto')
    args=p.parse_args()
    manifest=json.loads((root/'FILE_MANIFEST.json').read_text())
    full=all((root/f['path']).is_file() for f in manifest['files'] if f.get('full_only'))
    profile=('full' if full else 'core') if args.profile=='auto' else args.profile
    checked,optional_missing,failures=0,[],[]
    for item in manifest['files']:
        path=root/item['path']
        if not path.resolve().is_relative_to(root):
            failures.append({'path':item['path'],'error':'path outside package'});continue
        if not path.is_file():
            if profile=='core' and item.get('full_only'):optional_missing.append(item['path']);continue
            failures.append({'path':item['path'],'error':'missing'});continue
        with path.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
        if path.stat().st_size!=item['bytes'] or digest!=item['sha256']:
            failures.append({'path':item['path'],'error':'size/hash mismatch'})
        checked+=1
    out={'status':'PASS' if not failures else 'FAIL','profile':profile,'checked_files':checked,
         'optional_missing':optional_missing,'failures':failures,'snapshot_id':manifest['snapshot_id'],
         'note':'This verifies the starting bundle, not a completed exam or a final implementation.'}
    print(json.dumps(out,indent=2,ensure_ascii=False))
    raise SystemExit(0 if not failures else 1)


if __name__=='__main__':main()
