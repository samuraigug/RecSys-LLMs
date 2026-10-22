#!/usr/bin/env python3
"""Verify and unpack the original competition ZIP without overwriting data."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from zipfile import ZipFile


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    root=Path(__file__).resolve().parents[1]
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive',type=Path,default=root/'materials/originals/sweettv-tv-program-recommender.zip')
    p.add_argument('--data-dir',type=Path,default=root/'data/raw')
    args=p.parse_args()
    audit=json.loads((root/'evidence/preparation/raw-audit.json').read_text())
    expected={f['file']:f for f in audit['files']}
    if len(expected)!=5:raise ValueError('Expected a five-file data manifest')
    args.data_dir.mkdir(parents=True,exist_ok=True)
    existing=[]
    for name,meta in expected.items():
        dest=args.data_dir/name
        if dest.exists():
            if not dest.is_file() or dest.stat().st_size!=meta['bytes'] or digest(dest)!=meta['sha256']:
                raise ValueError(f'Existing file differs from official source; not overwriting: {dest}')
            existing.append(name)
    if len(existing)==len(expected):
        print(json.dumps({'status':'PASS','action':'reused verified raw files','files':existing,'data_dir':str(args.data_dir.resolve())},indent=2))
        return
    if not args.archive.is_file():
        raise FileNotFoundError('Official ZIP is not in this package. Pass --archive with the already downloaded sweettv-tv-program-recommender.zip; do not upload credentials.')
    archive_meta=json.loads((root/'materials/archive-manifest.json').read_text())
    if args.archive.stat().st_size!=archive_meta['bytes'] or digest(args.archive)!=archive_meta['sha256']:
        raise ValueError('Archive bytes/SHA-256 differ from the verified original')
    extracted=[]
    with ZipFile(args.archive) as z:
        if set(z.namelist())!=set(expected) or len(z.infolist())!=len(expected):
            raise ValueError('Unexpected archive members; no extraction performed')
        bad=z.testzip()
        if bad:raise ValueError('Archive CRC failed: '+bad)
        for name,meta in expected.items():
            if name in existing:continue
            dest=args.data_dir/name
            with tempfile.NamedTemporaryFile(prefix=name+'.',suffix='.partial',dir=args.data_dir,delete=False) as tmp:
                temporary=Path(tmp.name)
                with z.open(name) as src:shutil.copyfileobj(src,tmp)
            if temporary.stat().st_size!=meta['bytes'] or digest(temporary)!=meta['sha256']:
                raise ValueError('Extracted file hash failed; partial retained: '+str(temporary))
            if dest.exists():raise ValueError('Destination appeared during extraction; not overwriting: '+str(dest))
            temporary.rename(dest)
            extracted.append(name)
    print(json.dumps({'status':'PASS','archive_sha256':archive_meta['sha256'],'crc':'PASS','extracted':extracted,'reused':existing,'data_dir':str(args.data_dir.resolve())},indent=2))


if __name__=='__main__':main()
