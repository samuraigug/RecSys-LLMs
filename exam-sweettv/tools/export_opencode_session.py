#!/usr/bin/env python3
"""Save native OpenCode 1.x export stdout byte-for-byte; do not fabricate a log."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--session',required=True,help='Actual session ID from this implementation')
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if not args.session.startswith('ses_'):raise ValueError('Pass the actual OpenCode session ID')
    if args.output.exists():raise FileExistsError('Refusing to overwrite an existing native export')
    # OpenCode 1.18.30 can exit before its asynchronous pipe writer drains.
    # A regular-file descriptor preserves native stdout without editing JSON.
    with tempfile.TemporaryFile() as native_stdout:
        result=subprocess.run(['opencode','export',args.session],stdout=native_stdout,stderr=subprocess.PIPE,check=False)
        native_stdout.seek(0)
        original_bytes=native_stdout.read()
    if result.returncode!=0:
        print('OpenCode export failed. Check installed version and local help; no export was written.',file=sys.stderr)
        raise SystemExit(result.returncode)
    # Parse for verification, but write the ORIGINAL bytes, without dumping JSON again.
    data=json.loads(original_bytes)
    if not isinstance(data,dict) or data.get('info',{}).get('id')!=args.session:
        raise ValueError('Native export does not identify the requested session')
    messages=data.get('messages')
    if not isinstance(messages,list) or not messages:raise ValueError('Empty or unsupported export schema')
    tool_parts=sum(isinstance(part,dict) and part.get('type')=='tool' for message in messages for part in message.get('parts',[]))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('xb') as stream:stream.write(original_bytes)
    print(json.dumps({'status':'SAVED_NATIVE_BYTES','session_id':args.session,'messages':len(messages),
                      'tool_parts':tool_parts,'bytes':len(original_bytes),
                      'sha256':hashlib.sha256(original_bytes).hexdigest(),'path':str(args.output.resolve()),
                      'warning':'A saved export is not proof of personal verification or completeness of all claimed work.'},indent=2))


if __name__=='__main__':main()
