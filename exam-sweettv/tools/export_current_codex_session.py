#!/usr/bin/env python3
"""Retain the exact native Codex thread/read JSON response, without rewriting it."""
import argparse
import hashlib
import json
from pathlib import Path
import queue
import subprocess
import threading


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--thread', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--codex', required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    version = subprocess.check_output([args.codex, '--version'], text=True).strip()
    messages = queue.Queue()
    with args.output.with_suffix('.stderr.log').open('xb') as errors:
        process = subprocess.Popen([args.codex, 'app-server', '--listen', 'stdio://'],
                                   stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=errors)
        def reader():
            for raw in process.stdout:
                messages.put(raw)
            messages.put(None)
        threading.Thread(target=reader, daemon=True).start()
        def send(value):
            process.stdin.write((json.dumps(value)+'\n').encode())
            process.stdin.flush()
        def receive(request_id):
            while True:
                raw = messages.get(timeout=60)
                if raw is None:
                    raise RuntimeError('Native app-server closed before replying')
                data = json.loads(raw)
                if data.get('id') == request_id:
                    if 'error' in data:
                        raise RuntimeError(data['error'])
                    return raw, data
        try:
            send({'id': 1, 'method': 'initialize', 'params': {
                'clientInfo': {'name': 'exam-native-export', 'version': '1.0'},
                'capabilities': {'experimentalApi': True}}})
            receive(1)
            send({'method': 'initialized'})
            send({'id': 2, 'method': 'thread/read', 'params': {
                'threadId': args.thread, 'includeTurns': True}})
            raw, response = receive(2)
            thread = response['result']['thread']
            assert thread['id'] == args.thread and thread.get('turns')
            # Parse only to check identity; preserve every byte of the native response.
            with args.output.open('xb') as stream:
                stream.write(raw)
            result = {'status': 'NATIVE_RESPONSE_SAVED_UNMODIFIED', 'thread_id': thread['id'],
                      'codex_binary': args.codex, 'codex_version': version,
                      'native_method': 'thread/read', 'includeTurns': True,
                      'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest(),
                      'turns': len(thread['turns']),
                      'items': sum(len(t.get('items', [])) for t in thread['turns']),
                      'output': str(args.output.resolve()),
                      'scope': 'Current thread as returned by Codex; no synthetic or merged messages.',
                      'snapshot_note': 'The active turn can still be in progress at export time.'}
            args.output.with_suffix('.provenance.json').write_text(
                json.dumps(result, ensure_ascii=False, indent=2)+'\n')
            print(json.dumps(result, ensure_ascii=False))
        finally:
            process.stdin.close()
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


if __name__ == '__main__':
    main()
