#!/usr/bin/env python3
"""Download only pinned public JSON/text and an official ONNX encoder; never pickle/code."""
import hashlib,json,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
INFO=ROOT/'artifacts/semantic-research/hf-model-info.json'
x=json.loads(INFO.read_text());revision=x['sha'];repo=x['id']
files=['README.md','config.json','modules.json','sentence_bert_config.json','1_Pooling/config.json',
       'tokenizer.json','tokenizer_config.json','special_tokens_map.json','onnx/model_qint8_arm64.onnx']
target=ROOT/'artifacts/external-models'/repo.split('/')[-1]/revision
target.mkdir(parents=True,exist_ok=True);manifest=[]
for f in files:
 a=next(t for t in x['siblings'] if t['rfilename']==f);path=target/f
 url=f'https://huggingface.co/{repo}/resolve/{revision}/{f}'
 if not path.exists():
  path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.download')
  with urllib.request.urlopen(url,timeout=60) as source,tmp.open('wb') as out:
   while chunk:=source.read(1024*1024):out.write(chunk)
  tmp.rename(path)
 data=path.read_bytes();digest=hashlib.sha256(data).hexdigest()
 if len(data)!=a['size']:raise ValueError(f+' size mismatch')
 if 'lfs' in a and digest!=a['lfs']['sha256']:raise ValueError(f+' LFS mismatch')
 if 'lfs' not in a and hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()!=a['blobId']:raise ValueError(f+' git blob mismatch')
 manifest.append(dict(file=f,url=url,size=len(data),sha256=digest,git_blob=a['blobId'],lfs_sha256=a.get('lfs',{}).get('sha256')))
 print(f,len(data),digest,flush=True)
record=dict(repo=repo,revision=revision,license='Apache-2.0 (official model card)',license_url='https://www.apache.org/licenses/LICENSE-2.0',
 format='official ONNX qint8 ARM64 graph; no Python remote code, PyTorch or pickle files downloaded',files=manifest)
(target/'manifest.json').write_text(json.dumps(record,ensure_ascii=False,indent=2))
print(target)
