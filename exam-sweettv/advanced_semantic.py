"""Frozen external multilingual ONNX encoder on strictly pre-cutoff metadata.

The public pretrained encoder is not fine-tuned. New SWEET.TV content is never
encoded: metadata-train.json is the complete encoding/centering universe.
"""
from pathlib import Path
import re
import time

import numpy as np
from sklearn.preprocessing import normalize
from scipy import sparse

from advanced_collaborative import _matrix
from advanced_factorization import expanded_context
from models import METHODS, normalized_text
from repro import ROOT, dump, load, sha256

REVISION = 'e8f8c211226b894fcb81acc59f3b34ba3efd5f42'
REPOSITORY = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
MODEL_DIR = ROOT/'artifacts/external-models'/REPOSITORY.split('/')[-1]/REVISION
MANIFEST_SHA = '8ce34bc588e22a1616945868095dfd1d450984a1996b4494cebb8022107710de'
CACHE_VERSION = 'semantic-train-metadata-v2-stable-title-ties'


def verify_model(folder=MODEL_DIR):
    folder=Path(folder)
    if sha256(folder/'manifest.json')!=MANIFEST_SHA:
        raise ValueError('Pinned external model manifest changed')
    manifest=load(folder/'manifest.json')
    if manifest['revision']!=REVISION or manifest['repo']!=REPOSITORY:
        raise ValueError('External encoder identity changed')
    for f in manifest['files']:
        path=folder/f['file']
        if path.stat().st_size!=f['size'] or sha256(path)!=f['sha256']:
            raise ValueError('Pinned external encoder file changed: '+f['file'])
    return manifest


def document(doc,mode='metadata'):
    def values(fields):
        return sorted({v.strip() for f in fields for v in doc.get(f,[]) if normalized_text(v) not in ('','-','нет','n/a','null')},key=lambda v:(normalized_text(v),v))
    titles=values(['tv_show_title'])
    # A single shortest observed historical title avoids repeated episode listings
    # consuming the 128-token model window; no future title resolves this choice.
    title=min(titles,key=lambda x:(len(x),normalized_text(x),x)) if titles else ''
    if mode=='title':return title
    if mode!='metadata':raise ValueError('Unknown semantic document mode')
    fields=[title]
    for label,keys,limit in [
        ('Категория',['tv_show_category'],3),
        ('Жанры',['tv_show_genre_1','tv_show_genre_2','tv_show_genre_3'],8),
        ('Режиссёр',['director'],2),('Актёры',['actors'],2)]:
        v=values(keys)[:limit]
        if v:fields.append(label+': '+', '.join(v))
    return '. '.join(f for f in fields if f)


def masked_mean(tokens,mask):
    tokens=np.asarray(tokens,dtype=np.float32);mask=np.asarray(mask,dtype=np.float32)
    if tokens.ndim!=3 or mask.shape!=tokens.shape[:2]:raise ValueError('Invalid masked pooling shapes')
    return (tokens*mask[:,:,None]).sum(axis=1)/np.maximum(mask.sum(axis=1,keepdims=True),1e-9)


def encoder(folder=MODEL_DIR):
    import onnxruntime as ort
    from tokenizers import Tokenizer
    options=ort.SessionOptions();options.intra_op_num_threads=1;options.inter_op_num_threads=1
    options.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL
    session=ort.InferenceSession(str(folder/'onnx/model_qint8_arm64.onnx'),options,providers=['CPUExecutionProvider'])
    inputs={i.name:i.type for i in session.get_inputs()}
    if inputs!={'input_ids':'tensor(int64)','attention_mask':'tensor(int64)','token_type_ids':'tensor(int64)'}:
        raise ValueError('Unexpected official ONNX inputs')
    tokenizer=Tokenizer.from_file(str(folder/'tokenizer.json'))
    tokenizer.enable_truncation(max_length=128)
    tokenizer.enable_padding(pad_id=tokenizer.token_to_id('<pad>'),pad_token='<pad>')
    return session,tokenizer


def encode(documents,session,tokenizer,batch_size=32):
    output=np.empty((len(documents),384),np.float32);lengths=[]
    for start in range(0,len(documents),batch_size):
        batch=tokenizer.encode_batch(documents[start:start+batch_size])
        lengths.extend(sum(x.attention_mask) for x in batch)
        inputs={'input_ids':np.array([x.ids for x in batch],np.int64),
                'attention_mask':np.array([x.attention_mask for x in batch],np.int64),
                'token_type_ids':np.array([x.type_ids for x in batch],np.int64)}
        result=session.run(['last_hidden_state'],inputs)[0]
        output[start:start+len(batch)]=normalize(masked_mean(result,inputs['attention_mask']),norm='l2')
    for j,d in enumerate(documents):
        if not d.strip():output[j]=0
    if not np.isfinite(output).all():raise ValueError('Nonfinite external embeddings')
    return output,lengths


def train_embeddings(ctx,mode='metadata'):
    verify_model()
    key={'protocol_sha256':ctx.protocol_hash,'metadata_sha256':sha256(ctx.path/'metadata-train.json'),
         'model_manifest_sha256':MANIFEST_SHA,'source_sha256':sha256(__file__),
         'version':CACHE_VERSION,'document_mode':mode,'batch_size':32,'max_seq_length':128}
    import hashlib,json
    digest=hashlib.sha256(json.dumps(key,sort_keys=True).encode()).hexdigest()
    folder=ROOT/'artifacts/semantic-cache'/digest
    if folder.exists():
        meta=load(folder/'cache-manifest.json')
        if meta['key']!=key:raise ValueError('Semantic cache key mismatch')
        for name,expected in meta['files'].items():
            if sha256(folder/name)!=expected:raise ValueError('Semantic cache modified: '+name)
        mapping=load(folder/'documents.json');vectors=np.load(folder/'embeddings.npy',allow_pickle=False)
        if vectors.shape!=(len(mapping['items']),384):raise ValueError('Semantic cache orientation changed')
        if mapping['items']!=sorted(ctx.metadata,key=int):raise ValueError('Semantic cache item universe changed')
        return mapping['items'],vectors,{'path':str(folder),'manifest_sha256':sha256(folder/'cache-manifest.json'),**meta['details']}
    folder.mkdir(parents=True)
    items=sorted(ctx.metadata,key=int);documents=[document(ctx.metadata[i],mode) for i in items]
    session,tokenizer=encoder();started=time.perf_counter()
    vectors,lengths=encode(documents,session,tokenizer)
    # The same fixed batch/tokenisation is replayed before keeping the cache.
    probe,_=encode(documents[:32],session,tokenizer)
    np.testing.assert_array_equal(probe,vectors[:len(probe)])
    np.save(folder/'embeddings.npy',vectors)
    dump(folder/'documents.json',{'items':items,'documents':documents})
    dump(folder/'encoder-input-summary.json',{'lengths':lengths,'max_length':max(lengths,default=0),
        'truncated_or_exactly_limit':sum(n==128 for n in lengths),'batch_size':32,
        'onnx_inputs':[(i.name,i.type,i.shape) for i in session.get_inputs()],
        'onnx_outputs':[(i.name,i.type,i.shape) for i in session.get_outputs()]})
    np.testing.assert_array_equal(vectors,np.load(folder/'embeddings.npy',allow_pickle=False))
    details={'items':len(items),'dimensions':384,'seconds':time.perf_counter()-started,
             'same_batch_encoder_replay_exact':True,'embeddings_reload_exact':True,'future_metadata_used':False,
             'quantization_note':'Official qint8 ARM64 graph; fixed ordered batches of 32; no claim of float32-equivalent encoder outputs.'}
    files={f:sha256(folder/f) for f in ('embeddings.npy','documents.json','encoder-input-summary.json')}
    dump(folder/'cache-manifest.json',{'key':key,'files':files,'details':details})
    return items,vectors,{'path':str(folder),'manifest_sha256':sha256(folder/'cache-manifest.json'),**details}


def semantic_cbf(ctx,p,path):
    items,original,cache=train_embeddings(ctx,p.get('document_mode','metadata'))
    vectors=original.copy();center=np.zeros((1,384),np.float32)
    if p.get('center',True):
        active=np.linalg.norm(vectors,axis=1)>0
        center=vectors[active].mean(axis=0,keepdims=True)
        vectors[active]=normalize(vectors[active]-center,norm='l2')
    im={item:j for j,item in enumerate(items)}
    history=expanded_context(ctx) if p.get('weak_weight') is not None else ctx
    if any(i not in im for i in history.items):raise ValueError('Profile has item without pre-cutoff metadata')
    x=_matrix(history,{'strength':'count',**p})[history.target_rows]
    profiles=normalize(x@vectors[np.array([im[i] for i in history.items])],norm='l2').astype(np.float32)
    candidates=np.zeros((len(ctx.candidates),384),np.float32)
    for j,i in enumerate(ctx.candidates):
        if i in im:candidates[j]=vectors[im[i]]
    kernel_power=p.get('kernel_power')
    if kernel_power is not None:
        if kernel_power!=4:raise ValueError('Controlled semantic kernel fixes power=4; no power grid')
        history_vectors=vectors[np.array([im[i] for i in history.items])]
        kernel=np.clip(history_vectors@candidates.T,0,1)**kernel_power
        raw=np.asarray(x@kernel,dtype=np.float32)
        sparse.save_npz(path/'semantic-kernel-history.npz',x)
        np.save(path/'semantic-kernel-history-vectors.npy',history_vectors)
    else:
        raw=profiles@candidates.T
    factor=np.ones(len(ctx.candidates),np.float32)
    if p.get('opportunity_adjustment'):
        opp=load(ctx.path/'opportunities.json');prior=p.get('prior_strength',50)
        factor=np.array([opp['future'].get(i,0)/(opp['train'].get(i,0)+prior) for i in ctx.candidates],np.float32)
        raw*=factor[None,:]
    np.savez_compressed(path/'semantic-model.npz',profiles=profiles,candidates=candidates,center=center,exposure_factor=factor)
    dump(path/'semantic-mappings.json',{'items':items,'profile_items':history.items,'targets':ctx.targets,'candidates':ctx.candidates})
    restored=np.load(path/'semantic-model.npz',allow_pickle=False)
    if kernel_power is not None:
        stored_history=sparse.load_npz(path/'semantic-kernel-history.npz')
        stored_vectors=np.load(path/'semantic-kernel-history-vectors.npy',allow_pickle=False)
        replay=np.asarray(stored_history@(np.clip(stored_vectors@restored['candidates'].T,0,1)**kernel_power),dtype=np.float32)
        replay*=restored['exposure_factor'][None,:]
    else:
        replay=(restored['profiles']@restored['candidates'].T)*restored['exposure_factor'][None,:]
    np.testing.assert_allclose(raw,replay,rtol=1e-6,atol=1e-6)
    context_details=None
    if p.get('hourly_product'):
        from advanced_hourly_schedule import schedule_hourly
        schedule,context_details=schedule_hourly(ctx,{'joint_weight':.5,'neighbor_smooth':.25,'half_life_days':28},path)
        schedule=np.nan_to_num(schedule,nan=0);schedule/=np.maximum(schedule.max(axis=1,keepdims=True),1e-12)
        raw=np.maximum(raw,0)**p.get('content_power',.25)*schedule
        np.save(path/'semantic-hourly-modulation.npy',schedule)
        replay=np.maximum(replay,0)**p.get('content_power',.25)*np.load(path/'semantic-hourly-modulation.npy',allow_pickle=False)
        np.testing.assert_allclose(raw,replay,rtol=1e-5,atol=1e-6)
    raw[raw<=0]=np.nan
    details={'algorithm':'train-only centred multilingual semantic power-kernel sum' if kernel_power else 'train-only centred multilingual semantic profile cosine',
        'kernel_power':kernel_power,'kernel_self_similarity':1. if kernel_power else None,
        'kernel_scope':'Weighted exact repeats and semantic neighbours; not pure discovery' if kernel_power else None,
        'external_model':REPOSITORY,'external_revision':REVISION,'external_manifest_sha256':MANIFEST_SHA,
        'external_license':'Apache-2.0','model_card':'https://huggingface.co/'+REPOSITORY+'/tree/'+REVISION,
        'pooling':'attention-mask weighted token mean followed by L2 normalization',
        'fine_tuning':False,'training_metadata_sha256':sha256(ctx.path/'metadata-train.json'),
        'fit_cutoff':ctx.protocol['train_cutoff_exclusive'],'future_metadata_used':False,'future_views_used':False,
        'embedding_cache':cache,'centered':p.get('center',True),'supported_candidates':sum(i in im for i in ctx.candidates),
        'profile_items':len(history.items),'saved_model_reload_verified':True,'context_modulation':context_details,
        'disclosure':'External public pretrained multilingual encoder; no SWEET.TV validation labels/views/content supplied to encoder or profile fitting.'}
    dump(path/'semantic-disclosure.json',details)
    return raw,details


def register():METHODS['semantic_cbf']=semantic_cbf
