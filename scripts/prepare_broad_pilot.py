"""Assemble the completed collections, split checks and concrete pilot plan."""
from argparse import Namespace
from datetime import datetime, timezone
import json
from pathlib import Path
import math
from slm_lab.corpus import pack,file_hash
from slm_lab.data import stable_id
from slm_lab.mix_corpora import mix

ROOT=Path(__file__).resolve().parents[1]

def main():
    references=ROOT/'data/public-reference-v1';books=ROOT/'data/public-books-v1'
    source_manifests=[references/'manifest.json',books/'manifest.json']
    reference_manifest=json.loads(source_manifests[0].read_text())
    requested=json.loads((ROOT/'configs/public-reference.json').read_text())['domains']
    missing=set(requested)-set(reference_manifest['domain_characters'])
    if missing:raise ValueError(f'Reference domains missing: {sorted(missing)}')
    if reference_manifest['failures']:raise ValueError(f'Reference collection failures: {reference_manifest["failures"]}')
    combined=ROOT/'data/reference-books-v1';combined.mkdir(exist_ok=True)
    with (combined/'all.jsonl').open('w') as output:
        for source in [references/'all.jsonl',books/'all.jsonl']:
            with source.open() as stream:
                for line in stream:output.write(line)
    packed=ROOT/'data/packed-reference-books-v1'
    tokenizer=ROOT/'data/tokenizer-corpus/tokenizer.json'
    if not (packed/'corpus.json').exists():
        pack(Namespace(jsonl=str(combined/'all.jsonl'),dataset=None,tokenizer=str(tokenizer),
            output=str(packed),max_tokens=100_000_000,shard_tokens=10_000_000))
    manifest=json.loads((packed/'corpus.json').read_text())
    manifest['source']['collection_manifests']=[{'path':str(p),'sha256':file_hash(p)} for p in source_manifests]
    (packed/'corpus.json').write_text(json.dumps(manifest,indent=2)+'\n')
    if any(manifest['domain_tokens'].get(domain,{}).get('train',0)<1000 for domain in requested):
        raise ValueError('Every requested domain needs at least 1000 actual training tokens')
    # Balanced validation samples, all held out by the same source-group split used in packing.
    def sample(paths,field,kind=None,per_group=6):
        groups={};seen=set()
        for path in paths:
            with path.open() as source:
                for line in source:
                    row=json.loads(line);group=row.get('group_id');label=row.get(field)
                    if not group or not label or kind and row.get('kind')!=kind:continue
                    if int(stable_id(str(group))[:8],16)%10!=0 or group in seen:continue
                    bucket=groups.setdefault(label,[])
                    if len(bucket)>=per_group:continue
                    bucket.append(row);seen.add(group)
        return [row for label in sorted(groups) for row in groups[label]]
    eval_root=ROOT/'data/broad-evaluation-v1';eval_root.mkdir(exist_ok=True)
    samples={'code.valid.jsonl':sample([ROOT/'data/public-code-v1/all.jsonl',ROOT/'data/public-code-extra-v1/all.jsonl'],
                                    'language_extension','code'),
             'domains.valid.jsonl':sample([references/'all.jsonl'],'domain')}
    for name,rows in samples.items():
        if not rows:raise ValueError('Empty held-out evaluation set')
        (eval_root/name).write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in rows))
    mixture=ROOT/'data/broad-mixture-v1'
    if not (mixture/'corpus.json').exists():
        result=mix([('code',ROOT/'data/packed-code-combined-v1',None),
                    ('reference-books',packed,None),('general',ROOT/'data/pilot-100m',None)],mixture)
    else:result=json.loads((mixture/'corpus.json').read_text())
    target_batch=4*16*4095;train_tokens=result['splits']['train']['tokens']
    # A bounded almost-one-pass run; no manufactured corpus volume by repeating scarce data.
    steps=(train_tokens-1)//target_batch
    plan={'created_at':datetime.now(timezone.utc).isoformat(),'status':'prepared',
        'parameters':706012672,'gpus':4,'gpu_type':'NVIDIA L40','device':'cuda','bf16':True,
        'batch_size_per_gpu':1,'gradient_accumulation':16,'text_length':4096,'optimizer_steps':steps,
        'learning_rate':3e-4,'target_tokens_per_update':target_batch,
        'planned_training_target_tokens':steps*target_batch,'train_corpus_tokens':train_tokens,
        'valid_corpus_tokens':result['splits']['valid']['tokens'],
        'train_token_shares':result['train_token_shares'],
        'reference_domain_tokens':manifest['domain_tokens'],'corpus_manifest_sha256':file_hash(mixture/'corpus.json'),
        'initial_checkpoint':'checkpoints/full-first-pretrain/model.pt',
        'checkpoint_origin':'Our randomly initialized original model after one public-text update; no external model weights.',
        'tokenizer':'data/tokenizer-corpus/tokenizer.json','tokenizer_sha256':file_hash(tokenizer),
        'output':'checkpoints/broad-public-pilot-v1',
        'purpose':'First sustained code-heavy, broad-domain learning run; capability remains unverified.',
        'limitations':['Bounded public sources, not all open-source code or every knowledge domain.',
            'Current tokenizer is an initial general-text vocabulary; code/domain tokenizer ablation remains pending.',
            'Image capability and RLHF require subsequent training; this run trains the text decoder.',
            'Split groups are original files/articles/books; forks and cross-source near-duplicates are not yet audited.']}
    (ROOT/'reports/broad-public-pilot-plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    print(json.dumps(plan,indent=2),flush=True)

if __name__=='__main__':main()
