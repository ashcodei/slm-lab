"""Prepare public human-written demonstrations and genuine human preferences."""
import argparse
import base64
from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
from pathlib import Path
from .public_sources import api,fetch
from .corpus import file_hash
from .data import hh_pair,stable_id


def demonstrations(root):
    from datasets import load_dataset
    from huggingface_hub import HfApi
    dataset='HuggingFaceH4/no_robots';revision=HfApi().dataset_info(dataset,token=False).sha
    source=load_dataset(dataset,revision=revision,token=False)
    heldout_prompts=set();splits={};skipped=0
    for split in ['test','train']:
        rows=[]
        for entry in source[split]:
            messages=entry['messages']
            if not messages or messages[-1]['role']!='assistant':skipped+=1;continue
            parts=[]
            for turn in messages[:-1]:
                label={'user':'User','assistant':'Assistant','system':'System'}.get(turn['role'])
                if label is None:break
                parts.append(label+': '+turn['content'])
            else:
                prompt='\n'.join(parts)+'\nAssistant:';response=messages[-1]['content'];key=stable_id(prompt)
                if split=='train' and key in heldout_prompts:skipped+=1;continue
                if split=='test':heldout_prompts.add(key)
                rows.append({'id':key,'prompt':prompt,'response':response,'source':dataset,
                    'revision':revision,'license':'CC-BY-NC-4.0','category':entry.get('category'),
                    'annotation_origin':'Human-written instruction and demonstration, per dataset card.'})
                continue
            skipped+=1
        target='valid' if split=='test' else 'train';splits[target]=len(rows)
        (root/('sft.'+target+'.jsonl')).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    card=fetch(f'https://huggingface.co/datasets/{dataset}/raw/{revision}/README.md',1_000_000)
    (root/'no-robots-README.md').write_bytes(card)
    return {'dataset':dataset,'revision':revision,'license':'CC-BY-NC-4.0','splits':splits,
            'skipped':skipped,'card_sha256':file_hash(root/'no-robots-README.md')}


def preferences(root,max_train,max_valid):
    repo='anthropics/hh-rlhf';revision=api('https://api.github.com/repos/'+repo+'/commits/master')['sha']
    license_info=api(f'https://api.github.com/repos/{repo}/license?ref={revision}')
    license_text=base64.b64decode(license_info['content'])
    if license_info.get('license',{}).get('spdx_id')!='MIT':raise ValueError('Expected HH repository MIT license')
    (root/'HH-LICENSE.txt').write_bytes(license_text)
    seen=set();splits={};sources=[];skipped=0
    for split in ['test','train']:
        target='valid' if split=='test' else 'train';rows=[]
        budget=max_valid if target=='valid' else max_train
        for subset in ['helpful-base','harmless-base']:
            path=f'{subset}/{split}.jsonl.gz';url=f'https://raw.githubusercontent.com/{repo}/{revision}/{path}'
            raw=fetch(url,120_000_000)
            if raw.startswith(b'version https://git-lfs.github.com'):
                url=f'https://media.githubusercontent.com/media/{repo}/{revision}/{path}'
                raw=fetch(url,120_000_000)
            if not raw.startswith(b'\x1f\x8b'):raise ValueError('Expected gzip preference dataset, not an LFS pointer')
            used=0
            with gzip.open(io.BytesIO(raw),'rt') as stream:
                for line in stream:
                    try:pair=hh_pair(json.loads(line))
                    except ValueError:skipped+=1;continue
                    if pair['id'] in seen:skipped+=1;continue
                    seen.add(pair['id'])
                    if used>=budget//2:continue
                    used+=1
                    pair.update({'source':'GitHub/'+repo+'/'+subset,'revision':revision,'license':'MIT',
                        'source_url':f'https://github.com/{repo}/blob/{revision}/{path}',
                        'response_origin':'Responses from Anthropic models, ranked by human annotators.',
                        'feedback_source':'human','original_split':split})
                    rows.append(pair)
                    if split=='train' and used>=budget//2:break
            sources.append({'url':url,'subset':subset,'original_split':split,'used_pairs':used,
                            'download_sha256':hashlib.sha256(raw).hexdigest()})
        splits[target]=len(rows)
        (root/('preferences.'+target+'.jsonl')).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    return {'repo':repo,'revision':revision,'license':'MIT','splits':splits,'sources':sources,
            'skipped':skipped,'split_rule':'Original test split collected first; shared prompt hashes excluded from training.',
            'scope':'Human rankings of generated responses; not human-authored demonstration answers.'}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--preference-train',type=int,default=12000);p.add_argument('--preference-valid',type=int,default=1000)
    args=p.parse_args();root=Path(args.output)
    if root.exists() and any(root.iterdir()) and not args.resume:p.error('Use a new output directory or --resume.')
    root.mkdir(parents=True,exist_ok=True)
    demo_meta=root/'demonstrations.json'
    demos=json.loads(demo_meta.read_text()) if demo_meta.exists() else demonstrations(root)
    demo_meta.write_text(json.dumps(demos,indent=2)+'\n');print('demonstrations',demos,flush=True)
    prefs=preferences(root,args.preference_train,args.preference_valid);print('preferences',prefs,flush=True)
    manifest={'collected_at':datetime.now(timezone.utc).isoformat(),'demonstrations':demos,'preferences':prefs,
        'files_sha256':{f.name:file_hash(f) for f in root.glob('*') if f.is_file()}}
    (root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')

if __name__=='__main__':main()
