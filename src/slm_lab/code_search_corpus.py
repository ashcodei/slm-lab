"""Prepare licensed original CodeSearchNet text/code pairs and compact token shards.

The blocked S3 origin is replaced by a pinned original-format mirror, checked against
its Git-LFS hashes. Supplied license metadata is parsed as primitive data only.
Downloaded code and metadata constructors are never executed.
"""
import argparse
import ast
from collections import Counter
import gzip
import hashlib
import io
import json
from pathlib import Path,PurePosixPath
import re
import shutil
import sqlite3
import textwrap
import urllib.parse
import zipfile
from tokenizers import Tokenizer
from .corpus import ShardWriter,file_hash
from .data import stable_id
from .diverse_images import asset,atomic_json,jsonl
from .passive_pickle import load_primitives
from .wikipedia_corpus import protected_questions

REVISION='89d68535541a21f192969038c5000f3e1a43373b'
LANGUAGES=['python','javascript','java','go','php','ruby']
DISPLAY={'python':'Python','javascript':'JavaScript','java':'Java','go':'Go','php':'PHP','ruby':'Ruby'}


def license_identifier(text):
    head=' '.join(text[:6000].casefold().split())
    if 'permission is hereby granted, free of charge' in head and 'the above copyright notice' in head:return 'MIT'
    if 'apache license' in head and ('version 2.0' in head or 'version 2' in head):return 'Apache-2.0'
    if 'gnu affero general public license' in head and 'version 3' in head:return 'AGPL-3.0'
    if 'gnu lesser general public license' in head:
        if 'version 3' in head:return 'LGPL-3.0'
        if 'version 2.1' in head:return 'LGPL-2.1'
    if 'gnu general public license' in head:
        if 'version 3' in head:return 'GPL-3.0'
        if 'version 2' in head:return 'GPL-2.0'
    if 'redistribution and use in source and binary forms' in head and 'this software is provided' in head:
        return 'BSD-3-Clause' if 'neither the name' in head else 'BSD-2-Clause'
    if 'mozilla public license' in head and '2.0' in head:return 'MPL-2.0'
    if 'permission to use, copy, modify, and/or distribute' in head and 'with or without fee' in head:return 'ISC'
    if 'this is free and unencumbered software released into the public domain' in head:return 'Unlicense'
    return None


def applicable_license(repo,path,entries):
    """Choose the closest ancestor license directory; reject conflicting notices."""
    applicable=[];target=PurePosixPath(repo)/path
    for name,text in entries:
        location=PurePosixPath(name)
        if not isinstance(text,str) or location.is_absolute() or '..' in location.parts:continue
        if tuple(target.parts[:len(location.parent.parts)])!=location.parent.parts:continue
        license_id=license_identifier(text)
        if license_id:applicable.append((len(location.parent.parts),license_id,name,text))
    if not applicable:return None
    depth=max(r[0] for r in applicable);nearest=[r for r in applicable if r[0]==depth]
    if len({r[1] for r in nearest})!=1:return None
    _,identifier,name,text=nearest[0]
    return {'license':identifier,'license_path':name,'license_text_sha256':hashlib.sha256(text.encode('utf-8','surrogatepass')).hexdigest(),
            'license_text_hash_encoding':'UTF-8 with surrogatepass; original legacy catalogue strings retained.'}


def rows(archive,split):
    marker='/final/jsonl/'+split+'/'
    for name in sorted(archive.namelist()):
        if marker in name and name.endswith('.jsonl.gz'):
            with gzip.GzipFile(fileobj=archive.open(name)) as source:
                for line in source:
                    if len(line)>1_000_000:continue
                    if line.strip():yield json.loads(line)


def python_body(code):
    """Drop a repeated function docstring; parse only, without importing the code."""
    code=textwrap.dedent(code)
    try:tree=ast.parse(code)
    except (SyntaxError,ValueError):return None
    if len(tree.body)!=1 or not isinstance(tree.body[0],(ast.FunctionDef,ast.AsyncFunctionDef)):return None
    node=tree.body[0];lines=code.splitlines()
    if node.body and isinstance(node.body[0],ast.Expr) and isinstance(node.body[0].value,ast.Constant) and isinstance(node.body[0].value.value,str):
        doc=node.body[0]
        # Multi-statement one-line functions cannot be safely rewritten this way.
        if doc.lineno==node.lineno:return None
        lines[doc.lineno-1:doc.end_lineno]=[]
        if len(node.body)==1:lines.append(' '*node.body[0].col_offset+'pass')
    result='\n'.join(lines)
    try:ast.parse(result)
    except SyntaxError:return None
    return result


def instruction(row,license_info,tokenizer):
    language=row['language'];description=row.get('docstring','').strip();code=row.get('code','').strip()
    if not 25<=len(description)<=2400 or not 30<=len(code)<=16000:return None
    response=python_body(code) if language=='python' else code
    if not response:return None
    if language=='python':
        node=ast.parse(response).body[0]
        signature=('async ' if isinstance(node,ast.AsyncFunctionDef) else '')+'def '+node.name+'('+ast.unparse(node.args)+'):'
    else:signature=response.split('{',1)[0][:500]
    prompt=(f'Implement the documented {DISPLAY[language]} function or method from {row["repo"]}/{row["path"]}.\n'
            f'Signature/context: {signature}\n\n{description}\n\n'
            'The surrounding module or class supplies its existing imports, types and helpers. Return only the implementation.')
    answer='```'+language+'\n'+response+'\n```'
    prompt_ids=tokenizer.encode('User: '+prompt+'\nAssistant:').ids;answer_ids=tokenizer.encode(answer).ids
    if len(answer_ids)>2047 or len(prompt_ids)+len(answer_ids)+2>4096:return None
    return {'id':stable_id(row['url']+code),'group_id':'github:'+row['repo']+':'+row['path'],
            'prompt':prompt,'response':answer,'language':DISPLAY[language],'source':'CodeSearchNet',
            'source_url':row['url'],'original_partition':row['partition'],**license_info,
            'source_code_sha256':hashlib.sha256(code.encode()).hexdigest(),
            'annotation_origin':'Original source comment and code; static formatting, with Python docstring duplication removed. Authorship not independently verified.',
            'environment':'Original module/class context; imports/helpers may be needed. Implementations have not been executed.'}


def collect(output,cache,tokenizer_path,questions,old_inputs,max_tokens=1_000_000_000,max_pairs_per_language=200000):
    root=Path(output);cache=Path(cache);root.mkdir(parents=True,exist_ok=True);(root/'parts').mkdir(exist_ok=True)
    if (root/'corpus.json').exists():raise ValueError('Completed corpus exists; use a new output')
    files=json.loads((cache/'codesearchnet-source-files.json').read_text())['files']
    token_hash=file_hash(tokenizer_path);tokenizer=Tokenizer.from_file(str(tokenizer_path));eos=tokenizer.token_to_id('<eos>')
    if eos is None:raise ValueError('Tokenizer lacks EOS')
    matcher=protected_questions(questions);old_train=set()
    for path in old_inputs:
        for row in jsonl(path):
            group=row.get('group_id')
            if group and int(stable_id(group)[:8],16)%10!=0:old_train.add(group)
    configuration={'tokenizer_sha256':token_hash,'questions_sha256':file_hash(questions),'mirror_revision':REVISION,
                   'max_tokens':max_tokens,'max_pairs_per_language':max_pairs_per_language,
                   'old_inputs':[{ 'path':str(Path(p).resolve()),'sha256':file_hash(p)} for p in old_inputs]}
    cp=root/'collection-config.json'
    if cp.exists() and json.loads(cp.read_text())!=configuration:raise ValueError('Collection configuration changed')
    atomic_json(cp,configuration);archives={};sources=[];protected_repos=set()
    for language in LANGUAGES:
        metadata=files[language]
        path,record=asset(cache,'codesearchnet-'+language+'-mirror.zip',
            f'https://huggingface.co/datasets/code-search-net/code_search_net/resolve/{REVISION}/'+metadata['path'],metadata['bytes']+1)
        if record['sha256']!=metadata['sha256']:raise ValueError('Mirror Git-LFS checksum mismatch')
        archives[language]=path;sources.append({**record,'language':language,'mirror_git_lfs_sha256':metadata['sha256'],
            'original_url':f'https://s3.amazonaws.com/code-search-net/CodeSearchNet/v2/{language}.zip',
            'mirror_reason':'Original S3 endpoint returned HTTP 403','origin_equivalence_independently_verified':False})
        with zipfile.ZipFile(path) as archive:
            for split in ['valid','test']:
                protected_repos.update(row['repo'].casefold() for row in rows(archive,split))
        print('scanned official held-out repos',language,len(protected_repos),flush=True)
    (root/'protected-repositories.txt').write_text('\n'.join(sorted(protected_repos))+'\n')
    db=sqlite3.connect(root/'dedup.sqlite');db.execute('CREATE TABLE IF NOT EXISTS code (hash BLOB PRIMARY KEY, part TEXT, split TEXT)');db.commit()
    finished=[];total=0;dtype='<u2' if tokenizer.get_vocab_size()<=65536 else '<u4'
    for language in LANGUAGES:
        directory=root/'parts'/language;mp=directory/'corpus.json'
        if mp.exists():
            manifest=json.loads(mp.read_text());finished.append((directory,manifest));total+=manifest['total_tokens'];continue
        db.execute('DELETE FROM code WHERE part=?',(language,));db.commit()
        if directory.exists():shutil.rmtree(directory)
        directory.mkdir();writers={s:ShardWriter(directory,s,dtype,1_000_000) for s in ['train','valid']}
        excluded=Counter();counts=Counter();pair_counts=Counter();docs=0;used=0;license_counts=Counter()
        with zipfile.ZipFile(archives[language]) as archive:
            member=next(n for n in archive.namelist() if n.endswith('_licenses.pkl'))
            raw=archive.read(member);license_map=load_primitives(raw)
            with (directory/'provided-licenses.jsonl').open('w') as license_output:
                for repo,notices in license_map.items():
                    for name,text in notices:
                        license_output.write(json.dumps({'repo':repo,'path':name,'text':text},ensure_ascii=True)+'\n')
            del raw;db.execute('BEGIN')
            with (directory/'documents.jsonl').open('w') as journal, (directory/'sft.train.jsonl').open('w') as train_pairs,(directory/'sft.valid.jsonl').open('w') as valid_pairs:
                for original_split in ['valid','test','train']:
                    for row in rows(archive,original_split):
                        code=row.get('code','');description=row.get('docstring','');repo=row['repo'];path=row['path']
                        if not isinstance(code,str) or not 30<=len(code)<=50000:excluded['code_length']+=1;continue
                        try:(code+description+repo+path+row['url']).encode('utf-8')
                        except UnicodeError:excluded['invalid_source_unicode']+=1;continue
                        if not re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+',repo):excluded['repo_format']+=1;continue
                        notice=applicable_license(repo,path,license_map.get(repo,[]))
                        if notice is None:excluded['unknown_or_conflicting_license']+=1;continue
                        group='github:'+repo+':'+path;heldout=int(stable_id(group)[:8],16)%10==0
                        split='valid' if original_split!='train' or heldout or repo.casefold() in protected_repos else 'train'
                        if split=='valid' and group in old_train:excluded['previous_training_file_in_validation']+=1;continue
                        text=f'Language: {DISPLAY[language]}\nFile: {path}\n\nDescription: {description}\n\n{code}'
                        matches=[qid for _,qid in matcher.iter(' '.join(text.casefold().split()))]
                        if matches:excluded['reserved_knowledge_question']+=1;continue
                        digest=hashlib.sha256(code.strip().encode()).digest()
                        prior=db.execute('SELECT split FROM code WHERE hash=?',(digest,)).fetchone()
                        if prior:excluded['exact_code_duplicate']+=1;continue
                        ids=tokenizer.encode(text).ids+[eos]
                        if used+len(ids)>max_tokens//len(LANGUAGES):excluded['language_token_budget']+=1;continue
                        writer=writers[split];offset=writer.tokens+len(writer.buffer);writer.append(ids)
                        db.execute('INSERT INTO code VALUES (?,?,?)',(digest,language,split))
                        journal.write(json.dumps({'group_id':group,'source_url':row['url'],'language':language,
                            'split':split,'original_partition':original_split,'source_code_sha256':digest.hex(),
                            'token_start':offset,'token_count':len(ids),**notice},ensure_ascii=True)+'\n')
                        pair=instruction(row,notice,tokenizer)
                        if pair and pair_counts[split]<(max_pairs_per_language if split=='train' else max_pairs_per_language//5):
                            (train_pairs if split=='train' else valid_pairs).write(json.dumps(pair,ensure_ascii=True)+'\n');pair_counts[split]+=1
                        docs+=1;used+=len(ids);counts[split]+=1;license_counts[notice['license']]+=1
                        if docs%2000==0:
                            journal.flush();atomic_json(root/'progress.json',{'status':'processing original documented code',
                                'language':language,'language_functions':docs,'processed_tokens':total+used,'sft_pairs':dict(pair_counts)})
                            print('code',language,docs,'tokens',total+used,'pairs',dict(pair_counts),flush=True)
        for writer in writers.values():writer.flush()
        db.commit();manifest={'source':next(s for s in sources if s['language']==language),'dtype':dtype,
            'tokenizer_sha256':token_hash,'total_tokens':used,'functions':dict(counts),'sft_pairs':dict(pair_counts),
            'licenses':dict(license_counts),'excluded':dict(excluded),
            'splits':{s:{'tokens':w.tokens,'shards':w.files} for s,w in writers.items()},
            'files_sha256':{p.name:file_hash(p) for p in directory.glob('*.jsonl')}}
        atomic_json(mp,manifest);finished.append((directory,manifest));total+=used
        archives[language].unlink();print('completed code language',language,used,dict(pair_counts),flush=True)
    db.close();result={'source':{'dataset':'CodeSearchNet original code/comment pairs','mirror_revision':REVISION},
        'dtype':dtype,'tokenizer_sha256':token_hash,'total_tokens':total,'source_parts':[],
        'splits':{s:{'tokens':0,'shards':[]} for s in ['train','valid']},'language_tokens':{},
        'protected_questions_sha256':configuration['questions_sha256'],'protected_repositories':len(protected_repos),
        'split_rule':'Official validation/test repositories plus canonical source-file SHA256 modulo 10; prior training files excluded from validation.',
        'limitations':['Provided closest-ancestor license notices classified conservatively; not an exhaustive per-file rights audit.',
                       'Original code is source-era code; standalone execution, dependency contracts and authorship are unverified.',
                       'All original code is static data; legacy metadata globals/constructors are rejected.',
                       'Exact code deduplication within this collection; near-duplicate and previous-phase snippet overlap remain possible.',
                       'A pinned mirror of a blocked origin; source-origin hash equivalence not independently verified.']}
    for directory,manifest in finished:
        result['source_parts'].append({'path':str(directory),'manifest_sha256':file_hash(directory/'corpus.json')})
        result['language_tokens'][directory.name]=manifest['total_tokens']
        for split,values in manifest['splits'].items():
            result['splits'][split]['tokens']+=values['tokens']
            for shard in values['shards']:result['splits'][split]['shards'].append({**shard,'path':str((directory/shard['path']).relative_to(root))})
    for split in ['train','valid']:
        with (root/('sft.'+split+'.jsonl')).open('w') as target:
            for directory,_ in finished:
                with (directory/('sft.'+split+'.jsonl')).open() as source:
                    for line in source:target.write(line)
    if not all(values['tokens']>4096 for values in result['splits'].values()):raise ValueError('Insufficient licensed train/validation code')
    if any(tokens<=0 for tokens in result['language_tokens'].values()):raise ValueError('A requested code language has no licensed records')
    result['sft_files_sha256']={s:file_hash(root/('sft.'+s+'.jsonl')) for s in ['train','valid']}
    atomic_json(root/'corpus.json',result);atomic_json(root/'progress.json',{'status':'completed original code corpus','tokens':total})
    print('completed code corpus',total,flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);p.add_argument('--cache',required=True)
    p.add_argument('--tokenizer',required=True);p.add_argument('--questions',required=True);p.add_argument('--old-input',action='append',default=[])
    p.add_argument('--max-tokens',type=int,default=1_000_000_000);p.add_argument('--max-pairs-per-language',type=int,default=200000)
    a=p.parse_args();collect(a.output,a.cache,a.tokenizer,a.questions,a.old_input,a.max_tokens,a.max_pairs_per_language)

if __name__=='__main__':main()
