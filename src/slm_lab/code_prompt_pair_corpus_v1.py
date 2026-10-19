"""Immutable full-code pairing indices; original programme tokens stay in place."""
from collections import Counter
from datetime import datetime,timezone
import gzip
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

from .code_prompt_matching_v1 import pair_map,validate_pair_map,mismatch_sequence
from .codenet_checkpoint_v1 import FileBindings
from .codenet_token_pack_v1 import CompleteCodePack
from .corpus import file_hash
from .diverse_images import atomic_json

FORMAT='own-complete-code-source-pair-index-v1'


def prepare(pack_folder,output,seed=920031):
    pack_folder,output=Path(pack_folder).resolve(),Path(output).resolve()
    if output.exists():
        raise ValueError('Preserve prior pair index; use a new owned output')
    if shutil.disk_usage(output.parent).free<26*1024**3+64*1024**2:
        raise ValueError('Fund64MiB pair metadata above original26GiB reserve')
    packs={role:CompleteCodePack(pack_folder,role,purpose='training' if role=='train' else 'audit') for role in ['train','validation']}
    files={str(pack_folder/name):digest for name,digest in packs['train'].manifest['files_sha256'].items()}
    for name in ['manifest.json','admission.json']:
        files[str(pack_folder/name)]=file_hash(pack_folder/name)
    for path in [Path(__file__),Path(__file__).with_name('code_prompt_matching_v1.py')]:
        files[str(path)]=file_hash(path)
    guard=FileBindings(files)
    records={role:[] for role in packs}
    lookups={role:{int(global_position):local for local,global_position in enumerate(pack.positions)} for role,pack in packs.items()}
    with gzip.open(pack_folder/'original-rows.jsonl.gz','rt') as stream:
        for source in map(json.loads,stream):
            role=source['split']
            if role not in packs:
                continue
            pack=packs[role];global_position=source['token_pack_index'];local=lookups[role][global_position]
            index=pack.index[global_position]
            offset,length,start=(int(index[k]) for k in ['offset','length','target_start'])
            raw=pack.ids[offset:offset+length]
            digest=hashlib.sha256(raw.tobytes()).hexdigest()
            if digest!=source['complete_input_ids_sha256'] or length-start!=source['target_tokens'] or local!=len(records[role]):
                raise ValueError('Every original full source identity/token/mask/order must match')
            records[role].append(dict(id=source['id'],role=role,language=source['language'],group_id=source['group_id'],
                global_position=global_position,local_position=local,prefix_tokens=start,target_tokens=length-start,
                body_sha256=source['original_source_body_sha256'],prompt_sha256=hashlib.sha256(raw[:start].tobytes()).hexdigest(),
                full_input_sha256=digest))
    output.mkdir()
    summaries={}
    for number,role in enumerate(packs):
        values=records[role]
        if len(values)!=len(packs[role]):
            raise ValueError('Every original role row must remain in the new pair index')
        pairs=pair_map(values,seed+number)
        validate_pair_map(values,pairs)
        np.save(output/(role+'-negative-positions-i32.npy'),pairs,allow_pickle=False)
        with (output/(role+'-records.jsonl')).open('x') as stream:
            for value in values:
                stream.write(json.dumps(value,separators=(',',':'))+'\n')
        language=Counter(r['language'] for r in values)
        paired=Counter(r['language'] for r,j in zip(values,pairs) if j>=0)
        summaries[role]=dict(source_records=len(values),paired_records=int((pairs>=0).sum()),
            unpaired_original_rows_retained=int((pairs<0).sum()),by_language=dict(language),paired_by_language=dict(paired),
            original_response_targets=sum(r['target_tokens'] for r in values),maximum_context=max(r['prefix_tokens']+r['target_tokens'] for r in values))
    if sum(p.stat().st_size for p in output.iterdir())>64*1024**2:
        raise ValueError('Pair metadata exceeds its funded64MiB capacity')
    guard.check(full=True)
    atomic_json(output/'manifest.json',dict(format=FORMAT,prepared_at=datetime.now(timezone.utc).isoformat(),
        seed=seed,code_pack=str(pack_folder),code_manifest_sha256=file_hash(pack_folder/'manifest.json'),
        code_admission_sha256=file_hash(pack_folder/'admission.json'),tokenizer_sha256=file_hash(pack_folder/'tokenizer.json'),
        summaries=summaries,source_files_sha256=files,
        artifact_files_sha256={p.name:file_hash(p) for p in output.iterdir()},
        original_programme_or_token_bytes_copied=0,all_original_role_rows_preserved=True,
        full_question_and_code_EOS_targets_preserved=True,roles_mixed=False,
        labels='Original source associations only; mismatches are not execution-verified incorrect programmes or human preferences.',
        neural_updates=0,goal_achieved=False))
    return summaries


class SourcePairs:
    def __init__(self,folder,role):
        self.folder,self.role=Path(folder).resolve(),role
        if role not in {'train','validation'}:
            raise ValueError('Only sealed train or validation diagnostic roles supported')
        self.manifest=json.loads((self.folder/'manifest.json').read_text())
        if self.manifest['format']!=FORMAT or not self.manifest['all_original_role_rows_preserved']:
            raise ValueError('Complete immutable original-source pair index required')
        files=dict(self.manifest['source_files_sha256'])
        files[str(self.folder/'manifest.json')]=file_hash(self.folder/'manifest.json')
        files.update({str(self.folder/name):digest for name,digest in self.manifest['artifact_files_sha256'].items()})
        self.guard=FileBindings(files)
        self.pack=CompleteCodePack(self.manifest['code_pack'],role,purpose='training' if role=='train' else 'audit')
        self.records=list(map(json.loads,(self.folder/(role+'-records.jsonl')).open()))
        self.pairs=np.load(self.folder/(role+'-negative-positions-i32.npy'),mmap_mode='r',allow_pickle=False)
        if len(self.records)!=len(self.pack) or any(r['role']!=role or r['local_position']!=i
                or r['global_position']!=int(self.pack.positions[i]) for i,r in enumerate(self.records)):
            raise ValueError('Exact complete sealed role/source order required')
        validate_pair_map(self.records,self.pairs)
        if not np.array_equal(self.pairs,pair_map(self.records,self.manifest['seed']+(0 if role=='train' else 1))):
            raise ValueError('Every original deterministic pair index must reconstruct')

    def __len__(self):
        return len(self.records)

    def example(self,position):
        self.guard.check()
        if type(position) is not int or not 0<=position<len(self):
            raise ValueError('Local original sealed-source index required')
        source=self.records[position]
        positive=self.pack.sequence(position)
        if hashlib.sha256(np.asarray(positive[0],dtype='<u2').tobytes()).hexdigest()!=source['full_input_sha256']:
            raise ValueError('Every original positive code input byte must match')
        other=int(self.pairs[position])
        negative=None if other<0 else mismatch_sequence(positive,self.pack.sequence(other))
        return dict(id=source['id'],role=self.role,language=source['language'],group_id=source['group_id'],
            positive=positive,negative=negative,negative_source_id=None if other<0 else self.records[other]['id'])
