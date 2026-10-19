"""Read complete source-bound human-response packs with explicit source roles."""
import json
from pathlib import Path

import numpy as np
import torch

from .corpus import file_hash
from .whole_human_alignment_v2 import FORMAT


class WholeHumanData:
    def __init__(self, root, family, role, purpose):
        if family not in ('dialogue','preferences') or role not in ('train','valid') or purpose not in ('audit','capacity','optimization','evaluation'):
            raise ValueError('Exact human source family, role and input purpose required')
        if (purpose in ('optimization','capacity') and role!='train') or (purpose=='evaluation' and role!='valid'):
            raise ValueError('Original held sources never supply training or capacity gradients')
        self.root=Path(root).resolve();self.family=family;self.role=role;self.purpose=purpose
        self.manifest=json.loads((self.root/'manifest.json').read_text())
        if self.manifest['format']!=FORMAT or not self.manifest['full_original_responses_and_EOS_preserved']:
            raise ValueError('Whole source-response pack required')
        if purpose in ('optimization','capacity'):
            admission=json.loads((self.root/'admission.json').read_text())
            if (admission.get('training_admitted') is not True or admission['source_manifest_sha256']!=file_hash(self.root/'manifest.json')
                    or admission['independent_readback_sha256']!=file_hash(self.root/'independent-source-readback.json')
                    or admission['original_held_gradients_forbidden'] is not True):
                raise ValueError('Separate complete original-source readback and train-only admission required')
        name=family+'.'+role
        for suffix in ('.jsonl','.tokens.u16'):
            path=self.root/(name+suffix)
            if file_hash(path)!=self.manifest['files_sha256'][path.name]:
                raise ValueError('Every whole human index and native token byte must match')
        self.rows=[json.loads(line) for line in (self.root/(name+'.jsonl')).open()]
        self.tokens=np.memmap(self.root/(name+'.tokens.u16'),dtype='<u2',mode='r')
        self.fields=('response',) if family=='dialogue' else ('chosen','rejected')
        cursor=0
        for row in self.rows:
            if row['split']!=role or set(row['native'])!=set(self.fields):
                raise ValueError('Whole source row role and every response field required')
            for field in self.fields:
                item=row['native'][field]
                if (set(item)!=set(('offset','length','target_start')) or any(type(v) is not int for v in item.values())
                        or item['offset']!=cursor or not 1<=item['target_start']<item['length']<=self.manifest['config']['max_text_length']):
                    raise ValueError('Exact contiguous complete-response offsets and target boundary required')
                values=self.tokens[cursor:cursor+item['length']]
                if len(values)!=item['length'] or int(values[0])!=1 or int(values[-1])!=2:
                    raise ValueError('Whole own native BOS/response/EOS sequence required')
                cursor+=item['length']
        if len(self.rows)!=self.manifest['splits'][name]['examples'] or cursor!=len(self.tokens):
            raise ValueError('Every native token and original index row must be covered exactly once')

    def __len__(self):
        return len(self.rows)

    def sequence(self, position, field='response'):
        if type(position) is not int or not 0<=position<len(self) or field not in self.fields:
            raise ValueError('Exact original source position and response field required')
        item=self.rows[position]['native'][field]
        ids=self.tokens[item['offset']:item['offset']+item['length']].astype(np.int64).tolist()
        labels=[-100]*item['target_start']+ids[item['target_start']:]
        return ids,labels

    def batch(self, positions, field, device):
        sequences=[self.sequence(p,field) for p in positions]
        if not sequences:
            raise ValueError('Nonempty complete source batch required')
        length=max(len(ids) for ids,_ in sequences)
        ids=torch.zeros(len(sequences),length,dtype=torch.long)
        labels=torch.full_like(ids,-100);mask=torch.zeros_like(ids,dtype=torch.bool)
        for i,(seq,target) in enumerate(sequences):
            ids[i,:len(seq)]=torch.tensor(seq);labels[i,:len(seq)]=torch.tensor(target);mask[i,:len(seq)]=True
        return dict(input_ids=ids.to(device),labels=labels.to(device),attention_mask=mask.to(device))
