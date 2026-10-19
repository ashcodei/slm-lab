"""Admitted whole educational token streams with strict train/held purposes."""
import json
from pathlib import Path

import torch

from .corpus import file_hash
from .open_education_pack_v1 import FORMAT as CORPUS_FORMAT,WholeTokenWindows

ADMISSION_FORMAT='whole-original-open-education-train-only-admission-v1'


class EducationWindows:
    family='education'

    def __init__(self,root,tokenizer_sha256,length,role,purpose):
        if role not in ('train','valid') or purpose not in ('audit','optimization','capacity','evaluation'):
            raise ValueError('Explicit original educational role and purpose required')
        if (purpose in ('optimization','capacity') and role!='train') or (purpose=='evaluation' and role!='valid'):
            raise ValueError('Held original educational books may never supply optimization/capacity gradients')
        self.root=Path(root).resolve();self.role,self.purpose=role,purpose
        self.manifest=json.loads((self.root/'corpus.json').read_text())
        if self.manifest['format']!=CORPUS_FORMAT:raise ValueError('Exact complete original educational corpus format required')
        if purpose in ('optimization','capacity'):
            a=json.loads((self.root/'admission.json').read_text())
            expected={'corpus_sha256':'corpus.json','original_page_index_sha256':'original-pages.jsonl',
                'full_raw_token_readback_sha256':'independent-raw-source-and-token-readback.json',
                'retained_index_overlap_review_sha256':'cross-corpus-hash-index-review.json',
                'publisher_attribution_review_sha256':'publisher-attribution-review.json'}
            if (a.get('format')!=ADMISSION_FORMAT or a.get('training_admitted') is not True or a.get('admitted_role')!='train'
                    or a.get('held_original_gradients_forbidden') is not True
                    or any(a.get(key)!=file_hash(self.root/name) for key,name in expected.items())):
                raise ValueError('Complete original source/token/metadata readback and train-only admission required')
        self.windows=WholeTokenWindows(self.root,tokenizer_sha256,length,role)
        self.native_context=length

    def __len__(self):return len(self.windows)

    def window(self,position):return self.windows[position]

    def source_counts(self,positions):
        rows=[self.window(p) for p in positions]
        return dict(examples=len(rows),complete_context_tokens=sum(sum(r['attention_mask']) for r in rows),
                    response_EOS_targets=sum(r['real_next_token_targets'] for r in rows))

    def batch(self,positions,device='cpu'):
        if not positions:raise ValueError('Nonempty original educational window positions required')
        rows=[self.window(p) for p in positions]
        return {name:torch.tensor([r[name] for r in rows],dtype=torch.bool if name=='attention_mask' else torch.long,device=device)
                for name in ('input_ids','labels','attention_mask')}
