"""An explicit screened CodeContests CE-only optimization view.

Only admitted original publisher-train program positions can be requested.
Publisher accepted labels supply supervision; they are not execution rewards,
fresh human preferences, or an independent correctness certificate.
"""
import gzip
import json
from pathlib import Path

import torch

from .codenet_checkpoint_v1 import FileBindings
from .codenet_token_pack_v1 import CompleteCodePack
from .corpus import file_hash


class CodeContestsCEView:
    role='train'
    purpose='publisher_correct_complete_reference_CE_only'

    def __init__(self,root):
        self.root=Path(root).resolve();self.folder=self.root/'data/code-contests-text-screened-CE-view-v1'
        manifest_path=self.folder/'manifest.json';admission_path=self.folder/'admission.json'
        self.manifest=json.loads(manifest_path.read_text());admission=json.loads(admission_path.read_text())
        if (admission.get('admitted_to_CE_training') is not True or admission.get('admitted_to_execution_reward_training') is not False
                or admission['manifest_sha256']!=file_hash(manifest_path)
                or self.manifest['format']!='own-CodeContests-source-screened-complete-CE-view-v1'):
            raise ValueError('Exact complete CE-only source view admission required')
        self.guard=FileBindings(admission['source_files_sha256'])
        self.pack=CompleteCodePack(self.manifest['original_pack'],'train','audit')
        if file_hash(self.pack.folder/'manifest.json')!=self.manifest['original_pack_manifest_sha256']:
            raise ValueError('Exact original native pack required')
        positions_path=self.folder/'original-training-pack-positions.json'
        if file_hash(positions_path)!=self.manifest['positions_sha256']:
            raise ValueError('Every selected original training position must remain unchanged')
        self.positions=json.loads(positions_path.read_text())
        if (len(self.positions)!=self.manifest['native_counts']['presentations'] or len(set(self.positions))!=len(self.positions)
                or any(type(p) is not int or not 0<=p<len(self.pack.index) or int(self.pack.index[p]['role'])!=0 for p in self.positions)):
            raise ValueError('Distinct original publisher-train positions only; held roles cannot enter optimization')
        all_rows=[]
        with gzip.open(self.pack.folder/'original-rows.jsonl.gz','rt') as stream:
            all_rows=[json.loads(line) for line in stream]
        self.rows=[all_rows[p] for p in self.positions]
        if any(r['split']!='train' or r['publisher_split']!='train' or r['fresh_local_execution_pending'] is not True for r in self.rows):
            raise ValueError('Honest original train role and publisher-label execution scope required')
        self.tokenizer_path=self.pack.folder/'tokenizer.json'
        self.provenance=dict(view_manifest_sha256=file_hash(manifest_path),view_admission_sha256=file_hash(admission_path),
            original_complete_program_presentations=len(self.positions),original_unique_problem_sources=len({r['source_problem_id'] for r in self.rows}),
            tokenizer_sha256=file_hash(self.tokenizer_path),human_preferences=False,genuine_human_RLHF=False,
            local_execution_rewards_admitted=False,publisher_correctness_not_new_local_execution=True,
            source_semantic_and_full_ancestral_independence_not_certified=True,model_updates=0)

    def __len__(self):return len(self.positions)

    def row(self,position):
        if type(position) is not int or not 0<=position<len(self):raise ValueError('An admitted original train-view position is required')
        return self.rows[position]

    def sequence(self,position):
        self.row(position);entry=self.pack.index[self.positions[position]]
        offset=int(entry['offset']);length=int(entry['length']);start=int(entry['target_start'])
        ids=self.pack.ids[offset:offset+length].astype('int64').tolist()
        if ids[0]!=1 or ids[-1]!=2 or not 1<=start<len(ids)-1:raise ValueError('Whole original native input and EOS response span required')
        return ids,[-100]*start+ids[start:]

    def batch(self,position,device):
        ids,labels=self.sequence(position)
        return dict(input_ids=torch.tensor([ids],dtype=torch.long,device=device),
            labels=torch.tensor([labels],dtype=torch.long,device=device),
            attention_mask=torch.ones((1,len(ids)),dtype=torch.bool,device=device),pixel_values=None)
