"""Complete existing broad sources plus admitted systems code, QA and interfaces.

This integrates existing prepared sources into real native batches. It does not
assert a new raw download, semantic independence or that training has occurred.
Foundation and instruction phases remain explicit: generation-format scores
alone do not measure whether raw next-token pretraining has learned its corpus.
"""
import gzip
import json
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer

from .codenet_fact_replay_v1 import FactReplayCorpus
from .codenet_later_instruction_replay_v2 import LaterInstructionReplay
from .codenet_packed_replay_v1 import PackedReplayCorpus,PackedPretrainingEncoder,FAMILY as PACKED
from .codenet_runner_v1 import VerifiedReplayEncoder
from .codenet_token_pack_v1 import CompleteCodePack,packed_encoder
from .corpus import file_hash
from .fast_joint_data_v1 import FastJointSources,CompleteBucketSchedule


class ExpandedJointSources(FastJointSources):
    def __init__(self,root,config):
        super().__init__(root,config);root=self.root
        tokenizer_path=root/'data/tokenizer-corpus/tokenizer.json';tokenizer=Tokenizer.from_file(str(tokenizer_path))
        token_sha=file_hash(tokenizer_path);self.expanded_owners={}
        self.packed=PackedReplayCorpus(root/'data/codenet-packed-pretraining-replay-v1')
        self.packed_encoder=PackedPretrainingEncoder(tokenizer,config,root)
        if self.packed.manifest['tokenizer_sha256']!=token_sha or self.packed.manifest['maximum_length']!=4096:
            raise ValueError('Complete original4096 own-tokenizer packed pretraining required')
        self.later=LaterInstructionReplay(root/'data/codenet-later-instruction-replay-v2')
        audit=root/'reports/codenet-later-instruction-replay-v2-independent-audit/results.json'
        checked=json.loads(audit.read_text())
        if (checked.get('source_manifest_sha256')!=file_hash(self.later.folder/'manifest.json')
                or checked.get('every_original_input_ID_and_response_target_count_rederived') is not True
                or self.later.manifest['tokenizer_sha256']!=token_sha):raise ValueError('Every actual original later instruction input/mask must be separately read back')
        self.fact=FactReplayCorpus(root/'data/codenet-fact-replay-v1')
        self.expanded_encoder=VerifiedReplayEncoder(tokenizer,config,root,448)
        self.views=CompleteCodePack(root/'data/verified-code-views-v1','train','training')
        self.view_encoder=packed_encoder(self.views,tokenizer,config)
        if file_hash(self.views.folder/'tokenizer.json')!=token_sha:raise ValueError('Exact own view tokenizer required')
        with gzip.open(self.views.folder/'original-rows.jsonl.gz','rt') as f:self.view_rows=list(map(json.loads,f))
        self.view_positions={name:[i for i,r in enumerate(self.view_rows) if r['view']==name]
            for name in ['prefix_completion','verified_statement_repair']}
        def add(name,lengths,targets,owner):
            if name in self.lengths or len(lengths)==0 or len(lengths)!=len(targets):raise ValueError('Distinct complete nonempty source family required')
            self.lengths[name]=[int(n) for n in lengths];self.targets[name]=[int(n) for n in targets]
            self.kinds[name]='text';self.expanded_owners[name]=owner
        index=self.packed.index['train'];add('packed:'+PACKED,index['length'],index['length'].astype(np.int64)-1,('packed',PACKED))
        for i,name in enumerate(self.later.families):
            rows=self.later.index[self.later.positions[name]];repeat=4 if i<6 else 1
            add('later:'+name,np.tile(rows['context_tokens'],repeat),np.tile(rows['target_tokens'],repeat),('later',name))
        add('facts:'+self.fact.families[0],self.fact.index['context_tokens'],self.fact.index['target_tokens'],('facts',self.fact.families[0]))
        for name,positions in self.view_positions.items():
            rows=self.views.index[self.views.positions[positions]]
            add('code_views:'+name,rows['length'],rows['length'].astype(np.int64)-rows['target_start'],('views',name))
        self.populations={name:len(values) for name,values in self.lengths.items()}
        if any(n>config.max_text_length for values in self.lengths.values() for n in values):raise ValueError('No source may be shortened to fit the actual model')

    def batch(self,name,positions,device='cpu'):
        if name not in self.expanded_owners:return super().batch(name,positions,device)
        if not positions:return None
        if any(type(p) is not int or not 0<=p<self.populations[name] for p in positions):raise ValueError('Exact original train source positions required')
        kind,family=self.expanded_owners[name]
        if kind=='packed':batch=self.packed_encoder.batch([self.packed.row(family,p) for p in positions],None,'cpu')
        elif kind=='views':batch=self.view_encoder.batch([dict(packed_code_index=self.view_positions[family][p]) for p in positions],'response','cpu')
        else:
            corpus=self.later if kind=='later' else self.fact
            batch=self.expanded_encoder.batch([corpus.row(family,p) for p in positions],'response','cpu')
        if (int(batch['attention_mask'].sum())!=sum(self.lengths[name][p] for p in positions)
                or int((batch['labels'][:,1:]!=-100).sum())!=sum(self.targets[name][p] for p in positions)):
            raise ValueError('Every complete original native context and next/response/EOS target must reconstruct')
        return {k:(v.to(device) if isinstance(v,torch.Tensor) else v) for k,v in batch.items()}

    def phases(self):
        foundation={'education','variety','packed:'+PACKED}
        if not foundation<=self.populations.keys():raise ValueError('All declared raw/book foundation sources required')
        return dict(foundation=sorted(foundation),instruction_and_vision=sorted(set(self.populations)-foundation))


class CompletePhaseSchedule:
    """Every selected original source once: foundation then instructions/vision."""
    def __init__(self,data,widths,seed):
        self.entries=[];self.phase_for_step=[];self.phase_bounds={};self.seed=seed
        if set(widths)!=set(data.lengths):raise ValueError('An explicit width for every complete source family required')
        for i,(phase,names) in enumerate(data.phases().items()):
            schedule=CompleteBucketSchedule({n:data.lengths[n] for n in names},{n:widths[n] for n in names},seed+i*100003)
            self.phase_bounds[phase]=[len(self.entries),len(self.entries)+len(schedule)]
            self.entries.extend(schedule.entries);self.phase_for_step.extend([phase]*len(schedule))

    def __len__(self):return len(self.entries)

    def at(self,step):
        if type(step) is not int or not 0<=step<len(self):raise IndexError('Complete foundation/instruction schedule exhausted')
        name,positions=self.entries[step];return name,list(positions)

    def rank_positions(self,step,rank,world):
        if type(world) is not int or world<1 or type(rank) is not int or not 0<=rank<world:raise ValueError('Exact distributed replica required')
        name,positions=self.at(step);return name,positions[rank::world]
