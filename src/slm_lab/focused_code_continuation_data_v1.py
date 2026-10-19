"""Whole new code/execution supervision with explicit broad multimodal replay.

Execution candidates cannot enter optimization before their own complete pack
is independently admitted. A preparation-only instance can inspect the already
admitted code and replay sources while that upstream work remains live.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path
import random

import torch

from .code_contests_CE_view_v1 import CodeContestsCEView
from .codenet_token_pack_v1 import CompleteCodePack
from .corpus import file_hash
from .expanded_joint_sources_v1 import ExpandedJointSources
from .fast_joint_data_v1 import CompleteBucketSchedule

CODE='new:CodeContests_complete_code'
EXECUTION='new:public_execution_prediction'
REPLAY=('human','facts:human_fact_questions','replay:math','variety',
        'later:complete_JS_parameter_headers','later:publisher_exercises:Python',
        'images:additional_photo_captions','images:original_visual_reasoning','replay:image_reading','words')


def collate_complete(sequences,device='cpu'):
    if not sequences:return None
    for ids,labels in sequences:
        if (not 3<=len(ids)<=4096 or len(ids)!=len(labels) or ids[0]!=1 or ids[-1]!=2
                or any(type(v) is not int or not 0<=v<32768 for v in ids)
                or any(v<4 for v in ids[1:-1])):
            raise ValueError('Whole original native context within4096 with genuine boundaries required')
        start=next((i for i,v in enumerate(labels) if v!=-100),len(labels))
        if not 1<=start<len(ids)-1 or labels[:start]!=[-100]*start or labels[start:]!=ids[start:]:
            raise ValueError('Complete original response/EOS supervision without prefix targets required')
    width=max(len(ids) for ids,_ in sequences)
    ids=torch.zeros((len(sequences),width),dtype=torch.long,device=device)
    labels=torch.full_like(ids,-100);mask=torch.zeros_like(ids,dtype=torch.bool)
    for i,(original,target) in enumerate(sequences):
        ids[i,:len(original)]=torch.tensor(original,dtype=torch.long,device=device)
        labels[i,:len(target)]=torch.tensor(target,dtype=torch.long,device=device);mask[i,:len(original)]=True
    return dict(input_ids=ids,labels=labels,attention_mask=mask,pixel_values=None)


class FocusedCodeContinuationData:
    role='train'

    def __init__(self,root,config,*,purpose='optimization',execution_folder=None):
        if purpose not in {'optimization','source_preparation'}:raise ValueError('Explicit training or data-preparation purpose required')
        self.root=Path(root).resolve();self.purpose=purpose
        self.code=CodeContestsCEView(self.root);self.replay=ExpandedJointSources(self.root,config)
        self.lengths={name:list(self.replay.lengths[name]) for name in REPLAY}
        self.targets={name:list(self.replay.targets[name]) for name in REPLAY}
        self.kinds={name:self.replay.kinds[name] for name in REPLAY}
        entries=self.code.pack.index[self.code.positions]
        self.lengths[CODE]=[int(v) for v in entries['length']]
        self.targets[CODE]=[int(a)-int(b) for a,b in zip(entries['length'],entries['target_start'])]
        self.kinds[CODE]='text';self.execution=None
        if purpose=='optimization':
            folder=Path(execution_folder) if execution_folder is not None else self.root/'data/code-contests-public-execution-traces-admitted-v1'
            self.execution=CompleteCodePack(folder,'train','training')
            if file_hash(folder/'tokenizer.json')!=self.code.provenance['tokenizer_sha256']:
                raise ValueError('One exact own tokenizer for code, execution and multimodal replay required')
            provenance=self.execution.manifest['provenance']
            if provenance.get('no_private_generated_or_held_inputs_or_outputs_used_as_model_text') is not True:
                raise ValueError('Explicit public-input-only execution supervision required')
            entries=self.execution.index[self.execution.positions]
            self.lengths[EXECUTION]=[int(v) for v in entries['length']]
            self.targets[EXECUTION]=[int(a)-int(b) for a,b in zip(entries['length'],entries['target_start'])]
            self.kinds[EXECUTION]='text'
        self.populations={name:len(values) for name,values in self.lengths.items()}
        if (any(not n for n in self.populations.values()) or any(n>4096 for values in self.lengths.values() for n in values)
                or config.max_text_length!=4096):raise ValueError('Every original family must have complete native examples within the current model')
        phase=self.root/'reports/phase-joint-v3/plan.json'
        self.source_files_sha256=dict(json.loads(phase.read_text())['source_files_sha256'])
        self.source_files_sha256.update({str(p):v for p,v in self.code.guard.files.items()})
        self.source_files_sha256[str(phase)]=file_hash(phase)
        if self.execution is not None:
            for name in ['manifest.json','admission.json']:
                self.source_files_sha256[str(self.execution.folder/name)]=file_hash(self.execution.folder/name)
            self.source_files_sha256.update({str(self.execution.folder/name):value for name,value in self.execution.manifest['files_sha256'].items()})

    def batch(self,name,positions,device='cpu'):
        if name not in self.lengths or any(type(p) is not int or not 0<=p<self.populations[name] for p in positions):
            raise ValueError('Exact available train family and original local source positions required')
        if not positions:return None
        if name==CODE:result=collate_complete([self.code.sequence(p) for p in positions],device)
        elif name==EXECUTION:
            if self.purpose!='optimization' or self.execution is None:raise ValueError('Execution candidate not admitted to optimization')
            result=collate_complete([self.execution.sequence(p) for p in positions],device)
        else:result=self.replay.batch(name,positions,device)
        if (int(result['attention_mask'].sum())!=sum(self.lengths[name][p] for p in positions)
                or int((result['labels'][:,1:]!=-100).sum())!=sum(self.targets[name][p] for p in positions)):
            raise ValueError('Every original complete context, output and EOS target must reach the real batch')
        return result

    def budget(self):
        return {name:dict(examples=len(values),complete_context_tokens=sum(values),response_EOS_targets=sum(self.targets[name]),kind=self.kinds[name])
                for name,values in self.lengths.items()}


class FocusedContinuationSchedule:
    """One whole code/execution pass, with one fixed replay batch per code batch."""
    def __init__(self,data,*,seed=972317,width=16):
        if type(seed) is not int or type(width) is not int or width<1 or CODE not in data.lengths:
            raise ValueError('A fixed complete-source training width and seed required')
        if data.purpose=='optimization' and EXECUTION not in data.lengths:
            raise ValueError('Production cannot omit the required admitted execution-supervision task')
        def batches(name,salt):
            schedule=CompleteBucketSchedule({name:list(data.lengths[name])},{name:width},seed+salt,bucket_updates=32)
            return [list(positions) for _,positions in schedule.entries]
        code=batches(CODE,0);execution=batches(EXECUTION,1) if EXECUTION in data.lengths else []
        replay_orders={};cursors={name:0 for name in REPLAY}
        for i,name in enumerate(REPLAY):
            if len(data.lengths[name])<width:raise ValueError('Distinct complete original replay batch needs enough sources')
            replay_orders[name]=batches(name,100+i)
        self.entries=[];used_execution=0
        for i,positions in enumerate(code):
            self.entries.append((CODE,positions));name=REPLAY[i%len(REPLAY)];order=replay_orders[name]
            chosen=list(order[cursors[name]%len(order)]);cursors[name]+=1
            self.entries.append((name,chosen))
            until=(i+1)*len(execution)//len(code)
            while used_execution<until:
                self.entries.append((EXECUTION,execution[used_execution]));used_execution+=1
        if used_execution!=len(execution):raise ValueError('Every complete execution example must occupy the declared schedule')
        self.seed=seed;self.width=width
        self.phase_for_step=['instruction_and_vision']*len(self.entries)
        self.phase_bounds={'instruction_and_vision':[0,len(self.entries)]}
        self.identity_sha256=hashlib.sha256(json.dumps(self.entries,separators=(',',':')).encode()).hexdigest()
        for family in [CODE]+([EXECUTION] if execution else []):
            actual=[p for name,positions in self.entries if name==family for p in positions]
            if sorted(actual)!=list(range(len(data.lengths[family]))):raise ValueError('Whole new-source epoch must visit each original position exactly once')

    def __len__(self):return len(self.entries)

    def at(self,step):
        if type(step) is not int or not 0<=step<len(self):raise IndexError('Complete continuation source schedule exhausted')
        name,positions=self.entries[step];return name,list(positions)

    def selected_budget(self,data):
        counts={}
        for name,positions in self.entries:
            row=counts.setdefault(name,dict(presentations=0,complete_context_tokens=0,response_EOS_targets=0,updates=0))
            row['presentations']+=len(positions);row['complete_context_tokens']+=sum(data.lengths[name][p] for p in positions)
            row['response_EOS_targets']+=sum(data.targets[name][p] for p in positions);row['updates']+=1
        return counts
