"""Admitted whole public reasoning sources and non-executing native rewards.

This adapter exposes only the sealed train role. Reward includes the declared
native protocol: canonical answer, actual terminal EOS, and no emitted special
tokens besides that final EOS. A horizon stop is recorded, never repaired.
"""
import gzip
import hashlib
import json
from pathlib import Path
import random

import torch
from tokenizers import Tokenizer

from .codenet_checkpoint_v1 import FileBindings
from .codenet_token_pack_v1 import CompleteCodePack
from .corpus import file_hash
from .reasoning_source_oracles_v1 import expected,format_instruction,strict_reward


class PublicReasoningTrainData:
    role='train'
    purpose='optimization'

    def __init__(self,folder,proof):
        self.folder=Path(folder).resolve();proof=Path(proof).resolve()
        report=json.loads(proof.read_text());admission=json.loads((self.folder/'admission.json').read_text())
        self.pack=CompleteCodePack(self.folder,'train',purpose='training')
        if (not report.get('complete') or not report.get('admitted_to_training')
                or report.get('model_has_consumed_rows') is not False
                or report['admission_file_sha256']!=file_hash(self.folder/'admission.json')
                or report['dataset_manifest_sha256']!=file_hash(self.folder/'manifest.json')
                or report['actual_counts']!=self.pack.manifest['counts']
                or report['actual_counts']!=admission['actual_counts']
                or report['complete_original_draws_regenerated']!=3072
                or report['complete_protected_input_stems_rechecked']!=31349
                or report['actual_packed_sequences_rebuilt']!=2857
                or len(self.pack)!=2568):
            raise ValueError('The exact independently admitted full public train partition is required')
        self.source_files_sha256=dict(report['source_files_sha256'])
        self.source_files_sha256.update({str(proof):file_hash(proof),
            str(self.folder/'admission.json'):file_hash(self.folder/'admission.json')})
        self.guard=FileBindings(self.source_files_sha256)
        self.tokenizer=Tokenizer.from_file(str(self.folder/'tokenizer.json'))
        with gzip.open(self.folder/'original-rows.jsonl.gz','rt') as stream:all_rows=list(map(json.loads,stream))
        if len(all_rows)!=2857:raise ValueError('All original packed representatives are required')
        self.rows=[all_rows[int(position)] for position in self.pack.positions]
        self.slots=list(range(len(self.rows)))
        self.manifest_sha256=file_hash(self.folder/'manifest.json')
        for position,row in enumerate(self.rows):
            ids,labels=self.pack.sequence(position);start=next(i for i,v in enumerate(labels) if v!=-100)
            answer=expected(row['original_entry'])
            if (row['split']!='train' or row['response']!=answer or row['original_entry']['answer']!=answer
                    or row['prompt']!=row['original_entry']['question']+'\n'+format_instruction(row['task'])
                    or self.tokenizer.decode(ids[1:start],skip_special_tokens=False)!='User: '+row['prompt']+'\nAssistant:'
                    or self.tokenizer.decode(ids[start:-1],skip_special_tokens=False)!=answer
                    or ids[-1]!=2 or any(not 4<=v<32768 for v in ids[1:-1])):
                raise ValueError('Every complete original train question/answer/native mask must reconstruct')

    def __len__(self):return len(self.rows)

    def row(self,position):
        if type(position) is not int or not 0<=position<len(self):raise ValueError('Only an exact admitted local train position is allowed')
        return self.rows[position]

    def sequence(self,position):
        self.row(position)
        return self.pack.sequence(position)

    def information(self,position):
        row=self.row(position);ids,labels=self.sequence(position)
        start=next(i for i,v in enumerate(labels) if v!=-100)
        return dict(position=position,source_id=row['id'],task=row['task'],group_id=row['group_id'],
            original_prompt_positions=start,complete_context_tokens=len(ids),response_EOS_targets=len(ids)-start,
            complete_native_input_sha256=hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest(),
            full_training_action_budget=256 if row['task'] in ('binary_matrix','rotate_matrix') else 64)

    def complete_schedule(self,*,epochs=3,width=16,seed=982113):
        if any(type(value) is not int or value<1 for value in [epochs,width,seed]):
            raise ValueError('Explicit complete source epochs, width and seed are required')
        result=[]
        for epoch in range(epochs):
            order=list(self.slots);random.Random(seed+epoch).shuffle(order)
            result.extend(order[i:i+width] for i in range(0,len(order),width))
        return result

    def budget(self,entries):
        if (not isinstance(entries,list) or not entries
                or any(not isinstance(row,list) or not row or len(row)!=len(set(row)) for row in entries)):
            raise ValueError('Complete nonempty source batches without duplicate slots within a step are required')
        info=[self.information(position) for entry in entries for position in entry]
        return dict(updates=len(entries),source_presentations=len(info),
            complete_context_tokens=sum(row['complete_context_tokens'] for row in info),
            response_EOS_targets=sum(row['response_EOS_targets'] for row in info),
            full_native_generation_horizon_sum=sum(row['full_training_action_budget'] for row in info))

    def identity(self,entries):
        self.budget(entries)
        return hashlib.sha256(json.dumps(dict(admitted_manifest_sha256=self.manifest_sha256,
            complete_train_source_batches=entries),sort_keys=True,separators=(',',':')).encode()).hexdigest()

    def batch(self,position,device='cpu'):
        ids,labels=self.sequence(position)
        return dict(input_ids=torch.tensor([ids],device=device),labels=torch.tensor([labels],device=device),
            attention_mask=torch.ones((1,len(ids)),dtype=torch.bool,device=device))

    def score(self,position,actions,*,horizon):
        row=self.row(position)
        if (type(horizon) is not int or not 1<=horizon<=self.information(position)['full_training_action_budget']
                or not isinstance(actions,list) or not 1<=len(actions)<=horizon
                or any(type(v) is not int or not 0<=v<32768 for v in actions)
                or 2 in actions[:-1] or actions[-1]!=2 and len(actions)!=horizon):
            raise ValueError('Every actual bounded trajectory must end at its first EOS or declared horizon')
        ended=actions[-1]==2;body=actions[:-1] if ended else actions
        response=self.tokenizer.decode(body,skip_special_tokens=False)
        special=any(v<4 for v in body)
        correct_text=bool(strict_reward(response,row['original_entry']))
        reward=float(ended and not special and correct_text)
        status=('invalid_emitted_special_token' if special else 'horizon_without_EOS' if not ended
                else 'correct_canonical_answer_with_EOS' if correct_text else 'incorrect_canonical_answer')
        return dict(reward=reward,status=status,source_id=row['id'],source_group=row['group_id'],
            original_source_role='train',response=response,actions=list(actions),action_horizon=horizon,
            actually_stopped_on_EOS=ended,canonical_text_correct=correct_text,emitted_special_token_before_terminal_EOS=special,
            response_text_never_executed=True,emitted_tokens_never_filtered_or_repaired=True,
            externally_generated_teacher_or_human_preference=False)
