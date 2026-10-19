"""Complete source-group-reserved native CPU assessment, never optimization.

Reserved means withheld from this new curriculum only. All historical model
exposure and arbitrary semantic equivalence remain uncertified. Model input
contains only the original complete question and explicit formatting request.
"""
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path

import torch
from tokenizers import Tokenizer

from .codenet_checkpoint_v1 import FileBindings
from .codenet_token_pack_v1 import CompleteCodePack
from .corpus import file_hash
from .diverse_images import atomic_json
from .function_repair_evaluation_CPU_v1 import native_actions
from .inference_rotary_cache_v1 import inference_cache
from .reasoning_source_oracles_v1 import expected,format_instruction,strict_reward


class PublicReasoningReserved:
    role='validation'
    purpose='evaluation'
    optimization_forbidden=True

    def __init__(self,folder,proof):
        self.folder=Path(folder).resolve();proof=Path(proof).resolve();admitted=json.loads(proof.read_text())
        self.pack=CompleteCodePack(self.folder,'validation',purpose='audit')
        if (not admitted.get('complete') or not admitted.get('admitted_to_training') or len(self.pack)!=289
                or admitted['admission_file_sha256']!=file_hash(self.folder/'admission.json')
                or admitted['dataset_manifest_sha256']!=file_hash(self.folder/'manifest.json')
                or admitted['actual_counts']!=self.pack.manifest['counts']):
            raise ValueError('The exact independently admitted whole reserved partition is required')
        self.source_files_sha256=dict(admitted['source_files_sha256'])
        self.source_files_sha256.update({str(proof):file_hash(proof),str(self.folder/'admission.json'):file_hash(self.folder/'admission.json')})
        self.guard=FileBindings(self.source_files_sha256);self.tokenizer=Tokenizer.from_file(str(self.folder/'tokenizer.json'))
        with gzip.open(self.folder/'original-rows.jsonl.gz','rt') as stream:original=list(map(json.loads,stream))
        self.rows=[original[int(position)] for position in self.pack.positions];self.cases=[]
        for position,row in enumerate(self.rows):
            ids,labels=self.pack.sequence(position);boundary=next(i for i,v in enumerate(labels) if v!=-100)
            prompt=ids[:boundary];answer=expected(row['original_entry'])
            horizon=256 if row['task'] in ['binary_matrix','rotate_matrix'] else 64
            if (row['split']!='validation' or row['response']!=answer or row['original_entry']['answer']!=answer
                    or row['prompt']!=row['original_entry']['question']+'\n'+format_instruction(row['task'])
                    or self.tokenizer.decode(prompt[1:],skip_special_tokens=False)!='User: '+row['prompt']+'\nAssistant:'
                    or self.tokenizer.decode(ids[boundary:-1],skip_special_tokens=False)!=answer
                    or ids[-1]!=2 or len(prompt)+horizon>4096 or any(not 4<=v<32768 for v in ids[1:-1])):
                raise ValueError('Every whole original reserved input/answer and complete generation budget must reconstruct')
            self.cases.append(dict(position=position,source_id=row['id'],source_group=row['group_id'],task=row['task'],
                original_prompt_ids=prompt,complete_answer_EOS_ids=ids[boundary:],native_generation_budget=horizon,
                optimization_forbidden=True))
        if len(self.cases)!=289 or len({row['task'] for row in self.rows})!=12:raise ValueError('Every reserved example and all twelve task families are required')

    def grade(self,position,actions):
        if type(position) is not int or not 0<=position<len(self.cases):raise ValueError('Exact reserved-only position required')
        case=self.cases[position];horizon=case['native_generation_budget']
        if (not isinstance(actions,list) or not 1<=len(actions)<=horizon
                or any(type(v) is not int or not 0<=v<32768 for v in actions) or 2 in actions[:-1]
                or actions[-1]!=2 and len(actions)!=horizon):raise ValueError('Actual unchanged native termination and budget required')
        stopped=actions[-1]==2;body=actions[:-1] if stopped else actions;special=any(value<4 for value in body)
        response=self.tokenizer.decode(body,skip_special_tokens=False)
        correct=bool(strict_reward(response,self.rows[position]['original_entry']))
        return dict(correct=bool(stopped and not special and correct),canonical_text_correct=correct,
            actually_stopped_on_EOS=stopped,emitted_special_token_before_terminal_EOS=special,response=response,
            source_role='validation',optimization_forbidden=True,native_output_not_repaired=True)


@torch.no_grad()
def evaluate(model,checkpoint,folder,population,*,checked=lambda:None):
    checkpoint=Path(checkpoint).resolve();folder=Path(folder)
    if (type(population) is not PublicReasoningReserved or folder.exists() or model.training
            or next(model.parameters()).device.type!='cpu' or not checkpoint.is_file()
            or file_hash(checkpoint.parent/'tokenizer.json')!=file_hash(population.folder/'tokenizer.json')):
        raise ValueError('Exact eval-mode CPU model, immutable endpoint, tokenizer and complete reserved population required')
    population.guard.check(full=True);endpoint_sha=file_hash(checkpoint);folder.mkdir();rows=[];events=[]
    with (folder/'native-actions.jsonl').open('x') as journal,(folder/'actual-native-results.jsonl').open('x') as results:
        def record(event):events.append(event);journal.write(json.dumps(event)+'\n');journal.flush()
        with torch.autocast('cpu',dtype=torch.bfloat16),inference_cache(),native_actions(model,record):
            for case in population.cases:
                checked();population.guard.check();prefix=case['original_prompt_ids'];ids=torch.tensor([prefix])
                generated=model.generate(ids,max_new_tokens=case['native_generation_budget'],eos_token_id=2,use_cache=True)
                actions=generated[0,len(prefix):].tolist();event=events[-1]
                if (event['original_input_token_ids']!=prefix or event['actual_generated_token_ids']!=actions
                        or event['requested_native_action_budget']!=case['native_generation_budget']):
                    raise ValueError('Every actual delivered whole prefix and generated action must match its original case')
                score=population.grade(case['position'],actions)
                row=dict(position=case['position'],source_id=case['source_id'],source_group=case['source_group'],task=case['task'],
                    original_prompt_ids=prefix,actual_generated_token_ids=actions,original_native_generation_budget=case['native_generation_budget'],
                    actual_native_event_position=event['case_position'],score=score)
                rows.append(row);results.write(json.dumps(row)+'\n');results.flush()
    if len(rows)!=289 or len(events)!=289 or file_hash(checkpoint)!=endpoint_sha:raise ValueError('The complete exact endpoint and all original reserved deliveries must remain')
    population.guard.check(full=True)
    summary=dict(examples=289,correct=sum(row['score']['correct'] for row in rows),
        per_task={name:dict(examples=sum(row['task']==name for row in rows),correct=sum(row['task']==name and row['score']['correct'] for row in rows))
                  for name in sorted({row['task'] for row in rows})})
    report=dict(complete=True,checkpoint_sha256=endpoint_sha,source_files_sha256=population.source_files_sha256,
        **summary,all_original289cases_and_native64or256budgets_preserved=True,all_model_inputs_exclude_reference_answer_tokens=True,
        source_role='validation',optimization_forbidden=True,reserved_from_this_new_curriculum_only=True,
        historical_model_source_exposure_or_semantic_independence_not_certified=True,
        inference_device_type='cpu',inference_precision='bf16',actual_CPU_threads=torch.get_num_threads(),
        native_action_journal_sha256=file_hash(folder/'native-actions.jsonl'),native_results_sha256=file_hash(folder/'actual-native-results.jsonl'),
        new_model_updates=0,candidate_promoted=False,goal_achieved=False)
    atomic_json(folder/'complete.json',report)
    return report
