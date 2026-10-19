"""Native public-pretraining models initialized from a complete own code epoch.

Final loads use the complete packaged student/tokenizer only. Source digests and
training provenance are retained; no starting checkpoint is loaded for final
inference. A production controller must separately prove full data admission,
source scheduling, actual update counts, GPU ownership and capacity.
"""
import json
import re
from pathlib import Path

import torch

from .codenet_checkpoint_v1 import FileBindings
from .corpus import file_hash
from .own_dense_standalone_v1 import export as standalone_export,load as standalone_load

PROFILE='own-public-raw-function-and-multimodal-retention-pretraining-v1'


def validate_start(source,raw):
    from .focused_code_continuation_data_v3 import CODE,EXECUTION,STATE
    selected=source.get('selected_source_budget',{});budget=source.get('complete_source_budget',{})
    if (source.get('source_start_is_complete_main') is not True
            or source.get('complete_new_code_and_execution_source_epoch_once') is not True
            or source.get('complete_public_state_source_epoch_once') is not True
            or raw.get('completed_updates')!=source['planned_steps']
            or any(n not in selected or n not in budget or budget[n].get('examples',0)<1
                or selected[n].get('presentations')!=budget[n]['examples'] for n in [CODE,EXECUTION,STATE])):
        raise ValueError('Only the complete own code/output/state epoch may initialize public pretraining')


def fresh(plan,device='cpu'):
    inputs=plan['complete_code_first_inputs']
    checkpoint=Path(inputs['checkpoint']);source_plan=Path(inputs['plan'])
    guard=FileBindings({str(checkpoint):inputs['checkpoint_sha256'],str(source_plan):inputs['plan_sha256']})
    source=json.loads(source_plan.read_text());raw=torch.load(checkpoint,map_location='cpu',weights_only=True,mmap=True)
    validate_start(source,raw)
    expected_sources=dict(code_first_continuation_plan_sha256=inputs['plan_sha256'],
        own_complete_main_initializer_sha256=source['initial_dense_settings']['initializer_sha256'])
    del raw
    model,metadata=standalone_load(checkpoint,expected_checkpoint_sha256=inputs['checkpoint_sha256'],expected_sources=expected_sources)
    if (model.counts()!=plan['actual_counts'] or metadata['tokenizer_sha256']!=plan['tokenizer_sha256']
            or metadata['completed_updates']!=source['planned_steps']):
        raise ValueError('Every complete own starting value, tokenizer and update cursor must match')
    guard.check(full=True)
    return model.to(device),dict(metadata,starting_policy_is_complete_own_code_first=True)


def sources(plan,digest):
    return dict(public_pretraining_run_plan_sha256=digest,
        complete_own_code_first_standalone_sha256=plan['complete_code_first_inputs']['checkpoint_sha256'],
        original_public_raw_admission_sha256=plan['public_raw_admission_sha256'],
        original_verified_function_admission_sha256=plan['public_function_admission_sha256'])


def provenance(plan,arm,completed):
    step=re.fullmatch(r'step-(\d+)',arm)
    if arm not in {'initial','trained','capacity'} and step is None:
        raise ValueError('A declared public-pretraining native arm is required')
    if (type(completed) is not int or not 0<=completed<=plan['planned_steps']
            or arm=='initial' and completed!=0 or arm=='trained' and completed!=plan['planned_steps']
            or step is not None and int(step[1])!=completed):
        raise ValueError('Exact original public-pretraining arm and source cursor required')
    return dict(profile=PROFILE,arm=arm,source_schedule_sha256=plan['source_schedule_sha256'],
        planned_updates=plan['planned_steps'],actual_completed_updates=completed,
        whole_declared_source_epoch_completed=arm=='trained' and completed==plan['planned_steps'],
        complete_own_code_first_source_inputs=plan['complete_code_first_inputs'],
        public_raw_admission_sha256=plan['public_raw_admission_sha256'],
        public_function_admission_sha256=plan['public_function_admission_sha256'],
        no_external_pretrained_model_or_neural_teacher=True,new_student_parameters=0,
        actual_generated_or_all_round_capability_not_established_by_export=True)


def export(model,plan,digest,arm,completed):
    if model.counts()!=plan['actual_counts']:raise ValueError('Every actual own text/image/word/head value must match')
    return standalone_export(model,sources=sources(plan,digest),provenance=provenance(plan,arm,completed),
        completed_updates=completed,tokenizer_sha256=plan['tokenizer_sha256'])


def load(checkpoint,run_plan,arm,*,expected_checkpoint_sha256,expected_plan_sha256,device='cpu'):
    checkpoint,run_plan=Path(checkpoint),Path(run_plan)
    guard=FileBindings({str(checkpoint):expected_checkpoint_sha256,str(run_plan):expected_plan_sha256})
    plan=json.loads(run_plan.read_text());raw=torch.load(checkpoint,map_location='cpu',weights_only=True,mmap=True)
    if raw.get('provenance')!=provenance(plan,arm,raw.get('completed_updates')):
        raise ValueError('Exact public-pretraining source/arm/complete-cursor provenance required')
    del raw
    model,metadata=standalone_load(checkpoint,expected_checkpoint_sha256=expected_checkpoint_sha256,
        expected_sources=sources(plan,expected_plan_sha256))
    if model.counts()!=plan['actual_counts'] or metadata['tokenizer_sha256']!=plan['tokenizer_sha256']:
        raise ValueError('The actual complete under1B student and its own tokenizer must match')
    guard.check(full=True)
    return model.eval().to(device),dict(metadata,public_pretraining_arm=arm,no_starting_model_loaded_for_final_inference=True)
