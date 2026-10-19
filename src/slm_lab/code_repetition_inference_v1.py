"""Strict raw inference for each fixed endpoint of the repetition objective pilot."""
from dataclasses import asdict
from pathlib import Path

import torch

from .code_residual_branch_v1 import RoutedCodeResidual
from .code_residual_checkpoint_v1 import FORMAT
from .corpus import file_hash
from .recurrent_ablation import state_digest
from .training import load_checkpoint

PROFILES=(dict(name='CE_control',alpha=0.),dict(name='CE_plus_repetition_UL',alpha=.1))


def load_objective_policy(checkpoint,base_checkpoint,tokenizer,*,expected_checkpoint_sha256,
        expected_base_sha256,expected_plan_sha256,expected_starting_policy_sha256,
        expected_selection_sha256,expected_profile,device='cpu'):
    checkpoint,base_checkpoint,tokenizer=map(Path,(checkpoint,base_checkpoint,tokenizer))
    if (expected_profile not in PROFILES or file_hash(checkpoint)!=expected_checkpoint_sha256
            or file_hash(base_checkpoint)!=expected_base_sha256):
        raise ValueError('One exact frozen own objective endpoint/base/profile required')
    state=torch.load(checkpoint,map_location='cpu',weights_only=True,mmap=True)
    token_sha=file_hash(tokenizer)
    if (state.get('format')!=FORMAT or state.get('stage')!='code_repetition_objective_pilot_v1'
            or state.get('world_size')!=1 or state.get('steps')!=128 or state.get('planned_steps')!=128
            or state.get('objective_profile')!=expected_profile or state.get('fresh_optimizer') is not True
            or state.get('training_sources')!=512 or state.get('training_rollout_sources')!=128
            or state.get('rollout_generation_budget')!=128 or state.get('regularizer_ngram')!=16
            or state.get('reference_repeated_ngrams_exempt') is not True
            or state.get('base_checkpoint_sha256')!=expected_base_sha256
            or state.get('starting_policy_checkpoint_sha256')!=expected_starting_policy_sha256
            or state.get('original_selection_sha256')!=expected_selection_sha256
            or state.get('run_plan_sha256')!=expected_plan_sha256 or state.get('tokenizer_sha256')!=token_sha
            or state.get('candidate_promoted') is not False or state.get('goal_achieved') is not False):
        raise ValueError('Exact complete128-update objective/selection/plan/tokenizer contract required')
    state.pop('optimizer',None);state.pop('rng',None)
    base,metadata=load_checkpoint(base_checkpoint,'cpu',training_state=False)
    if (metadata['tokenizer_sha256']!=token_sha or asdict(base.config)!=state.get('config')
            or state_digest(base.state_dict())!=state.get('frozen_backbone_state_sha256')):
        raise ValueError('Every original own frozen base tensor/config/tokenizer must match')
    routed=RoutedCodeResidual(base,width=state['residual_width'])
    if (routed.parameter_counts()!=state.get('model_counts')
            or [n for n,_ in routed.residuals.named_parameters()]!=state.get('residual_parameter_names')):
        raise ValueError('Exact unique parameter count and residual inventory required')
    routed.load_adapter_state(state['residuals']);routed.eval();routed.to(device)
    if sum(p.numel() for p in routed.code.parameters())!=routed.parameter_counts()['total_unique']:
        raise ValueError('Returned code route must count every unique parameter')
    return routed.code,dict(stage='code_repetition_objective_pilot_v1',steps=128,planned_steps=128,
        objective_profile=expected_profile,tokenizer_sha256=token_sha,data_sha256=state['source_manifest_sha256'],
        world_size=1,own_base_checkpoint_sha256=expected_base_sha256,run_plan_sha256=expected_plan_sha256,
        model_counts=state['model_counts'],route='code',optimizer_loaded_on_GPU=False,candidate_promoted=False)
