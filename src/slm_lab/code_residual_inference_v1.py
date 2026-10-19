"""Load one completed own residual pilot into the unchanged decoder interface."""
from dataclasses import asdict
from pathlib import Path

import torch

from .code_residual_branch_v1 import RoutedCodeResidual
from .code_residual_checkpoint_v1 import FORMAT
from .code_residual_pilot_v1 import PROFILES
from .corpus import file_hash
from .recurrent_ablation import state_digest
from .training import load_checkpoint


def load_residual_policy(checkpoint, base_checkpoint, tokenizer, *, expected_checkpoint_sha256,
                         expected_base_sha256, expected_plan_sha256, device='cpu'):
    checkpoint, base_checkpoint, tokenizer = map(Path,(checkpoint,base_checkpoint,tokenizer))
    if (file_hash(checkpoint) != expected_checkpoint_sha256 or file_hash(base_checkpoint) != expected_base_sha256):
        raise ValueError('Exact complete own residual and original base checkpoint bytes required')
    state = torch.load(checkpoint,map_location='cpu',weights_only=True,mmap=True)
    token_sha = file_hash(tokenizer)
    if (state.get('format') != FORMAT or state.get('stage') != 'code_residual_lr_pilot_v1'
            or state.get('world_size') != 1 or state.get('base_checkpoint_sha256') != expected_base_sha256
            or state.get('tokenizer_sha256') != token_sha or state.get('run_plan_sha256') != expected_plan_sha256
            or state.get('pilot_profile') not in PROFILES
            or type(state.get('steps')) is not int or state['steps'] < 1
            or state['steps'] != state.get('planned_steps')
            or state.get('candidate_promoted') is not False or state.get('goal_achieved') is not False):
        raise ValueError('Exact completed internally selected own pilot/plan/tokenizer required')
    # Optimizer moments remain CPU-mapped and are discarded before device movement.
    state.pop('optimizer',None);state.pop('rng',None)
    base,prior = load_checkpoint(base_checkpoint,'cpu',training_state=False)
    if (prior['tokenizer_sha256'] != token_sha or asdict(base.config) != state.get('config')
            or state_digest(base.state_dict()) != state.get('frozen_backbone_state_sha256')):
        raise ValueError('Every original own frozen base tensor/config/tokenizer must match')
    routed = RoutedCodeResidual(base,width=state['residual_width'])
    if (routed.parameter_counts() != state.get('model_counts')
            or [n for n,_ in routed.residuals.named_parameters()] != state.get('residual_parameter_names')):
        raise ValueError('Exact unique parameter count and residual tensor order required')
    routed.load_adapter_state(state['residuals']);routed.eval();routed.to(device)
    if sum(p.numel() for p in routed.code.parameters()) != routed.parameter_counts()['total_unique']:
        raise ValueError('The returned inference route must include every unique model parameter')
    metadata = dict(stage='code_residual_lr_pilot_v1',steps=state['steps'],planned_steps=state['planned_steps'],
                    tokenizer_sha256=token_sha,data_sha256=state['source_manifest_sha256'],world_size=1,
                    own_base_checkpoint_sha256=expected_base_sha256,run_plan_sha256=expected_plan_sha256,
                    model_counts=state['model_counts'],pilot_profile=state['pilot_profile'],
                    route='code',optimizer_loaded_on_GPU=False,candidate_promoted=False)
    return routed.code,metadata
