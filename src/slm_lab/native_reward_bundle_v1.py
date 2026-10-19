"""Attach an exactly completed current-native reward head without training data.

The caller supplies the bundle's expected bytes, source identities and fixed
training options. Complete pair cursors, all head/Adam/RNG values and the live
native basis are validated before copying only the three reward tensors.
"""
import math
from pathlib import Path

import torch

from .codenet_stage_v1 import EpochOrder
from .code_residual_checkpoint_v1 import _cpu, _validate_adam, _validate_rng
from .corpus import file_hash
from .native_human_reward_v1 import (HEAD_FORMAT, STATE_FORMAT, ROUTE, POPULATIONS,
    FrozenSharedNativeReward, NativeRewardHeadStage, signature, sources_ok)
from .own_dense_growth_v1 import OwnDenseAssistant
from .whole_human_sft_evidence_v1 import tensor_digest

FORMAT = 'own-complete-current-native-human-reward-head-bundle-v1'


def export(stage):
    if type(stage) is not NativeRewardHeadStage: raise ValueError('The actual current-native reward head stage is required')
    return dict(format=FORMAT, head=stage.finish(), training_state=stage.state(), options=dict(stage.options),
        sources=dict(stage.features.sources), native_checkpoint_and_tokenizer_still_required=True,
        training_features_not_required_for_head_attachment=True, candidate_promoted=False, goal_achieved=False)


def attach(base, path, *, expected_sha256, expected_sources, expected_options, allow_mechanics=False):
    path = Path(path)
    if type(base) is not OwnDenseAssistant or file_hash(path) != expected_sha256 or not sources_ok(expected_sources):
        raise ValueError('Exact native, completed head bytes and external expected source identities are required')
    bundle = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    head, state = bundle.get('head', {}), bundle.get('training_state', {})
    population = head.get('actual_train_pairs'); completed = state.get('completed')
    full = base.counts()['total_unique'] == 984399690
    small = allow_mechanics is True and base.counts()['total_unique'] < 10**7 and next(base.parameters()).device.type == 'cpu'
    if (not full and not small or bundle.get('format') != FORMAT or bundle.get('options') != expected_options
            or bundle.get('sources') != expected_sources or bundle.get('candidate_promoted') is not False
            or bundle.get('goal_achieved') is not False or head.get('format') != HEAD_FORMAT or head.get('complete') is not True
            or head.get('sources') != expected_sources or head.get('feature_route') != ROUTE
            or state.get('format') != STATE_FORMAT or state.get('options') != expected_options or state.get('sources') != expected_sources
            or state.get('feature_route') != ROUTE or state.get('candidate_promoted') is not False or state.get('goal_achieved') is not False
            or type(population) is not int or population < 1 or type(completed) is not int or completed != expected_options['steps']
            or completed != math.ceil(population/expected_options['batch_size']) or head.get('actual_updates') != completed
            or head.get('actual_native_values') != base.counts()['total_unique']
            or full and (population != POPULATIONS['train'] or head.get('full_original_train_epoch_complete') is not True
                or state.get('full_original_role_population') is not True)
            or state.get('initial_head_sha256') != tensor_digest(base.heads['reward'].named_parameters())
            or head.get('initial_head_sha256') != state['initial_head_sha256']
            or head.get('trained_head_sha256') == head['initial_head_sha256']
            or head.get('frozen_native_encoder_sha256') != expected_sources['frozen_native_encoder_sha256']
            or tensor_digest(base.network.named_parameters()) != expected_sources['frozen_native_encoder_sha256']):
        raise ValueError('Complete original human epoch, exact head/native feature basis and source/options are required before mutation')
    order = EpochOrder(population, expected_options['seed']); metrics = state.get('metrics', [])
    if len(metrics) != completed: raise ValueError('Every complete original reward update must be retained')
    for step, row in enumerate(metrics):
        positions = [order.at(i) for i in range(step*expected_options['batch_size'], min((step+1)*expected_options['batch_size'], population))]
        if (row.get('step') != step+1 or row.get('source_positions') != positions or row.get('pairs') != len(positions)
                or any(not isinstance(row.get(k), (int, float)) or not math.isfinite(row[k]) or row[k] < 0
                    for k in ['pair_mean_nll', 'gradient_norm_before_clip'])):
            raise ValueError('Every original unrepeated human pair and finite reward metric must reconstruct')
    named = list(base.heads['reward'].named_parameters()); values = head.get('learned_head', {}); saved = state.get('head', {})
    if (not isinstance(values, dict) or not isinstance(saved, dict)
            or values.keys() != dict(named).keys() or saved.keys() != values.keys()
            or any(not isinstance(t, torch.Tensor) or t.shape != dict(named)[n].shape or t.dtype != dict(named)[n].dtype
                or not bool(torch.isfinite(t).all()) or not torch.equal(t, saved[n]) for n, t in values.items())
            or head.get('trained_head_sha256') != tensor_digest(values.items())
            or head.get('head_values') != sum(p.numel() for _, p in named)):
        raise ValueError('Every exact completed native reward tensor and saved head must agree')
    _validate_adam(state.get('optimizer'), [p for _, p in named], completed); _validate_rng(state.get('rng'), 'cpu')
    group = state['optimizer']['param_groups'][0]
    if (group['lr'] != expected_options['learning_rate'] or tuple(group['betas']) != tuple(expected_options['betas'])
            or group['weight_decay'] != expected_options['weight_decay'] or group.get('foreach') is not False
            or group.get('fused') not in (False, None)):
        raise ValueError('The exact original complete-head ordinary AdamW options are required')
    ids = {id(p) for _, p in named}; fixed = [(n, p) for n, p in base.named_parameters() if id(p) not in ids]
    before = signature(p for _, p in fixed)
    if tensor_digest(fixed) != state.get('fixed_values_sha256') or file_hash(path) != expected_sha256:
        raise ValueError('Every original fixed native value and complete head file must remain exact before mutation')
    base.heads['reward'].load_state_dict(values, strict=True)
    if signature(p for _, p in fixed) != before or tensor_digest(fixed) != state['fixed_values_sha256']:
        raise ValueError('Only the trained reward tensors may change during attachment')
    return FrozenSharedNativeReward(base, _cpu(head), allow_mechanics=allow_mechanics)
