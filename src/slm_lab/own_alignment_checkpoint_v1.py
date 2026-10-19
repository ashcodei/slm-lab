"""New-only extension weights, exact Adam, RNG and source-cursor recovery.

This version supports one ordinary fixed-hyperparameter AdamW optimizer and a
caller-validated source cursor. It never restores or moves original tensors.
"""
from .codenet_checkpoint_v1 import capture_rng, restore_rng
from .code_residual_checkpoint_v1 import _cpu, _validate_adam, _validate_rng
from .point_grounding_evaluation_v1 import state_digest

import torch

FORMAT = 'own-alignment-new-only-Adam-RNG-source-cursor-v1'


def parameters(model, optimizer, groups):
    expected = model.extension_parameters(*groups)
    if (type(optimizer) is not torch.optim.AdamW or len(optimizer.param_groups) != 1
            or [id(p) for p in optimizer.param_groups[0]['params']] != [id(p) for p in expected]):
        raise ValueError('One exact-order explicitly selected new-only AdamW parameter group required')
    group = optimizer.param_groups[0]
    if (group.get('foreach') is not False or group.get('fused') not in (None, False)
            or any(group.get(k, False) for k in ('amsgrad', 'maximize', 'capturable', 'differentiable'))):
        raise ValueError('Ordinary non-foreach AdamW execution required')
    return expected


def make_state(model, optimizer, *, groups, sources, completed_steps, planned_steps, sampler_state, metrics, device):
    selected = parameters(model, optimizer, groups)
    if (type(completed_steps) is not int or type(planned_steps) is not int
            or not 0 < completed_steps <= planned_steps or not isinstance(metrics, list)
            or len(metrics) != completed_steps or not isinstance(sampler_state, dict) or not sampler_state):
        raise ValueError('Complete fixed update budget, source cursors and every recorded update required')
    adam = _cpu(optimizer.state_dict())
    _validate_adam(adam, selected, completed_steps)
    rng = capture_rng(device); _validate_rng(rng, device)
    names = {id(p): name for name, p in model.parts.named_parameters()}
    return dict(format=FORMAT, model=model.export(sources), optimizer=adam, groups=list(groups),
        optimized_names=[names[id(p)] for p in selected], rng=rng, world_size=1,
        steps=completed_steps, planned_steps=planned_steps, sampler_state=_cpu(sampler_state), metrics=_cpu(metrics),
        fixed_original_state_sha256=state_digest(model.original),
        candidate_promoted=False, goal_achieved=False)


def restore_state(model, optimizer, state, *, groups, expected_sources, planned_steps,
                  expected_sampler_state, device, allow_completed=False):
    """Reject any bad field before changing a weight, optimizer or global RNG."""
    selected = parameters(model, optimizer, groups)
    names = {id(p): name for name, p in model.parts.named_parameters()}
    if (state.get('format') != FORMAT or state.get('world_size') != 1 or state.get('groups') != list(groups)
            or state.get('optimized_names') != [names[id(p)] for p in selected]
            or type(planned_steps) is not int or state.get('planned_steps') != planned_steps
            or type(state.get('steps')) is not int or not 0 < state['steps'] <= planned_steps
            or (state['steps'] == planned_steps and not allow_completed)
            or state.get('sampler_state') != expected_sampler_state or not expected_sampler_state
            or not isinstance(state.get('metrics'), list) or len(state['metrics']) != state['steps']
            or state.get('candidate_promoted') is not False or state.get('goal_achieved') is not False):
        raise ValueError('Exact source cursor, architecture, optimizer order and complete update budget required')
    model.validate_export(state['model'], expected_sources=expected_sources)
    _validate_adam(state.get('optimizer'), selected, state['steps'])
    if state['optimizer']['param_groups'] != optimizer.state_dict()['param_groups']:
        raise ValueError('Exact fixed optimizer hyperparameters required')
    _validate_rng(state.get('rng'), device)
    if state.get('fixed_original_state_sha256') != state_digest(model.original):
        raise ValueError('Every original backbone/code/grounding tensor must remain identical')
    model.import_export(state['model'], expected_sources=expected_sources)
    optimizer.load_state_dict(state['optimizer'])
    optimizer.zero_grad(set_to_none=True)
    restore_rng(state['rng'], device)
    return state['steps'], _cpu(state['metrics'])
