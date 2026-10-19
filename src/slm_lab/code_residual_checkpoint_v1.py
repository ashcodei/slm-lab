"""Complete adapter/Adam/RNG state bound to one exact own frozen backbone."""
from dataclasses import asdict
import math
import random

import numpy as np
import torch

from .codenet_checkpoint_v1 import capture_rng, restore_rng
from .recurrent_ablation import state_digest


FORMAT = 'own-code-residual-training-state-v1'


def _digest(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def _optimizer_parameters(model, optimizer):
    model.assert_partition()
    expected = model.adapter_parameters()
    if (type(optimizer) is not torch.optim.AdamW or len(optimizer.param_groups) != 1
            or [id(p) for p in optimizer.param_groups[0]['params']] != [id(p) for p in expected]):
        raise ValueError('One AdamW group in exact residual-only parameter order required')
    group = optimizer.param_groups[0]
    if (group.get('foreach') is not False or group.get('fused') not in [None, False]
            or any(group.get(k, False) for k in ['amsgrad', 'maximize', 'capturable', 'differentiable'])):
        raise ValueError('Explicit ordinary non-foreach AdamW contract required')
    return expected


def _cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().to(device='cpu', copy=True)
    if isinstance(value, dict):
        return {k: _cpu(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_cpu(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_cpu(v) for v in value)
    return value


def _validate_rng(state, device):
    if not isinstance(state, dict) or set(state) != {'python', 'torch', 'numpy', 'cuda'}:
        raise ValueError('Complete original Python/NumPy/CPU/CUDA RNG state required')
    try:
        random.Random().setstate(state['python'])
        n = state['numpy']
        if set(n) != {'kind', 'keys', 'position', 'has_gauss', 'cached_gaussian'}:
            raise ValueError('NumPy generator inventory differs')
        np.random.RandomState().set_state((n['kind'], np.asarray(n['keys'], dtype=np.uint32),
                                           n['position'], n['has_gauss'], n['cached_gaussian']))
        torch.Generator(device='cpu').set_state(state['torch'])
        if torch.device(device).type == 'cuda':
            if state['cuda'] is None:
                raise ValueError('Original CUDA generator state missing')
            torch.Generator(device=device).set_state(state['cuda'])
        elif state['cuda'] is not None:
            raise ValueError('CPU restore cannot consume a CUDA training state')
    except (TypeError, RuntimeError, KeyError, ValueError, IndexError) as error:
        raise ValueError('Original RNG state or CPU/CUDA resume contract differs') from error


def _validate_adam(state, parameters, completed_steps):
    if not isinstance(state, dict) or set(state) != {'state', 'param_groups'} or len(state['param_groups']) != 1:
        raise ValueError('Complete ordinary AdamW state required')
    group = state['param_groups'][0]
    indices = list(range(len(parameters)))
    if group.get('params') != indices or set(state['state']) != set(indices):
        raise ValueError('Complete exact-order residual Adam moment inventory required')
    if (group.get('foreach') is not False or group.get('fused') not in [None, False]
            or any(group.get(k, False) for k in ['amsgrad', 'maximize', 'capturable', 'differentiable'])):
        raise ValueError('Saved AdamW execution contract differs')
    if (not all(type(group.get(k)) in [float, int] and math.isfinite(group[k])
                and group[k] >= 0 for k in ['lr', 'eps', 'weight_decay'])
            or group['lr'] == 0 or group['eps'] == 0
            or not isinstance(group.get('betas'), (tuple, list)) or len(group['betas']) != 2
            or any(type(v) not in [float, int] or not math.isfinite(v) or not 0 <= v < 1 for v in group['betas'])):
        raise ValueError('Finite saved AdamW hyperparameters required')
    for i, parameter in enumerate(parameters):
        value = state['state'][i]
        if not isinstance(value, dict) or set(value) != {'step', 'exp_avg', 'exp_avg_sq'}:
            raise ValueError('Every residual needs both Adam moments and step')
        step = value['step']
        if (not isinstance(step, torch.Tensor) or step.numel() != 1
                or not bool(torch.isfinite(step).all()) or float(step) != completed_steps):
            raise ValueError('Every residual Adam step must match the completed update count')
        for name in ['exp_avg', 'exp_avg_sq']:
            tensor = value[name]
            if (not isinstance(tensor, torch.Tensor) or tensor.shape != parameter.shape
                    or tensor.dtype != parameter.dtype or not bool(torch.isfinite(tensor).all())):
                raise ValueError('Finite exact-shape/dtype Adam moments required')
        if bool(torch.any(value['exp_avg_sq'] < 0)):
            raise ValueError('Squared Adam moment cannot be negative')


def make_training_state(model, optimizer, *, base_checkpoint_sha256, tokenizer_sha256,
                        run_plan_sha256, completed_steps, planned_steps, device, metrics):
    parameters = _optimizer_parameters(model, optimizer)
    if (not all(_digest(v) for v in [base_checkpoint_sha256, tokenizer_sha256, run_plan_sha256])
            or type(completed_steps) is not int or type(planned_steps) is not int
            or not 0 < completed_steps <= planned_steps or not isinstance(metrics, list)):
        raise ValueError('Exact sources, update budget and recorded metrics required')
    state = _cpu(optimizer.state_dict())
    _validate_adam(state, parameters, completed_steps)
    rng = capture_rng(device)
    _validate_rng(rng, device)
    return dict(format=FORMAT, base_checkpoint_sha256=base_checkpoint_sha256,
                tokenizer_sha256=tokenizer_sha256, run_plan_sha256=run_plan_sha256,
                residual_width=model.width, config=asdict(model.config), model_counts=model.parameter_counts(),
                frozen_backbone_state_sha256=state_digest(model.base.state_dict()),
                residual_parameter_names=[name for name, _ in model.residuals.named_parameters()],
                residuals=model.adapter_state(), optimizer=state, rng=rng, world_size=1,
                steps=completed_steps, planned_steps=planned_steps, metrics=_cpu(metrics),
                candidate_promoted=False, goal_achieved=False)


def restore_training_state(model, optimizer, bundle, *, base_checkpoint_sha256, tokenizer_sha256,
                           run_plan_sha256, planned_steps, device, allow_completed=False):
    """Validate the complete state before changing residual weights or global RNG."""
    parameters = _optimizer_parameters(model, optimizer)
    if (not all(_digest(v) for v in [base_checkpoint_sha256, tokenizer_sha256, run_plan_sha256])
            or bundle.get('format') != FORMAT or bundle.get('world_size') != 1
            or bundle.get('base_checkpoint_sha256') != base_checkpoint_sha256
            or bundle.get('tokenizer_sha256') != tokenizer_sha256
            or bundle.get('run_plan_sha256') != run_plan_sha256
            or bundle.get('residual_width') != model.width or bundle.get('config') != asdict(model.config)
            or bundle.get('model_counts') != model.parameter_counts()
            or bundle.get('residual_parameter_names') != [n for n, _ in model.residuals.named_parameters()]
            or type(bundle.get('steps')) is not int or type(planned_steps) is not int
            or bundle.get('planned_steps') != planned_steps or not 0 < bundle['steps'] <= planned_steps
            or (bundle['steps'] == planned_steps and not allow_completed)
            or not isinstance(bundle.get('metrics'), list)
            or bundle.get('candidate_promoted') is not False or bundle.get('goal_achieved') is not False):
        raise ValueError('Exact own residual checkpoint, source/plan, parameter layout and update budget required')
    expected = model.residuals.state_dict()
    residuals = bundle.get('residuals')
    if (not isinstance(residuals, dict) or set(residuals) != set(expected)
            or any(not isinstance(residuals[k], torch.Tensor) or residuals[k].shape != expected[k].shape
                   or residuals[k].dtype != expected[k].dtype or not bool(torch.isfinite(residuals[k]).all()) for k in expected)):
        raise ValueError('Complete finite exact residual tensor inventory required')
    _validate_adam(bundle.get('optimizer'), parameters, bundle['steps'])
    _validate_rng(bundle.get('rng'), device)
    if (not _digest(bundle.get('frozen_backbone_state_sha256'))
            or bundle['frozen_backbone_state_sha256'] != state_digest(model.base.state_dict())):
        raise ValueError('Every original frozen backbone tensor must match the saved own base')
    model.load_adapter_state(residuals)
    optimizer.load_state_dict(bundle['optimizer'])
    restore_rng(bundle['rng'], device)
    model.assert_partition()
    return bundle['steps'], bundle['metrics'].copy()
