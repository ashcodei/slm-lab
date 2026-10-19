"""Private vision/matching/word Adam recovery without copying frozen language."""
import copy

import torch

from .codenet_checkpoint_v1 import capture_rng, restore_rng
from .code_residual_checkpoint_v1 import _cpu, _validate_adam, _validate_rng
from .own_actor_vision_replay_CPU_v1 import OwnActorVisionReplayCPU
from .whole_human_sft_evidence_v1 import tensor_digest

FORMAT = 'own-vision-matching-word-replay-selected-CPU-Adam-state-v1'


def selected(view, optimizer):
    if type(view) is not OwnActorVisionReplayCPU or next(view.parameters()).device.type != 'cpu':
        raise ValueError('Exact own CPU-only vision/word replay graph required')
    view.assert_shared(); named = view.named_selected()
    if (type(optimizer) is not torch.optim.AdamW or len(optimizer.param_groups) != 1
            or optimizer.param_groups[0].get('foreach') is not False
            or optimizer.param_groups[0].get('fused') not in (None, False)
            or [id(p) for p in optimizer.param_groups[0]['params']] != [id(p) for _, p in named]):
        raise ValueError('Original exact vision/matching/word Adam parameter order required')
    return named


def fixed_digest(view):
    active = {id(p) for _, p in view.named_selected()}
    return tensor_digest((name, p) for name, p in view.actor.named_parameters() if id(p) not in active)


def snapshot(view, optimizer, *, completed, planned, sources, settings, metrics):
    named = selected(view, optimizer)
    if (type(completed) is not int or type(planned) is not int or not 0 < completed <= planned
            or not isinstance(metrics, list) or len(metrics) != completed or not sources or not settings):
        raise ValueError('Exact actual update count, all metrics, settings and source bindings required')
    adam = _cpu(optimizer.state_dict()); _validate_adam(adam, [p for _, p in named], completed)
    rng = capture_rng('cpu'); _validate_rng(rng, 'cpu')
    return dict(format=FORMAT, actual_counts=view.actor.counts(), completed=completed, planned=planned,
        sources=copy.deepcopy(sources), settings=copy.deepcopy(settings), optimized_names=[n for n, _ in named],
        selected_values={n: p.detach().to(device='cpu', copy=True) for n, p in named},
        fixed_language_actor_critic_and_other_heads_sha256=fixed_digest(view), optimizer=adam, rng=rng,
        metrics=copy.deepcopy(metrics), full_neural_reference_or_old_base_values_included=False,
        candidate_promoted=False, goal_achieved=False)


def restore(view, optimizer, raw, *, planned, expected_sources, expected_settings):
    named = selected(view, optimizer); completed = raw.get('completed')
    if (raw.get('format') != FORMAT or raw.get('actual_counts') != view.actor.counts()
            or raw.get('planned') != planned or type(completed) is not int or not 0 < completed < planned
            or raw.get('sources') != expected_sources or raw.get('settings') != expected_settings
            or raw.get('optimized_names') != [n for n, _ in named]
            or not isinstance(raw.get('metrics'), list) or len(raw['metrics']) != completed
            or raw.get('full_neural_reference_or_old_base_values_included') is not False
            or raw.get('candidate_promoted') is not False or raw.get('goal_achieved') is not False):
        raise ValueError('Exact selected CPU replay state/profile/source/cursor required before mutation')
    values = raw.get('selected_values')
    if (not isinstance(values, dict) or list(values) != [n for n, _ in named]
            or any(not isinstance(values[n], torch.Tensor) or values[n].shape != p.shape
                   or values[n].dtype != p.dtype or not bool(torch.isfinite(values[n]).all()) for n, p in named)):
        raise ValueError('Every actual finite selected tensor with exact original shape/dtype required')
    _validate_adam(raw.get('optimizer'), [p for _, p in named], completed); _validate_rng(raw.get('rng'), 'cpu')
    if (raw['optimizer']['param_groups'] != optimizer.state_dict()['param_groups']
            or fixed_digest(view) != raw['fixed_language_actor_critic_and_other_heads_sha256']):
        raise ValueError('Actual original optimizer and every fixed language/actor/critic/head value required')
    with torch.no_grad():
        for name, p in named: p.copy_(values[name])
    optimizer.load_state_dict(_cpu(raw['optimizer'])); optimizer.zero_grad(set_to_none=True); restore_rng(raw['rng'], 'cpu')
    return completed, copy.deepcopy(raw['metrics'])
