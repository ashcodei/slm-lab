"""Selected-only joint policy/vision/word Adam and private RNG recovery.

Bind the complete declared schedule and retain each leaf's actual sparse Adam
clock. The immutable language, critic and other auxiliary values are checked,
never rewritten or duplicated in this payload. This primitive grants no data,
full-size GPU, bounded-history or production training admission.
"""
import copy
from dataclasses import asdict
import math

import torch
from torch import distributed as dist

from .codenet_checkpoint_v1 import capture_rng, restore_rng
from .code_residual_checkpoint_v1 import _cpu, _validate_rng
from .parallel_function_repair_stage_v1 import value_digest
from .shared_joint_actor_vision_v1 import SharedJointActorVisionV1
from .whole_human_sft_evidence_v1 import tensor_digest

FORMAT = 'own-joint-actor-vision-word-selected-Adam-private-RNG-v1'
SOURCE_KEYS = {'text_initializer_sha256', 'visual_initializer_sha256', 'tokenizer_sha256',
    'run_plan_sha256', 'source_admission_sha256', 'complete_schedule_sha256', 'fixed_values_sha256'}


def sha(value):
    return type(value) is str and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def selected(view, optimizer):
    if type(view) is not SharedJointActorVisionV1:
        raise ValueError('The exact shared own joint policy/vision/word graph is required')
    view.assert_shared(); named = view.named_selected()
    if (type(optimizer) is not torch.optim.AdamW or len(optimizer.param_groups) != 1
            or [id(p) for p in optimizer.param_groups[0]['params']] != [id(p) for _, p in named]):
        raise ValueError('One exact-order joint ordinary AdamW group is required')
    group = optimizer.param_groups[0]
    if (group.get('foreach') is not False or group.get('fused') not in [False, None]
            or any(group.get(k, False) for k in ['amsgrad', 'maximize', 'capturable', 'differentiable'])
            or any(type(group.get(k)) not in [float, int] or not math.isfinite(group[k])
                or group[k] < 0 or k != 'weight_decay' and group[k] == 0 for k in ['lr', 'eps', 'weight_decay'])
            or type(group.get('betas')) not in [tuple, list] or len(group['betas']) != 2
            or any(type(v) not in [float, int] or not math.isfinite(v) or not 0 <= v < 1 for v in group['betas'])):
        raise ValueError('Finite valid ordinary joint AdamW settings are required')
    return named


class FixedGuard:
    def __init__(self, view, optimizer):
        self.view, self.optimizer = view, optimizer
        self.selected_ids = {id(p) for _, p in selected(view, optimizer)}
        self.sha256 = tensor_digest(self.fixed())
        self.signature = self.current_signature()
        self.metadata = self.current_metadata()

    def fixed(self):
        return [(n, p) for n, p in self.view.named_parameters() if id(p) not in self.selected_ids]

    def current_signature(self):
        return [(n, id(p), p.data_ptr(), p._version, p.requires_grad, str(p.dtype), str(p.device), tuple(p.shape))
            for n, p in self.fixed()]

    def current_metadata(self):
        return copy.deepcopy(dict(configuration=asdict(self.view.actor.config), actor_settings=self.view.actor.settings,
            actual_counts=self.view.actor.counts(), task_parameter_names={task: self.view.task_names(task)
                for task in ['text', 'native_image', 'image_matching', 'word_CTC']}))

    def check(self, view, optimizer, *, full=False):
        if (view is not self.view or optimizer is not self.optimizer or self.current_signature() != self.signature
                or self.current_metadata() != self.metadata or full and tensor_digest(self.fixed()) != self.sha256):
            raise ValueError('Every fixed language/critic/auxiliary value and original metadata must remain exact')


def control(group):
    if not dist.is_initialized():
        if group is not None: raise ValueError('An actual initialized joint control group is required')
        return 0, 1
    if dist.get_backend(group) != 'gloo': raise ValueError('An explicit actual Gloo joint control group is required')
    return dist.get_rank(group), dist.get_world_size(group)


def gather(value, group, world):
    if world == 1: return [value]
    rows = [None] * world
    dist.all_gather_object(rows, value, group=group)
    return rows


def agree(error, common, group, world):
    rows = gather((error, common), group, world)
    if any(r[0] is not None for r in rows):
        raise ValueError('Coordinated joint checkpoint rejection: ' + repr([r[0] for r in rows]))
    if any(r[1] != rows[0][1] for r in rows):
        raise ValueError('Coordinated joint source/cursor/weights/Adam disagreement: ' + repr([r[1] for r in rows]))


def context(view, optimizer, fixed, sources, entries, rank, world, device, group):
    if (type(rank) is not int or type(world) is not int or (rank, world) != control(group) or not 1 <= world <= 4
            or type(sources) is not dict or set(sources) != SOURCE_KEYS or not all(sha(v) for v in sources.values())
            or sources['fixed_values_sha256'] != fixed.sha256 or type(entries) is not list or not entries
            or sources['complete_schedule_sha256'] != value_digest(entries)):
        raise ValueError('Exact actual ranks, fixed source identities and complete declared joint schedule are required')
    named = selected(view, optimizer); fixed.check(view, optimizer, full=True)
    if (any(p.grad is not None or p.dtype != torch.float32 or p.device != torch.device(device) for p in view.parameters())
            or any(not p.requires_grad for _, p in named)):
        raise ValueError('A clear-gradient complete FP32 joint optimizer boundary is required')
    fields = {'task', 'source_family', 'source_role', 'source_sha256', 'original_positions',
        'complete_context_tokens', 'response_EOS_targets', 'word_UTF8_bytes', 'normalization_denominators'}
    for index, row in enumerate(entries, 1):
        if (type(row) is not dict or set(row) != {'step', 'objectives'} or type(row['step']) is not int or row['step'] != index
                or type(row['objectives']) is not list or not row['objectives']):
            raise ValueError('Every complete declared joint step and original task source must remain bound')
        for objective in row['objectives']:
            if (type(objective) is not dict or set(objective) != fields
                    or objective['task'] not in fixed.metadata['task_parameter_names']
                    or type(objective['source_family']) is not str or not objective['source_family']
                    or objective['source_role'] != 'train' or objective['source_sha256'] not in sources.values()
                    or type(objective['original_positions']) is not list or not objective['original_positions']
                    or any(type(p) is not int or p < 0 for p in objective['original_positions'])
                    or len(set(objective['original_positions'])) != len(objective['original_positions'])
                    or any(type(objective[k]) is not int or objective[k] < 0 for k in
                        ['complete_context_tokens', 'response_EOS_targets', 'word_UTF8_bytes'])
                    or type(objective['normalization_denominators']) is not dict or not objective['normalization_denominators']
                    or any(type(k) is not str or not k or type(v) is not int or v < 1
                        for k, v in objective['normalization_denominators'].items())):
                raise ValueError('Each original train source, full target budget and independent denominator must remain explicit')
            if objective['task'] == 'word_CTC':
                valid = objective['word_UTF8_bytes'] > 0 and objective['response_EOS_targets'] == objective['complete_context_tokens'] == 0
            else:
                valid = 0 < objective['response_EOS_targets'] <= objective['complete_context_tokens'] and objective['word_UTF8_bytes'] == 0
            if not valid: raise ValueError('Native response/EOS targets and whole-word byte targets cannot be interchanged')
    return named


def cursor(view, entries, completed, metrics):
    if (type(completed) is not int or not 0 <= completed <= len(entries)
            or type(metrics) is not list or len(metrics) != completed):
        raise ValueError('The complete actual update prefix and every objective metric are required')
    clocks = {n: 0 for n, _ in view.named_selected()}
    for row, metric in zip(entries[:completed], metrics):
        active = set(n for task in row['objectives'] for n in view.task_names(task['task']))
        for name in active: clocks[name] += 1
        if (type(metric) is not dict or set(metric) != {'step', 'entry_sha256', 'active_parameter_names',
                'normalized_task_losses', 'gradient_norm_before_clip', 'globally_reduced_task_gradients_before_combining', 'gradient_interaction'}
                or type(metric['step']) is not int or metric['step'] != row['step'] or metric['entry_sha256'] != value_digest(row)
                or metric['active_parameter_names'] != [n for n in clocks if n in active]
                or metric['globally_reduced_task_gradients_before_combining'] is not True
                or type(metric['normalized_task_losses']) is not list or len(metric['normalized_task_losses']) != len(row['objectives'])
                or any(type(v) not in [int, float] or not math.isfinite(v) or v < 0
                    for v in [*metric['normalized_task_losses'], metric['gradient_norm_before_clip']])
                or type(metric['gradient_interaction']) is not dict or not metric['gradient_interaction']):
            raise ValueError('Every actual joint objective, active leaf and global gradient boundary must reconstruct')
        value_digest(metric['gradient_interaction'])
    return dict(completed=completed, planned=len(entries), actual_per_parameter_Adam_steps=clocks,
        complete_schedule_sha256=value_digest(entries), consumed_source_prefix_sha256=value_digest(entries[:completed]),
        next_entry_sha256=None if completed == len(entries) else value_digest(entries[completed]),
        actual_global_metrics_sha256=value_digest(metrics))


def validate_adam(saved, optimizer, named, clocks):
    if (type(saved) is not dict or set(saved) != {'state', 'param_groups'}
            or saved['param_groups'] != optimizer.state_dict()['param_groups']
            or saved['param_groups'][0]['params'] != list(range(len(named)))):
        raise ValueError('Every exact joint AdamW setting and selected parameter order must match')
    wanted = set()
    for index, (name, parameter) in enumerate(named):
        if clocks[name] == 0:
            if index in saved['state']: raise ValueError('A never-active joint leaf cannot fabricate Adam state')
            continue
        wanted.add(index); value = saved['state'].get(index)
        if (type(value) is not dict or set(value) != {'step', 'exp_avg', 'exp_avg_sq'}
                or not isinstance(value['step'], torch.Tensor) or value['step'].device.type != 'cpu'
                or value['step'].numel() != 1 or not bool(torch.isfinite(value['step']).all()) or float(value['step']) != clocks[name]):
            raise ValueError('Every joint leaf must retain its exact task-derived sparse Adam clock')
        for field in ['exp_avg', 'exp_avg_sq']:
            tensor = value[field]
            if (not isinstance(tensor, torch.Tensor) or tensor.device.type != 'cpu' or tensor.shape != parameter.shape
                    or tensor.dtype != parameter.dtype or not bool(torch.isfinite(tensor).all())):
                raise ValueError('Every joint leaf requires complete finite exact-shape/dtype Adam moments')
        if bool(torch.any(value['exp_avg_sq'] < 0)): raise ValueError('Squared joint Adam moments cannot be negative')
    if set(saved['state']) != wanted: raise ValueError('Exactly the genuinely active joint Adam inventory is required')


def common_digest(state):
    return value_digest({k: v for k, v in state.items() if k not in ['rank', 'private_RNG', 'private_RNG_sha256']})


def make_state(view, optimizer, fixed, *, sources, entries, completed, metrics, rank=0, world=1,
               device='cpu', control_group=None):
    _, actual_world = control(control_group); error = common = state = None
    try:
        named = context(view, optimizer, fixed, sources, entries, rank, world, device, control_group)
        boundary = cursor(view, entries, completed, metrics)
        if any(not bool(torch.isfinite(p).all()) for _, p in named): raise ValueError('All live joint weights must be finite before publication')
        adam = _cpu(optimizer.state_dict()); validate_adam(adam, optimizer, named, boundary['actual_per_parameter_Adam_steps'])
        rng = _cpu(capture_rng(device)); _validate_rng(rng, device)
        state = dict(format=FORMAT, sources=copy.deepcopy(sources), world_size=world, rank=rank,
            model_metadata=copy.deepcopy(fixed.metadata), cursor=boundary, metrics=_cpu(metrics),
            optimized_names=[n for n, _ in named], selected_weights={n: _cpu(p) for n, p in named},
            optimizer=adam, private_RNG=rng, private_RNG_sha256=value_digest(rng),
            grants_production_training_admission=False, candidate_promoted=False, goal_achieved=False)
        common = {k: value_digest(v) for k, v in state.items() if k not in ['rank', 'private_RNG', 'private_RNG_sha256']}
    except Exception as exc: error = type(exc).__name__ + ': ' + str(exc)
    agree(error, common, control_group, actual_world)
    return state


def restore_state(view, optimizer, fixed, state, *, sources, entries, expected_common_sha256,
                  expected_private_RNG_sha256, expected_completed, rank=0, world=1, device='cpu',
                  control_group=None, allow_completed=False):
    _, actual_world = control(control_group); error = common = boundary = None
    try:
        named = context(view, optimizer, fixed, sources, entries, rank, world, device, control_group)
        boundary = cursor(view, entries, state.get('cursor', {}).get('completed'), state.get('metrics'))
        if (state.get('format') != FORMAT or state.get('sources') != sources or type(state.get('world_size')) is not int or state['world_size'] != world
                or type(state.get('rank')) is not int or state['rank'] != rank or state.get('model_metadata') != fixed.metadata
                or state.get('cursor') != boundary or type(expected_completed) is not int or boundary['completed'] != expected_completed
                or boundary['completed'] == len(entries) and not allow_completed
                or state.get('optimized_names') != [n for n, _ in named] or list(state.get('selected_weights', {})) != [n for n, _ in named]
                or any(state.get(k) is not False for k in ['grants_production_training_admission', 'candidate_promoted', 'goal_achieved'])):
            raise ValueError('Exact joint source/model/task/full-prefix/rank recovery is required before mutation')
        for name, parameter in named:
            value = state['selected_weights'][name]
            if (not isinstance(value, torch.Tensor) or value.device.type != 'cpu' or value.shape != parameter.shape
                    or value.dtype != parameter.dtype or not bool(torch.isfinite(value).all())):
                raise ValueError('Complete finite selected joint weights are required before recovery')
        validate_adam(state.get('optimizer'), optimizer, named, boundary['actual_per_parameter_Adam_steps'])
        _validate_rng(state.get('private_RNG'), device)
        if (not sha(expected_private_RNG_sha256) or state.get('private_RNG_sha256') != value_digest(state['private_RNG'])
                or state['private_RNG_sha256'] != expected_private_RNG_sha256):
            raise ValueError('The exact independently committed private joint RNG boundary must match')
        common = common_digest(state)
        if not sha(expected_common_sha256) or common != expected_common_sha256:
            raise ValueError('Every joint value, sparse moment and source metric must match the external complete identity')
    except Exception as exc: error = type(exc).__name__ + ': ' + str(exc)
    agree(error, common, control_group, actual_world)
    with torch.no_grad():
        for name, parameter in named: parameter.copy_(state['selected_weights'][name])
    optimizer.load_state_dict(_cpu(state['optimizer'])); optimizer.zero_grad(set_to_none=True)
    restore_rng(state['private_RNG'], device); torch.clear_autocast_cache()
    fixed.check(view, optimizer, full=True)
    return boundary['completed'], _cpu(state['metrics'])
