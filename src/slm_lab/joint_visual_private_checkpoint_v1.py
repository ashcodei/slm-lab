"""Private selected-only recovery for the three visual optimizer groups.

Store existing vision/pointing/word weights, their ordinary Adam states, the
actual per-worker RNG and complete source cursor. Do not duplicate or import
the frozen language/actor/critic values. The full data/runtime/GPU controller
must separately prove admission and capacity; this primitive grants neither.
"""
from collections import Counter
import copy
from dataclasses import asdict
import hashlib
import json
import math

import torch
from torch import distributed as dist

from .codenet_checkpoint_v1 import capture_rng, restore_rng
from .code_residual_checkpoint_v1 import _cpu, _validate_rng
from .full_visual_source_admission_v1 import FAMILIES as POPULATIONS
from .joint_visual_objective_update_v1 import FAMILIES, optimizer_groups, optimizer_clocks
from .whole_human_sft_evidence_v1 import tensor_digest

FORMAT = 'own-joint-visual-selected-weights-three-group-Adam-private-RNG-v1'
SOURCE_KEYS = {'base_checkpoint_sha256', 'tokenizer_sha256', 'run_plan_sha256',
    'source_admission_sha256', 'complete_schedule_sha256', 'corrected_index_sha256', 'fixed_values_sha256'}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


class FixedGuard:
    def __init__(self, view, optimizer):
        groups = optimizer_groups(view, optimizer)
        self.view, self.optimizer = view, optimizer
        self.selected = {id(p) for rows in groups.values() for _, p in rows}
        self.sha256 = tensor_digest(self.fixed())
        self.signature = self.current_signature()
        self.metadata = self.current_metadata()

    def fixed(self):
        return [(name, p) for name, p in self.view.named_parameters() if id(p) not in self.selected]

    def current_signature(self):
        return [(name, id(p), p.data_ptr(), p._version, p.requires_grad, str(p.dtype), str(p.device), tuple(p.shape))
            for name, p in self.fixed()]

    def current_metadata(self):
        return copy.deepcopy(dict(config=asdict(self.view.actor.config), actor_settings=self.view.actor.settings,
            actor_counts=self.view.actor.counts()))

    def check(self, view, optimizer):
        if (view is not self.view or optimizer is not self.optimizer or self.current_signature() != self.signature
                or self.current_metadata() != self.metadata):
            raise ValueError('Exact unchanged own fixed values, model structure and trainability metadata required')


def _control(group):
    if not dist.is_initialized():
        if group is not None:
            raise ValueError('An uninitialized private checkpoint cannot declare a collective control group')
        return 0, 1
    if dist.get_backend(group) != 'gloo':
        raise ValueError('Actual explicit Gloo private checkpoint control group required')
    return dist.get_rank(group), dist.get_world_size(group)


def _agree(error, common, group, world):
    if world == 1:
        rows = [(error, common)]
    else:
        rows = [None] * world
        dist.all_gather_object(rows, (error, common), group=group)
    if any(row[0] is not None for row in rows):
        raise ValueError('Coordinated private visual checkpoint rejection: ' + repr([r[0] for r in rows]))
    if any(row[1] != rows[0][1] for row in rows):
        raise ValueError('Coordinated private visual source/cursor/weights/Adam disagreement')


def _context(view, optimizer, fixed, sources, entries, mode, rank, world, device, actual_rank, actual_world):
    fixed.check(view, optimizer)
    groups = optimizer_groups(view, optimizer)
    if (type(rank) is not int or type(world) is not int or rank != actual_rank or world != actual_world
            or not 1 <= world <= 4 or mode not in ['capacity', 'train']
            or set(sources) != SOURCE_KEYS or sources['fixed_values_sha256'] != fixed.sha256
            or any(not isinstance(s, str) or len(s) != 64 or any(c not in '0123456789abcdef' for c in s) for s in sources.values())
            or not isinstance(entries, list) or not entries
            or any(not p.requires_grad for rows in groups.values() for _, p in rows)
            or any(p.grad is not None or p.dtype != torch.float32 or p.device != torch.device(device) for p in view.parameters())):
        raise ValueError('Actual rank/world, explicit source SHA bindings, full entries and clear-gradient FP32 model required')
    required = {'step', 'family', 'original_local_positions', 'complete_context_tokens',
        'response_EOS_targets', 'proposed_rank_source_counts'}
    counts = {name: Counter() for name in POPULATIONS}
    for i, row in enumerate(entries, 1):
        if (set(row) != required or row['step'] != i or row['family'] not in FAMILIES
                or not isinstance(row['original_local_positions'], list)
                or not 1 <= len(row['original_local_positions']) <= 64
                or len(row['original_local_positions']) != len(set(row['original_local_positions']))
                or any(type(p) is not int or p < 0 for p in row['original_local_positions'])
                or row['proposed_rank_source_counts'] != [len(row['original_local_positions'][r::world]) for r in range(world)]
                or any(type(row[k]) is not int or row[k] < 1 for k in ['complete_context_tokens', 'response_EOS_targets'])
                or row['response_EOS_targets'] > row['complete_context_tokens']):
            raise ValueError('Every complete original family/source/mask budget and real rank assignment required')
        counts[row['family']].update(row['original_local_positions'])
    if mode == 'train' and (world != 4 or len(entries) != 4733
            or view.actor.counts()['actual_unique'] != 999706442
            or view.actor.base.counts()['total_unique'] != 984399690
            or any(counts[name] != Counter(range(population)) for name, population in POPULATIONS.items())
            or sum(e['complete_context_tokens'] for e in entries) != 17681556
            or sum(e['response_EOS_targets'] for e in entries) != 4669243):
        raise ValueError('Production snapshots require the full actual under1B graph and every original4733step/302627view budget')
    return groups


def _cursor(entries, completed, metrics):
    if type(completed) is not int or not 0 <= completed <= len(entries) or not isinstance(metrics, list) or len(metrics) != completed:
        raise ValueError('Exact completed source prefix and every actual optimizer metric required')
    clocks = dict(vision=0, pointing=0, word=0)
    global_metrics = []
    for row, metric in zip(entries[:completed], metrics):
        kind = FAMILIES[row['family']]
        active = ['vision', 'pointing'] if kind == 'photo' else ['vision', 'word'] if kind == 'word' else ['vision']
        for group in active:
            clocks[group] += 1
        if (metric.get('family') != row['family'] or metric.get('global_positions') != row['original_local_positions']
                or metric.get('active_groups') != active or metric.get('optimizer_group_clocks') != clocks
                or metric.get('global_context_tokens') != row['complete_context_tokens']
                or metric.get('response_EOS_targets') != row['response_EOS_targets']
                or metric.get('global_words') != (len(row['original_local_positions']) if kind == 'word' else 0)
                or metric.get('one_global_gradient_SUM_and_one_optimizer_commit') is not True
                or metric.get('all_inactive_weights_moments_clocks_and_fixed_model_metadata_preserved') is not True
                or any(type(metric.get(k)) not in [int, float] or not math.isfinite(metric[k]) or metric[k] < 0
                    for k in ['native_CE_sum', 'byte_normalized_CTC_sum', 'gradient_norm_before_clip'])):
            raise ValueError('Every actual global source objective and sparse Adam clock must reconstruct from the full prefix')
        global_metrics.append({k: metric[k] for k in ['family', 'global_positions', 'active_groups', 'optimizer_group_clocks',
            'global_context_tokens', 'response_EOS_targets', 'global_words', 'native_CE_sum', 'byte_normalized_CTC_sum', 'gradient_norm_before_clip']})
    return dict(completed=completed, planned=len(entries), group_clocks=clocks,
        next_entry_sha256=None if completed == len(entries) else digest(entries[completed]),
        whole_schedule_entries_sha256=digest(entries), actual_global_metrics_sha256=digest(global_metrics))


def _adam(saved, optimizer, groups, clocks):
    if (not isinstance(saved, dict) or set(saved) != {'state', 'param_groups'}
            or saved['param_groups'] != optimizer.state_dict()['param_groups']):
        raise ValueError('Every actual ordinary three-group Adam option, order and current learning rate must match')
    wanted = set()
    for group, (name, rows) in zip(saved['param_groups'], groups.items()):
        for key, (_, parameter) in zip(group['params'], rows):
            value = saved['state'].get(key)
            if clocks[name] == 0:
                if value is not None:
                    raise ValueError('A never-active visual head must not fabricate Adam weights or clocks')
                continue
            wanted.add(key)
            if (not isinstance(value, dict) or set(value) != {'step', 'exp_avg', 'exp_avg_sq'}
                    or not isinstance(value['step'], torch.Tensor) or value['step'].numel() != 1
                    or not bool(torch.isfinite(value['step']).all()) or float(value['step']) != clocks[name]):
                raise ValueError('Every active visual leaf must retain its exact sparse group Adam clock')
            for field in ['exp_avg', 'exp_avg_sq']:
                tensor = value[field]
                if (not isinstance(tensor, torch.Tensor) or tensor.device.type != 'cpu' or tensor.shape != parameter.shape
                        or tensor.dtype != parameter.dtype or not bool(torch.isfinite(tensor).all())):
                    raise ValueError('Complete finite exact-shape/dtype private visual moments required')
            if bool(torch.any(value['exp_avg_sq'] < 0)):
                raise ValueError('Squared private visual Adam moments cannot be negative')
    if set(saved['state']) != wanted:
        raise ValueError('Only every actually active original selected visual leaf may have Adam state')


def _common(state):
    return digest(dict(sources=state['sources'], mode=state['mode'], world=state['world_size'],
        model_metadata=state['model_metadata'], cursor=state['cursor'],
        selected_sha256=tensor_digest(state['selected_weights'].items()),
        Adam_sha256=tensor_digest((str(k) + '.' + field, value[field])
            for k, value in state['optimizer']['state'].items() for field in ['step', 'exp_avg', 'exp_avg_sq']),
        Adam_settings=state['optimizer']['param_groups']))


def private_rng_sha256(rng):
    metadata = dict(python=rng['python'], numpy=rng['numpy'], CUDA_profile=rng['cuda'] is not None)
    arrays = [('CPU_RNG', rng['torch'])]
    if rng['cuda'] is not None:
        arrays.append(('CUDA_RNG', rng['cuda']))
    return digest(dict(metadata=metadata, actual_generator_arrays_sha256=tensor_digest(arrays)))


def make_state(view, optimizer, fixed, *, sources, entries, completed, metrics, mode, rank=0, world=1,
               device='cpu', control_group=None):
    actual_rank, actual_world = _control(control_group)
    error = common = result = None
    try:
        groups = _context(view, optimizer, fixed, sources, entries, mode, rank, world, device, actual_rank, actual_world)
        cursor = _cursor(entries, completed, metrics)
        if optimizer_clocks(view, optimizer) != cursor['group_clocks']:
            raise ValueError('Live visual optimizer clocks must equal their exact completed family prefix')
        saved = _cpu(optimizer.state_dict())
        _adam(saved, optimizer, groups, cursor['group_clocks'])
        rng = _cpu(capture_rng(device))
        _validate_rng(rng, device)
        result = dict(format=FORMAT, sources=dict(sources), mode=mode, rank=rank, world_size=world,
            model_metadata=fixed.metadata, cursor=cursor, metrics=_cpu(metrics),
            optimized_names=[name for rows in groups.values() for name, _ in rows],
            selected_weights={name: _cpu(p) for rows in groups.values() for name, p in rows},
            optimizer=saved, private_RNG=rng, private_RNG_sha256=private_rng_sha256(rng),
            candidate_promoted=False, goal_achieved=False)
        common = _common(result)
    except Exception as exc:
        error = type(exc).__name__ + ': ' + str(exc)
    _agree(error, common, control_group, actual_world)
    return result


def restore_state(view, optimizer, fixed, state, *, sources, entries, mode, rank=0, world=1,
                  device='cpu', control_group=None, expected_completed, expected_private_RNG_sha256, allow_completed=False):
    actual_rank, actual_world = _control(control_group)
    error = common = cursor = groups = None
    try:
        groups = _context(view, optimizer, fixed, sources, entries, mode, rank, world, device, actual_rank, actual_world)
        completed = state.get('cursor', {}).get('completed')
        cursor = _cursor(entries, completed, state.get('metrics'))
        names = [name for rows in groups.values() for name, _ in rows]
        if (type(expected_completed) is not int or completed != expected_completed
                or state.get('format') != FORMAT or state.get('sources') != sources or state.get('mode') != mode
                or state.get('rank') != rank or state.get('world_size') != world
                or state.get('model_metadata') != fixed.metadata or state.get('cursor') != cursor
                or state.get('optimized_names') != names or list(state.get('selected_weights', {})) != names
                or state.get('candidate_promoted') is not False or state.get('goal_achieved') is not False
                or completed == len(entries) and not allow_completed):
            raise ValueError('Exact private rank/source/model/full-cursor/selected-order recovery required')
        for rows in groups.values():
            for name, parameter in rows:
                value = state['selected_weights'][name]
                if (not isinstance(value, torch.Tensor) or value.device.type != 'cpu' or value.shape != parameter.shape
                        or value.dtype != parameter.dtype or not bool(torch.isfinite(value).all())):
                    raise ValueError('Only complete finite exact-shape/dtype existing selected weights may be restored')
        _adam(state.get('optimizer'), optimizer, groups, cursor['group_clocks'])
        _validate_rng(state.get('private_RNG'), device)
        if (state.get('private_RNG_sha256') != private_rng_sha256(state['private_RNG'])
                or state['private_RNG_sha256'] != expected_private_RNG_sha256):
            raise ValueError('The exact private rank RNG must match its independently committed boundary hash')
        common = _common(state)
    except Exception as exc:
        error = type(exc).__name__ + ': ' + str(exc)
    # Every actual rank validates its entire private source/Adam/RNG envelope
    # before any peer copies weights or changes optimizer/global RNG state.
    _agree(error, common, control_group, actual_world)
    with torch.no_grad():
        for rows in groups.values():
            for name, parameter in rows:
                parameter.copy_(state['selected_weights'][name])
    optimizer.load_state_dict(_cpu(state['optimizer']))
    optimizer.zero_grad(set_to_none=True)
    restore_rng(state['private_RNG'], device)
    torch.clear_autocast_cache()
    fixed.check(view, optimizer)
    return cursor['completed'], _cpu(state['metrics'])
