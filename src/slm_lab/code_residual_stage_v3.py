"""Expanded original code/instruction/image replay with explicit objectives.

The29 prepared families are supported here. Final paired photos, the final own
base, capability baselines and a production launcher remain separate work.
"""
from dataclasses import asdict
import math

import torch
from torch import distributed as dist

from .code_residual_checkpoint_v1 import _digest, _optimizer_parameters
from .code_residual_collective_checkpoint_v2 import fingerprint, _same, _validated
from .code_residual_stage_v2 import (FULL_REPLAY_FAMILIES, ResidualStageKernel,
    open_full_replay, validate_options)
from .codenet_fact_replay_v1 import CombinedReplay
from .codenet_later_instruction_replay_v2 import FAMILIES as LATER_FAMILIES, LaterInstructionReplay
from .codenet_packed_replay_v1 import (FAMILY as PACKED_FAMILY, OBJECTIVE as PACKED_OBJECTIVE,
    PackedPretrainingEncoder, PackedReplayCorpus)
from .codenet_replay_schedule_v1 import coverage_replay_schedule
from .codenet_runner_v1 import FixedReductionModel, VerifiedReplayEncoder, average_gradients
from .codenet_stage_v1 import EpochOrder, Objective, exposure_budget, objective_gradients
from .recurrent_ablation import state_digest


EXPANDED_REPLAY_FAMILIES = FULL_REPLAY_FAMILIES + LATER_FAMILIES + (PACKED_FAMILY,)


def replay_field(name):
    if name not in EXPANDED_REPLAY_FAMILIES:
        raise ValueError('Known explicitly admitted replay family required')
    return None if name == PACKED_FAMILY else 'response'


def replay_objective(name):
    return PACKED_OBJECTIVE if replay_field(name) is None else 'response_only'


def open_expanded_replay(folders, tokenizer_sha256):
    if set(folders) != {'joint', 'facts', 'dialogue', 'images', 'later', 'packed'}:
        raise ValueError('All six original expanded replay components required')
    original = open_full_replay({name: folders[name] for name in ('joint', 'facts', 'dialogue', 'images')}, tokenizer_sha256)
    later, packed = LaterInstructionReplay(folders['later']), PackedReplayCorpus(folders['packed'])
    if any(component.manifest['tokenizer_sha256'] != tokenizer_sha256 for component in (later, packed)):
        raise ValueError('Exact same own tokenizer across later and packed replay required')
    replay = CombinedReplay(original, later, packed)
    if tuple(replay.families) != EXPANDED_REPLAY_FAMILIES:
        raise ValueError('All twenty-nine ordered original replay families required')
    return replay


class ExpandedReplayEncoder(VerifiedReplayEncoder):
    """Retain original verified pixels/full responses and explicit packed labels."""

    def sequence(self, row, response_field=None):
        if 'tokens' in row or row.get('_replay_objective') == PACKED_OBJECTIVE:
            return PackedPretrainingEncoder.sequence(self, row, response_field)
        return super().sequence(row, response_field)


class _EncodedBatches:
    """Validate local inputs collectively before any objective's all-reduces.

    Reuse those exact batches once, rather than rereading images or reencoding
    text after local preflight. No generated replacement examples are supplied.
    """

    def __init__(self, encoder, groups, field, device):
        self.groups, self.field, self.device = groups, field, str(device)
        self.batches = [encoder.batch(group, field, device) for group in groups]
        self.cursor = 0

    def batch(self, rows, field, device):
        if (self.cursor >= len(self.groups) or rows is not self.groups[self.cursor]
                or field != self.field or str(device) != self.device):
            raise ValueError('Use each original preflight batch exactly once under its own objective')
        value = self.batches[self.cursor]
        self.cursor += 1
        return value


def expanded_training_update(model, optimizer, code, code_encoder, replay, replay_encoder,
                             step, options, device, amp, rank, world):
    if (not isinstance(model, FixedReductionModel) or not dist.is_initialized()
            or world != 4 or dist.get_world_size() != world or dist.get_rank() != rank):
        raise ValueError('Four actual workers and the fixed gradient reduction wrapper required')

    def prepare():
        total = exposure_budget(len(code), options['passes'], world,
            options['batch_size'], options['gradient_accumulation'])['optimizer_steps']
        if type(step) is not int or not 0 <= step < total:
            raise ValueError('Optimizer update outside declared complete exposure budget')
        populations = tuple((name, replay.size(name)) for name in replay.families)
        schedule = coverage_replay_schedule(populations, total,
            world * options['replay_batch_size'] * options['replay_accumulation'])
        name, occurrence = schedule.at(step)
        field = replay_field(name)
        code_groups = [[{'packed_code_index': i} for i in group] for group in EpochOrder(len(code), options['seed']).rank_groups(
            step, rank, world, options['batch_size'], options['gradient_accumulation'])]
        order = EpochOrder(replay.size(name), options['seed'] + 1000 + replay.families.index(name))
        replay_groups = [[replay.row(name, i) for i in group] for group in order.rank_groups(
            occurrence, rank, world, options['replay_batch_size'], options['replay_accumulation'])]
        encoded_code = _EncodedBatches(code_encoder, code_groups, 'response', device)
        encoded_replay = _EncodedBatches(replay_encoder, replay_groups, field, device)
        return total, name, occurrence, field, code_groups, replay_groups, encoded_code, encoded_replay

    total, name, occurrence, field, code_groups, replay_groups, encoded_code, encoded_replay = _validated(prepare)
    _same(fingerprint(dict(step=step, total=total, family=name, occurrence=occurrence,
        objective=replay_objective(name), options=options)))
    learning_rate = options['learning_rate'] * min(1., (step + 1) / max(1, total // 10))
    learning_rate *= .1 + .9 * .5 * (1 + math.cos(math.pi * step / total))
    for group in optimizer.param_groups:
        group['lr'] = learning_rate
    optimizer.zero_grad(set_to_none=True)
    metrics = objective_gradients(model, [Objective('complete_code', encoded_code, code_groups, 'response', 1.),
        Objective(name, encoded_replay, replay_groups, field, options['replay_coefficient'])], device, amp, world=world)
    average_gradients(model, device, world)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), options['gradient_clip'], error_if_nonfinite=True)
    optimizer.step()
    return {**metrics, 'step': step + 1, 'replay_family': name, 'replay_occurrence': occurrence,
        'replay_objective': replay_objective(name), 'replay_response_field': field,
        'learning_rate': learning_rate, 'gradient_norm_before_clipping': float(norm)}


class ExpandedResidualStageKernel(ResidualStageKernel):
    """Complete per-update history and four-rank continuation for all29 sources."""

    def __init__(self, model, optimizer, code, code_encoder, replay, replay_encoder,
                 options, sources, device, amp, *, guard=None):
        if not dist.is_initialized() or dist.get_world_size() != 4:
            raise ValueError('Four actual initialized workers required')
        self.rank, self.world = dist.get_rank(), dist.get_world_size()

        def preflight():
            validate_options(options)
            if tuple(replay.families) != EXPANDED_REPLAY_FAMILIES or not isinstance(replay_encoder, ExpandedReplayEncoder):
                raise ValueError('All twenty-nine ordered families and their explicit full-objective encoder required')
            if set(sources) != {'base_checkpoint_sha256', 'tokenizer_sha256', 'run_plan_sha256'} or not all(_digest(v) for v in sources.values()):
                raise ValueError('Exact own base/tokenizer/run-plan source bindings required')
            _optimizer_parameters(model, optimizer)
            if any(parameter.grad is not None for parameter in model.parameters()):
                raise ValueError('Initialize at a clean optimizer boundary')
            if (code_encoder.config != model.config or replay_encoder.config != model.config
                    or replay_encoder.image_size != options['image_size'] or code_encoder.truncated or replay_encoder.truncated):
                raise ValueError('Actual matching complete encoders/config/image size required; no clipping')
            populations = tuple((name, replay.size(name)) for name in replay.families)
            budget = exposure_budget(len(code), options['passes'], self.world,
                options['batch_size'], options['gradient_accumulation'])
            schedule = coverage_replay_schedule(populations, budget['optimizer_steps'],
                self.world * options['replay_batch_size'] * options['replay_accumulation'])
            signature = fingerprint(dict(kernel='expanded-own-code-replay-v3', config=asdict(model.config),
                base=state_digest(model.base.state_dict()), residuals=model.residuals.state_dict(),
                optimizer=optimizer.state_dict(), options=options, sources=sources, code_size=len(code),
                replay_populations=populations, replay_objectives=[replay_objective(name) for name in replay.families], budget=budget))
            return budget, schedule, signature

        budget, schedule, signature = _validated(preflight)
        _same(signature)
        self.options, self.sources = options.copy(), sources.copy()
        self.model, self.optimizer, self.code, self.code_encoder = model, optimizer, code, code_encoder
        self.replay, self.replay_encoder = replay, replay_encoder
        self.device, self.amp, self.guard = device, amp, guard
        self.wrapped, self.budget = FixedReductionModel(model.code), budget
        self.steps, self.schedule = budget['optimizer_steps'], schedule
        self.completed, self.metrics = 0, []

    def sampler_state(self, step):
        return {**super().sampler_state(step), 'kernel': 'expanded-own-code-replay-v3',
            'replay_objectives': {name: replay_objective(name) for name in self.replay.families}}

    def update(self):
        def boundary():
            if type(self.completed) is not int or not 0 <= self.completed < self.steps or len(self.metrics) != self.completed:
                raise ValueError('The declared training budget is complete or its history differs')
            if self.code_encoder.truncated or self.replay_encoder.truncated:
                raise ValueError('Clipped sequences cannot enter the complete-source stage')
            return str(self.completed)
        _same(_validated(boundary))
        self._guard()
        self.model.train()
        self.model.code.gradient_checkpointing = True
        result = expanded_training_update(self.wrapped, self.optimizer, self.code, self.code_encoder,
            self.replay, self.replay_encoder, self.completed, self.options, self.device, self.amp, self.rank, self.world)
        if self.code_encoder.truncated or self.replay_encoder.truncated:
            raise ValueError('Any clipped original sequence invalidates the complete-source stage')
        if any(parameter.grad is not None for parameter in self.model.base.parameters()):
            raise ValueError('Every original backbone tensor must remain frozen')
        self.optimizer.zero_grad(set_to_none=True)
        self.completed += 1
        self.metrics.append(result)
        return result
