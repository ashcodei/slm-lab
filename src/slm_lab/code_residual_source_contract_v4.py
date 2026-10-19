"""Explicit complete-source replay contracts, including original paired photos.

This kernel accepts the full required source set and explicitly declared later
additions. It never supplies a default objective for an unknown source. Source
finalization, final-base baselines, resources and a production launcher remain
external requirements.
"""
from copy import deepcopy
from dataclasses import asdict
import math
from pathlib import Path

import torch
from torch import distributed as dist

from .code_residual_checkpoint_v1 import _digest, _optimizer_parameters
from .code_residual_collective_checkpoint_v2 import fingerprint, _same, _validated, save_collective_residual
from .code_residual_stage_v2 import ResidualStageKernel, validate_options
from .code_residual_stage_v3 import (EXPANDED_REPLAY_FAMILIES, ExpandedReplayEncoder,
    _EncodedBatches, open_expanded_replay)
from .codenet_fact_replay_v1 import CombinedReplay
from .codenet_packed_replay_v1 import FAMILY as PACKED_FAMILY
from .codenet_paired_photo_replay_v1 import FAMILY as PAIRED_FAMILY, PairedPhotoReplay
from .codenet_replay_schedule_v1 import coverage_replay_schedule
from .codenet_runner_v1 import FixedReductionModel, average_gradients
from .codenet_stage_v1 import EpochOrder, Objective, exposure_budget, objective_gradients
from .corpus import file_hash
from .recurrent_ablation import state_digest


REQUIRED_FAMILIES = EXPANDED_REPLAY_FAMILIES + (PAIRED_FAMILY,)
COMPONENT_NAMES = ('joint', 'facts', 'dialogue', 'images', 'later', 'packed', 'paired')
REQUIRED_OWNERS = dict(zip(REQUIRED_FAMILIES,
    ['joint'] * 11 + ['facts', 'dialogue'] + ['images'] * 6 + ['later'] * 9 + ['packed', 'paired']))
FORMAT = 'own-complete-code-replay-source-contract-v4'
FIELDS = {'response_only': 'response', 'causal_pretraining': None}


def validate_source_contract(replay, contract, tokenizer_sha256):
    if (set(contract) != {'format', 'tokenizer_sha256', 'components', 'families'}
            or contract['format'] != FORMAT or contract['tokenizer_sha256'] != tokenizer_sha256
            or not _digest(tokenizer_sha256) or not isinstance(contract['components'], dict)
            or not set(COMPONENT_NAMES) <= set(contract['components'])
            or any(not isinstance(name, str) or not name or not _digest(digest)
                for name, digest in contract['components'].items())):
        raise ValueError('Complete exact own-tokenizer/source-manifest objective contract required')
    if not isinstance(contract['families'], list):
        raise ValueError('Explicit ordered complete source families required')
    names = [item.get('name') for item in contract['families']]
    if (len(names) != len(set(names)) or names != list(replay.families)
            or tuple(names[:len(REQUIRED_FAMILIES)]) != REQUIRED_FAMILIES):
        raise ValueError('All thirty ordered required families must remain in the declared source contract')
    used, objectives = set(), {}
    for item in contract['families']:
        if (set(item) != {'name', 'objective', 'source_records', 'component'}
                or not isinstance(item['name'], str) or not item['name']
                or item['objective'] not in FIELDS or type(item['source_records']) is not int
                or item['source_records'] < 1 or item['source_records'] != replay.size(item['name'])
                or item['component'] not in contract['components']):
            raise ValueError('Every declared family needs an explicit objective, actual population and bound component')
        name = item['name']
        if name in REQUIRED_OWNERS:
            expected = 'causal_pretraining' if name == PACKED_FAMILY else 'response_only'
            if item['component'] != REQUIRED_OWNERS[name] or item['objective'] != expected:
                raise ValueError('Required original sources must retain their own component and causal/response objective')
        objectives[name] = item['objective']
        used.add(item['component'])
    if used != set(contract['components']):
        raise ValueError('Every bound original source component must have declared replay coverage')
    return objectives


def build_required_contract(replay, tokenizer_sha256, component_digests):
    if tuple(replay.families) != REQUIRED_FAMILIES or set(component_digests) != set(COMPONENT_NAMES):
        raise ValueError('All seven current original source components required')
    contract = dict(format=FORMAT, tokenizer_sha256=tokenizer_sha256, components=dict(component_digests),
        families=[dict(name=name, objective='causal_pretraining' if name == PACKED_FAMILY else 'response_only',
            source_records=replay.size(name), component=REQUIRED_OWNERS[name]) for name in replay.families])
    validate_source_contract(replay, contract, tokenizer_sha256)
    return contract


def open_final_source_replay(folders, tokenizer_sha256):
    if set(folders) != set(COMPONENT_NAMES):
        raise ValueError('All seven admitted original replay components required before opening')
    paired = PairedPhotoReplay(folders['paired'])
    if paired.manifest['tokenizer_sha256'] != tokenizer_sha256:
        raise ValueError('One exact own tokenizer across final paired photo questions required')
    previous = open_expanded_replay({name: folders[name] for name in COMPONENT_NAMES if name != 'paired'}, tokenizer_sha256)
    replay = CombinedReplay(previous, paired)
    contract = build_required_contract(replay, tokenizer_sha256,
        {name: file_hash(Path(folder) / 'manifest.json') for name, folder in folders.items()})
    return replay, contract


def source_contract_update(model, optimizer, code, code_encoder, replay, replay_encoder,
                           objectives, step, options, device, amp, rank, world):
    if (not isinstance(model, FixedReductionModel) or not dist.is_initialized()
            or world != 4 or dist.get_world_size() != world or dist.get_rank() != rank):
        raise ValueError('Four actual workers and original fixed gradient reduction required')

    def prepare():
        total = exposure_budget(len(code), options['passes'], world,
            options['batch_size'], options['gradient_accumulation'])['optimizer_steps']
        if type(step) is not int or not 0 <= step < total:
            raise ValueError('Optimizer update outside declared complete source budget')
        schedule = coverage_replay_schedule(tuple((name, replay.size(name)) for name in replay.families),
            total, world * options['replay_batch_size'] * options['replay_accumulation'])
        name, occurrence = schedule.at(step)
        field = FIELDS[objectives[name]]
        code_groups = [[{'packed_code_index': i} for i in group] for group in EpochOrder(len(code), options['seed']).rank_groups(
            step, rank, world, options['batch_size'], options['gradient_accumulation'])]
        order = EpochOrder(replay.size(name), options['seed'] + 1000 + replay.families.index(name))
        replay_groups = [[replay.row(name, i) for i in group] for group in order.rank_groups(
            occurrence, rank, world, options['replay_batch_size'], options['replay_accumulation'])]
        return (total, name, occurrence, field, code_groups, replay_groups,
            _EncodedBatches(code_encoder, code_groups, 'response', device),
            _EncodedBatches(replay_encoder, replay_groups, field, device))

    total, name, occurrence, field, code_groups, replay_groups, encoded_code, encoded_replay = _validated(prepare)
    _same(fingerprint(dict(step=step, total=total, family=name, occurrence=occurrence,
        objectives=objectives, options=options)))
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
        'replay_objective': objectives[name], 'replay_response_field': field,
        'learning_rate': learning_rate, 'gradient_norm_before_clipping': float(norm)}


class SourceContractResidualKernel(ResidualStageKernel):
    """Complete per-update history/recovery under an immutable source contract."""

    def __init__(self, model, optimizer, code, code_encoder, replay, replay_encoder,
                 options, sources, device, amp, *, replay_contract, expected_contract_fingerprint, guard=None):
        if not dist.is_initialized() or dist.get_world_size() != 4:
            raise ValueError('Four actual initialized workers required')
        self.rank, self.world = dist.get_rank(), dist.get_world_size()

        def preflight():
            validate_options(options)
            if (set(sources) != {'base_checkpoint_sha256', 'tokenizer_sha256', 'run_plan_sha256'}
                    or not all(_digest(value) for value in sources.values())):
                raise ValueError('Exact own base/tokenizer/run-plan hashes required')
            objectives = validate_source_contract(replay, replay_contract, sources['tokenizer_sha256'])
            if not _digest(expected_contract_fingerprint) or fingerprint(replay_contract) != expected_contract_fingerprint:
                raise ValueError('Exact externally bound complete replay contract required')
            _optimizer_parameters(model, optimizer)
            if (not isinstance(replay_encoder, ExpandedReplayEncoder)
                    or code_encoder.config != model.config or replay_encoder.config != model.config
                    or replay_encoder.image_size != options['image_size'] or code_encoder.truncated or replay_encoder.truncated
                    or any(parameter.grad is not None for parameter in model.parameters())):
                raise ValueError('Clean complete code/causal/image encoders and matching original config required')
            populations = tuple((name, replay.size(name)) for name in replay.families)
            budget = exposure_budget(len(code), options['passes'], self.world, options['batch_size'], options['gradient_accumulation'])
            schedule = coverage_replay_schedule(populations, budget['optimizer_steps'],
                self.world * options['replay_batch_size'] * options['replay_accumulation'])
            signature = fingerprint(dict(kernel=FORMAT, contract=replay_contract, config=asdict(model.config),
                base=state_digest(model.base.state_dict()), residuals=model.residuals.state_dict(),
                optimizer=optimizer.state_dict(), options=options, sources=sources, code_size=len(code), budget=budget))
            return objectives, budget, schedule, signature

        objectives, budget, schedule, signature = _validated(preflight)
        _same(signature)
        self.contract = deepcopy(replay_contract)
        self.contract_fingerprint = expected_contract_fingerprint
        self.objectives = objectives
        self.options, self.sources = options.copy(), sources.copy()
        self.model, self.optimizer, self.code, self.code_encoder = model, optimizer, code, code_encoder
        self.replay, self.replay_encoder = replay, replay_encoder
        self.device, self.amp, self.guard = device, amp, guard
        self.wrapped, self.budget = FixedReductionModel(model.code), budget
        self.steps, self.schedule = budget['optimizer_steps'], schedule
        self.completed, self.metrics = 0, []

    def sampler_state(self, step):
        if (fingerprint(self.contract) != self.contract_fingerprint
                or validate_source_contract(self.replay, self.contract, self.sources['tokenizer_sha256']) != self.objectives):
            raise ValueError('The externally bound complete source/objective contract must remain unchanged')
        return {**super().sampler_state(step), 'kernel': FORMAT,
            'replay_contract_fingerprint': self.contract_fingerprint, 'replay_contract': deepcopy(self.contract)}

    def snapshot(self, path, *, maximum_bytes, minimum_free_bytes):
        self._guard(full=True)
        progress = _validated(lambda: self.sampler_state(self.completed))
        return save_collective_residual(path, self.model, self.optimizer, **self.sources,
            completed_steps=self.completed, planned_steps=self.steps,
            sampler_state=progress, metrics=self.metrics, device=self.device,
            maximum_bytes=maximum_bytes, minimum_free_bytes=minimum_free_bytes)

    def update(self):
        def boundary():
            if (type(self.completed) is not int or not 0 <= self.completed < self.steps or len(self.metrics) != self.completed
                    or fingerprint(self.contract) != self.contract_fingerprint
                    or validate_source_contract(self.replay, self.contract, self.sources['tokenizer_sha256']) != self.objectives
                    or self.code_encoder.truncated or self.replay_encoder.truncated):
                raise ValueError('Complete source contract/update budget/encoder state must remain unchanged')
            return fingerprint(dict(step=self.completed, contract=self.contract, objectives=self.objectives))
        _same(_validated(boundary))
        self._guard()
        self.model.train()
        self.model.code.gradient_checkpointing = True
        result = source_contract_update(self.wrapped, self.optimizer, self.code, self.code_encoder,
            self.replay, self.replay_encoder, self.objectives, self.completed, self.options,
            self.device, self.amp, self.rank, self.world)
        if self.code_encoder.truncated or self.replay_encoder.truncated:
            raise ValueError('Original full-source inputs cannot be clipped')
        if any(parameter.grad is not None for parameter in self.model.base.parameters()):
            raise ValueError('Every original backbone tensor must remain frozen')
        self.optimizer.zero_grad(set_to_none=True)
        self.completed += 1
        self.metrics.append(result)
        return result
