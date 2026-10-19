"""Complete-code/replay update kernel with exact collective continuation.

This is not a production launcher or a readiness bypass. An adopted run still
needs an exact final own base, complete later-source coverage, baseline/evaluation
bindings, exclusive resources and a funded immutable run plan.
"""
from collections import Counter
from dataclasses import asdict
import math

import torch
from torch import distributed as dist

from .code_residual_checkpoint_v1 import _digest, _optimizer_parameters
from .code_residual_collective_checkpoint_v2 import (fingerprint, _same, _validated,
    save_collective_residual, restore_collective_residual)
from .codenet_dialogue_replay_v1 import DialogueReplayCorpus
from .codenet_fact_replay_v1 import CombinedReplay, FactReplayCorpus
from .codenet_image_replay_v1 import ImageReplayCorpus
from .codenet_replay_v1 import ReplayCorpus
from .codenet_replay_schedule_v1 import coverage_replay_schedule
from .codenet_runner_v1 import FLOAT_OPTIONS, INTEGER_OPTIONS, FixedReductionModel, training_update
from .codenet_stage_v1 import exposure_budget
from .recurrent_ablation import state_digest


FULL_REPLAY_FAMILIES = (
    'human_instruction', 'code:Python', 'code:JavaScript', 'code:Java', 'code:Go',
    'code:PHP', 'code:Ruby', 'math', 'generic_visual', 'image_reading', 'verified_programs',
    'human_fact_questions', 'reviewed_human_dialogue', 'additional_photo_captions',
    'additional_real_words', 'additional_commons', 'original_visual_reasoning',
    'human_photo_reading', 'own_context_photo_reading')


def open_full_replay(folders, tokenizer_sha256):
    """Use the actual audited readers; preserve all source rows/masks/aliases."""
    readers = dict(joint=ReplayCorpus, facts=FactReplayCorpus,
        dialogue=DialogueReplayCorpus, images=ImageReplayCorpus)
    if set(folders) != set(readers) or not _digest(tokenizer_sha256):
        raise ValueError('All four original replay components and exact own tokenizer required')
    components = [reader(folders[name]) for name, reader in readers.items()]
    if any(component.manifest['tokenizer_sha256'] != tokenizer_sha256 for component in components):
        raise ValueError('One exact own tokenizer across every replay component required')
    replay = CombinedReplay(*components)
    if tuple(replay.families) != FULL_REPLAY_FAMILIES:
        raise ValueError('All nineteen ordered original source families required')
    return replay


def validate_options(options):
    if set(options) != INTEGER_OPTIONS | FLOAT_OPTIONS:
        raise ValueError('Exact original complete-code optimization options required')
    if any(type(options[k]) is not int or options[k] < (0 if k == 'seed' else 1) for k in INTEGER_OPTIONS):
        raise ValueError('Positive integer layout and nonnegative integer seed required')
    if any(type(options[k]) not in {float, int} or not math.isfinite(options[k]) or options[k] <= 0 for k in FLOAT_OPTIONS):
        raise ValueError('Finite positive optimization coefficients required')


class ResidualStageKernel:
    """Same production update function, with full per-update history and cursors."""

    def __init__(self, model, optimizer, code, code_encoder, replay, replay_encoder,
                 options, sources, device, amp, *, guard=None):
        if not dist.is_initialized() or dist.get_world_size() != 4:
            raise ValueError('Four actual initialized workers required')
        self.rank, self.world = dist.get_rank(), dist.get_world_size()
        def preflight():
            validate_options(options)
            if tuple(replay.families) != FULL_REPLAY_FAMILIES:
                raise ValueError('All nineteen original replay families required')
            if set(sources) != {'base_checkpoint_sha256', 'tokenizer_sha256', 'run_plan_sha256'} or not all(_digest(v) for v in sources.values()):
                raise ValueError('Exact own base/tokenizer/run-plan source bindings required')
            _optimizer_parameters(model, optimizer)
            if any(parameter.grad is not None for parameter in model.parameters()):
                raise ValueError('Initialize at a clean optimizer boundary')
            if code_encoder.config != model.config or replay_encoder.config != model.config or replay_encoder.image_size != options['image_size']:
                raise ValueError('Actual original encoders must match model/configured image size')
            if code_encoder.truncated or replay_encoder.truncated:
                raise ValueError('Clipped sequences cannot enter the complete-source stage')
            populations = tuple((name, replay.size(name)) for name in replay.families)
            budget = exposure_budget(len(code), options['passes'], self.world,
                options['batch_size'], options['gradient_accumulation'])
            schedule = coverage_replay_schedule(populations, budget['optimizer_steps'],
                self.world * options['replay_batch_size'] * options['replay_accumulation'])
            signature = fingerprint(dict(config=asdict(model.config), base=state_digest(model.base.state_dict()),
                residuals=model.residuals.state_dict(), optimizer=optimizer.state_dict(),
                options=options, sources=sources, code_size=len(code), replay_populations=populations, budget=budget))
            return budget, schedule, signature
        budget, schedule, signature = _validated(preflight)
        _same(signature)
        self.options, self.sources = options.copy(), sources.copy()
        self.model, self.optimizer = model, optimizer
        self.code, self.code_encoder = code, code_encoder
        self.replay, self.replay_encoder = replay, replay_encoder
        self.device, self.amp, self.guard = device, amp, guard
        self.wrapped = FixedReductionModel(model.code)
        self.budget = budget
        self.steps = self.budget['optimizer_steps']
        self.schedule = schedule
        self.completed, self.metrics = 0, []

    def sampler_state(self, step):
        if type(step) is not int or not 0 <= step <= self.steps:
            raise ValueError('Sampler update must be in the exact declared budget')
        families = Counter(self.schedule.at(i)[0] for i in range(step))
        code_exposures = step * self.world * self.options['batch_size'] * self.options['gradient_accumulation']
        replay_width = self.world * self.options['replay_batch_size'] * self.options['replay_accumulation']
        return dict(updates=step, code_source_size=len(self.code), code_examples=code_exposures,
            code_epoch_and_cursor=list(divmod(code_exposures, len(self.code))),
            replay_examples=step * replay_width,
            replay_families={name: dict(source_size=self.replay.size(name), updates=families[name], examples=families[name] * replay_width,
                epoch_and_cursor=list(divmod(families[name] * replay_width, self.replay.size(name))))
                for name in self.replay.families})

    def _guard(self, full=False):
        def check():
            if self.guard is not None:
                self.guard.check(full=full)
        _validated(check)

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
        result = training_update(self.wrapped, self.optimizer, self.code, self.code_encoder,
            self.replay, self.replay_encoder, self.completed, self.options,
            self.device, self.amp, self.rank, self.world)
        if self.code_encoder.truncated or self.replay_encoder.truncated:
            raise ValueError('Any clipped original sequence invalidates this complete-source stage')
        if any(p.grad is not None for p in self.model.base.parameters()):
            raise ValueError('Every original backbone tensor must stay frozen')
        self.optimizer.zero_grad(set_to_none=True)
        self.completed += 1
        self.metrics.append(result)
        return result

    def snapshot(self, path, *, maximum_bytes, minimum_free_bytes):
        self._guard(full=True)
        return save_collective_residual(path, self.model, self.optimizer, **self.sources,
            completed_steps=self.completed, planned_steps=self.steps,
            sampler_state=self.sampler_state(self.completed), metrics=self.metrics,
            device=self.device, maximum_bytes=maximum_bytes, minimum_free_bytes=minimum_free_bytes)

    def restore(self, path, expected_checkpoint_sha256, *, allow_completed=False):
        self._guard(full=True)
        step, progress, history = restore_collective_residual(path, self.model, self.optimizer,
            **self.sources, expected_checkpoint_sha256=expected_checkpoint_sha256,
            planned_steps=self.steps, sampler_state_for_step=self.sampler_state,
            device=self.device, allow_completed=allow_completed)
        self.completed, self.metrics = step, history
        return step, progress
