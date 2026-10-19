"""Source-aware sequencing and durable recovery for existing visual groups.

This internal stage never acquires a GPU, creates a model, changes trainability,
or grants runtime admission. A production controller must own the lease and
prove full-size capacity before selecting train mode. Explicit mechanics mode
is CPU-only and can never be exported as full-size capacity/training evidence.
"""
import copy

import torch

from . import joint_visual_durable_state_v2 as durable
from .joint_visual_objective_update_v1 import update, optimizer_clocks
from .joint_visual_private_checkpoint_v1 import _context, _cursor
from .joint_visual_private_checkpoint_v2 import (SOURCE_KEYS, FixedGuard, digest,
    settings, _control, _agree)
from .joint_visual_source_delivery_v1 import OriginalVisualSources
from .whole_human_sft_evidence_v1 import tensor_digest

FORMAT = 'own-complete-source-three-objective-visual-stage-v1'


def adam_settings(optimizer):
    return [{key: value for key, value in group.items() if key != 'params'}
        for group in optimizer.param_groups]


def make_plan(view, optimizer, fixed, *, sources, entries, options, mode):
    """Return canonical JSON content; its digest binds every stage setting."""
    if set(sources) != SOURCE_KEYS - {'run_plan_sha256'} or mode not in ['mechanics', 'capacity', 'train']:
        raise ValueError('Explicit stage mode and every independent non-plan source hash are required')
    fixed.check(view, optimizer)
    groups = settings(view, optimizer)
    return dict(format=FORMAT, mode=mode, sources=copy.deepcopy(sources),
        complete_entries_sha256=digest(entries), options=copy.deepcopy(options),
        optimizer=copy.deepcopy(adam_settings(optimizer)),
        initial_selected_values_sha256=tensor_digest((n, p) for rows in groups.values() for n, p in rows),
        actual_actor_parameters=view.actor.counts()['actual_unique'],
        actual_base_parameters=view.actor.base.counts()['total_unique'],
        whole_source_entry_count=len(entries), candidate_promoted=False, goal_achieved=False)


class JointVisualStage:
    def __init__(self, view, optimizer, provider, *, plan, sources, entries,
                 rank=0, world=1, device='cpu', control_group=None, fixed=None):
        self.rank, self.world = _control(control_group)
        self.device = torch.device(device)
        self.group = control_group
        error = specification = None
        try:
            self.view, self.optimizer, self.provider = view, optimizer, provider
            self.fixed = fixed if fixed is not None else FixedGuard(view, optimizer)
            self._plan, self._sources, self._entries = copy.deepcopy(plan), dict(sources), copy.deepcopy(entries)
            self.mode = plan['mode']
            self.private_mode = 'train' if self.mode == 'train' else 'capacity'
            expected = make_plan(view, optimizer, self.fixed, sources={k: v for k, v in sources.items()
                if k != 'run_plan_sha256'}, entries=entries, options=plan['options'], mode=self.mode)
            if (set(sources) != SOURCE_KEYS or digest(plan) != sources['run_plan_sha256']
                    or digest(expected) != digest(plan) or rank != self.rank or world != self.world):
                raise ValueError('The exact canonical plan, all source hashes and actual ranks must agree')
            _context(view, optimizer, self.fixed, sources, entries, self.private_mode,
                rank, world, self.device, self.rank, self.world)
            if optimizer.state or optimizer_clocks(view, optimizer) != dict(vision=0, pointing=0, word=0):
                raise ValueError('A new stage must begin at the bound initializer with fresh visual Adam state')
            if self.mode == 'mechanics':
                if self.device.type != 'cpu' or torch.cuda.is_initialized():
                    raise ValueError('Explicit small mechanics stages must remain CPU-only')
            elif (type(provider) is not OriginalVisualSources or not provider.training_admitted
                    or provider.purpose != ('optimization' if self.mode == 'train' else 'capacity')
                    or any(provider.bindings[k] != sources[k] for k in provider.bindings)
                    or self.world != 4 or self.device.type != 'cuda'
                    or view.actor.counts()['actual_unique'] != 999706442
                    or view.actor.base.counts()['total_unique'] != 984399690):
                raise ValueError('Full-size visual stages require actual admitted complete sources and four CUDA replicas')
            self._options = copy.deepcopy(plan['options'])
            self._settings = copy.deepcopy(plan['optimizer'])
            self._schedule_sha256 = digest(self._entries)
            self._binding_sha256 = digest(self._sources)
            self._plan_sha256 = digest(self._plan)
            self._metrics, self.completed = [], 0
            self.failed = False
            self._clock_prefixes = [dict(vision=0, pointing=0, word=0)]
            for entry in self._entries:
                clocks = dict(self._clock_prefixes[-1])
                clocks['vision'] += 1
                clocks['pointing'] += int(entry['family'] == 'images:additional_photo_captions')
                clocks['word'] += int(entry['family'] == 'words')
                self._clock_prefixes.append(clocks)
            specification = digest(dict(plan=plan, sources=sources, entries=entries))
        except Exception as exc:
            error = type(exc).__name__ + ': ' + str(exc)
        _agree(error, specification, self.group, self.world)

    @property
    def metrics(self):
        return copy.deepcopy(self._metrics)

    @property
    def next_entry(self):
        return None if self.completed == len(self._entries) else copy.deepcopy(self._entries[self.completed])

    def _identity(self):
        if (digest(self._entries) != self._schedule_sha256
                or digest(self._sources) != self._binding_sha256 or digest(self._plan) != self._plan_sha256
                or digest(self._options) != digest(self._plan['options']) or self.mode != self._plan['mode']
                or self.private_mode != ('train' if self.mode == 'train' else 'capacity')
                or digest(adam_settings(self.optimizer)) != digest(self._settings)):
            raise ValueError('The exact bound sources, plan, mode and ordinary optimizer settings must remain unchanged')
        self.fixed.check(self.view, self.optimizer)
        settings(self.view, self.optimizer)
        if any(p.grad is not None for p in self.view.parameters()):
            raise ValueError('A complete clear-gradient source boundary is required')

    def _boundary(self, *, require_next=False):
        error = common = None
        try:
            if (self.failed or type(self.completed) is not int or not 0 <= self.completed <= len(self._entries)
                    or require_next and self.completed == len(self._entries)):
                raise ValueError('One healthy exact source-prefix stage with unchanged bound settings is required')
            self._identity()
            cursor = _cursor(self._entries, self.completed, self._metrics)
            if (optimizer_clocks(self.view, self.optimizer) != self._clock_prefixes[self.completed]
                    or cursor['group_clocks'] != self._clock_prefixes[self.completed]):
                raise ValueError('Every actual group Adam clock/settings/gradient must match the exact completed prefix')
            common = digest(dict(cursor=cursor, options=self._options, optimizer=self._settings,
                sources=self._sources, mode=self.mode))
        except Exception as exc:
            error = type(exc).__name__ + ': ' + str(exc)
        _agree(error, common, self.group, self.world)

    def next_update(self):
        self._boundary(require_next=True)
        error = source_sha256 = prepared = source = None
        entry = self.next_entry
        try:
            prepared, source = self.provider.prepare(entry, rank=self.rank, world=self.world, device=self.device)
            if (source['family'] != entry['family'] or source['global_positions'] != entry['original_local_positions']
                    or sum(source['global_context_lengths']) != entry['complete_context_tokens']
                    or sum(source['global_response_EOS_targets']) != entry['response_EOS_targets']):
                raise ValueError('Every delivered objective must preserve its bound complete source entry')
            source_sha256 = digest(source)
        except Exception as exc:
            error = type(exc).__name__ + ': ' + str(exc)
        # Coordinate decode/source exceptions before any rank enters the
        # objective's nested kernels, gradient collectives or optimizer step.
        _agree(error, source_sha256, self.group, self.world)
        try:
            metric = update(self.view, self.optimizer, prepared, source, self._options, control_group=self.group)
            tentative = self._metrics + [metric]
            error = common = None
            try:
                cursor = _cursor(self._entries, self.completed + 1, tentative)
                self.fixed.check(self.view, self.optimizer)
                if optimizer_clocks(self.view, self.optimizer) != cursor['group_clocks']:
                    raise ValueError('Actual optimizer clocks must equal the next exact source prefix')
                common = digest(cursor)
            except Exception as exc:
                error = type(exc).__name__ + ': ' + str(exc)
            _agree(error, common, self.group, self.world)
            self._metrics = tentative
            self.completed += 1
            return copy.deepcopy(metric)
        except BaseException:
            # A failed numerical/update boundary may have partially changed
            # selected values. Recovery is explicit; do not retry that state.
            self.failed = True
            raise

    def commit(self, directory, *, maximum_bytes, minimum_free_bytes):
        self._boundary()
        return durable.commit(self.view, self.optimizer, self.fixed, directory,
            sources=self._sources, entries=self._entries, completed=self.completed, metrics=self._metrics,
            mode=self.private_mode, maximum_bytes=maximum_bytes, minimum_free_bytes=minimum_free_bytes,
            rank=self.rank, world=self.world, device=self.device, control_group=self.group)

    def restore(self, directory, *, expected_pointer_sha256, allow_completed=False):
        error = common = None
        try:
            self._identity()
            if (not isinstance(expected_pointer_sha256, str) or len(expected_pointer_sha256) != 64
                    or any(c not in '0123456789abcdef' for c in expected_pointer_sha256)):
                raise ValueError('Recovery requires an independently retained exact durable pointer hash')
            common = digest(dict(pointer_sha256=expected_pointer_sha256, plan=self._plan, sources=self._sources,
                allow_completed=allow_completed))
        except Exception as exc:
            error = type(exc).__name__ + ': ' + str(exc)
        _agree(error, common, self.group, self.world)
        completed, metrics, receipt = durable.restore_latest(self.view, self.optimizer, self.fixed, directory,
            sources=self._sources, entries=self._entries, mode=self.private_mode, rank=self.rank, world=self.world,
            device=self.device, control_group=self.group, expected_pointer_sha256=expected_pointer_sha256,
            allow_completed=allow_completed)
        self.completed, self._metrics, self.failed = completed, metrics, False
        self._boundary()
        return receipt
