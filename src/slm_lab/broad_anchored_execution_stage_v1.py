"""Whole selected-source epoch, compact private recovery and own-policy KL.

Each global update draws one original source. All replicas supervise the same
complete reference with a global denominator and produce independent samples.
Full trajectory records are returned to an external durable journal; private
checkpoints retain their deterministic semantic digest and compact metrics.
"""
import copy
import hashlib
import json
import math
import random

import torch

from .anchored_execution_step_v2 import update
from .code_probability_guard_v8 import load as load_calibration
from .own_dense_actor_checkpoint_v1 import make_state, restore_state

FORMAT = 'own-broad-anchored-execution-private-source-epoch-v1'


def semantic_digest(trajectories):
    rows = []
    for item in trajectories:
        result = item['actual_execution_reward']
        actual = result.get('actual_execution', result.get('raw_candidate_execution', {}))
        rows.append(dict(response_ids=item['response_ids'], cached=item['actual_cached_behavior_logps'],
            full=item['actual_full_behavior_logps'], response=item['own_response'],
            reward=result['reward'], status=result['status'],
            source_program_sha256=actual.get('original_program_sha256'),
            original_tests_sha256=actual.get('original_tests_sha256'),
            case_passes=actual.get('case_passes', actual.get('example_passes'))))
    return hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def source_order(corpus, seed):
    # A complete, predetermined easy-to-hard curriculum. No held score or
    # sampled reward influences which original programs are consumed.
    strata = {'documented': [], 'introductory': [], 'interview': [], 'competition': []}
    for position, row in enumerate(corpus.rows):
        key = 'documented' if row['kind'] == 'documented' else row['original']['difficulty']
        if key not in strata: raise ValueError('Every original difficulty requires an explicit curriculum disposition')
        strata[key].append(position)
    result = []
    for i, values in enumerate(strata.values()):
        random.Random(seed + i * 100003).shuffle(values); result.extend(values)
    if sorted(result) != list(range(len(corpus))):
        raise ValueError('Every admitted original source must occur exactly once')
    return result


class BroadAnchoredExecutionStage:
    def __init__(self, model, optimizer, corpus, own_reference, *, root, method, seed,
                 sources, world=1, rank=0, control_group=None, gradient_group=None,
                 local_group_size=1, KL_weight=.01, anchor_weight=1.,
                 cache_absolute_bound, cache_calibration_sha256,
                 CPU_control_positions=None, CPU_mechanics_horizon=None):
        if (method not in ['anchor_CE', 'anchor_execution'] or type(seed) is not int or seed < 1
                or not sources or any(not isinstance(k, str) or not isinstance(v, str) or len(v) != 64
                                     or any(c not in '0123456789abcdef' for c in v) for k, v in sources.items())):
            raise ValueError('Explicit complete source epoch, method, seed and immutable SHA bindings required')
        self.model, self.optimizer, self.corpus, self.reference = model, optimizer, corpus, own_reference
        own_reference.assert_bound(model)
        device = next(model.parameters()).device
        if device.type == 'cuda':
            if CPU_control_positions is not None or CPU_mechanics_horizon is not None:
                raise ValueError('CPU source/horizon controls cannot silently reduce GPU production')
            calibrated = load_calibration(root, model)
            if calibrated['bound'] != cache_absolute_bound or calibrated['sha256'] != cache_calibration_sha256:
                raise ValueError('Actual same-architecture rounding admission must match the bound source plan')
        self.order = source_order(corpus, seed)
        if CPU_control_positions is not None:
            if (device.type != 'cpu' or not CPU_control_positions
                    or len(set(CPU_control_positions)) != len(CPU_control_positions)
                    or any(type(p) is not int or not 0 <= p < len(corpus) for p in CPU_control_positions)):
                raise ValueError('Explicit native CPU source controls required')
            self.order = list(CPU_control_positions)
        self.sources = copy.deepcopy(sources)
        self.settings = dict(method=method, seed=seed, world=world, rank=rank,
            local_group_size=local_group_size, KL_weight=KL_weight, anchor_weight=anchor_weight,
            cache_absolute_bound=cache_absolute_bound, cache_calibration_sha256=cache_calibration_sha256,
            CPU_control_positions=CPU_control_positions, CPU_mechanics_horizon=CPU_mechanics_horizon,
            original_source_population=len(corpus), planned_updates=len(self.order),
            curriculum='documented_then_introductory_then_interview_then_competition_seeded_within_stratum',
            complete_original_reference_supervision_every_update=True,
            reference_is_frozen_own_starting_actor=True, precision='FP32_TF32_disabled',
            gradient_transport='Gloo_host_SUM_already_global_normalized')
        self.kwargs = dict(method=method, own_reference=own_reference, rank=rank, world=world,
            local_group_size=local_group_size, KL_weight=KL_weight, anchor_weight=anchor_weight,
            cache_absolute_bound=cache_absolute_bound, control_group=control_group, gradient_group=gradient_group,
            CPU_mechanics_horizon=CPU_mechanics_horizon)
        self.completed = 0; self.metrics = []

    def cursor(self, completed):
        return dict(completed_updates=completed, planned_updates=len(self.order),
            source_order=list(self.order), source_population=len(self.corpus),
            outer_world_size=self.settings['world'], replica_rank=self.settings['rank'], seed=self.settings['seed'])

    def update(self):
        if self.completed >= len(self.order): raise ValueError('The entire declared source epoch is already complete')
        position = self.order[self.completed]
        record = update(self.model, self.optimizer, self.corpus, position, **self.kwargs)
        record.update(step=self.completed + 1, actual_counts=self.model.counts(),
            source_epoch_order_position=self.completed, planned_source_updates=len(self.order))
        # Exclude wall-time-dependent execution payloads and long action logs
        # from every private checkpoint, retaining a semantic replay binding.
        compact = {k: copy.deepcopy(v) for k, v in record.items() if k not in ['trajectories', 'complete_prompt_ids']}
        compact['global_trajectory_semantic_sha256'] = semantic_digest(record['trajectories'])
        compact['original_source_context_positions'] = self.corpus.row(position)['complete_context_positions']
        self.completed += 1; self.metrics.append(compact)
        return record

    def state(self):
        self.reference.assert_bound(self.model)
        core = make_state(self.model, self.optimizer, groups=('actor',), sources=self.sources,
            completed=self.completed, planned=len(self.order), cursor=self.cursor(self.completed),
            metrics=self.metrics, device=next(self.model.parameters()).device)
        return dict(format=FORMAT, settings=copy.deepcopy(self.settings), core=core,
            corpus_provenance=copy.deepcopy(self.corpus.provenance), reference_binding=self.reference.binding(),
            private_state_is_not_whole_distributed_checkpoint=True,
            full_trajectory_records_require_external_durable_journal=True,
            candidate_promoted=False, goal_achieved=False)

    def resume(self, payload, *, allow_completed=False):
        self.reference.assert_bound(self.model)
        core = payload.get('core', {}); completed = core.get('completed')
        if (payload.get('format') != FORMAT or payload.get('settings') != self.settings
                or payload.get('corpus_provenance') != self.corpus.provenance
                or payload.get('reference_binding') != self.reference.binding()
                or payload.get('private_state_is_not_whole_distributed_checkpoint') is not True
                or payload.get('full_trajectory_records_require_external_durable_journal') is not True
                or payload.get('candidate_promoted') is not False or payload.get('goal_achieved') is not False
                or type(completed) is not int or not 0 < completed <= len(self.order)
                or not isinstance(core.get('metrics'), list) or len(core['metrics']) != completed):
            raise ValueError('Exact private source epoch, frozen own reference and compact trajectory bindings required')
        for i, record in enumerate(core['metrics']):
            position = self.order[i]; row = self.corpus.row(position)
            if (record.get('step') != i + 1 or record.get('position') != position or record.get('source_id') != row['id']
                    or record.get('original_reference_used_regardless_of_rewards') is not True
                    or record.get('source_reference_targets') != row['response_EOS_targets']
                    or record.get('method') != self.settings['method']
                    or record.get('reference_binding') != self.reference.binding()
                    or len(record.get('global_rewards', [])) != self.settings['world'] * self.settings['local_group_size']
                    or any(not math.isfinite(x) or not 0 <= x <= 1 for x in record['global_rewards'])
                    or len(record.get('global_trajectory_semantic_sha256', '')) != 64):
                raise ValueError('Every actual consumed native source and objective record must reconstruct before mutation')
        self.completed, self.metrics = restore_state(self.model, self.optimizer, core, groups=('actor',),
            expected_sources=self.sources, planned=len(self.order), expected_cursor=self.cursor(completed),
            device=next(self.model.parameters()).device, allow_completed=allow_completed)
