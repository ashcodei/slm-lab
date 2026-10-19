"""Combine actual native workload and fixed full-horizon capacity evidence.

A complete source-state readback and an owned production launcher are still
required. Explicit small CPU fixtures remain unable to admit production.
"""
import copy
from pathlib import Path
from .corpus import file_hash

import torch

from .native_suffix_capacity_controller_v2 import live, history_inventory
from .codenet_checkpoint_v1 import capture_rng
from .native_suffix_capacity_cases_v1 import choose_admitted
from .native_suffix_capacity_controller_v2 import FORMAT as WORKLOAD
from .native_suffix_full_horizon_capacity_v2 import stress, FORMAT as STRESS
from .native_suffix_train_data_v1 import NativeSuffixTrainData, LEVELS, FORMAT as DATA_FORMAT
from .native_suffix_validation_stage_v2 import PROTOCOL
from .native_suffix_indexed_stage_v4 import HISTORY
from .native_suffix_batched_history_stage_v5 import NativeSuffixBatchedHistoryStageV5, PROTOCOL as HISTORY_BATCH
from .parallel_function_repair_stage_v1 import value_digest
from .whole_human_sft_evidence_v1 import tensor_digest

FORMAT = 'own-native-suffix-indexed-complete-selected-full-horizon-stress-evidence-v2'


def stress_positions(data):
    if (type(data) is not NativeSuffixTrainData or data.role != 'train' or not data.full_source_data_admitted
            or data.purpose not in ['capacity', 'optimization'] or len(data) != 14884 or not data.admission_sha256):
        raise ValueError('All actual admitted native source views are required for full-horizon selection')
    data.guard.check()
    rows = [data.information(p) for p in data.slots]
    for row in rows:
        row['prefix_plus_full_native_horizon'] = row['conditional_prefix_tokens'] + row['full_native_action_budget']
    result = set()
    for kind in ['documented', 'apps']:
        for level in LEVELS:
            eligible = [r for r in rows if r['kind'] == kind and r['level'] == level and r['native_execution_eligible']]
            if eligible:
                for key in ['conditional_prefix_tokens', 'full_native_action_budget', 'prefix_plus_full_native_horizon']:
                    result.add(max(eligible, key=lambda r: (r[key], -r['position']))['position'])
    data.guard.check()
    return sorted(result)


def check_complete_capacity_plan(stage, plan):
    expected = choose_admitted(stage.data, seed=stage.options['seed'], epochs=stage.options['epochs'])
    if stage.data.purpose == 'optimization':
        # The workload was selected by the capacity API. Training retains its
        # own distinct source identity, while this comparison reconstructs the
        # original capacity identity from exactly the same closed bytes/views.
        entries = stage.data.complete_schedule(epochs=stage.options['epochs'], width=4, seed=stage.options['seed'])
        expected['complete_training_source_identity'] = value_digest(dict(format=DATA_FORMAT, purpose='capacity',
            manifest_sha256=stage.data.manifest_sha256, admission_sha256=stage.data.admission_sha256,
            full_source_data_admitted=True, complete_source_view_schedule=entries))
    if plan != expected:
        raise ValueError('The exact complete original capacity plan must reconstruct across the explicit data API roles')


def run_stress_cases(stage, *, allow_mechanics=False, mechanics_positions=None):
    error = None
    positions = None
    try:
        if (type(stage) is not NativeSuffixBatchedHistoryStageV5 or stage.world != 4 or stage.failed
                or stage.completed != len(stage.entries) or not stage.optimizer.state
                or stage.mode != 'capacity' and not (stage.mode == 'mechanics' and allow_mechanics is True)):
            raise ValueError('A completed actual four-replica disposable workload with allocated Adam is required')
        if stage.mode == 'capacity':
            if mechanics_positions is not None: raise ValueError('Production horizon selection cannot be overridden')
            positions = stress_positions(stage.data)
        else:
            if (not isinstance(mechanics_positions, list) or not mechanics_positions
                    or any(type(p) is not int for p in mechanics_positions)
                    or len(mechanics_positions) != len(set(mechanics_positions))):
                raise ValueError('Explicit small-fixture stress positions are required')
            positions = list(mechanics_positions)
        if not set(positions) <= {p for row in stage.entries for p in row}:
            raise ValueError('Every worst native source must also be in the actual selected workload')
        stage.intact()
    except Exception as exc: error = repr(exc)
    stage.coordinated(error, 'complete selected native horizon preflight')
    if any(p != positions for p in stage.gather(positions)):
        raise ValueError('All actual ranks must stress exactly the same complete worst-source selection')
    before = live(stage)
    rows = []
    for position in positions:
        error = None
        result = None
        try: result = stress(stage, position, allow_mechanics=allow_mechanics)
        except Exception as exc: error = repr(exc)
        stage.coordinated(error, 'actual native full-group fixed horizon allocation')
        rows.append(result)
    stage.coordinated(None if live(stage) == before else 'Fixed horizon stress mutated a live replica', 'native horizon unchanged state')
    return dict(format=FORMAT, complete=True, rank=stage.rank, world=stage.world, mode=stage.mode, profile=stage.profile,
        actual_device_type=stage.device.type, actual_device_index=stage.device.index,
        actual_research_actor_parameters=stage.model.counts()['actual_unique'], actual_native_parameters=stage.model.base.counts()['total_unique'],
        original_source_bindings=copy.deepcopy(stage.sources), original_objective_options=copy.deepcopy(stage.options),
        original_own_reference_binding=copy.deepcopy(stage.reference_binding), actual_complete_selected_horizon_positions=positions,
        actual_completed_workload_updates=stage.completed, actual_actor_sha256=tensor_digest(stage.model.parts.named_parameters()),
        actual_Adam_sha256=value_digest(stage.optimizer.state_dict()), actual_private_RNG_sha256=value_digest(capture_rng(stage.device)),
        indexed_history_protocol=HISTORY, source_checked_history_batch_protocol=HISTORY_BATCH, actual_history_budgets=copy.deepcopy(stage.history_budgets),
        actual_durable_indexed_history_inventory=history_inventory(stage),
        actual_fixed_horizon_cases=rows, all_selected_weights_Adam_source_cursor_and_private_RNG_unchanged=True,
        actual_GPU_peak_allocated_bytes=torch.cuda.max_memory_allocated(stage.device) if stage.device.type == 'cuda' else None,
        actual_GPU_peak_reserved_bytes=torch.cuda.max_memory_reserved(stage.device) if stage.device.type == 'cuda' else None,
        production_training_admitted=False, new_optimizer_calls=0, new_generated_actions=0,
        fixed_capacity_actions_are_not_policy_rollouts_or_rewards=True,
        full_size_selected_horizon_allocation_passed=stage.mode == 'capacity', model_quality_gain=False, goal_achieved=False)


def require_capacity(stage, reports, horizon_reports, plan, *, allow_mechanics=False):
    if (type(stage) is not NativeSuffixBatchedHistoryStageV5 or stage.world != 4
            or stage.profile not in ['anchor_CE', 'anchor_execution']
            or not isinstance(reports, list) or len(reports) != 4
            or not isinstance(horizon_reports, list) or len(horizon_reports) != 4
            or stage.sources.get(str(Path(__file__).resolve())) != file_hash(Path(__file__).resolve())):
        raise ValueError('Every actual native workload and full-horizon replica report is required')
    full = stage.mode != 'mechanics'
    if full:
        if (stage.mode not in ['capacity', 'train'] or stage.options['CPU_mechanics_horizon'] is not None
                or stage.options['epochs'] != 3 or stage.options['group_size'] != 8):
            raise ValueError('The entire admitted native-source plan, three epochs and eight full-horizon attempts must remain')
        check_complete_capacity_plan(stage, plan)
        positions = stress_positions(stage.data)
        if not positions: raise ValueError('A native execution experiment requires actual qualified source views')
    else:
        if allow_mechanics is not True or plan.get('mechanics_only') is not True:
            raise ValueError('Small CPU fixtures require explicit opt-in and do not admit production')
        positions = horizon_reports[0].get('actual_complete_selected_horizon_positions')
        if not isinstance(positions, list) or not positions or any(type(p) is not int for p in positions):
            raise ValueError('Every explicit native mechanics horizon case must remain')
    budget = stage.data.budget(plan['capacity_entries'])
    for rank, (work, horizon) in enumerate(zip(reports, horizon_reports)):
        common = dict(rank=rank, world=4, mode='capacity' if full else 'mechanics', profile=stage.profile,
            actual_research_actor_parameters=stage.model.counts()['actual_unique'], actual_native_parameters=stage.model.base.counts()['total_unique'],
            original_source_bindings=stage.sources, original_objective_options=stage.options,
            original_own_reference_binding=stage.reference_binding)
        if (any(work.get(k) != v or horizon.get(k) != v for k, v in common.items())
                or work.get('format') != WORKLOAD or work.get('complete') is not True or work.get('passed') is not True
                or work.get('actual_stage_class') != 'NativeSuffixBatchedHistoryStageV5' or work.get('recorded_source_validation') != PROTOCOL
                or work.get('capacity_plan_sha256') != value_digest(plan)
                or work.get('actual_complete_capacity_source_positions') != plan['capacity_entries']
                or work.get('actual_complete_capacity_updates') != len(plan['capacity_entries'])
                or work.get('actual_physical_optimizer_calls') != len(plan['capacity_entries']) + 1
                or work.get('actual_source_budget') != budget
                or work.get('actual_complete_CE_only_presentations') != budget['CE_only_presentations']
                or work.get('actual_execution_eligible_presentations') != budget['execution_eligible_presentations']
                or work.get('actual_unconditional_original_CE_targets') != budget['original_response_EOS_targets']
                or work.get('actual_single_source_three_empty_rank_steps') != budget['actual_single_source_three_empty_rank_steps']
                or work.get('actual_committed_prefix_trajectory_count') != budget['execution_eligible_presentations'] * stage.options['group_size']
                or horizon.get('format') != FORMAT or horizon.get('complete') is not True
                or horizon.get('actual_complete_selected_horizon_positions') != positions
                or horizon.get('actual_completed_workload_updates') != len(plan['capacity_entries'])
                or horizon.get('new_optimizer_calls') != 0 or horizon.get('new_generated_actions') != 0
                or horizon.get('production_training_admitted') is not False
                or horizon.get('all_selected_weights_Adam_source_cursor_and_private_RNG_unchanged') is not True):
            raise ValueError('Every actual native source/target/disposition/workload/replay/horizon binding must match')
        declared = work.get('actual_history_budgets', {})
        if (work.get('indexed_history_protocol') != HISTORY or horizon.get('indexed_history_protocol') != HISTORY
                or work.get('source_checked_history_batch_protocol') != HISTORY_BATCH
                or horizon.get('source_checked_history_batch_protocol') != HISTORY_BATCH
                or declared.get('maximum_records') != len(plan['capacity_entries'])
                or horizon.get('actual_history_budgets') != declared
                or not work.get('actual_durable_indexed_history_inventory')
                or not horizon.get('actual_durable_indexed_history_inventory')
                or work.get('no_complete_neural_history_materialized_by_capacity_controller') is not True
                or not all(declared.get(k) == stage.history_budgets[k] for k in
                    ['maximum_compact_record_bytes', 'maximum_expanded_record_bytes', 'minimum_free_bytes'])):
            raise ValueError('Every exact indexed record bound and actual allocated history/failure inventory must remain')
        expected = work['actual_per_rank_intervals'][rank]
        if any(horizon.get(k) != expected[v] for k, v in [('actual_actor_sha256', 'actor_sha256'),
                ('actual_Adam_sha256', 'Adam_sha256'), ('actual_private_RNG_sha256', 'private_RNG_sha256')]):
            raise ValueError('Fixed allocation must preserve the exact corresponding completed workload state')
        cases = horizon.get('actual_fixed_horizon_cases', [])
        if len(cases) != len(positions): raise ValueError('Every selected native fixed-horizon case must be retained')
        for position, case in zip(positions, cases):
            info = stage.data.information(position)
            actual_horizon = stage.options['CPU_mechanics_horizon'] or info['full_native_action_budget']
            if (case.get('format') != STRESS or case.get('complete') is not True or case.get('original_source') != info
                    or case.get('actual_fixed_non_EOS_actions_per_history') != actual_horizon
                    or case.get('full_original_action_horizon') != info['full_native_action_budget']
                    or case.get('actual_native_group_size') != stage.options['group_size']
                    or case.get('actual_prefill_batch_sizes') != [stage.options['group_size']]
                    or case.get('actual_decode_batch_sizes') != [stage.options['group_size']] * (actual_horizon - 1)
                    or case.get('actual_complete_original_CE_targets') != info['original_response_EOS_targets']
                    or case.get('actual_original_prefix_tokens') != info['conditional_prefix_tokens']
                    or case.get('actual_finite_actor_gradient_tensors') != len(stage.parameters)
                    or case.get('actual_peak_logical_KV_bytes') != (2 * stage.options['group_size'] * stage.model.config.num_layers
                        * stage.model.config.num_kv_heads * (stage.model.config.hidden_size // stage.model.config.num_heads)
                        * (info['conditional_prefix_tokens'] + actual_horizon - 1) * 4)
                    or case.get('actual_peak_logical_KV_bytes') != case.get('exact_full_group_horizon_logical_KV_bytes')
                    or not 0 <= case.get('actual_maximum_cached_full_logp_difference', float('inf')) <= stage.options['cache_absolute_bound']
                    or case.get('actual_fixed_actions') != [[4 + (i + row) % (stage.model.config.vocab_size - 4)
                        for i in range(actual_horizon)] for row in range(stage.options['group_size'])]
                    or case.get('no_execution_reward_claimed') is not True
                    or case.get('fixed_actions_not_sampled_trajectories') is not True
                    or case.get('full_size_GPU_horizon_allocation_passed') is not full
                    or case.get('no_optimizer_step') is not True or case.get('source_cursor_Adam_weights_and_all_private_RNG_unchanged') is not True):
                raise ValueError('Full original histories, complete CE and finite reference/actor gradients must remain exact')
        if full and (work.get('actual_device_type') != 'cuda' or work.get('actual_device_index') != rank
                or horizon.get('actual_device_type') != 'cuda' or horizon.get('actual_device_index') != rank
                or work.get('actual_Adam_states') != 561 or work.get('full_size_actual_selected_workload_passed') is not True
                or horizon.get('full_size_selected_horizon_allocation_passed') is not True
                or not all(work.get(k) is True for k in ['all_actual_next_actions_execution_semantics_unclipped_gradients_Adam_private_RNG_exact',
                    'invalid_peer_CPU_RNG_rejected_before_mutation', 'invalid_peer_CUDA_RNG_rejected_before_mutation',
                    'peer_only_valid_foreign_RNG_rejected_before_mutation'])
                or not 0 < horizon.get('actual_GPU_peak_allocated_bytes', 0) <= horizon.get('actual_GPU_peak_reserved_bytes', 0)):
            raise ValueError('Every full-size actual GPU/private state and worst-horizon allocation must pass')
    for key in ['final_actual_shared_checkpoint_sha256', 'final_actual_pointer_sha256']:
        if len({p[key] for p in reports}) != 1: raise ValueError('All actual capacity replicas must share one complete committed state')
    return dict(all_actual_workload_and_full_horizon_reports_match=True, production_training_admitted=False,
        independent_complete_source_state_readback_and_owned_launcher_still_required=True, goal_achieved=False)
