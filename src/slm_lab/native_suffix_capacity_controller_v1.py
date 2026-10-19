"""Disposable whole-source native-code capacity and independent actor cold load.

This controller uses the qualified native stage, including real code workers,
complete CE fallback and real empty-rank tails. It owns no resource lease and
cannot authorize production. Early EOS in actual sampling does not prove
worst-case full-horizon allocation; a separate synthetic capacity probe and
complete source-state readback remain required before production admission.
"""
import copy
import time

import torch

from .batched_reasoning_capacity_controller_v1 import _cold, live, draws, sync
from .codenet_checkpoint_v1 import capture_rng
from .native_suffix_capacity_cases_v1 import choose_admitted
from .native_suffix_validation_stage_v2 import NativeSuffixValidationStageV2, PROTOCOL
from .parallel_function_repair_stage_v1 import manifest, value_digest
from .parallel_native_suffix_stage_v1 import CE_ONLY, KERNEL, record_semantics
from .parallel_repair_durable_state_v1 import commit, read_latest, restore_latest
from .whole_human_sft_evidence_v1 import tensor_digest

FORMAT = 'own-native-suffix-whole-selected-capacity-private-replay-actor-cold-v1'


def run_capacity(stage, directory, *, capacity_plan, independent_actor, cold_inputs,
                 maximum_bytes, minimum_free_bytes, allow_mechanics=False):
    error = common = None
    try:
        if (type(stage) is not NativeSuffixValidationStageV2 or stage.completed or stage.failed or stage.world != 4
                or len(stage.entries) < 2 or stage.entries != capacity_plan['capacity_entries']
                or not callable(independent_actor)
                or stage.mode != 'capacity' and not (allow_mechanics is True and stage.mode == 'mechanics')):
            raise ValueError('A fresh four-replica optimized native-code capacity stage is required')
        if stage.mode == 'capacity':
            if capacity_plan != choose_admitted(stage.data, seed=stage.options['seed'], epochs=stage.options['epochs']):
                raise ValueError('Every complete actual admitted native worst-source batch and physical rotation must match')
        elif capacity_plan.get('mechanics_only') is not True:
            raise ValueError('Own-small controller fixtures must be explicitly mechanics-only')
        stage.intact()
        common = dict(plan=value_digest(capacity_plan), binding=stage.initial_binding)
    except Exception as exc:
        error = repr(exc)
    stage.coordinated(error, 'whole native capacity preflight')
    if any(v != common for v in stage.gather(common)):
        raise ValueError('Every actual rank must run the identical complete native workload')
    if stage.device.type == 'cuda': torch.cuda.reset_peak_memory_stats(stage.device)
    timings, physical_calls = [], 0

    def step():
        nonlocal physical_calls
        sync(stage)
        began = time.monotonic()
        value = stage.update()
        sync(stage)
        physical_calls += 1
        timings.append(dict(actual_global_step=stage.completed, actual_seconds=time.monotonic() - began,
            actual_generated_actions=sum(t['kernel'].get('actual_sampled_action_tokens', 0)
                for t in value['actual_rank_observations'] if t['kernel'] is not None)))
        return value

    def save(): return commit(stage, directory, maximum_bytes=maximum_bytes, minimum_free_bytes=minimum_free_bytes)

    step()
    first = save()
    error = None
    saved = receipt = pointer = None
    try:
        saved, receipt, pointer = read_latest(directory)
        if first != receipt: raise ValueError('The complete shared file and private-worker external receipt must reconcile')
    except Exception as exc:
        error = repr(exc)
    stage.coordinated(error, 'native first durable readback')
    wanted_draws, wanted = draws(stage.device), step()
    expected = live(stage)
    expected_semantic_metrics = value_digest([record_semantics(r) for r in stage.metrics])
    for kind in ['invalid_CPU', 'invalid_CUDA', 'peer_foreign']:
        if kind == 'invalid_CUDA' and stage.device.type != 'cuda': continue
        bad = copy.deepcopy(saved)
        if kind == 'peer_foreign':
            if stage.rank == 2: bad['private_workers'][0]['rng'] = copy.deepcopy(saved['private_workers'][1]['rng'])
            expected_manifest = receipt['semantic_manifest']
        else:
            bad['private_workers'][3]['rng']['cuda' if kind == 'invalid_CUDA' else 'torch'] = torch.zeros(1, dtype=torch.uint8)
            expected_manifest = manifest(bad)
        rejected = False
        try: stage.resume(bad, expected_manifest=expected_manifest)
        except ValueError: rejected = True
        stage.coordinated(None if rejected and live(stage) == expected else 'Bad peer RNG mutated live native state',
            'native ' + kind + ' private RNG rejection')
    restore_latest(stage, directory, expected_pointer_sha256=pointer)
    actual_draws, actual = draws(stage.device), step()
    current = live(stage)
    error = None
    # Wall times/thread IDs remain in the raw evidence. Compare actual execution
    # semantics separately, preserving every program/test/output/verdict field.
    if (actual_draws != wanted_draws or record_semantics(actual) != record_semantics(wanted)
            or value_digest([record_semantics(r) for r in stage.metrics]) != expected_semantic_metrics
            or any(current[k] != expected[k] for k in ['actor', 'Adam', 'RNG', 'completed', 'failed'])):
        error = 'Next actual native actions/program verdicts/logps/gradients/weights/Adam/private RNG differ'
    stage.coordinated(error, 'exact next native source-group replay')
    while stage.completed < len(stage.entries): step()
    final = save()
    error = None
    last = last_receipt = final_pointer = None
    try:
        last, last_receipt, final_pointer = read_latest(directory)
        if (last_receipt != final or last['completed'] != len(stage.entries)
                or value_digest(last['metrics']) != value_digest(stage.metrics)
                or tensor_digest(last['model']['parts'].items()) != tensor_digest(stage.model.parts.named_parameters())):
            raise ValueError('Every final native source observation and selected value must equal its committed file')
    except Exception as exc:
        error = repr(exc)
    stage.coordinated(error, 'complete native final capacity readback')
    cold = _cold(stage, independent_actor, last, cold_inputs)
    stage.intact()
    observations = [a['kernel'] for r in stage.metrics for a in r['actual_rank_observations'] if a['kernel'] is not None]
    native = [r for r in observations if r['format'] == KERNEL]
    ce_only = [r for r in observations if r['format'] == CE_ONLY]
    samplers = [r['complete_batched_sampling_observation'] for r in native]
    budget = stage.data.budget(stage.entries)
    if (len(native) != budget['execution_eligible_presentations'] or len(ce_only) != budget['CE_only_presentations']
            or sum(r['unconditional_original_CE_anchor']['local_original_response_EOS_targets'] for r in observations)
                != budget['original_response_EOS_targets']):
        raise ValueError('All failed/abstaining native views must retain exactly their complete unconditional CE anchors')
    workers = stage.gather(dict(actual_rank=stage.rank, private_RNG_sha256=value_digest(capture_rng(stage.device)),
        actor_sha256=tensor_digest(stage.model.parts.named_parameters()), Adam_sha256=value_digest(stage.optimizer.state_dict()),
        actual_intervals=timings))
    if (len({v['private_RNG_sha256'] for v in workers}) != 4 or len({v['actor_sha256'] for v in workers}) != 1
            or len({v['Adam_sha256'] for v in workers}) != 1):
        raise ValueError('Four actual native workers require distinct private RNGs and identical model/Adam')
    return dict(format=FORMAT, complete=True, passed=True, rank=stage.rank, world=4, mode=stage.mode,
        actual_device_type=stage.device.type, actual_device_index=stage.device.index,
        actual_stage_class=type(stage).__name__, recorded_source_validation=PROTOCOL,
        actual_research_actor_parameters=stage.model.counts()['actual_unique'],
        actual_native_parameters=stage.model.base.counts()['total_unique'], profile=stage.profile,
        original_source_bindings=copy.deepcopy(stage.sources), original_objective_options=copy.deepcopy(stage.options),
        original_own_reference_binding=copy.deepcopy(stage.reference_binding), capacity_plan_sha256=value_digest(capacity_plan),
        actual_complete_capacity_updates=stage.completed, actual_physical_optimizer_calls=physical_calls,
        actual_source_budget=budget, actual_Adam_states=len(stage.optimizer.state),
        actual_complete_capacity_source_positions=copy.deepcopy(stage.entries),
        actual_execution_eligible_presentations=len(native), actual_complete_CE_only_presentations=len(ce_only),
        actual_unconditional_original_CE_targets=budget['original_response_EOS_targets'],
        actual_single_source_three_empty_rank_steps=budget['actual_single_source_three_empty_rank_steps'],
        actual_committed_prefix_trajectory_count=sum(len(r['trajectories']) for r in native),
        actual_committed_prefix_generated_action_count=sum(r['actual_sampled_action_tokens'] for r in native),
        actual_all_physical_global_generated_actions_including_replay=sum(t['actual_generated_actions'] for t in timings),
        actual_genuine_EOS_stops=sum(t['actual_stopped_on_EOS'] for r in native for t in r['trajectories']),
        actual_reward1_trajectories=sum(x == 1. for r in native for x in r['actual_rewards']),
        actual_reward_varying_groups=sum(len({x for x in r['actual_rewards'] if x is not None}) > 1 for r in native),
        actual_abstaining_groups=sum(r['policy_gradient_abstained_due_to_missing_verdict'] for r in native),
        actual_maximum_cached_full_logp_difference=max((s['cached_full_max_absolute_difference'] for s in samplers), default=0.),
        actual_peak_logical_KV_bytes=max((s['actual_peak_logical_KVcache_bytes'] for s in samplers), default=0),
        actual_conservative_full_horizon_logical_KV_bytes=max((s['conservative_full_horizon_logical_KVcache_bytes'] for s in samplers), default=0),
        actual_per_rank_intervals=workers, all_actual_next_actions_execution_semantics_unclipped_gradients_Adam_private_RNG_exact=True,
        execution_timings_and_thread_IDs_retained_outside_replay_equality=True,
        invalid_peer_CPU_RNG_rejected_before_mutation=True, invalid_peer_CUDA_RNG_rejected_before_mutation=stage.device.type == 'cuda',
        peer_only_valid_foreign_RNG_rejected_before_mutation=True, independent_complete_actor_cold=cold,
        final_actual_shared_checkpoint_sha256=final['checkpoint_sha256'], final_actual_pointer_sha256=final_pointer,
        actual_GPU_peak_allocated_bytes=torch.cuda.max_memory_allocated(stage.device) if stage.device.type == 'cuda' else None,
        actual_GPU_peak_reserved_bytes=torch.cuda.max_memory_reserved(stage.device) if stage.device.type == 'cuda' else None,
        full_size_actual_selected_workload_passed=stage.mode == 'capacity', full_size_GPU_capacity_passed=False,
        synthetic_worst_full_horizon_allocation_still_required=True, independent_complete_source_state_readback_still_required=True,
        production_training_admitted=False, production_updates=0, new_model_quality_gain=False,
        final_standalone_and_full_assessments_still_required=True, candidate_promoted=False, goal_achieved=False)
