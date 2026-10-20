"""Whole public reasoned-code capacity, durable replay and actor cold load.

Owns no lease and starts no production run. The independently admitted full
source selection and actual full GPUs remain mandatory for production proofs.
"""
import copy
from pathlib import Path
import time

import torch

from .batched_reasoning_capacity_controller_v1 import _cold, draws, live, sync
from .public_reasoned_code_capacity_cases_v1 import choose
from .complete_response_stopping_loss_v1 import PROFILES
from .parallel_function_repair_stage_v1 import manifest, value_digest
from .parallel_public_reasoned_code_SFT_stage_v1 import ParallelPublicReasonedCodeSFTStage
from .parallel_repair_durable_state_v1 import commit, read_latest, restore_latest
from .codenet_checkpoint_v1 import capture_rng
from .whole_human_sft_evidence_v1 import tensor_digest

FORMAT = 'own-public-reasoned-code-SFT-full-capacity-private-recovery-cold-v1'


def run_capacity(stage, directory, *, capacity_plan, independent_actor, cold_inputs,
                 maximum_bytes, minimum_free_bytes, allow_mechanics=False):
    error = None
    try:
        if (type(stage) is not ParallelPublicReasonedCodeSFTStage or stage.completed or stage.failed or stage.world != 4
                or len(stage.entries) < 2 or stage.entries != capacity_plan['capacity_entries']
                or not callable(independent_actor)
                or stage.mode != 'capacity' and not (allow_mechanics is True and stage.mode == 'mechanics')
                or Path(directory).exists() and any(Path(directory).iterdir())):
            raise ValueError('Fresh actual four-replica stage and unused capacity directory required')
        if stage.mode == 'capacity':
            if capacity_plan != choose(stage.data, seed=stage.options['seed'], epochs=stage.options['epochs']):
                raise ValueError('All actual complete worst source batches and rank rotations must match')
        elif capacity_plan.get('mechanics_only') is not True:
            raise ValueError('An own-small fixture requires explicit mechanics-only marking')
        stage.intact()
        common = dict(plan=value_digest(capacity_plan), binding=stage.initial_binding, directory=str(Path(directory).resolve()))
    except Exception as exc: error = repr(exc)
    stage.coordinated(error, 'whole reasoned-code capacity preflight')
    if any(v != common for v in stage.gather(common)): raise ValueError('All actual replicas must run the same entire capacity workload')
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
            actual_original_context_tokens=value['source_metadata']['complete_context_tokens'],
            actual_response_EOS_targets=value['source_metadata']['response_EOS_targets']))
        return value
    def save(): return commit(stage, directory, maximum_bytes=maximum_bytes, minimum_free_bytes=minimum_free_bytes)
    step()
    first = save()
    error = None
    try:
        saved, receipt, pointer = read_latest(directory)
        if first != receipt: raise ValueError('Every actual first durable checkpoint/receipt must reconcile')
    except Exception as exc: error = repr(exc)
    stage.coordinated(error, 'first whole reasoned-code capacity readback')
    wanted_draws = draws(stage.device)
    wanted_row = step()
    expected = live(stage)
    # Valid foreign private RNGs still need the original external manifest;
    # malformed peer generators remain invalid even with a new manifest.
    for kind in ['CPU', 'valid_foreign'] + (['CUDA'] if stage.device.type == 'cuda' else []):
        bad = copy.deepcopy(saved)
        if kind == 'CPU': bad['private_workers'][3]['rng']['torch'] = torch.zeros(1, dtype=torch.uint8)
        if kind == 'CUDA': bad['private_workers'][3]['rng']['cuda'] = torch.zeros(1, dtype=torch.uint8)
        if kind == 'valid_foreign' and stage.rank == 2:
            bad['private_workers'][0]['rng'] = copy.deepcopy(saved['private_workers'][1]['rng'])
        rejected = False
        retained = receipt['semantic_manifest'] if kind == 'valid_foreign' else manifest(bad)
        try: stage.resume(bad, expected_manifest=retained)
        except ValueError: rejected = True
        stage.coordinated(None if rejected and live(stage) == expected else 'A corrupt peer private RNG mutated a live value',
            kind + ' private RNG rejection proof')
    restore_latest(stage, directory, expected_pointer_sha256=pointer)
    got_draws = draws(stage.device)
    got_row = step()
    current = live(stage)
    if (got_draws != wanted_draws or got_row != wanted_row
            or any(current[k] != expected[k] for k in ['actor', 'Adam', 'RNG', 'metrics', 'completed', 'failed'])):
        error = 'Actual next complete source loss/unclipped gradients/weights/Adam/private RNG differ'
    else: error = None
    stage.coordinated(error, 'exact next complete coding source replay')
    while stage.completed < len(stage.entries): step()
    final = save()
    error = None
    try:
        last, final_receipt, final_pointer = read_latest(directory)
        if (final_receipt != final or last['completed'] != len(stage.entries)
                or value_digest(last['metrics']) != value_digest(stage.metrics)
                or tensor_digest(last['model']['parts'].items()) != tensor_digest(stage.model.parts.named_parameters())):
            raise ValueError('Every final coding capacity source and actual selected value must match durable storage')
    except Exception as exc: error = repr(exc)
    stage.coordinated(error, 'entire final coding capacity state readback')
    cold = _cold(stage, independent_actor, last, cold_inputs)
    stage.intact()
    workers = stage.gather(dict(rank=stage.rank, private_RNG_sha256=value_digest(capture_rng(stage.device)),
        actor_sha256=tensor_digest(stage.model.parts.named_parameters()), Adam_sha256=value_digest(stage.optimizer.state_dict()),
        actual_intervals=timings))
    if (len({r['private_RNG_sha256'] for r in workers}) != 4 or len({r['actor_sha256'] for r in workers}) != 1
            or len({r['Adam_sha256'] for r in workers}) != 1):
        raise ValueError('All actual four private RNGs must remain distinct with one identical complete actor/Adam')
    return dict(format=FORMAT, complete=True, passed=True, rank=stage.rank, world=4, mode=stage.mode, profile=stage.profile,
        actual_device_type=stage.device.type, actual_device_index=stage.device.index,
        actual_research_actor_parameters=stage.model.counts()['actual_unique'], actual_native_parameters=stage.model.base.counts()['total_unique'],
        original_source_bindings=copy.deepcopy(stage.sources), original_objective_options=copy.deepcopy(stage.options),
        capacity_plan_sha256=value_digest(capacity_plan), actual_complete_capacity_source_positions=copy.deepcopy(stage.entries),
        actual_complete_capacity_updates=stage.completed, actual_physical_optimizer_calls=physical_calls,
        actual_source_budget=stage.data.budget(stage.entries), actual_Adam_states=len(stage.optimizer.state),
        actual_source_views=sorted({r['view'] for e in stage.entries for r in [stage.data.information(p) for p in e]}),
        actual_all_physical_context_tokens_including_replay=sum(r['actual_original_context_tokens'] for r in timings),
        actual_all_physical_response_EOS_targets_including_replay=sum(r['actual_response_EOS_targets'] for r in timings),
        actual_per_rank_intervals=workers, actual_complete_metrics=stage.metrics,
        exact_next_losses_unclipped_gradients_weights_Adam_private_RNG=True,
        invalid_peer_CPU_RNG_rejected_before_mutation=True, invalid_peer_CUDA_RNG_rejected_before_mutation=stage.device.type == 'cuda',
        peer_only_valid_foreign_RNG_rejected_before_mutation=True, independent_complete_actor_cold=cold,
        final_actual_shared_checkpoint_sha256=final['checkpoint_sha256'], final_actual_pointer_sha256=final_pointer,
        actual_GPU_peak_allocated_bytes=torch.cuda.max_memory_allocated(stage.device) if stage.device.type == 'cuda' else None,
        actual_GPU_peak_reserved_bytes=torch.cuda.max_memory_reserved(stage.device) if stage.device.type == 'cuda' else None,
        full_size_GPU_capacity_passed=stage.mode == 'capacity', production_updates=0, new_model_quality_gain=False,
        final_standalone_export_and_full_assessments_still_required=True, candidate_promoted=False, goal_achieved=False)


def require_full_capacity_reports(proofs, *, capacity_plan, profile, expected_sources, expected_options):
    if (not isinstance(proofs, list) or len(proofs) != 4 or profile not in PROFILES or not expected_sources
            or capacity_plan.get('complete_source_selection') is not True or capacity_plan.get('mechanics_only') is not False
            or capacity_plan.get('source_selection_only') is not True or capacity_plan.get('full_size_capacity_passed') is not False
            or capacity_plan.get('original_views') != ['code_only', 'reasoning']
            or capacity_plan.get('all_four_replicas_receive_each_selected_source_in_declared_plan') is not True
            or not isinstance(capacity_plan.get('actual_complete_training_tail_sources_per_rank'), list)
            or len(capacity_plan['actual_complete_training_tail_sources_per_rank']) != 4
            or expected_options.get('width') != 16 or expected_options.get('maximum_padded_tokens', 0) < 4096):
        raise ValueError('All four actual full-source/full-size coding capacity reports required')
    for rank, r in enumerate(proofs):
        if (r.get('format') != FORMAT or r.get('complete') is not True or r.get('passed') is not True
                or r.get('rank') != rank or r.get('world') != 4 or r.get('mode') != 'capacity'
                or r.get('profile') != profile or r.get('actual_device_type') != 'cuda' or r.get('actual_device_index') != rank
                or r.get('actual_research_actor_parameters') != 999706442 or r.get('actual_native_parameters') != 984399690
                or r.get('original_source_bindings') != expected_sources or r.get('original_objective_options') != expected_options
                or r.get('capacity_plan_sha256') != value_digest(capacity_plan)
                or r.get('actual_complete_capacity_source_positions') != capacity_plan['capacity_entries']
                or r.get('actual_source_budget') != capacity_plan['capacity_budget']
                or r.get('actual_source_views') != capacity_plan['original_views']
                or r.get('actual_complete_capacity_updates') != len(capacity_plan['capacity_entries'])
                or r.get('actual_physical_optimizer_calls') != len(capacity_plan['capacity_entries']) + 1
                or r.get('actual_Adam_states') != 561 or r.get('full_size_GPU_capacity_passed') is not True
                or r.get('production_updates') != 0 or not all(r.get(k) is True for k in [
                    'exact_next_losses_unclipped_gradients_weights_Adam_private_RNG', 'invalid_peer_CPU_RNG_rejected_before_mutation',
                    'invalid_peer_CUDA_RNG_rejected_before_mutation', 'peer_only_valid_foreign_RNG_rejected_before_mutation'])
                or not 0 < r.get('actual_GPU_peak_allocated_bytes', 0) <= r.get('actual_GPU_peak_reserved_bytes', 0)):
            raise ValueError('Every actual whole-capacity GPU/source/optimizer/private recovery proof must reconcile')
        cold = r.get('independent_complete_actor_cold', {})
        if (cold.get('complete') is not True or cold.get('mode') != 'capacity' or cold.get('mechanics_only') is not False
                or cold.get('actual_text_positions') != 4096 or cold.get('actual_photo_side') != 448
                or cold.get('actual_word_shape') != [64, 448] or cold.get('actual_word_readout_dtype') != 'torch.float32'
                or not all(cold.get(k) is True for k in ['every_base_head_actor_critic_parameter_equal', 'no_shared_parameter_storage',
                    'complete_text_photo_word_outputs_bit_exact', 'all_private_RNGs_unchanged'])):
            raise ValueError('All actual complete full-size text/photo/word cold output and value checks required')
    if len({r['final_actual_shared_checkpoint_sha256'] for r in proofs}) != 1:
        raise ValueError('All four actual reports must bind the same complete shared capacity checkpoint')
    return dict(all_four_complete_full_size_capacity_report_schemas_match=True, production_training_admitted=False,
        independent_complete_capacity_state_readback_and_owned_training_admission_still_required=True, goal_achieved=False)
