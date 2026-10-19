"""Connect complete source capacity, private recovery, native cold and training.

Resource ownership and full baseline/final assessments belong to the external
GPU controller. This layer admits full training only after four matching real
full-size capacity reports. Explicit CPU mechanics can test the same flow but
cannot produce that admission or any production update.
"""
import copy

import torch

from .joint_visual_cold_overlay_v1 import prove as prove_cold, fixed_digest
from .joint_visual_durable_state_v2 import read_latest
from .joint_visual_experiment_v1 import prove_next_update, train as train_loop
from .joint_visual_objective_update_v1 import FAMILIES, named_groups, optimizer_clocks
from .joint_visual_private_checkpoint_v2 import _agree, digest, private_rng_sha256
from .joint_visual_training_stage_v1 import JointVisualStage
from .codenet_checkpoint_v1 import capture_rng
from .whole_human_sft_evidence_v1 import tensor_digest

FORMAT = 'own-whole-nine-family-visual-runtime-capacity-v1'


def bindings(stage):
    return {k: v for k, v in stage._sources.items() if k != 'run_plan_sha256'}


def run_capacity(stage, directory, *, capacity_plan, independent_native, cold_inputs,
                 maximum_bytes, minimum_free_bytes, allow_mechanics=False):
    error = common = None
    try:
        if (type(stage) is not JointVisualStage or stage.completed or stage.failed
                or stage.mode != 'capacity' and not (allow_mechanics is True and stage.mode == 'mechanics')
                or capacity_plan['capacity_entries'] != stage._entries
                or set(e['family'] for e in stage._entries) != set(FAMILIES)
                or not callable(independent_native)):
            raise ValueError('A fresh admitted capacity stage with all9complete source families is required')
        if stage.mode == 'capacity' and (capacity_plan['actual_original_epoch_entries'] != 4733
                or capacity_plan['actual_original_epoch_source_views'] != 302627
                or len(stage._entries) != 10 or sum(len(e['original_local_positions']) for e in stage._entries) != 510
                or capacity_plan['maximum_original_native_context'] != 2952
                or capacity_plan['maximum_original_photo_negative_set'] != 64
                or capacity_plan['maximum_original_word_UTF8_bytes'] != 50
                or stage._entries[-1]['original_local_positions'] != [21055]
                or stage._entries[-1]['proposed_rank_source_counts'] != [1, 0, 0, 0]):
            raise ValueError('All original full-source worst-shape capacity cases and actual singleton tail are required')
        common = digest(dict(plan=capacity_plan, bindings=bindings(stage), mode=stage.mode))
    except Exception as exc:
        error = type(exc).__name__ + ': ' + str(exc)
    _agree(error, common, stage.group, stage.world)
    recovery = prove_next_update(stage, directory, maximum_bytes=maximum_bytes, minimum_free_bytes=minimum_free_bytes)
    while stage.next_entry is not None:
        stage.next_update()
    receipt = stage.commit(directory, maximum_bytes=maximum_bytes, minimum_free_bytes=minimum_free_bytes)
    error = common = saved = None
    try:
        saved, actual_receipt, pointer = read_latest(directory)
        if (receipt != actual_receipt or saved['common']['cursor']['completed'] != len(stage._entries)
                or saved['private_states'][stage.rank]['metrics'] != stage.metrics
                or tensor_digest(saved['common']['selected_weights'].items()) !=
                tensor_digest((n, p) for rows in named_groups(stage.view).values() for n, p in rows)):
            raise ValueError('Every complete actual capacity source metric and committed selected value must match')
        common = digest(dict(receipt=receipt, pointer=pointer))
    except Exception as exc:
        error = type(exc).__name__ + ': ' + str(exc)
    _agree(error, common, stage.group, stage.world)
    error = None
    other = cold = None
    before_RNG = private_rng_sha256(capture_rng(stage.device))
    try:
        other = independent_native()
        if private_rng_sha256(capture_rng(stage.device)) != before_RNG:
            raise ValueError('The independent native loader must preserve every actual private generator')
    except Exception as exc:
        error = type(exc).__name__ + ': ' + str(exc)
    _agree(error, None, stage.group, stage.world)
    error = None
    try:
        cold = prove_cold(stage.view, other, cold_inputs, mode='mechanics' if stage.mode == 'mechanics' else 'capacity',
            precision=stage._options['precision'], expected_fixed_sha256=fixed_digest(stage.view.actor.base))
    except Exception as exc:
        error = type(exc).__name__ + ': ' + str(exc)
    _agree(error, None, stage.group, stage.world)
    stage._boundary()
    groups = named_groups(stage.view)
    result = dict(format=FORMAT, complete=True, passed=True, rank=stage.rank, world=stage.world,
        mode=stage.mode, actual_device_type=stage.device.type, actual_device_index=stage.device.index,
        actual_actor_parameters=stage.view.actor.counts()['actual_unique'],
        actual_native_parameters=stage.view.actor.base.counts()['total_unique'], original_source_bindings=bindings(stage),
        capacity_plan_sha256=digest(capacity_plan), actual_complete_capacity_entries=stage.completed,
        actual_capacity_source_presentations=sum(len(e['original_local_positions']) for e in stage._entries),
        actual_complete_source_families=sorted({e['family'] for e in stage._entries}),
        actual_physical_optimizer_calls=len(stage._entries) + 1,
        actual_selected_Adam_leaves=len(stage.optimizer.state),
        actual_selected_parameter_leaves=sum(len(rows) for rows in groups.values()),
        actual_final_sparse_group_clocks=optimizer_clocks(stage.view, stage.optimizer),
        recovery=recovery, independent_native_cold=cold, final_actual_shared_checkpoint_sha256=receipt['checkpoint_sha256'],
        all_original_worst_cases_and_real_empty_tail_observed=True,
        historical_different_word_and_prefix_numeric_failures_not_waived=True,
        full_size_GPU_capacity_passed=stage.mode == 'capacity', production_updates=0,
        final_complete_native_export_and_full_assessments_still_required=True, goal_achieved=False)
    del other, saved
    if stage.device.type == 'cuda':
        torch.cuda.empty_cache()
    return result


def require_full_capacity(proofs, *, expected_bindings, expected_capacity_plan_sha256):
    if not isinstance(proofs, list) or len(proofs) != 4:
        raise ValueError('All four actual complete GPUcapacity reports are required')
    common_checkpoint = set()
    for rank, proof in enumerate(proofs):
        if (proof.get('format') != FORMAT or proof.get('complete') is not True or proof.get('passed') is not True
                or proof.get('rank') != rank or proof.get('world') != 4 or proof.get('mode') != 'capacity'
                or proof.get('actual_device_type') != 'cuda' or proof.get('actual_device_index') != rank
                or proof.get('actual_actor_parameters') != 999706442 or proof.get('actual_native_parameters') != 984399690
                or proof.get('original_source_bindings') != expected_bindings
                or proof.get('capacity_plan_sha256') != expected_capacity_plan_sha256
                or proof.get('actual_complete_capacity_entries') != 10 or proof.get('actual_capacity_source_presentations') != 510
                or proof.get('actual_physical_optimizer_calls') != 11
                or proof.get('actual_complete_source_families') != sorted(FAMILIES)
                or proof.get('actual_selected_Adam_leaves') != 130 or proof.get('actual_selected_parameter_leaves') != 130
                or proof.get('actual_final_sparse_group_clocks') != dict(vision=10, pointing=1, word=2)
                or proof.get('full_size_GPU_capacity_passed') is not True
                or proof.get('historical_different_word_and_prefix_numeric_failures_not_waived') is not True
                or proof.get('all_original_worst_cases_and_real_empty_tail_observed') is not True
                or proof.get('production_updates') != 0):
            raise ValueError('Every full-size actual physical GPU/source/budget/Adam capacity report must match')
        recovery, cold = proof['recovery'], proof['independent_native_cold']
        if (recovery.get('exact_next_source_weights_Adam_metrics_and_private_RNG') is not True
                or recovery.get('valid_foreign_rank_RNG_rejected_before_mutation') is not True
                or recovery.get('invalid_source_active_moment_and_nonfinite_saved_values_rejected_before_mutation') is not True
                or cold.get('mode') != 'capacity' or cold.get('mechanics_only') is not False
                or cold.get('actual_text_context_tokens') != 4096 or cold.get('actual_photo_side') != 448
                or cold.get('actual_word_shape') != [64, 448] or cold.get('actual_word_readout_dtype') != 'torch.float32'
                or cold.get('every_original_fixed_and_selected_native_value_equal') is not True
                or cold.get('actual_text_photo_logits_and_word_readout_bit_exact') is not True
                or cold.get('independent_native_storage_without_aliasing') is not True):
            raise ValueError('Actual full4096text/448photo/word cold and private recovery must pass on every GPU')
        common_checkpoint.add(proof['final_actual_shared_checkpoint_sha256'])
    if len(common_checkpoint) != 1:
        raise ValueError('All four GPUcapacity reports must bind the same verified common selected checkpoint')
    return dict(complete=True, passed=True, actual_original_GPUs=4,
        source_bindings=copy.deepcopy(expected_bindings), capacity_plan_sha256=expected_capacity_plan_sha256,
        all_actual_full_size_capacity_recovery_and_native_cold_gates_passed=True,
        final_complete_native_export_and_full_assessments_still_required=True,
        source_positive_targets_or_original_canonical_scope_not_reduced=True, production_updates=0, goal_achieved=False)


def run_training(stage, directory, journal, *, capacity_proofs, capacity_plan_sha256,
                 every, maximum_bytes, minimum_free_bytes, maximum_journal_bytes, preflight, progress):
    error = common = None
    try:
        if type(stage) is not JointVisualStage or stage.mode != 'train':
            raise ValueError('Only a complete actual4733step train stage may enter production')
        admission = require_full_capacity(capacity_proofs, expected_bindings=bindings(stage),
            expected_capacity_plan_sha256=capacity_plan_sha256)
        common = digest(admission)
    except Exception as exc:
        error = type(exc).__name__ + ': ' + str(exc)
    _agree(error, common, stage.group, stage.world)
    return train_loop(stage, directory, journal, every=every, maximum_bytes=maximum_bytes,
        minimum_free_bytes=minimum_free_bytes, maximum_journal_bytes=maximum_journal_bytes,
        preflight=preflight, progress=progress)
