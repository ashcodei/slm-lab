"""Run the complete joint epoch after actual capacity and native GPU cold.

Ownership, physical funding and completed initializer admission are supplied
by the owned launcher. The full path retains every original5362step source.
Explicit small CPU mechanics exercise the same loop without GPU admission.
"""
import copy
import os
import time

import torch

from .codenet_checkpoint_v1 import capture_rng
from .complete_joint_capacity_run_v2 import FORMAT as CAPACITY
from .complete_joint_indexed_training_v4 import CompleteJointIndexedTrainingV4
from .joint_native_GPU_cold_v1 import FORMAT as COLD
from .joint_source_checked_training_stage_v3 import JointSourceCheckedStageV3
from .parallel_function_repair_stage_v1 import value_digest
from .shared_joint_checkpoint_v2 import agree
from .whole_human_sft_evidence_v1 import tensor_digest
from .joint_indexed_durable_state_v3 import read_latest

FORMAT = 'own-complete-original-joint-training-loop-after-full-capacity-and-native-cold-v1'


def require_capacity(stage, capacity_reports, cold_reports, capacity_plan):
    if (type(capacity_reports) is not list or len(capacity_reports) != 4 or type(cold_reports) is not list or len(cold_reports) != 4
            or capacity_plan.get('actual_original_joint_steps') != 5362 or capacity_plan.get('actual_complete_capacity_cases') != 47
            or capacity_plan.get('source_identity_sha256') != stage.sources['source_admission_sha256']):
        raise ValueError('All four actual full-size capacity and independent native GPUcold reports are required')
    selected = stage.view.named_selected(); clocks = {n: 0 for n, _ in selected}
    for case in capacity_plan['cases']:
        names = {n for objective in case['checkpoint_entry']['objectives'] for n in stage.view.task_names(objective['task'])}
        for name in names:
            clocks[name] += 1
    source_bindings = {k: v for k, v in stage.sources.items() if k != 'complete_schedule_sha256'}
    common = set(); native = set(); private = set()
    for rank, (report, cold) in enumerate(zip(capacity_reports, cold_reports)):
        if (report.get('format') != CAPACITY or report.get('complete') is not True or report.get('rank') != rank or report.get('world') != 4
                or report.get('actual_device_type') != 'cuda' or report.get('actual_device_index') != rank
                or report.get('actual_research_parameters') != 999706442 or report.get('actual_native_parameters') != 984399690
                or report.get('capacity_plan_sha256') != value_digest(capacity_plan)
                or {k: v for k, v in report.get('source_bindings', {}).items() if k != 'complete_schedule_sha256'} != source_bindings
                or report.get('actual_complete_capacity_cases') != 47 or report.get('actual_physical_AdamW_calls') != 48
                or report.get('actual_selected_parameter_leaves') != len(selected) or report.get('actual_Adam_states') != len(selected)
                or report.get('actual_final_sparse_parameter_clocks') != clocks
                or report.get('full_size_GPU_capacity_passed') is not True or report.get('production_updates') != 0
                or any(report.get(k) is not True for k in ['exact_next_source_values_Adam_private_RNG_and_full_metric_replay',
                    'foreign_actual_rank_private_RNG_rejected_before_any_mutation',
                    'all_original_visual_cases_corrective_batches_and_real_empty_tasks_preserved',
                    'every_actual_complete_capacity_record_read_and_final_state_restored'])):
            raise ValueError('Every actual full-size original-source capacity,recovery,private RNG and sparse Adam report must match')
        if (cold.get('format') != COLD or cold.get('complete') is not True or cold.get('physical_GPU') != rank
                or cold.get('actual_device_type') != 'cuda' or cold.get('actual_native_values') != 984399690
                or cold.get('sources') != report['source_bindings'] or cold.get('full_size_GPU_native_cold_passed') is not True
                or cold.get('actual_source_selected_checkpoint_sha256') != report.get('actual_final_checkpoint_sha256')
                or any(cold.get(k) is not True for k in ['all_original_native_values_and_text_photo_FP32word_cold_outputs_bit_exact',
                    'original_native_critic_and_current_joint_visual_word_values_preserved', 'independent_source_and_candidate_storage',
                    'source_native_derived_without_exported_candidate_weights', 'private_RNG_and_both_model_values_and_metadata_unchanged'])):
            raise ValueError('Every independent full-size native GPUcold must derive from the actual committed joint capacity state')
        observations = cold['actual_observations']
        if (observations['text']['actual_output_shape'][1] != 4096
                or observations['photo']['actual_original_photo_shape'][-2:] != [448, 448]
                or observations['word']['actual_output_shape'][1] != 140 or observations['word']['actual_output_dtype'] != 'torch.float32'
                or any(observations[name].get('bit_exact') is not True for name in ['text', 'photo', 'word'])):
            raise ValueError('The full original4096text and140frame FP32word cold observations cannot be reduced')
        common.add(report['actual_final_checkpoint_sha256'])
        native.add((cold['actual_checkpoint_sha256'], cold['actual_native_parameters_sha256']))
        private.add(report['final_actual_state']['private_RNG_sha256'])
    if len(common) != 1 or len(native) != 1 or len(private) != 4:
        raise ValueError('Every actual GPU must agree on the same selected/native values while retaining its own private generators')
    return dict(actual_four_L40_capacity_and_independent_native_cold_passed=True,
        capacity_plan_sha256=value_digest(capacity_plan), actual_common_capacity_checkpoint_sha256=next(iter(common)),
        actual_common_native_capacity_checkpoint_sha256=next(iter(native))[0],
        complete_training_export_and_all_original_assessments_still_required=True)


def run(stage, directory, *, capacity_reports, cold_reports, capacity_plan, checkpoint_every,
        maximum_checkpoint_bytes, minimum_free_bytes, preflight, progress, allow_mechanics=False, compute_next=None):
    error = identity = admission = None
    try:
        count = stage.view.actor.counts()['actual_unique']
        full = (type(stage) is CompleteJointIndexedTrainingV4 and stage.device.type == 'cuda'
            and count == 999706442 and stage.world == 4 and len(stage.entries) == 5362)
        mechanics = (allow_mechanics is True and type(stage) is JointSourceCheckedStageV3
            and stage.device.type == 'cpu' and count < 10**7 and callable(compute_next))
        if (not full and not mechanics or stage.completed or stage.failed or type(checkpoint_every) is not int or checkpoint_every < 1
                or minimum_free_bytes < 40 * 1024**3 or not callable(preflight) or not callable(progress)
                or full and (compute_next is not None or os.nice(0) != 0 or os.sched_getscheduler(0) != os.SCHED_OTHER
                    or torch.cuda.get_device_name(stage.device) != 'NVIDIA L40' or stage.device.index != stage.rank)):
            raise ValueError('A fresh full original joint GPUstage or an explicit own-small CPUloop control is required')
        preflight()
        admission = require_capacity(stage, capacity_reports, cold_reports, capacity_plan) if full else dict(explicit_small_CPU_mechanics=True, GPU_training_admitted=False)
        identity = value_digest(dict(sources=stage.sources, admission=admission, checkpoint_every=checkpoint_every,
            maximum_checkpoint_bytes=maximum_checkpoint_bytes, minimum_free_bytes=minimum_free_bytes))
    except Exception as exc:
        error = repr(exc)
    agree(error, identity, stage.control_group, stage.world)
    feed = stage.train_next if full else compute_next
    stage.open_history(); started = time.monotonic(); contexts = targets = ctc = 0; final_receipt = None
    while stage.completed < len(stage.entries):
        error = None
        try:
            preflight()
        except Exception as exc:
            error = repr(exc)
        agree(error, None, stage.control_group, stage.world)
        metric = feed()
        entry = stage.entries[stage.completed - 1]
        contexts += sum(o['complete_context_tokens'] for o in entry['objectives'])
        targets += sum(o['response_EOS_targets'] for o in entry['objectives'])
        ctc += sum(o['word_UTF8_bytes'] for o in entry['objectives'])
        error = None
        try:
            progress(dict(actual_completed_updates=stage.completed, planned_updates=len(stage.entries),
                actual_complete_native_context_tokens=contexts, actual_native_response_EOS_targets=targets,
                actual_separate_CTC_UTF8_bytes=ctc, elapsed_seconds=time.monotonic() - started,
                actual_last_complete_metric_sha256=value_digest(metric), full_size_GPU_training=full, goal_achieved=False))
        except Exception as exc:
            error = repr(exc)
        agree(error, None, stage.control_group, stage.world)
        if stage.completed % checkpoint_every == 0 or stage.completed == len(stage.entries):
            final_receipt = stage.commit(directory, maximum_bytes=maximum_checkpoint_bytes, minimum_free_bytes=minimum_free_bytes)
    error = common = None
    try:
        shared, actual_receipt, pointer = read_latest(directory)
        if (actual_receipt != final_receipt or shared['common']['cursor']['completed'] != len(stage.entries)
                or tensor_digest(stage.view.named_selected()) != tensor_digest(shared['common']['selected_weights'].items())
                or value_digest(stage.optimizer.state_dict()) != value_digest(shared['common']['optimizer'])
                or value_digest(capture_rng(stage.device)) != shared['private_states'][stage.rank]['private_RNG_sha256']
                or full and (contexts, targets, ctc) != (91100747, 34500161, 644063)):
            raise ValueError('Every complete original source budget and final actual selected/Adam/private RNG state must match the durable endpoint')
        stage.guard.fixed.check(stage.view, stage.optimizer, full=True); stage.reference.assert_bound(full=True); stage.source_guard(); preflight()
        common = value_digest(actual_receipt)
    except Exception as exc:
        error = repr(exc)
    agree(error, common, stage.control_group, stage.world)
    return dict(format=FORMAT, complete=True, mode='train' if full else 'mechanics', rank=stage.rank, world=stage.world,
        actual_completed_updates=stage.completed, planned_updates=len(stage.entries), actual_counts=stage.view.actor.counts(),
        actual_complete_native_context_tokens=contexts, actual_native_response_EOS_targets=targets, actual_separate_CTC_UTF8_bytes=ctc,
        actual_final_checkpoint_sha256=final_receipt['checkpoint_sha256'], actual_final_pointer_sha256=pointer,
        actual_admission=copy.deepcopy(admission), actual_final_selected_parameters_sha256=tensor_digest(stage.view.named_selected()),
        actual_final_Adam_sha256=value_digest(stage.optimizer.state_dict()), every_original_source_prefix_and_final_private_state_durable=True,
        actual_full_size_GPU_joint_updates=stage.completed if full else 0, new_native_export_and_all_original_assessments_required=True,
        automatic_retry=False, candidate_promoted=False, goal_achieved=False)
