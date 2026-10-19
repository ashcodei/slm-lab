"""Physical allocations for one full original joint experiment.

Cover both immutable capacity attempts, one full training attempt, all tensor
generations, three complete natives and both full assessment populations.
Further recovery attempts need fresh funding. Logical gzip limits are never
treated as physical allocation or evidence that training has completed.
"""
GIB = 1024**3
MIB = 1024**2
RESERVE = 40 * GIB
STATE = 2 * GIB
NATIVE = 5 * GIB
ASSESSMENTS = 4 * GIB


def history(mode):
    if mode not in ['capacity', 'train']:
        raise ValueError('Only the complete original joint capacity or training history may be budgeted')
    records = 47 if mode == 'capacity' else 5362
    return dict(maximum_records=records, maximum_compact_record_bytes=4 * MIB,
        maximum_expanded_record_bytes=4 * MIB, maximum_logical_bytes=records * 4 * MIB + 16 * MIB,
        maximum_physical_bytes=(2 if mode == 'capacity' else 8) * GIB, minimum_free_bytes=RESERVE)


def budget(measurement):
    if (measurement.get('complete') is not True or measurement.get('actual_full_selected_values') != 68123713
            or measurement.get('actual_selected_parameter_leaves') != 690
            or measurement.get('actual_original_two_update_actor_Adam_states') != 561
            or measurement.get('actual_additional_size_only_visual_moment_tensors') != 258
            or measurement.get('all_original5362oversized_receipt_blocks_retained') is not True
            or type(measurement.get('actual_complete_torch_save_bytes')) is not int
            or not 0 < measurement['actual_complete_torch_save_bytes'] <= STATE
            or measurement.get('ordinary_serialized_byte_limit') != STATE or measurement.get('persistent_checkpoint_bytes_written') != 0
            or measurement.get('visual_buffers_not_learned_Adam_and_no_joint_update_or_resume_claim') is not True
            or measurement.get('full_size_GPU_capacity_recovery_training_funding_or_quality_proved') is not False):
        raise ValueError('The actual complete joint tensor/oversized-metadata serialization measurement is required')
    parts = dict(unchanged_other_writer_reserve=RESERVE,
        complete_capacity_and_training_generations_plus_one_atomic_replacement=5 * STATE,
        assembled_initial_capacity_and_final_natives=3 * NATIVE,
        all_original_initial_and_final_assessments=2 * ASSESSMENTS,
        both_complete_capacity_history_attempts=2 * history('capacity')['maximum_physical_bytes'],
        one_complete_full5362step_training_history=history('train')['maximum_physical_bytes'],
        complete_controller_events=512 * MIB, original_complete_cold_inputs=512 * MIB,
        complete_source_and_result_metadata_headroom=512 * MIB)
    return dict(parts=parts, required_free_bytes_before_launch=sum(parts.values()),
        maximum_actual_checkpoint_bytes=STATE, maximum_complete_native_bytes=NATIVE,
        maximum_each_complete_assessment_bytes=ASSESSMENTS, capacity_history=history('capacity'), training_history=history('train'),
        actual_size_measurement_bytes=measurement['actual_complete_torch_save_bytes'],
        actual_size_measurement_remaining_headroom_bytes=STATE - measurement['actual_complete_torch_save_bytes'],
        additional_resume_attempts_require_fresh_physical_funding=True,
        logical_expanded_quota_does_not_preallocate_physical_blocks=True,
        no_future_compression_ratio_assumed=True, actual_future_GPU_state_still_must_fit_enforced_limit=True,
        quotas_and_size_measurement_do_not_prove_capacity_or_training_completion=True,
        production_training_admitted=False, goal_achieved=False)
