"""Indexed continuation admission without weakening original source scopes.

The unchanged version2 preparation proves the complete prior populations.
Its historical storage estimate is not the indexed runner's storage recipe.
Both complete new physical quotas and the original reserve must be available;
actual record, state and full-horizon bounds still require GPU measurement.
This source-only contract imports no neural runtime and grants no GPU lease.
"""
from pathlib import Path

from .native_suffix_owned_runtime_contract_v2 import require_complete_preparation as require_original, FULL
from .native_suffix_preparation_v2 import read, sha
from .exact_waiting_process_retirement_v1 import terminal
from .metadata_observer_recovery_v1 import ORIGINAL_OUT, read as recovery_read

PROTOCOL = 'own-native-indexed-full-source-runtime-recipe-with-metadata-recovery-v5'
INDEXED_HISTORY = dict(maximum_compact_record_bytes=16 * 1024**2,
    maximum_expanded_record_bytes=64 * 1024**2, reference_logical_headroom_bytes=16 * 1024**2,
    capacity_attempt_physical_bytes=2 * 1024**3, training_attempt_physical_bytes=8 * 1024**3,
    minimum_free_bytes=40 * 1024**3)
PRESERVED = ['exact_predecessors', 'initializer', 'cold_inputs', 'maximum_native_bytes', 'cached_baseline',
    'cached_own_reference', 'pulse_sources', 'full_native_budget', 'native_stage_options', 'native_prefill']
RUNTIME_FILES = ['native_training_runtime_bindings_v1.py', 'native_indexed_history_inventory_v3.py', 'native_suffix_training_controller_v3.py',
    'native_suffix_standalone_export_v3.py', 'native_suffix_batched_history_stage_v5.py',
    'native_suffix_indexed_stage_v4.py', 'indexed_native_history_v1.py', 'indexed_native_history_v2.py',
    'native_suffix_capacity_controller_v2.py', 'native_suffix_capacity_evidence_v2.py',
    'native_suffix_capacity_readback_v2.py', 'native_suffix_full_horizon_capacity_v2.py',
    'native_history_summary_v1.py', 'native_allocated_history_guard_v1.py',
    'native_suffix_owned_runtime_contract_v4.py', 'native_suffix_preparation_v4.py']


def history_budget(recipe, mode, records):
    if (recipe != INDEXED_HISTORY or mode not in ['capacity', 'train']
            or type(records) is not int or records < 1 or mode == 'train' and records != FULL['updates']):
        raise ValueError('The exact complete original indexed capacity or11172-update training recipe is required')
    return dict(maximum_records=records, maximum_compact_record_bytes=recipe['maximum_compact_record_bytes'],
        maximum_expanded_record_bytes=recipe['maximum_expanded_record_bytes'],
        maximum_logical_bytes=records * recipe['maximum_expanded_record_bytes'] + recipe['reference_logical_headroom_bytes'],
        maximum_physical_bytes=recipe['capacity_attempt_physical_bytes' if mode == 'capacity' else 'training_attempt_physical_bytes'],
        minimum_free_bytes=recipe['minimum_free_bytes'])


def storage_budget(*, maximum_state_bytes, maximum_native_bytes, maximum_assessment_bytes,
                   maximum_event_raw_bytes, reserve_bytes, indexed_history):
    values = [maximum_state_bytes, maximum_native_bytes, maximum_assessment_bytes, maximum_event_raw_bytes, reserve_bytes]
    if (any(type(v) is not int or v < 1 for v in values) or indexed_history != INDEXED_HISTORY
            or maximum_state_bytes < 1024**3 or maximum_native_bytes < 4 * 1024**3
            or maximum_assessment_bytes < 4 * 1024**3 or maximum_event_raw_bytes < 4096 * FULL['updates']
            or reserve_bytes != indexed_history['minimum_free_bytes']):
        raise ValueError('Complete actor/Adam, native, assessment, timing and indexed physical budgets must remain funded')
    # Two profiles retain two capacity attempts and one complete training
    # attempt each. Nine state slots include the live atomic replacement.
    parts = dict(unchanged_other_writer_floor=reserve_bytes, complete_state_slots=9 * maximum_state_bytes,
        complete_native_candidates=2 * maximum_native_bytes, all_original_assessments=2 * maximum_assessment_bytes,
        all_four_capacity_attempts=4 * indexed_history['capacity_attempt_physical_bytes'],
        both_complete_training_attempts=2 * indexed_history['training_attempt_physical_bytes'],
        both_complete_controller_events=2 * maximum_event_raw_bytes, complete_metadata_headroom=512 * 1024**2)
    return dict(parts=parts, required_free_bytes=sum(parts.values()), additional_resume_attempts_require_new_funding=True,
        lossless_logical_expanded_quota_does_not_allocate_physical_blocks=True,
        compression_ratio_not_assumed_for_next_record_or_failure_reservation=True,
        whole_training_completion_not_inferred_from_quota_or_capacity=True)


def require_complete_preparation(root, prepared, *, options, prefill, maximum_state_bytes,
                                 maximum_event_raw_bytes, maximum_assessment_bytes, reserve_bytes, indexed_history):
    root = Path(root).resolve()
    old_path = ORIGINAL_OUT / 'ready.json'
    recovery_plan = ORIGINAL_OUT / 'plan.json'
    recovered = recovery_read(recovery_plan)
    if (not terminal(recovered['owner']) or recovered.get('versioned_metadata_recovery', {}).get('phase') != 'original'
            or recovered.get('versioned_metadata_recovery', {}).get('all_original_scopes_preserved') is not True):
        raise ValueError('The exact recovered original full source owner must finish successfully without relabelling its retired predecessor')
    original_spec = prepared.get('original_full_scope_preparation', {})
    if (prepared.get('format') != PROTOCOL or prepared.get('complete') is not True
            or Path(original_spec.get('path', '')).resolve() != old_path
            or not old_path.is_file() or old_path.is_symlink() or sha(old_path) != original_spec.get('sha256')):
        raise ValueError('The exact unchanged complete original full-scope preparation must remain independently bound')
    original = read(old_path)
    if original.get('owner') != recovered['owner'] or prepared.get('source_files_sha256', {}).get(str(recovery_plan)) != sha(recovery_plan):
        raise ValueError('The actual recovered full-source owner and its immutable plan must remain bound')
    require_original(root, original, options=options, prefill=prefill, maximum_state_bytes=12 * 1024**3,
        maximum_journal_raw_bytes=32 * 1024**3, maximum_assessment_bytes=4 * 1024**3, reserve_bytes=40 * 1024**3)
    if (any(prepared.get(k) != original.get(k) for k in PRESERVED)
            or prepared.get('native_stage_options') != options or prepared.get('native_prefill') != prefill
            or prepared.get('indexed_history') != indexed_history or indexed_history != INDEXED_HISTORY
            or prepared.get('GPU_work_started') is not False or prepared.get('production_updates') != 0
            or prepared.get('original_canonical_visual_mixed_reasoning_scopes_preserved') is not True):
        raise ValueError('No original source, epoch, CE target, action horizon, fixed initializer or assessment may change')
    budget = storage_budget(maximum_state_bytes=maximum_state_bytes, maximum_native_bytes=prepared['maximum_native_bytes'],
        maximum_assessment_bytes=maximum_assessment_bytes, maximum_event_raw_bytes=maximum_event_raw_bytes,
        reserve_bytes=reserve_bytes, indexed_history=indexed_history)
    if prepared.get('indexed_storage_budget') != budget or prepared.get('required_free_bytes_before_launch') != budget['required_free_bytes']:
        raise ValueError('All complete indexed physical allocations, atomic generations and original reserve must be explicitly funded')
    files = prepared.get('source_files_sha256', {})
    native = prepared.get('native_source_files_sha256', {})
    runtime = prepared.get('additional_runtime_sources', {})
    if (not files or not native or not runtime or files.get(str(old_path)) != original_spec['sha256']
            or any(files.get(p) != h for p, h in original['source_files_sha256'].items())
            or any(native.get(p) != h for p, h in original['native_source_files_sha256'].items())
            or any(files.get(p) != h for p, h in native.items())
            or any(files.get(p) != h or native.get(p) != h for p, h in runtime.items())):
        raise ValueError('Every original source and every added policy/controller/export byte must remain in the closed recipe')
    for name in RUNTIME_FILES:
        path = root / 'src/slm_lab' / name
        if runtime.get(str(path)) != sha(path):
            raise ValueError('The actual indexed implementation must belong to both policy and runtime bindings')
    for name in ['scripts/recover_native_recipe_observers_CPU_v1.py', 'scripts/train_native_code_suffix_aligned_GPU_v5.py',
            'src/slm_lab/native_suffix_owned_runtime_contract_v5.py', 'src/slm_lab/metadata_observer_recovery_v1.py']:
        path = root / name
        if runtime.get(str(path)) != sha(path):raise ValueError('Every actual source-recovery adapter and unchanged-driver selector must remain runtime-bound')
    driver = root / 'scripts/train_native_code_suffix_aligned_GPU_v4.py'
    if runtime.get(str(driver)) != sha(driver): raise ValueError('Only the actual indexed owned driver may use this recipe')
    proof_root = root / 'reports/native-indexed-training-v3-proof-v1'
    proof_paths = [proof_root / 'plan.json', proof_root / 'complete.json',
        proof_root / 'pytest/test_actual_four_replica_compl0/complete.json']
    if any(not p.is_file() or p.is_symlink() or files.get(str(p)) != sha(p) for p in proof_paths):
        raise ValueError('The completed actual four-replica training/recovery/export proof must remain source-bound')
    proof_plan, wrapper, proof = [read(p) for p in proof_paths]
    if not terminal(proof_plan['owner']):
        raise ValueError('The exact actual four-replica proof owner must finish before runtime preparation')
    for name in ['native_suffix_training_controller_v3.py', 'native_suffix_standalone_export_v3.py',
            'native_indexed_history_inventory_v3.py']:
        path = root / 'src/slm_lab' / name
        if proof_plan.get('source_files_sha256', {}).get(str(path)) != runtime[str(path)]:
            raise ValueError('The completed proof must bind the actual corrected training/export/consumed-failure code')
    if (wrapper.get('pytest_exit_code') != 0 or wrapper.get('all_actual_capacity_and_runtime_source_bytes_unchanged') is not True
            or proof.get('actual_physical_small_CPU_AdamW_calls') != 56
            or proof.get('exact_uninterrupted_and_post_Adam_failure_recovered_actor_Adam_private_RNG_and_complete_history') is not True
            or len(proof.get('actual_independent_complete_native_exports', [])) != 4
            or any(p.get('complete_text_photo_word_cold_outputs_exact') is not True
                for p in proof.get('actual_independent_complete_native_exports', []))):
        raise ValueError('All declared complete CPU training, post-Adam failure recovery and independent native cold checks must pass')
    size_root = root / 'reports/indexed-full-actor-state-size-CPU-v1'
    size_paths = [size_root / 'plan.json', size_root / 'complete.json']
    if any(not p.is_file() or p.is_symlink() or files.get(str(p)) != sha(p) for p in size_paths):
        raise ValueError('The actual full-actor/Adam complete receipt-population serialization measurement must remain bound')
    size_plan, measured = [read(p) for p in size_paths]
    size_script = root / 'scripts/measure_indexed_actor_checkpoint_size_CPU_v1.py'
    if (not terminal(size_plan['owner']) or runtime.get(str(size_script)) != sha(size_script)
            or size_plan.get('source_files_sha256', {}).get(str(size_script)) != runtime[str(size_script)]
            or any(files.get(p) != h for p, h in size_plan.get('source_files_sha256', {}).items())
            or measured.get('complete') is not True or type(measured.get('actual_complete_torch_save_bytes')) is not int
            or not 0 < measured['actual_complete_torch_save_bytes'] <= maximum_state_bytes
            or measured.get('actual_original_actor_values') != 15305216
            or measured.get('actual_original_Adam_states') != 561
            or measured.get('actual_oversized_receipt_population') != FULL['updates']
            or measured.get('persistent_checkpoint_bytes_written') != 0
            or measured.get('full_size_GPU_capacity_or_recovery_or_quality_proved') is not False
            or measured.get('actual_future_native_full_state_still_requires_runtime_measurement') is not True):
        raise ValueError('The actual full actor/Adam and all11172 oversized receipt stress cases must fit the declared complete state limit')
    for p, h in files.items():
        if sha(p) != h: raise ValueError('An exact original or added immutable runtime source changed')
    return True
