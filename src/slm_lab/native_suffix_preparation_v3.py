"""Close an indexed runtime around the unchanged complete original recipe."""
import copy
from pathlib import Path

from .native_suffix_preparation_v2 import read, sha, merge
from .native_suffix_owned_runtime_contract_v3 import (PROTOCOL, PRESERVED, INDEXED_HISTORY,
    storage_budget, require_complete_preparation)


def assemble(root, *, owner, original_path, source_files, runtime_sources, driver):
    root = Path(root).resolve(); original_path = Path(original_path).resolve()
    original = read(original_path)
    files, native = {}, {}
    merge(files, original['source_files_sha256']); merge(files, source_files)
    merge(files, {str(original_path): sha(original_path)}); merge(files, runtime_sources)
    merge(native, original['native_source_files_sha256']); merge(native, runtime_sources)
    prepared = {k: copy.deepcopy(original[k]) for k in PRESERVED}
    budget = storage_budget(maximum_state_bytes=driver.MAX_STATE, maximum_native_bytes=original['maximum_native_bytes'],
        maximum_assessment_bytes=driver.MAX_ASSESSMENTS, maximum_event_raw_bytes=driver.MAX_EVENT_RAW,
        reserve_bytes=driver.RESERVE, indexed_history=INDEXED_HISTORY)
    prepared.update(format=PROTOCOL, complete=True, owner=owner,
        original_full_scope_preparation=dict(path=str(original_path), sha256=sha(original_path)),
        source_files_sha256=files, native_source_files_sha256=native,
        additional_runtime_sources=copy.deepcopy(runtime_sources), indexed_history=copy.deepcopy(INDEXED_HISTORY),
        indexed_storage_budget=budget, required_free_bytes_before_launch=budget['required_free_bytes'],
        original_canonical_visual_mixed_reasoning_scopes_preserved=True,
        GPU_work_started=False, models_constructed=0, neural_forwards=0, GPU_contexts=0, GPU_leases=0, production_updates=0,
        original_legacy_storage_estimate_preserved_as_historical_evidence=True,
        final_11172_update_state_and_record_bounds_require_actual_GPU_checks=True,
        later_fresh_normal_nice0_owned_launch_required=True, candidate_promoted=False, goal_achieved=False)
    require_complete_preparation(root, prepared, options=driver.STAGE_OPTIONS, prefill=driver.NATIVE_PREFILL,
        maximum_state_bytes=driver.MAX_STATE, maximum_event_raw_bytes=driver.MAX_EVENT_RAW,
        maximum_assessment_bytes=driver.MAX_ASSESSMENTS, reserve_bytes=driver.RESERVE, indexed_history=INDEXED_HISTORY)
    return prepared
