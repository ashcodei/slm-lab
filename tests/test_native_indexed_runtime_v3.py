"""Full-population metadata fixtures for indexed runtime assembly and funding.

The opaque fixture weights and declared CPU receipts cannot train a model.
These cases verify preservation/rejection and source-only imports, not actual
GPU capacity, completed neural training or learned model quality.
"""
import copy
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

import train_native_code_suffix_aligned_GPU_v2 as old_driver
import train_native_code_suffix_aligned_GPU_v3 as driver
from slm_lab.native_suffix_owned_runtime_contract_v3 import (INDEXED_HISTORY, RUNTIME_FILES,
    require_complete_preparation, storage_budget, history_budget)
from slm_lab.native_suffix_preparation_v3 import assemble
from slm_lab.native_suffix_preparation_v2 import assemble as old_assemble, completed_stages, sha, read
from test_native_suffix_preparation_v2 import fixture as old_fixture, write

ROOT = Path(__file__).resolve().parents[1]


def fixture(root):
    original, capacity, files, _ = old_fixture(root)
    legacy = old_assemble(root, original, capacity, completed_stages(root, {}),
        owner=dict(pid=99999999, ticks='1'), source_files=files, driver=old_driver)
    original_path = root / 'reports/native-code-suffix-aligned-preparation-CPU-v2/ready.json'
    write(original_path, legacy)
    paths = [Path('src/slm_lab') / name for name in RUNTIME_FILES]
    paths.append(Path('scripts/train_native_code_suffix_aligned_GPU_v3.py'))
    runtime = {}
    for name in paths:
        path = root / name; path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, path); runtime[str(path)] = sha(path)
    proof = root / 'reports/native-indexed-training-v2-proof-v1'
    write(proof / 'plan.json', dict(fixture_metadata_only=True))
    write(proof / 'complete.json', dict(pytest_exit_code=0, all_actual_capacity_and_runtime_source_bytes_unchanged=True,
        fixture_metadata_only=True))
    write(proof / 'pytest/test_actual_four_replica_compl0/complete.json', dict(actual_physical_small_CPU_AdamW_calls=56,
        exact_uninterrupted_and_post_Adam_failure_recovered_actor_Adam_private_RNG_and_complete_history=True,
        actual_independent_complete_native_exports=[dict(complete_text_photo_word_cold_outputs_exact=True) for _ in range(4)],
        fixture_metadata_only=True))
    sources = {str(p): sha(p) for p in proof.rglob('*.json')}
    return original_path, legacy, runtime, sources


def build(root):
    path, legacy, runtime, sources = fixture(root)
    result = assemble(root, owner=dict(pid=99999998, ticks='1'), original_path=path,
        source_files=sources, runtime_sources=runtime, driver=driver)
    return result, legacy


def require(root, prepared):
    return require_complete_preparation(root, prepared, options=driver.STAGE_OPTIONS, prefill=driver.NATIVE_PREFILL,
        maximum_state_bytes=driver.MAX_STATE, maximum_event_raw_bytes=driver.MAX_EVENT_RAW,
        maximum_assessment_bytes=driver.MAX_ASSESSMENTS, reserve_bytes=driver.RESERVE, indexed_history=INDEXED_HISTORY)


def test_indexed_recipe_preserves_every_original_source_scope_and_funds_all_retained_attempts(tmp_path):
    new, legacy = build(tmp_path)
    assert require(tmp_path, new)
    assert new['full_native_budget'] == legacy['full_native_budget']
    assert all(new['native_source_files_sha256'].get(p) == h for p, h in legacy['native_source_files_sha256'].items())
    assert all(new['source_files_sha256'].get(p) == h for p, h in legacy['source_files_sha256'].items())
    assert new['initializer'] == legacy['initializer'] and new['cached_baseline'] == legacy['cached_baseline']
    parts = new['indexed_storage_budget']['parts']
    assert parts['all_four_capacity_attempts'] == 8 * 1024**3
    assert parts['both_complete_training_attempts'] == 16 * 1024**3
    assert parts['complete_state_slots'] == 9 * 1024**3
    assert parts['unchanged_other_writer_floor'] == 40 * 1024**3
    assert new['required_free_bytes_before_launch'] == sum(parts.values())
    assert new['required_free_bytes_before_launch'] < legacy['required_free_bytes_before_launch']
    assert new['production_updates'] == new['GPU_leases'] == new['models_constructed'] == 0


@pytest.mark.parametrize('kind', ['partial_targets', 'changed_initializer', 'drop_original_native', 'drop_runtime',
    'underfund', 'alter_bounds', 'wrong_CPU_calls', 'missing_cold', 'changed_same_mtime_source'])
def test_rehashed_partial_sources_bounds_and_incomplete_actual_proof_cannot_admit_runtime(tmp_path, kind):
    new, legacy = build(tmp_path)
    if kind == 'partial_targets': new['full_native_budget']['original_response_EOS_targets'] -= 1
    elif kind == 'changed_initializer': new['initializer']['checkpoint_sha256'] = 'e' * 64
    elif kind == 'drop_original_native': del new['native_source_files_sha256'][next(iter(legacy['native_source_files_sha256']))]
    elif kind == 'drop_runtime': del new['additional_runtime_sources'][str(tmp_path / 'src/slm_lab/native_suffix_training_controller_v2.py')]
    elif kind == 'underfund': new['required_free_bytes_before_launch'] -= 1
    elif kind == 'alter_bounds': new['indexed_history']['maximum_expanded_record_bytes'] //= 2
    elif kind in ['wrong_CPU_calls', 'missing_cold']:
        path = tmp_path / 'reports/native-indexed-training-v2-proof-v1/pytest/test_actual_four_replica_compl0/complete.json'
        row = read(path)
        if kind == 'wrong_CPU_calls': row['actual_physical_small_CPU_AdamW_calls'] -= 1
        else: row['actual_independent_complete_native_exports'][2]['complete_text_photo_word_cold_outputs_exact'] = False
        write(path, row); new['source_files_sha256'][str(path)] = sha(path)
    else:
        path = Path(next(iter(legacy['native_source_files_sha256']))); stat = path.stat()
        path.write_bytes(b'changed source'); os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    with pytest.raises(ValueError): require(tmp_path, new)


def test_logical_full_history_quota_is_separate_from_real_physical_funding_and_never_shortens_schedule():
    history = history_budget(INDEXED_HISTORY, 'train', 11172)
    assert history['maximum_logical_bytes'] == 11172 * 64 * 1024**2 + 16 * 1024**2
    assert history['maximum_physical_bytes'] == 8 * 1024**3
    assert history['minimum_free_bytes'] == 40 * 1024**3
    with pytest.raises(ValueError): history_budget(INDEXED_HISTORY, 'train', 11171)
    with pytest.raises(ValueError): history_budget(dict(INDEXED_HISTORY, minimum_free_bytes=0), 'capacity', 132)
    with pytest.raises(ValueError): storage_budget(maximum_state_bytes=driver.MAX_STATE, maximum_native_bytes=4 * 1024**3,
        maximum_assessment_bytes=driver.MAX_ASSESSMENTS, maximum_event_raw_bytes=4096 * 11172 - 1,
        reserve_bytes=driver.RESERVE, indexed_history=INDEXED_HISTORY)


def test_indexed_driver_and_preparation_imports_do_not_construct_a_neural_runtime_or_GPU_job():
    script = ("import runpy,sys;runpy.run_path('scripts/train_native_code_suffix_aligned_GPU_v3.py');"
        "runpy.run_path('scripts/prepare_native_code_suffix_aligned_CPU_v3.py');"
        "import slm_lab.native_suffix_preparation_v3;assert 'torch' not in sys.modules;print('source-only imports verified')")
    result = subprocess.run([sys.executable, '-c', script], cwd=ROOT, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0 and result.stdout.strip() == 'source-only imports verified'
