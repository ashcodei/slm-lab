"""Additional runtime provenance and rejection of unsupported indexed training."""
from pathlib import Path

import pytest

from slm_lab.corpus import file_hash
from slm_lab import native_training_runtime_bindings_v1 as runtime_code
from slm_lab.native_training_runtime_bindings_v1 import RuntimeBindings
from slm_lab.native_suffix_batched_history_stage_v5 import NativeSuffixBatchedHistoryStageV5
from slm_lab.native_suffix_training_controller_v2 import run_training
from slm_lab.native_suffix_standalone_export_v2 import export


def runtime(path):
    required = Path(runtime_code.__file__).resolve()
    return {str(required): file_hash(required), str(path): file_hash(path)}


def test_later_runtime_bindings_extend_without_replacing_any_original_neural_source(tmp_path):
    source = tmp_path / 'controller.py';source.write_text('declared runtime source\n')
    files = runtime(source)
    guard = RuntimeBindings(files, required=[source])
    policy = {'original_neural_input_sha256': 'a' * 64}
    combined = guard.combine(policy)
    assert combined['original_neural_input_sha256'] == policy['original_neural_input_sha256']
    assert all(combined[k] == v for k, v in files.items())
    assert policy == {'original_neural_input_sha256': 'a' * 64}
    with pytest.raises(ValueError): guard.combine(dict(policy, **{str(source): '0' * 64}))


def test_runtime_source_or_closed_map_changes_are_rejected(tmp_path):
    source = tmp_path / 'controller.py';source.write_text('original bytes\n')
    files = runtime(source)
    guard = RuntimeBindings(files, required=[source])
    source.write_text('changed  bytes\n')
    with pytest.raises(ValueError): guard.check()
    with pytest.raises(ValueError): RuntimeBindings(files, required=[source])
    with pytest.raises(ValueError): RuntimeBindings({str(source): file_hash(source)}, required=[source])


def test_partial_single_replica_cannot_enter_complete_indexed_training_or_create_artifacts(tmp_path):
    stage = NativeSuffixBatchedHistoryStageV5.__new__(NativeSuffixBatchedHistoryStageV5)
    stage.rank, stage.world, stage.control_group = 0, 1, None
    stage.completed, stage.failed, stage.failure_observations = 0, False, []
    with pytest.raises(ValueError):
        run_training(stage, tmp_path / 'private', tmp_path / 'history', tmp_path / 'events.gz',
            capacity_plan={}, reports=[], horizon_reports=[], readbacks=[], runtime_sources={},
            checkpoint_every=1, maximum_state_bytes=32 * 1024**2, minimum_free_bytes=0,
            maximum_event_raw_bytes=65536, preflight=lambda: None, allow_mechanics=True)
    assert not list(tmp_path.iterdir()) and stage.completed == 0


def test_foreign_or_partial_export_cannot_create_a_standalone_candidate():
    with pytest.raises(ValueError):
        export(object(), object(), {}, {}, sources={}, tokenizer_sha256='0' * 64,
            working_budget_bytes=256 * 1024**2, runtime_sources={}, allow_mechanics=True)
