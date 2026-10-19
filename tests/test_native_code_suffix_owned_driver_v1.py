"""Use the real native driver's optimizer/prefill with actual code executions.

The small fixture repeats one human program. It is not full-size GPU evidence,
an independently diverse coding benchmark, or a completed production stage.
"""
import copy
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

import train_native_code_suffix_aligned_GPU_v1 as driver
from slm_lab.corpus import file_hash
from slm_lab.native_suffix_train_data_v1 import NativeSuffixTrainData
from slm_lab.native_suffix_validation_stage_v2 import NativeSuffixValidationStageV2
from slm_lab.parallel_function_repair_stage_v1 import value_digest
from slm_lab.parallel_native_suffix_stage_v1 import record_semantics
from slm_lab.whole_human_sft_evidence_v1 import tensor_digest
from test_native_suffix_capacity_controller_v2 import student
from test_parallel_native_suffix_stage_v1 import ROOT, CONTROL


@pytest.mark.parametrize('profile', driver.PROFILES)
def test_real_owned_recipe_updates_actual_native_stage_and_replays_every_private_attempt(profile, tmp_path):
    data = NativeSuffixTrainData.CPU_mechanics(ROOT, CONTROL)
    model = student(data)
    options = dict(driver.STAGE_OPTIONS, CPU_mechanics_horizon=5)
    optimizer = driver.make_optimizer(model, options, driver.OPTIMIZER)
    prefill = driver.attach_native_prefill(model, purpose='CPU-mechanics', settings=driver.NATIVE_PREFILL)
    sources = {str(p): file_hash(p) for p in [Path(driver.__file__).resolve(),
        ROOT / 'src/slm_lab/native_text_chunked_prefill_v1.py', CONTROL]}
    stage = NativeSuffixValidationStageV2(model, optimizer, data, [[0], [1]], profile, options, sources, mode='mechanics')
    before_base = tensor_digest(model.base.named_parameters())
    before_actor = tensor_digest(model.parts.named_parameters())
    first = stage.update()
    saved, manifest = stage.snapshot()
    expected = stage.update()
    wanted = dict(actor=tensor_digest(model.parts.named_parameters()), Adam=value_digest(optimizer.state_dict()),
        metrics=value_digest([record_semantics(r) for r in stage.metrics]))
    stage.resume(saved, expected_manifest=manifest)
    repeated = stage.update()
    assert record_semantics(repeated) == record_semantics(expected)
    assert dict(actor=tensor_digest(model.parts.named_parameters()), Adam=value_digest(optimizer.state_dict()),
        metrics=value_digest([record_semantics(r) for r in stage.metrics])) == wanted
    assert wanted['actor'] != before_actor and tensor_digest(model.base.named_parameters()) == before_base
    assert type(optimizer) is torch.optim.AdamW and len(optimizer.state) == 17
    assert all(float(s['step']) == 2 for s in optimizer.state.values())
    assert stage.completed == 2 and prefill.last_observation['actual_logical_histories'] == 8
    assert prefill.last_observation['no_prefix_history_or_attempt_dropped']
    for row in stage.metrics:
        kernel = row['actual_rank_observations'][0]['kernel']
        assert len(kernel['trajectories']) == 8
        assert kernel['unconditional_original_CE_anchor']['local_original_response_EOS_targets'] == 68
    prefill.assert_current()
    assert not torch.cuda.is_initialized()
    (tmp_path / ('actual-owned-native-' + profile + '.json')).write_text(json.dumps(dict(
        profile=profile, actual_own_small_parameters=model.counts()['actual_unique'],
        actual_optimizer_class=type(optimizer).__name__, actual_physical_optimizer_calls=3,
        actual_complete_logical_updates=2, actual_unique_human_programs=1,
        actual_native_private_attempts_per_source=8, explicit_CPU_mechanics_horizon=5,
        actual_original_native_horizon=512, actual_original_CE_targets_per_source=68,
        original_base_unchanged=True, exact_next_actions_execution_semantics_actor_Adam_and_private_RNG_recovered=True,
        actual_first_update=first, actual_final_update=expected, prefill=prefill.last_observation,
        production_updates=0, full_size_GPU_capacity_or_quality_gain=False, goal_achieved=False), indent=2))


@pytest.mark.parametrize('case', ['idle_parent', 'live_preparer', 'incomplete_canonical', 'incomplete_native_scope', 'disk_shortfall'])
def test_launch_fails_before_any_lease_model_or_outputs_when_a_real_required_gate_fails(case, tmp_path, monkeypatch):
    from slm_lab import exact_waiting_process_retirement_v1 as retirement
    from slm_lab import joint_visual_runtime_contract_v1 as canonical
    from slm_lab import native_suffix_owned_runtime_contract_v1 as native
    prep = tmp_path / 'prep'; prep.mkdir()
    (prep / 'ready.json').write_text(json.dumps(dict(exact_predecessors={}, required_free_bytes_before_launch=100)))
    (prep / 'plan.json').write_text(json.dumps(dict(owner=dict(pid=1, ticks='100'))))
    monkeypatch.setattr(driver, 'PREP', prep)
    monkeypatch.setattr(driver, 'OUT', tmp_path / 'output')
    monkeypatch.setattr(driver, 'DEST', tmp_path / 'checkpoints')
    monkeypatch.setattr(driver.os, 'sched_getscheduler', lambda pid: driver.os.SCHED_IDLE if case == 'idle_parent' else driver.os.SCHED_OTHER)
    monkeypatch.setattr(driver.os, 'nice', lambda delta: 0)
    monkeypatch.setattr(retirement, 'terminal', lambda owner: case != 'live_preparer')
    monkeypatch.setattr(canonical, 'ready', lambda *a, **k: case != 'incomplete_canonical')
    def contract(*args, **kwargs):
        if case == 'incomplete_native_scope': raise ValueError('The full actual native source and predecessor scope is incomplete')
        return True
    monkeypatch.setattr(native, 'require_complete_preparation', contract)
    monkeypatch.setattr(driver.shutil, 'disk_usage', lambda path: SimpleNamespace(free=99))
    with pytest.raises(ValueError): driver.parent()
    assert not driver.OUT.exists() and not driver.DEST.exists() and not torch.cuda.is_initialized()


def test_driver_import_does_not_start_a_neural_runtime_or_GPU_job():
    command = "import runpy,sys;runpy.run_path('scripts/train_native_code_suffix_aligned_GPU_v1.py');assert 'torch' not in sys.modules;print('source-only import verified')"
    result = subprocess.run([sys.executable, '-c', command], cwd=ROOT, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0 and result.stdout.strip() == 'source-only import verified'


def test_production_scope_full_sources_context_attempts_and_assessments_remains_explicit():
    from slm_lab.native_suffix_owned_runtime_contract_v1 import require_complete_preparation, FULL
    assert driver.PROFILES == ('anchor_CE', 'anchor_execution')
    assert driver.STAGE_OPTIONS['group_size'] == 8 and driver.STAGE_OPTIONS['CPU_mechanics_horizon'] is None
    assert driver.STAGE_OPTIONS['epochs'] == 3 and driver.STAGE_OPTIONS['seed'] == 989711
    assert driver.STAGE_OPTIONS['maximum_cache_bytes'] >= 2 * 8 * 35 * 8 * 64 * 4095 * 4
    assert driver.SUITES == ('code', 'broad', 'preferences', 'QA', 'AI2D', 'MBPP', 'words')
    assert driver.NATIVE_PREFILL['chunk_tokens'] == 128
    assert FULL['updates'] == 11172 and FULL['source_presentations'] == 44652
    assert FULL['original_response_EOS_targets'] == 7010616
    # An actual unfinished current data/preparation state cannot be upgraded.
    with pytest.raises(ValueError):
        require_complete_preparation(ROOT, {}, options=driver.STAGE_OPTIONS, prefill=driver.NATIVE_PREFILL,
            maximum_state_bytes=driver.MAX_STATE, maximum_journal_raw_bytes=driver.MAX_JOURNAL,
            maximum_assessment_bytes=driver.MAX_ASSESSMENTS, reserve_bytes=driver.RESERVE)


def test_modified_prefill_recipe_is_rejected_without_mutating_the_actual_actor():
    own = student(NativeSuffixTrainData.CPU_mechanics(ROOT, CONTROL))
    before, original = tensor_digest(own.parts.named_parameters()), own.policy.prefill
    with pytest.raises(ValueError):
        driver.attach_native_prefill(own, purpose='CPU-mechanics', settings=dict(driver.NATIVE_PREFILL, chunk_tokens=64))
    assert tensor_digest(own.parts.named_parameters()) == before and own.policy.prefill == original
