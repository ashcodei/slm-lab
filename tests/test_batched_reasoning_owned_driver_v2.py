"""Consume the real driver optimizer; own-small CPU evidence is mechanics only."""
import copy
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

import train_batched_reasoning_aligned_GPU_v2 as driver
from slm_lab.parallel_batched_reasoning_stage_v2 import ParallelBatchedReasoningStage
from slm_lab.parallel_function_repair_stage_v1 import value_digest
from slm_lab.public_reasoning_train_data_v1 import PublicReasoningTrainData
from slm_lab.whole_human_sft_evidence_v1 import tensor_digest
from test_batched_reasoning_capacity_controller_v2 import make_student
from test_parallel_batched_reasoning_stage_v2 import SOURCES, positions

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def data():
    return PublicReasoningTrainData(driver.DATA, driver.ADMITTED)


@pytest.mark.parametrize('profile', driver.PROFILES)
def test_real_owned_AdamW_updates_actual_stage_and_replays_next_actions_and_state(data, tmp_path, profile):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    own = make_student(data)
    options = dict(driver.STAGE_OPTIONS, CPU_mechanics_horizon=2)
    adam = driver.make_optimizer(own, options, driver.OPTIMIZER)
    assert type(adam) is torch.optim.AdamW
    assert adam.param_groups[0]['foreach'] is False and adam.param_groups[0]['fused'] is False
    assert adam.param_groups[0]['lr'] == 3e-5 and adam.param_groups[0]['betas'] == (.9, .95)
    selected = positions(data)[:2]
    entries = [[p] for p in selected]
    stage = ParallelBatchedReasoningStage(own, adam, data, entries, profile, options, SOURCES, mode='mechanics')
    base_before = tensor_digest(own.base.named_parameters())
    actor_before = tensor_digest(own.parts.named_parameters())
    first = stage.update()
    saved, manifest = stage.snapshot()
    wanted = stage.update()
    wanted_actor = tensor_digest(own.parts.named_parameters())
    wanted_adam = value_digest(adam.state_dict())
    wanted_metrics = copy.deepcopy(stage.metrics)
    stage.resume(saved, expected_manifest=manifest)
    assert stage.update() == wanted
    assert tensor_digest(own.parts.named_parameters()) == wanted_actor
    assert value_digest(adam.state_dict()) == wanted_adam and stage.metrics == wanted_metrics
    assert wanted_actor != actor_before and tensor_digest(own.base.named_parameters()) == base_before
    assert stage.completed == 2 and len(adam.state) == 17
    assert all(float(row['step']) == 2 for row in adam.state.values())
    assert all(len(row['actual_rank_observations'][0]['kernel']['trajectories']) == 8 for row in stage.metrics)
    assert all(row['original_response_EOS_targets'] == data.information(selected[i])['response_EOS_targets']
        for i, row in enumerate(stage.metrics))

    bad = make_student(data)
    wrong = torch.optim.Adam([p for _, p in bad.named_selected(('actor',))], lr=options['learning_rate'])
    unchanged = tensor_digest(bad.parts.named_parameters())
    with pytest.raises(ValueError, match='AdamW'):
        ParallelBatchedReasoningStage(bad, wrong, data, entries, profile, options, SOURCES, mode='mechanics')
    assert not wrong.state and tensor_digest(bad.parts.named_parameters()) == unchanged
    assert not torch.cuda.is_initialized()
    (tmp_path / ('actual-owned-optimizer-' + profile + '.json')).write_text(json.dumps(dict(
        profile=profile, actual_optimizer_class=type(adam).__name__, actual_own_small_parameters=own.counts()['actual_unique'],
        actual_physical_optimizer_calls=3, complete_logical_source_updates=2, actual_source_positions=selected,
        actual_complete_response_EOS_targets=sum(row['original_response_EOS_targets'] for row in stage.metrics),
        actual_original_context_tokens=sum(row['complete_original_context_tokens'] for row in stage.metrics),
        actual_first_update=first, actual_second_update=wanted, group_size=8,
        actual_selected_AdamW_states=17, final_AdamW_clock=2, next_actual_actions_verifiers_weights_AdamW_replayed_exactly=True,
        original_base_unchanged=True, wrong_Adam_rejected_before_update=True,
        explicit_CPU_mechanics_horizon=2, production_native_horizons_unchanged=[64, 256],
        full_size_GPUcapacity_or_training=False, quality_gain_established=False, production_updates=0,
        goal_achieved=False), indent=2))


@pytest.mark.parametrize('case', ['idle_inherited', 'preparation_live', 'canonical_incomplete', 'disk_shortfall'])
def test_real_launch_rejects_before_lease_model_or_output_on_required_gate_failure(case, tmp_path, monkeypatch):
    from slm_lab import exact_waiting_process_retirement_v1 as retirement
    from slm_lab import joint_visual_runtime_contract_v1 as contract
    prep = tmp_path / 'prep'
    prep.mkdir()
    (prep / 'ready.json').write_text(json.dumps(dict(exact_predecessors=dict(pipeline_owner='fixed'), required_free_bytes_before_launch=100)))
    (prep / 'plan.json').write_text(json.dumps(dict(owner=dict(pid=1, ticks='100'))))
    monkeypatch.setattr(driver, 'PREP', prep)
    monkeypatch.setattr(driver, 'OUT', tmp_path / 'output')
    monkeypatch.setattr(driver, 'DEST', tmp_path / 'checkpoints')
    monkeypatch.setattr(driver.os, 'sched_getscheduler', lambda pid: driver.os.SCHED_IDLE if case == 'idle_inherited' else driver.os.SCHED_OTHER)
    monkeypatch.setattr(driver.os, 'nice', lambda delta: 0)
    monkeypatch.setattr(retirement, 'terminal', lambda owner: case != 'preparation_live')
    monkeypatch.setattr(contract, 'ready', lambda *a, **k: case != 'canonical_incomplete')
    monkeypatch.setattr(driver.shutil, 'disk_usage', lambda path: SimpleNamespace(free=99))
    with pytest.raises(ValueError): driver.parent()
    assert not driver.OUT.exists() and not driver.DEST.exists() and not torch.cuda.is_initialized()


def test_source_preparation_and_driver_import_do_not_start_neural_runtime_or_GPUwork():
    command = "import runpy,sys;runpy.run_path('scripts/prepare_batched_reasoning_aligned_CPU_v2.py');runpy.run_path('scripts/train_batched_reasoning_aligned_GPU_v2.py');assert 'torch' not in sys.modules;print('source-only imports verified')"
    result = subprocess.run([sys.executable, '-c', command], cwd=ROOT, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0 and result.stdout.strip() == 'source-only imports verified'


def test_full_production_recipe_keeps_every_original_source_epoch_native_horizon_and_assessment():
    plan = json.loads((driver.CASES / 'actual-complete-source-capacity-plan.json').read_text())
    assert driver.PROFILES == ('anchor_CE', 'anchor_RL')
    assert driver.STAGE_OPTIONS['epochs'] == plan['epochs'] == 3
    assert driver.STAGE_OPTIONS['seed'] == plan['seed'] == 989113
    assert driver.STAGE_OPTIONS['group_size'] == 8 and driver.STAGE_OPTIONS['CPU_mechanics_horizon'] is None
    assert driver.OPTIMIZER['algorithm'] == 'torch.optim.AdamW'
    assert driver.SUITES == ('code', 'broad', 'preferences', 'QA', 'AI2D', 'MBPP', 'words')
    assert plan['complete_training_budget'] == dict(updates=1926, source_presentations=7704,
        complete_context_tokens=735570, response_EOS_targets=57240, full_native_generation_horizon_sum=726336)
    assert len(plan['capacity_entries']) == 109 and plan['full_original_action_budgets'] == [64, 256]
    assert len(plan['actual_train_task_families']) == 12


def test_optimizer_factory_rejects_an_altered_algorithm_or_execution_recipe_before_any_update(data):
    own = make_student(data)
    before = tensor_digest(own.parts.named_parameters())
    for altered in [dict(algorithm='torch.optim.Adam'), dict(foreach=True), dict(fused=True)]:
        with pytest.raises(ValueError, match='AdamW'):
            driver.make_optimizer(own, driver.STAGE_OPTIONS, dict(driver.OPTIMIZER, **altered))
    assert tensor_digest(own.parts.named_parameters()) == before and all(p.grad is None for p in own.parameters())
