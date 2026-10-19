"""Real optimizer consumption; no GPU launch or full-size proof is fabricated."""
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

import train_mixed_coding_SFT_aligned_GPU_v1 as driver
from slm_lab.mixed_coding_SFT_data_v1 import MixedCodingSFTData
from slm_lab.parallel_mixed_coding_SFT_stage_v1 import ParallelMixedCodingSFTStage
from slm_lab.whole_human_sft_evidence_v1 import tensor_digest
from test_parallel_mixed_coding_SFT_stage_v1 import ROOT, SOURCES, source_positions
from test_mixed_coding_SFT_training_and_export_v1 import student


@pytest.fixture(scope='module')
def data(): return MixedCodingSFTData(ROOT)


@pytest.mark.parametrize('profile', driver.PROFILES)
def test_real_owned_optimizer_is_consumed_by_actual_stage_and_updates_only_actor(data, tmp_path, profile):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    own = student()
    adam = driver.make_optimizer(own, driver.STAGE_OPTIONS, driver.OPTIMIZER)
    assert type(adam) is torch.optim.AdamW and adam.param_groups[0]['foreach'] is False and adam.param_groups[0]['fused'] is False
    positions = source_positions(data)[:3]
    stage = ParallelMixedCodingSFTStage(own, adam, data, [positions], profile, driver.STAGE_OPTIONS, SOURCES, mode='mechanics')
    original_base = tensor_digest(own.base.named_parameters())
    original_actor = tensor_digest(own.parts.named_parameters())
    row = stage.update()
    assert stage.completed == 1 and len(adam.state) == 17 and all(float(v['step']) == 1 for v in adam.state.values())
    assert tensor_digest(own.parts.named_parameters()) != original_actor and tensor_digest(own.base.named_parameters()) == original_base
    bad = student()
    wrong = torch.optim.Adam([p for _, p in bad.named_selected(('actor',))], lr=driver.STAGE_OPTIONS['learning_rate'])
    unchanged = tensor_digest(bad.parts.named_parameters())
    with pytest.raises(ValueError): ParallelMixedCodingSFTStage(bad, wrong, data, [positions], profile,
        driver.STAGE_OPTIONS, SOURCES, mode='mechanics')
    assert not wrong.state and tensor_digest(bad.parts.named_parameters()) == unchanged
    assert not torch.cuda.is_initialized()
    (tmp_path / ('actual-owned-optimizer-' + profile + '.json')).write_text(json.dumps(dict(profile=profile,
        actual_optimizer_class=type(adam).__name__, actual_complete_native_sources=positions, actual_observed_update=row,
        actual_own_small_parameters=own.counts()['actual_unique'], actual_selected_Adam_states=17,
        wrong_Adam_rejected_before_any_optimizer_update=True, full_size_GPUcapacity_or_training=False,
        quality_gain_established=False, production_updates=0, goal_achieved=False), indent=2))


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


def test_owned_source_preparation_and_driver_import_open_no_neural_runtime_or_GPUwork():
    command = "import runpy,sys;runpy.run_path('scripts/prepare_mixed_coding_SFT_aligned_CPU_v1.py');runpy.run_path('scripts/train_mixed_coding_SFT_aligned_GPU_v1.py');assert 'torch' not in sys.modules;print('source-only imports verified')"
    completed = subprocess.run([sys.executable, '-c', command], cwd=ROOT, capture_output=True, text=True, timeout=30)
    assert completed.returncode == 0 and completed.stdout.strip() == 'source-only imports verified'


def test_full_recipe_preserves_all_original_sources_epochs_and_native_full_assessments():
    plan = json.loads((driver.CASES / 'actual-complete-source-capacity-plan.json').read_text())
    assert driver.PROFILES == ('plain_tokens', 'balanced_sources')
    assert driver.STAGE_OPTIONS['epochs'] == plan['epochs'] == 3
    assert driver.STAGE_OPTIONS['seed'] == plan['seed'] == 989923 and driver.STAGE_OPTIONS['width'] == 16
    assert driver.STAGE_OPTIONS['maximum_padded_tokens'] >= 4096 and driver.STAGE_OPTIONS['maximum_microbatch_examples'] == 2
    assert driver.SUITES == ('code', 'broad', 'preferences', 'QA', 'AI2D', 'MBPP', 'words')
    assert plan['complete_training_budget']['source_presentations'] == 84084
    assert plan['complete_training_budget']['response_EOS_targets'] == 21316734
    assert plan['complete_training_budget']['updates'] == 5256 and len(plan['capacity_entries']) == 53
    assert plan['actual_complete_training_tail_sources_per_rank'] == [3, 3, 3, 3]
