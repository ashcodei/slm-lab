"""Real own-small exported values/native transport; GPU admission stays closed."""
import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

import train_batched_reasoning_aligned_GPU_v1 as driver
from slm_lab.codenet_checkpoint_v1 import capture_rng
from slm_lab.corpus import file_hash
from slm_lab.dense_foundation_inference_v1 import DenseNativeView
from slm_lab.own_dense_standalone_v1 import load, export as export_dense
from slm_lab.parallel_function_repair_stage_v1 import value_digest
from slm_lab.parallel_repair_durable_state_v1 import read_latest
from slm_lab.public_reasoning_reserved_evaluation_v1 import PublicReasoningReserved
from slm_lab.public_reasoning_train_data_v1 import PublicReasoningTrainData
from slm_lab.reasoning_native_export_v1 import export
from slm_lab.reasoning_reserved_shards_v1 import evaluate, reconcile
from slm_lab.whole_human_sft_evidence_v1 import tensor_digest
from test_batched_reasoning_capacity_controller_v2 import make_student, cold_inputs

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/public-reasoning-curriculum-v1'
ADMITTED = ROOT / 'reports/public-reasoning-curriculum-v1-readback-admission/complete.json'
PROOFS = ROOT / 'reports/batched-reasoning-training-controller-v1-pytest/test_actual_four_replica_capac0'


@pytest.fixture(scope='module')
def data(): return PublicReasoningTrainData(DATA, ADMITTED)


@pytest.mark.parametrize('profile', ['anchor_CE', 'anchor_RL'])
def test_actual_trained_parts_export_complete_native_values_critic_and_real_cold_outputs(data, tmp_path, profile):
    torch.set_num_threads(1)
    state, _, _ = read_latest(PROOFS / profile / 'complete/durable')
    receipt = json.loads((PROOFS / (profile + '-rank-0.json')).read_text())['actual_complete_training']
    model = make_student(data).eval()
    sources = state['model']['sources']
    model.import_export(state['model'], expected_sources=sources)
    before, rng = tensor_digest(model.named_parameters()), value_digest(capture_rng('cpu'))
    critic = model.base.network.value_head.weight.detach().clone()
    assert bool((model.parts['critic'].weight == 0).all())
    native, payload, proof = export(model, state, receipt, sources=sources,
        tokenizer_sha256=file_hash(DATA / 'tokenizer.json'), working_budget_bytes=256 * 1024**2, allow_mechanics=True)
    assert tensor_digest(model.named_parameters()) == before
    assert torch.equal(native.network.value_head.weight, critic)
    assert native.counts()['total_unique'] == 267430 and model.counts()['actual_unique'] == 267718
    assert not ({p.data_ptr() for p in model.parameters()} & {p.data_ptr() for p in native.parameters()})
    assert not proof['whole_production_source_schedule_completed'] and proof['hot_policy_FP32_BF16_or_generation_equivalence_not_assumed']
    checkpoint = tmp_path / 'model.pt'
    shutil.copyfile(DATA / 'tokenizer.json', tmp_path / 'tokenizer.json')
    torch.save(payload, checkpoint)
    cold, metadata = load(checkpoint, expected_checkpoint_sha256=file_hash(checkpoint), expected_sources=sources)
    assert metadata['completed_updates'] == 3 and tensor_digest(cold.named_parameters()) == tensor_digest(native.named_parameters())
    inputs = cold_inputs(data)
    errors = {}
    with torch.no_grad(), torch.autocast('cpu', enabled=False):
        for name in ['text', 'photo']:
            a, b, hot = native(**inputs[name])['logits'], cold(**inputs[name])['logits'], model(**inputs[name])['logits']
            assert torch.equal(a, b) and bool(torch.isfinite(a).all())
            errors[name] = float((hot - a).abs().max())
        a, b = native.word_logits(inputs['word']['word_pixels']), cold.word_logits(inputs['word']['word_pixels'])
        assert a.dtype == torch.float32 and torch.equal(a, b)
    assert value_digest(capture_rng('cpu')) == rng
    for case in ['no_opt_in', 'partial', 'wrong_Adam', 'production_claim']:
        changed = copy.deepcopy(receipt)
        if case == 'partial': changed['actual_complete_updates'] -= 1
        if case == 'wrong_Adam': changed['actual_final_Adam_sha256'] = 'c' * 64
        if case == 'production_claim': changed['mode'] = 'train'
        with pytest.raises(ValueError):
            export(model, state, changed, sources=sources, tokenizer_sha256=file_hash(DATA / 'tokenizer.json'),
                working_budget_bytes=256 * 1024**2, allow_mechanics=case != 'no_opt_in')
        assert tensor_digest(model.named_parameters()) == before and value_digest(capture_rng('cpu')) == rng
    (tmp_path / 'actual-trained-reasoning-native-export.json').write_text(json.dumps(dict(profile=profile,
        actual_small_research_values=267718, actual_small_native_values=267430, actual_completed_source_groups=3,
        actual_all_native_values_and_text_photo_word_cold_exact=True, actual_original_dense_critic_preserved=True, original_dense_critic_was_nonzero=bool((critic != 0).any()),
        source_model_and_private_RNG_unchanged=True, actual_hot_to_folded_FP32_logit_errors=errors,
        checkpoint_sha256=file_hash(checkpoint), tokenizer_sha256=file_hash(tmp_path / 'tokenizer.json'),
        full_size_GPUcapacity_quality_or_hot_generation_equivalence=False, production_updates=0), indent=2))


def test_actual_whole289_reserved_native_shards_original_inputs_budgets_and_fresh_regrading(data, tmp_path):
    torch.set_num_threads(1)
    population = PublicReasoningReserved(DATA, ADMITTED)
    model = make_student(data).base.eval()
    # Deliberately make every greedy response true EOS. This is a mechanics
    # fixture, not a trained reasoning result; every actual neural call runs.
    with torch.no_grad():
        model.network.embedding.weight.fill_(1.)
        model.network.embedding.weight[2].fill_(100.)
        for block in model.network.blocks:
            for name in ['q_proj', 'k_proj', 'v_proj', 'o_proj']: getattr(block.attn, name).weight.zero_()
            for name in ['gate', 'up', 'down']: getattr(block, name).weight.zero_()
    native = DenseNativeView(model).eval()
    checkpoint = tmp_path / 'EOS-fixture.pt'
    torch.save(export_dense(model, sources=dict(test_source_sha256=file_hash(ADMITTED)),
        provenance=dict(deliberately_initialized_EOS_only_CPU_mechanics=True), completed_updates=0,
        tokenizer_sha256=file_hash(DATA / 'tokenizer.json')), checkpoint)
    before, rng = tensor_digest(model.named_parameters()), value_digest(capture_rng('cpu'))
    paths, events = [], []
    for rank in range(4):
        path = tmp_path / ('rank-' + str(rank) + '.json')
        report = evaluate(native, population, path, rank=rank, world=4,
            checkpoint_sha256=file_hash(checkpoint), checked=lambda: population.guard.check(),
            record=events.append, allow_CPU_mechanics=True)
        assert all(r['actual_generated_token_ids'] == [2] for r in report['results'])
        paths.append(path)
    result = reconcile(population, paths, tmp_path / 'complete.json')
    assert result['examples'] == 289 and len(result['per_task']) == 12 and result['correct'] == 0
    assert len(events) == 289 and sorted(e['requested_native_action_budget'] for e in events) == sorted(
        c['native_generation_budget'] for c in population.cases)
    for row in result['results']:
        case = population.cases[row['position']]
        assert row['original_prompt_ids'] == case['original_prompt_ids']
        assert row['score'] == population.grade(row['position'], row['actual_generated_token_ids'])
    assert tensor_digest(model.named_parameters()) == before and value_digest(capture_rng('cpu')) == rng
    wrong = json.loads(paths[1].read_text())
    wrong['results'][0]['original_prompt_ids'].append(2)
    altered = tmp_path / 'altered.json'
    altered.write_text(json.dumps(wrong))
    with pytest.raises(ValueError): reconcile(population, [paths[0], altered, *paths[2:]], tmp_path / 'invalid.json')
    with pytest.raises(ValueError):
        evaluate(native, population, tmp_path / 'forbidden.json', rank=0, world=4,
            checkpoint_sha256=file_hash(checkpoint), checked=lambda: None, record=lambda value: None)
    (tmp_path / 'actual-complete-reserved-native-transport.json').write_text(json.dumps(dict(
        actual_original_reserved_sources=289, actual_task_families=12, actual_native_greedy_neural_deliveries=289,
        all_actual_unchanged_whole_prompts_64or256budgets_and_true_EOS_retained=True,
        every_original_source_freshly_regraded=True, source_model_and_private_RNG_unchanged=True,
        deliberately_initialized_EOS_only_fixture=True, actual_fixture_correct=0,
        CPU_mechanics_does_not_admit_GPU_or_quality=True, checkpoint_sha256=file_hash(checkpoint),
        production_updates=0), indent=2))


@pytest.mark.parametrize('case', ['idle_inherited', 'preparation_live', 'canonical_incomplete', 'disk_shortfall'])
def test_real_launch_entry_rejects_before_lease_model_or_output_when_a_required_gate_fails(case, tmp_path, monkeypatch):
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
    assert not driver.OUT.exists() and not driver.DEST.exists()


def test_source_preparation_and_driver_import_create_no_neural_runtime_or_GPUwork():
    command = "import runpy,sys;runpy.run_path('scripts/prepare_batched_reasoning_aligned_CPU_v1.py');runpy.run_path('scripts/train_batched_reasoning_aligned_GPU_v1.py');assert 'torch' not in sys.modules;print('source-only imports verified')"
    completed = subprocess.run([sys.executable, '-c', command], cwd=ROOT, capture_output=True, text=True, timeout=30)
    assert completed.returncode == 0 and completed.stdout.strip() == 'source-only imports verified'


def test_full_driver_recipe_keeps_matched_profiles_full_native_budgets_and_all_original_suites():
    plan = json.loads((driver.CASES / 'actual-complete-source-capacity-plan.json').read_text())
    assert driver.PROFILES == ('anchor_CE', 'anchor_RL')
    assert driver.STAGE_OPTIONS['epochs'] == plan['epochs'] == 3
    assert driver.STAGE_OPTIONS['seed'] == plan['seed']
    assert driver.STAGE_OPTIONS['CPU_mechanics_horizon'] is None
    assert driver.SUITES == ('code', 'broad', 'preferences', 'QA', 'AI2D', 'MBPP', 'words')
    assert plan['complete_training_budget']['source_presentations'] == 7704
    assert plan['complete_training_budget']['response_EOS_targets'] == 57240
    assert plan['full_original_action_budgets'] == [64, 256]
