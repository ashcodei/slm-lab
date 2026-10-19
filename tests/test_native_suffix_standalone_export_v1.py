"""Actual completed native CPU actors become independent multimodal packages."""
import copy
import json
from pathlib import Path
import shutil

import pytest
import torch

from slm_lab.codenet_checkpoint_v1 import capture_rng
from slm_lab.corpus import file_hash
from slm_lab.exact_waiting_process_retirement_v1 import terminal
from slm_lab.native_suffix_standalone_export_v1 import export
from slm_lab.native_suffix_train_data_v1 import NativeSuffixTrainData
from slm_lab.own_dense_standalone_v1 import load
from slm_lab.parallel_function_repair_stage_v1 import value_digest
from slm_lab.parallel_repair_durable_state_v1 import read_latest
from slm_lab.whole_human_sft_evidence_v1 import tensor_digest
from test_native_suffix_capacity_controller_v2 import student, inputs
from test_parallel_native_suffix_stage_v1 import ROOT, CONTROL, SOURCES

TRAINING = ROOT / 'reports/native-suffix-training-controller-v1-full-pytest/test_actual_four_replica_capac0'
CAPACITY = ROOT / 'reports/native-suffix-capacity-controller-v2-pytest/test_actual_four_replicas_both0'


@pytest.fixture(scope='module')
def actual():
    assert terminal(dict(pid=194586, ticks='57262246'))
    complete = json.loads((ROOT / 'reports/native-suffix-training-controller-v1-queue/complete.json').read_text())
    assert complete['actual_four_replica_small_native_controller_integration_passed']
    data = NativeSuffixTrainData.CPU_mechanics(ROOT, CONTROL)
    data._information[4]['native_execution_eligible'] = False
    return data


@pytest.mark.parametrize('profile', ['anchor_CE', 'anchor_execution'])
def test_actual_trained_native_actor_folds_preserving_all_values_and_independent_text_photo_word_cold(actual, tmp_path, profile):
    state, _, _ = read_latest(TRAINING / profile / 'complete/durable')
    receipt = json.loads((TRAINING / (profile + '-rank-0.json')).read_text())['actual_complete_training']
    with torch.random.fork_rng(devices=[]): hot = student(actual).eval()
    hot.import_export(state['model'], expected_sources=SOURCES)
    before = tensor_digest(hot.named_parameters()); rng = value_digest(capture_rng('cpu'))
    original_critic = hot.base.network.value_head.weight.detach().clone()
    native, payload, proof = export(hot, actual, state, receipt, sources=SOURCES,
        tokenizer_sha256=file_hash(actual.corpus.tokenizer_path), working_budget_bytes=256*1024**2, allow_mechanics=True)
    assert tensor_digest(hot.named_parameters()) == before and value_digest(capture_rng('cpu')) == rng
    assert torch.equal(native.network.value_head.weight, original_critic)
    assert native.counts()['total_unique'] == 267430 and hot.counts()['actual_unique'] == 267718
    assert not ({p.data_ptr() for p in native.parameters()} & {p.data_ptr() for p in hot.parameters()})
    assert not proof['whole_production_source_schedule_completed'] and proof['actual_native_data_admission_sha256'] is None
    assert proof['complete_training_budget']['source_presentations'] == 9
    assert proof['complete_training_budget']['distinct_original_human_sources'] == 1
    assert proof['complete_training_budget']['original_response_EOS_targets'] == 612
    assert proof['complete_training_budget']['CE_only_presentations'] == 1
    path = tmp_path / 'standalone'; path.mkdir()
    shutil.copyfile(actual.corpus.tokenizer_path, path / 'tokenizer.json'); torch.save(payload, path / 'model.pt')
    cold, metadata = load(path / 'model.pt', expected_checkpoint_sha256=file_hash(path / 'model.pt'), expected_sources=SOURCES)
    assert tensor_digest(native.named_parameters()) == tensor_digest(cold.named_parameters())
    assert metadata['no_old_base_actor_optimizer_or_training_reference_loaded']
    batches = inputs(actual); differences = {}
    with torch.no_grad(), torch.autocast('cpu', enabled=False):
        for name in ['text', 'photo']:
            expected, loaded, unfused = native(**batches[name])['logits'], cold(**batches[name])['logits'], hot(**batches[name])['logits']
            assert torch.equal(expected, loaded) and bool(torch.isfinite(expected).all())
            differences[name] = float((expected-unfused).abs().max())
        expected, loaded = native.word_logits(batches['word']['word_pixels']), cold.word_logits(batches['word']['word_pixels'])
        assert expected.dtype == torch.float32 and torch.equal(expected, loaded)
    for kind in ['no_opt_in', 'partial', 'wrong_Adam', 'production_claim', 'tokenizer', 'dropped_CE']:
        changed, changed_state = copy.deepcopy(receipt), copy.deepcopy(state)
        if kind == 'partial': changed['actual_complete_updates'] -= 1
        if kind == 'wrong_Adam': changed['actual_final_Adam_sha256'] = 'e'*64
        if kind == 'production_claim': changed['mode'] = 'train'
        if kind == 'dropped_CE': changed_state['metrics'][0]['original_response_EOS_targets'] -= 1
        with pytest.raises(ValueError): export(hot, actual, changed_state, changed, sources=SOURCES,
            tokenizer_sha256='e'*64 if kind == 'tokenizer' else file_hash(actual.corpus.tokenizer_path),
            working_budget_bytes=256*1024**2, allow_mechanics=kind != 'no_opt_in')
    assert tensor_digest(hot.named_parameters()) == before and value_digest(capture_rng('cpu')) == rng
    (tmp_path / 'actual-standalone-native-multimodal-export.json').write_text(json.dumps(dict(profile=profile,
        actual_trained_research_parameters=267718, actual_standalone_native_parameters=267430,
        actual_original_human_programs=1, actual_source_presentations=9, complete_original_CE_targets=612,
        exact_complete_native_values_and_text_photo_word_cold_outputs=True, source_model_and_all_private_RNG_unchanged=True,
        original_dense_critic_preserved_and_unused_research_critic_excluded=True,
        actual_hot_to_folded_FP32_logit_errors=differences, hot_generation_equivalence_not_assumed=True,
        actual_standalone_checkpoint_sha256=file_hash(path / 'model.pt'), actual_tokenizer_sha256=file_hash(path / 'tokenizer.json'),
        full_size_GPUcapacity_or_model_quality_proved=False, production_updates=0, goal_achieved=False), indent=2))


def test_actual_disposable_capacity_state_cannot_claim_completed_native_training_export(actual):
    state, _, _ = read_latest(CAPACITY / 'anchor_CE/durable')
    receipt = json.loads((CAPACITY / 'anchor_CE-rank-0.json').read_text())
    with torch.random.fork_rng(devices=[]): hot = student(actual).eval()
    hot.import_export(state['model'], expected_sources=SOURCES)
    before = tensor_digest(hot.named_parameters()); rng = value_digest(capture_rng('cpu'))
    with pytest.raises(ValueError): export(hot, actual, state, receipt, sources=SOURCES,
        tokenizer_sha256=file_hash(actual.corpus.tokenizer_path), working_budget_bytes=256*1024**2, allow_mechanics=True)
    assert tensor_digest(hot.named_parameters()) == before and value_digest(capture_rng('cpu')) == rng
