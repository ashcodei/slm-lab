import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from slm_lab.complete_joint_training_loop_v1 import run
from slm_lab.corpus import file_hash
from slm_lab.frozen_own_text_reference_v1 import FrozenOwnTextReferenceV1
from slm_lab.joint_native_GPU_cold_v1 import prove
from slm_lab.joint_native_source_v1 import reconstruct
from slm_lab.joint_selected_native_export_v1 import export
from slm_lab.joint_source_checked_training_stage_v3 import JointSourceCheckedStageV3
from slm_lab.own_dense_standalone_v1 import load
from slm_lab.parallel_function_repair_stage_v1 import value_digest
from slm_lab.whole_human_sft_evidence_v1 import tensor_digest
from test_joint_indexed_training_stage_v1 import budgets, compute, RESERVE
from test_joint_selected_native_state_v1 import completed, fresh
from test_joint_task_gradients_v1 import fixture, word_fixture


@pytest.fixture(scope='module')
def trained_loop(completed):
    trial = SimpleNamespace(**vars(completed['trial']))
    trial.view, trial.optimizer = fresh(completed)
    trial.reference = FrozenOwnTextReferenceV1(trial.view.actor.base, sources={'fixture_sources': trial.data.identity_sha256}, tokenizer_sha256=trial.sources['tokenizer_sha256'])
    folder = completed['directory'] / 'connected-original-loop'; folder.mkdir()
    stage = JointSourceCheckedStageV3(trial.view, trial.reference, trial.optimizer, trial.entries, trial.sources, trial.options,
        history_root=folder / 'history', history_budgets=budgets(), source_guard=completed['stage'].source_guard,
        source_files=completed['files'], maximum_attempts=3)
    observed = []; probes = []
    before = tensor_digest(trial.view.named_parameters())
    with pytest.raises(ValueError, match='full original joint GPUstage'):
        run(stage, folder / 'state', capacity_reports=[], cold_reports=[], capacity_plan={}, checkpoint_every=2,
            maximum_checkpoint_bytes=8 * 1024**2, minimum_free_bytes=RESERVE,
            preflight=stage.source_guard, progress=observed.append)
    assert stage.completed == 0 and not trial.optimizer.state and tensor_digest(trial.view.named_parameters()) == before
    report = run(stage, folder / 'state', capacity_reports=[], cold_reports=[], capacity_plan={}, checkpoint_every=2,
        maximum_checkpoint_bytes=8 * 1024**2, minimum_free_bytes=RESERVE, preflight=stage.source_guard,
        progress=observed.append, allow_mechanics=True,
        compute_next=lambda: stage.step(lambda index, hook: compute(trial, stage, probes, index, hook)))
    assert report['actual_full_size_GPU_joint_updates'] == 0 and report['mode'] == 'mechanics'
    assert [r['actual_completed_updates'] for r in observed] == [1, 2, 3, 4]
    assert tensor_digest(trial.view.named_parameters()) == tensor_digest(completed['trial'].view.named_parameters())
    assert value_digest(trial.optimizer.state_dict()) == value_digest(completed['trial'].optimizer.state_dict())
    (folder / 'actual-complete-loop-proof.json').write_text(json.dumps(report, indent=2))
    return dict(completed=completed, trial=trial, stage=stage, report=report, folder=folder)


def test_connected_actual_four_objective_source_loop_commits_complete_selected_Adam_and_private_state(trained_loop):
    report = trained_loop['report']
    assert report['actual_completed_updates'] == report['planned_updates'] == 4
    assert report['every_original_source_prefix_and_final_private_state_durable']
    assert report['new_native_export_and_all_original_assessments_required']
    assert (trained_loop['folder'] / 'state/state-slot-0.pt').exists()


@pytest.fixture(scope='module')
def native_pair(trained_loop):
    original = trained_loop['completed']; trial = trained_loop['trial']; folder = trained_loop['folder']
    photo, _ = fixture(1); _, word, _ = word_fixture(); code = trial.data.batch([trial.positions[0]])
    view, optimizer = fresh(original)
    receipt = export(view, optimizer, folder / 'state', folder / 'complete-native', trial.data.weak.folder / 'tokenizer.json',
        sources=trial.sources, entries=trial.entries, source_files=original['files'], expected_world=1, expected_completed=4,
        expected_pointer_sha256=trained_loop['report']['actual_final_pointer_sha256'], mode='mechanics',
        probes=[('actual_original_code', code), ('synthetic_photo', photo)], word_pixels=word['word_pixels'],
        working_budget_bytes=256 * 1024**2, maximum_checkpoint_bytes=8 * 1024**2, minimum_free_bytes=RESERVE,
        maximum_logit_error=1e-5, allow_mechanics=True)
    other, adam = fresh(original)
    native, source_proof = reconstruct(other, adam, folder / 'state', sources=trial.sources, entries=trial.entries,
        source_files=original['files'], expected_world=1, expected_completed=4,
        expected_pointer_sha256=trained_loop['report']['actual_final_pointer_sha256'], mode='mechanics',
        working_budget_bytes=256 * 1024**2, allow_mechanics=True)
    cold, _ = load(receipt['checkpoint'], expected_checkpoint_sha256=receipt['checkpoint_sha256'], expected_sources=trial.sources)
    inputs = dict(text={k: v for k, v in code.items() if k != 'labels'}, photo={k: v for k, v in photo.items() if k != 'labels'}, word={'word_pixels': word['word_pixels']})
    return dict(native=native, cold=cold, source_proof=source_proof, receipt=receipt, inputs=inputs, sources=trial.sources, folder=folder)


def test_independent_selected_state_and_exported_natives_have_exact_BF16_text_photo_and_FP32word_CPU_outputs(native_pair):
    pair = native_pair
    report = prove(pair['native'], pair['cold'], pair['source_proof'], checkpoint_sha256=pair['receipt']['checkpoint_sha256'],
        sources=pair['sources'], inputs=pair['inputs'], physical_GPU=None, allow_mechanics=True)
    assert report['all_original_native_values_and_text_photo_FP32word_cold_outputs_bit_exact']
    assert report['actual_observations']['word']['actual_output_dtype'] == 'torch.float32'
    assert report['actual_observations']['text']['actual_output_dtype'] == 'torch.bfloat16'
    assert report['full_size_GPU_native_cold_passed'] is False
    (pair['folder'] / 'independent-native-cold-CPU-proof.json').write_text(json.dumps(report, indent=2))


def test_an_unused_native_head_value_cannot_be_lost_even_if_text_image_and_word_outputs_would_match(native_pair):
    pair = native_pair; parameter = pair['cold'].heads['detector'][-1].bias
    original = parameter.detach().clone()
    try:
        with torch.no_grad():
            parameter.add_(.25)
        with pytest.raises(ValueError, match='Every actual native value'):
            prove(pair['native'], pair['cold'], pair['source_proof'], checkpoint_sha256=pair['receipt']['checkpoint_sha256'],
                sources=pair['sources'], inputs=pair['inputs'], physical_GPU=None, allow_mechanics=True)
    finally:
        with torch.no_grad():
            parameter.copy_(original)
