import copy
import json
from pathlib import Path

import pytest
import torch

from slm_lab.codenet_checkpoint_v1 import capture_rng, FileBindings
from slm_lab.corpus import file_hash
from slm_lab.frozen_own_text_reference_v1 import FrozenOwnTextReferenceV1
from slm_lab.joint_objective_step_v1 import JointReplicaGuardV1
from slm_lab.joint_selected_native_state_v1 import install
from slm_lab.joint_source_checked_training_stage_v3 import JointSourceCheckedStageV3
from slm_lab.mixed_coding_SFT_data_v1 import MixedCodingSFTData
from slm_lab.parallel_function_repair_stage_v1 import value_digest
from slm_lab.whole_human_sft_evidence_v1 import tensor_digest
from test_joint_indexed_training_stage_v1 import budgets, compute, RESERVE
from test_joint_objective_step_v1 import ROOT, Trial


@pytest.fixture(scope='module')
def completed(tmp_path_factory):
    torch.set_num_threads(1); directory = tmp_path_factory.mktemp('actual-joint-selected-native')
    data = MixedCodingSFTData(ROOT); trial = Trial(data, 'bf16')
    with torch.no_grad():
        trial.view.actor.base.network.value_head.weight.fill_(.125)
    trial.guard = JointReplicaGuardV1(trial.view, trial.optimizer)
    initial = copy.deepcopy(trial.view).eval()
    trial.sources.update(text_initializer_sha256=tensor_digest(trial.view.actor.base.language_named_parameters()),
        visual_initializer_sha256=tensor_digest((n, p) for n, p in trial.view.actor.base.named_parameters() if n.startswith(('network.vision.', 'heads.pointing.', 'heads.word.'))),
        source_admission_sha256='a' * 64, fixed_values_sha256=trial.guard.fixed.sha256)
    trial.reference = FrozenOwnTextReferenceV1(trial.view.actor.base, sources={'fixture_sources': data.identity_sha256}, tokenizer_sha256=trial.sources['tokenizer_sha256'])
    sentinel = directory / 'original-source-control.bin'; sentinel.write_bytes(b'orig')
    files = dict(data.source_files_sha256); files[str(sentinel)] = file_hash(sentinel)
    source_guard = FileBindings(files)
    stage = JointSourceCheckedStageV3(trial.view, trial.reference, trial.optimizer, trial.entries, trial.sources, trial.options,
        history_root=directory / 'history', history_budgets=budgets(), source_guard=source_guard.check,
        source_files=files, maximum_attempts=3)
    stage.open_history(); probes = []
    for _ in range(4):
        stage.step(lambda index, hook: compute(trial, stage, probes, index, hook))
    receipt = stage.commit(directory / 'state', maximum_bytes=8 * 1024**2, minimum_free_bytes=RESERVE)
    return dict(directory=directory, trial=trial, initial=initial, files=files, sentinel=sentinel,
        pointer_sha256=file_hash(directory / 'state/LATEST.json'), receipt=receipt, stage=stage)


def fresh(completed):
    view = copy.deepcopy(completed['initial']).eval()
    optimizer = torch.optim.AdamW([p for _, p in view.named_selected()], lr=1e-4, eps=1e-5, weight_decay=.01, foreach=False)
    return view, optimizer


def run(completed, view, optimizer, **extra):
    return install(view, optimizer, completed['directory'] / 'state', sources=completed['trial'].sources,
        entries=completed['trial'].entries, source_files=completed['files'], expected_world=1, expected_completed=4,
        expected_pointer_sha256=completed['pointer_sha256'], mode='mechanics', allow_mechanics=True, **extra)


def state(view, optimizer):
    return tensor_digest(view.named_parameters()), value_digest(optimizer.state_dict()), value_digest(capture_rng('cpu'))


def test_actual_complete_selected_weights_and_four_original_metrics_reconstruct_without_restoring_Adam_or_RNG(completed):
    view, optimizer = fresh(completed); rng = value_digest(capture_rng('cpu'))
    report = run(completed, view, optimizer)
    assert tensor_digest(view.named_parameters()) == tensor_digest(completed['trial'].view.named_parameters())
    assert report['every_original_metric_and_sparse_Adam_clock_reconstructed']
    assert report['actual_Adam_states_validated'] == len(completed['trial'].optimizer.state)
    assert report['actual_complete_cursor']['completed'] == 4
    assert not optimizer.state and value_digest(capture_rng('cpu')) == rng
    assert report['new_optimizer_updates'] == 0 and report['full_size_GPU_capacity_or_training_admission'] is False
    assert not torch.cuda.is_initialized()


def test_actual_selected_joint_model_exports_every_value_and_original_code_photo_word_outputs_to_one_cold_native(completed):
    from slm_lab.joint_selected_native_export_v1 import export
    from test_joint_task_gradients_v1 import fixture, word_fixture
    view, optimizer = fresh(completed)
    photo, _ = fixture(1); _, word, _ = word_fixture()
    trial = completed['trial']
    code = trial.data.batch([trial.positions[0]])
    output = completed['directory'] / 'independent-selected-native'
    report = export(view, optimizer, completed['directory'] / 'state', output, trial.data.weak.folder / 'tokenizer.json',
        sources=trial.sources, entries=trial.entries, source_files=completed['files'], expected_world=1, expected_completed=4,
        expected_pointer_sha256=completed['pointer_sha256'], mode='mechanics',
        probes=[('actual_complete_original_code', code), ('synthetic_image_control', photo)], word_pixels=word['word_pixels'],
        working_budget_bytes=256 * 1024**2, maximum_checkpoint_bytes=8 * 1024**2, minimum_free_bytes=RESERVE,
        maximum_logit_error=1e-5, allow_mechanics=True)
    assert report['every_complete_native_parameter_and_text_photo_word_output_cold_load_bit_exact']
    assert report['actual_reconstruction']['actual_Adam_states_validated'] == len(trial.optimizer.state)
    assert report['actual_fold']['original_native_critic_preserved']
    assert report['standalone_inference_requires_only_native_and_tokenizer']
    assert report['actual_counts']['total_unique'] == trial.view.actor.base.counts()['total_unique']
    (completed['directory'] / 'independent-selected-native-proof.json').write_text(json.dumps(report, indent=2))


def test_same_size_same_mtime_source_change_at_final_validation_rejects_before_any_selected_copy(completed, monkeypatch):
    from slm_lab import joint_selected_native_state_v1 as exporter
    view, optimizer = fresh(completed); before = state(view, optimizer)
    sentinel = completed['sentinel']; original = sentinel.stat(); scan = exporter.scan
    def corrupt(*args, **kwargs):
        result = scan(*args, **kwargs)
        sentinel.write_bytes(b'edit')
        __import__('os').utime(sentinel, ns=(original.st_atime_ns, original.st_mtime_ns))
        return result
    monkeypatch.setattr(exporter, 'scan', corrupt)
    try:
        with pytest.raises(ValueError, match='Bound runtime/data changed'):
            run(completed, view, optimizer)
        assert state(view, optimizer) == before
    finally:
        sentinel.write_bytes(b'orig')
        __import__('os').utime(sentinel, ns=(original.st_atime_ns, original.st_mtime_ns))


def test_incorrect_task_Adam_clock_is_rejected_even_with_a_matching_in_memory_common_digest(completed, monkeypatch):
    from slm_lab import joint_selected_native_state_v1 as exporter
    view, optimizer = fresh(completed); before = state(view, optimizer); original = exporter.read_latest
    def wrong_clock(*args, **kwargs):
        shared, receipt, pointer = original(*args, **kwargs)
        shared = copy.deepcopy(shared); receipt = copy.deepcopy(receipt)
        next(iter(shared['common']['optimizer']['state'].values()))['step'].fill_(9)
        receipt['semantic_manifest']['actual_common_sha256'] = exporter.common_digest(shared['common'])
        return shared, receipt, pointer
    monkeypatch.setattr(exporter, 'read_latest', wrong_clock)
    with pytest.raises(ValueError, match='sparse Adam clock'):
        run(completed, view, optimizer)
    assert state(view, optimizer) == before


def test_wrong_original_fixed_critic_is_rejected_before_any_model_or_generator_copy(completed):
    view, optimizer = fresh(completed)
    with torch.no_grad():
        view.actor.base.network.value_head.weight.add_(.5)
    before = state(view, optimizer)
    with pytest.raises(ValueError, match='original own native initializer'):
        run(completed, view, optimizer)
    assert state(view, optimizer) == before
