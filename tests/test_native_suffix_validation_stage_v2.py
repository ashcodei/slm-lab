"""Actual recorded worker evidence, byte-check costs, and real own-small updates."""
import copy
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import time

import pytest
import torch

from slm_lab.native_suffix_train_data_v1 import NativeSuffixTrainData
from slm_lab.native_suffix_validation_stage_v2 import NativeSuffixValidationStageV2
from slm_lab.parallel_native_suffix_stage_v1 import ParallelNativeSuffixStage, record_semantics
from slm_lab.parallel_function_repair_stage_v1 import value_digest
from slm_lab.whole_human_sft_evidence_v1 import tensor_digest
from test_parallel_native_suffix_stage_v1 import ROOT, CONTROL, OPTIONS, SOURCES, student, optimizer

ACTUAL = ROOT / 'reports/parallel-native-suffix-stage-v1-pytest/test_actual_single_replica_com0/actual-single-replica-native-recovery.json'


def make(public, profile='anchor_CE'):
    own = student(public)
    return NativeSuffixValidationStageV2(own, optimizer(own), public, [[0], [1], [2]], profile,
        OPTIONS, SOURCES, mode='mechanics')


@pytest.fixture(scope='module')
def controls():
    public = NativeSuffixTrainData.CPU_mechanics(ROOT, CONTROL)
    stage = make(public)
    old = ParallelNativeSuffixStage(stage.model, stage.optimizer, public, [[0], [1], [2]], 'anchor_CE',
        OPTIONS, SOURCES, mode='mechanics')
    row = json.loads(ACTUAL.read_text())['actual_first_group']
    return public, old, stage, row


def io():
    return {k:int(v) for k,v in [line.split(':') for line in Path('/proc/self/io').read_text().splitlines()]}


def measured(action):
    before = io(); wall = time.monotonic(); cpu = time.process_time()
    action()
    elapsed, CPU = time.monotonic() - wall, time.process_time() - cpu
    after = io()
    return dict(wall_seconds=elapsed, CPU_seconds=CPU, read_character_bytes=after['rchar']-before['rchar'],
        physical_read_bytes=after['read_bytes']-before['read_bytes'])


def test_actual_recorded_full_native_worker_observation_passes_both_validators_with_fewer_byte_reads(controls, tmp_path):
    public, old, stage, row = controls
    before = tensor_digest(stage.model.parts.named_parameters()), value_digest(stage.optimizer.state_dict())
    old_cost = measured(lambda: old.validate_record(row, 0))
    new_cost = measured(lambda: stage.validate_record(row, 0))
    assert new_cost['read_character_bytes'] < old_cost['read_character_bytes'] / 2
    assert (tensor_digest(stage.model.parts.named_parameters()), value_digest(stage.optimizer.state_dict())) == before
    assert stage._validation_owner is None and not torch.cuda.is_initialized()
    with stage.checked_validation_sources():
        prepared, binding = stage.validation_source(0)
    assert binding == row['actual_rank_observations'][0]['kernel']['episode_binding']
    assert prepared['original_response_EOS_targets'] == 68 and prepared['full_native_action_budget'] == 512
    (tmp_path / 'actual-recorded-source-validation-cost.json').write_text(json.dumps(dict(
        actual_record_file=str(ACTUAL), original_full_native_8_worker_group_passes_both_validators=True,
        old_validation=old_cost, new_validation=new_cost,
        observed_read_character_reduction=1-new_cost['read_character_bytes']/old_cost['read_character_bytes'],
        observed_wall_time_not_an_isolated_hardware_benchmark=True, actual_source_scope_and_every_CE_target_preserved=True,
        actual_original_episode_binding_reconstructed_exactly=True, optimizer_updates=0,
        full_size_GPUcapacity_or_model_quality=False), indent=2))


@pytest.mark.parametrize('case', ['targets', 'worker_program', 'control_proof', 'native_horizon', 'native_actions', 'nonfinite_logp'])
def test_forged_masks_programs_proofs_horizons_actions_or_logps_reject_without_mutating_live_state(controls, case):
    public, old, stage, row = controls
    bad = copy.deepcopy(row)
    kernel = bad['actual_rank_observations'][0]['kernel']
    if case == 'targets': bad['original_response_EOS_targets'] -= 1
    if case == 'worker_program':
        item = next(t for t in kernel['trajectories']
            if t['actual_native_execution'].get('actual_original_oracle_observation', {}).get('actual_execution'))
        raw = item['actual_native_execution']['actual_original_oracle_observation']['actual_execution']
        key = next(k for k in ['actual_program_sha256', 'original_program_sha256', 'program_sha256'] if k in raw)
        raw[key] = 'd' * 64
    if case == 'control_proof': kernel['episode_binding']['native_control_proof_sha256'] = 'e' * 64
    if case == 'native_horizon': kernel['full_original_native_horizon'] -= 1
    if case == 'native_actions': kernel['trajectories'][0]['actual_generated_token_ids'] = [0, 2]
    if case == 'nonfinite_logp': kernel['trajectories'][0]['actual_cached_behavior_logps'][0] = float('nan')
    before = tensor_digest(stage.model.parts.named_parameters()), value_digest(stage.optimizer.state_dict())
    with pytest.raises(ValueError): stage.validate_record(bad, 0)
    assert stage._validation_owner is None
    assert (tensor_digest(stage.model.parts.named_parameters()), value_digest(stage.optimizer.state_dict())) == before


def test_an_in_memory_source_descriptor_cannot_replace_the_actual_original_CE_mask(controls):
    public, old, stage, row = controls
    original = public._information[0]['original_full_CE_labels_sha256']
    public._information[0]['original_full_CE_labels_sha256'] = 'c' * 64
    try:
        with pytest.raises(ValueError, match='descriptor'): stage.validate_record(row, 0)
    finally: public._information[0]['original_full_CE_labels_sha256'] = original
    assert stage._validation_owner is None


def test_additional_real_control_source_bytes_are_checked_even_with_same_size_and_timestamp(controls, tmp_path):
    public, old, original_stage, row = controls
    extra = tmp_path / 'extra-closed-source.txt'
    original = b'original source bytes'
    extra.write_bytes(original)
    proof = json.loads(CONTROL.read_text())
    proof['source_files_sha256'][str(extra)] = hashlib.sha256(original).hexdigest()
    control = tmp_path / 'closed-actual-source-controls.json'
    control.write_text(json.dumps(proof, ensure_ascii=True, allow_nan=False) + '\n')
    copied = NativeSuffixTrainData.CPU_mechanics(ROOT, control)
    stage = make(copied)
    delivered = copy.deepcopy(row)
    delivered['actual_rank_observations'][0]['kernel']['episode_binding']['native_control_proof_sha256'] = hashlib.sha256(control.read_bytes()).hexdigest()
    stage.validate_record(delivered, 0)
    assert extra.resolve() in stage._extra_source_files
    stat = extra.stat()
    changed = b'changed! source bytes'
    assert len(changed) == len(original)
    extra.write_bytes(changed); os.utime(extra, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    try:
        with pytest.raises(ValueError, match='changed'): stage.validate_record(delivered, 0)
    finally:
        extra.write_bytes(original); os.utime(extra, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    assert stage._validation_owner is None and not stage.optimizer.state


def test_an_unrelated_thread_cannot_borrow_an_active_source_validation_batch(controls):
    public, old, stage, row = controls
    def borrow():
        with stage.checked_validation_sources(): pass
    with stage.checked_validation_sources():
        with ThreadPoolExecutor(max_workers=1) as pool:
            with pytest.raises(ValueError, match='thread'): pool.submit(borrow).result()
    assert stage._validation_owner is None


@pytest.mark.parametrize('profile', ['anchor_CE', 'anchor_execution'])
def test_actual_new_stage_updates_and_exact_next_native_actions_worker_outcomes_and_AdamW_recovery(profile, tmp_path):
    public = NativeSuffixTrainData.CPU_mechanics(ROOT, CONTROL)
    stage = make(public, profile)
    fixed = tensor_digest(stage.model.base.named_parameters())
    stage.update()
    saved, receipt = stage.snapshot()
    next_row = stage.update()
    wanted = tensor_digest(stage.model.parts.named_parameters()), value_digest(stage.optimizer.state_dict())
    stage.resume(saved, expected_manifest=receipt)
    replay = stage.update()
    assert record_semantics(replay) == record_semantics(next_row)
    assert (tensor_digest(stage.model.parts.named_parameters()), value_digest(stage.optimizer.state_dict())) == wanted
    assert tensor_digest(stage.model.base.named_parameters()) == fixed and stage._validation_owner is None
    assert stage.completed == 2 and len(stage.optimizer.state) == 17
    assert all(float(v['step']) == 2 for v in stage.optimizer.state.values())
    assert not torch.cuda.is_initialized()
    (tmp_path / ('actual-v2-native-recovery-' + profile + '.json')).write_text(json.dumps(dict(
        profile=profile, actual_small_parameters=stage.model.counts()['actual_unique'], physical_optimizer_calls=3,
        logical_complete_source_updates=2, original_human_sources=1, original_response_EOS_targets=136,
        actual_next_group=next_row, actual_replayed_group=replay,
        exact_next_actions_actual_worker_semantics_logps_gradients_weights_AdamW=True,
        unchanged_original_base=True, explicit_fixture_horizon=5, original_native_horizon_unchanged=512,
        full_size_GPUcapacity_or_model_quality=False, production_updates=0), indent=2, allow_nan=False))
