"""Real neural checkpoint equivalence and full final-source corruption guard."""
import os
from pathlib import Path
import random

import numpy as np
import pytest
import torch

from slm_lab import corpus
from slm_lab import joint_indexed_checkpoint_v2 as private
from slm_lab.frozen_own_text_reference_v1 import FrozenOwnTextReferenceV1
from slm_lab.joint_source_checked_training_stage_v2 import JointSourceCheckedStageV2
from slm_lab.mixed_coding_SFT_data_v1 import MixedCodingSFTData
from slm_lab.parallel_function_repair_stage_v1 import value_digest
from test_joint_indexed_training_stage_v1 import budgets,compute,actual_live,RESERVE
from test_joint_objective_step_v1 import ROOT,Trial


def test_batched_source_checkpoint_exact_replay_and_same_size_same_mtime_rejection_before_mutation(tmp_path,monkeypatch):
    torch.set_num_threads(1);data=MixedCodingSFTData(ROOT);trial=Trial(data,'bf16')
    trial.reference=FrozenOwnTextReferenceV1(trial.view.actor.base,sources={'actual_fixture_data_sha256':data.identity_sha256},tokenizer_sha256=trial.sources['tokenizer_sha256'])
    sentinel=tmp_path/'original-source-control.bin';sentinel.write_bytes(b'abcd');original_stat=sentinel.stat()
    files=dict(data.source_files_sha256);files[str(sentinel)]=corpus.file_hash(sentinel)
    stage=JointSourceCheckedStageV2(trial.view,trial.reference,trial.optimizer,trial.entries,trial.sources,trial.options,
        history_root=tmp_path/'history',history_budgets=budgets(),source_guard=data.guard.check,source_files=files,maximum_attempts=3)
    stage.open_history();probes=[]
    run=lambda:stage.step(lambda index,hook:compute(trial,stage,probes,index,hook))
    random.seed(996100);np.random.seed(996100);torch.manual_seed(996100)
    run();hashes=[];resources=[];original_hash=corpus.file_hash;original_resource=stage.resource_guard
    def counted_resource():
        resources.append(True);original_resource()
    stage.resource_guard=counted_resource
    def counted(path):
        if Path(path)==sentinel:hashes.append(str(path))
        return original_hash(path)
    monkeypatch.setattr(corpus,'file_hash',counted)
    stage.commit(tmp_path/'state',maximum_bytes=8*1024**2,minimum_free_bytes=RESERVE)
    assert len(hashes)==2 and len(resources)==2
    expected_metric=run();expected=actual_live(trial);expected_probe=probes[-1]
    original_scan=private.scan
    def corrupt_after_record_validation(*args,**kwargs):
        result=original_scan(*args,**kwargs)
        sentinel.write_bytes(b'efgh');os.utime(sentinel,ns=(original_stat.st_atime_ns,original_stat.st_mtime_ns))
        return result
    monkeypatch.setattr(private,'scan',corrupt_after_record_validation)
    with pytest.raises(ValueError,match='Bound runtime/data changed'):
        stage.restore(tmp_path/'state')
    assert actual_live(trial)==expected and sentinel.stat().st_mtime_ns==original_stat.st_mtime_ns
    sentinel.write_bytes(b'abcd');os.utime(sentinel,ns=(original_stat.st_atime_ns,original_stat.st_mtime_ns))
    original_sentinel_sha=stage.source_files[str(sentinel)]
    def corrupt_binding_after_records(*args,**kwargs):
        result=original_scan(*args,**kwargs);stage.source_files[str(sentinel)]='d'*64;return result
    monkeypatch.setattr(private,'scan',corrupt_binding_after_records)
    with pytest.raises(ValueError,match='original joint source/model/reference/history binding'):
        stage.restore(tmp_path/'state')
    assert actual_live(trial)==expected
    stage.source_files[str(sentinel)]=original_sentinel_sha
    monkeypatch.setattr(private,'scan',original_scan);hashes.clear();resources.clear()
    stage.restore(tmp_path/'state');assert len(hashes)==2 and len(resources)==2
    stage.open_history();assert run()==expected_metric and actual_live(trial)==expected and probes[-1]==expected_probe
    assert not torch.cuda.is_initialized()
    (tmp_path/'actual-source-checked-state-recovery.json').write_text(__import__('json').dumps(dict(
        actual_own_small_counts=trial.view.actor.counts(),actual_AdamW_calls=3,actual_logical_fixture_updates=2,
        exact_next_values_Adam_private_RNG_and_full_metric_replay=True,same_size_same_mtime_source_change_rejected_before_any_restore_mutation=True,
        complete_source_hash_passes_per_checkpoint_make=2,complete_source_hash_passes_per_restore=2,
        complete_resource_checks_per_source_only_batch=2,changed_in_memory_source_bindings_reject_before_restore_mutation=True,
        all_original_record_bytes_and_semantics_still_reconstructed=True,neural_source_guards_not_batched=True,full_size_GPU_capacity_or_speed_or_model_quality=False),indent=2))
