"""Allocated journal/state controls, including actual four-replica neural work."""
import copy
from datetime import timedelta
import json
import math
import os
from pathlib import Path
import random

import numpy as np
import pytest
import torch
from torch import distributed as dist
from torch.multiprocessing import spawn

from slm_lab.codenet_checkpoint_v1 import capture_rng,restore_rng
from slm_lab.frozen_own_text_reference_v1 import FrozenOwnTextReferenceV1
from slm_lab.indexed_native_history_v2 import read_failure,observation_digest
from slm_lab.joint_indexed_history_v1 import Writer,Reader,JointRecords
from slm_lab.joint_indexed_training_stage_v1 import JointIndexedStageV1
from slm_lab.joint_objective_step_v1 import JointReplicaGuardV1
from slm_lab.joint_objective_step_v2 import update
from slm_lab.mixed_coding_SFT_data_v1 import MixedCodingSFTData
from slm_lab.parallel_function_repair_stage_v1 import value_digest
from slm_lab.whole_human_sft_evidence_v1 import tensor_digest
from test_joint_objective_step_v1 import ROOT,Trial,visual_packet,check_native
from test_joint_task_gradients_v1 import view

RESERVE=40*1024**3


def budgets(records=4):
    return dict(maximum_records=records,maximum_compact_record_bytes=1024**2,maximum_expanded_record_bytes=1024**2,
        maximum_logical_bytes=16*1024**2,maximum_physical_bytes=16*1024**2,minimum_free_bytes=RESERVE)


def binding():
    return dict(sources={'synthetic_storage_control':'a'*64},model_metadata={},own_text_reference={},options={},world_size=1,optimizer_settings_sha256='b'*64)


def test_allocated_joint_records_reconstruct_full_values_and_independent_failure_survives_bad_guard(tmp_path):
    state={'bad':False}
    def guard():
        if state['bad']:raise ValueError('changed actual source')
    writer=Writer(tmp_path/'attempt',reference_binding=binding(),starting_completed=0,budgets=budgets(),guard=guard)
    reader=Reader(writer.root,expected_header_sha256=writer.header_sha256,maximum_compact_record_bytes=1024**2,maximum_expanded_record_bytes=1024**2,guard=guard)
    original=dict(step=1,all_original_observations={'values':[1,.25,None,True],'negative':-1.5})
    allocated=writer.reserve_next();receipt=writer.append(original)
    assert reader.compact(receipt)==original and allocated['success_allocated_bytes']>2*1024**2
    writer.acknowledge_success(1);writer.reserve_next();state['bad']=True
    with pytest.raises(ValueError,match='changed actual source'):writer.append(dict(step=2))
    raw=dict(actual_nonfinite_observation=float('nan'),tensor=torch.tensor([1.,float('inf')]),all_original_values=original)
    failed=writer.preserve_failure(raw)
    restored=read_failure(writer.root,failed,maximum_bytes=1024**2)['original_observation']
    assert observation_digest(restored)==observation_digest(raw) and reader.header['reference_binding']==binding()
    assert failed['successful_optimizer_commit_certified'] is False and writer.failed


@pytest.mark.parametrize('bad',[{'step':1,'value':(1,2)},{'step':1,'value':{1:'integer_key'}},{'step':1,'value':float('nan')}])
def test_original_joint_record_types_cannot_be_silently_converted(bad):
    with pytest.raises(ValueError):JointRecords(binding()).pack(bad)


@pytest.mark.parametrize('fault',['raise','RNG','record'])
def test_pre_Adam_observation_failure_never_calls_optimizer_and_clears_gradients(fault):
    current=view();optimizer=torch.optim.AdamW([p for _,p in current.named_selected()],lr=1e-4,foreach=False)
    guard=JointReplicaGuardV1(current,optimizer);before=tensor_digest(current.named_parameters())
    packet=visual_packet('native',0,1)
    from test_joint_objective_step_v1 import OPTIONS
    def corrupt(record):
        if fault=='raise':raise ValueError('deliberate failed durable publication')
        if fault=='RNG':torch.rand(())
        else:record['gradient_norm_before_clip']+=1
    with pytest.raises(ValueError,match='Coordinated joint checkpoint rejection'):
        update(current,None,optimizer,guard,visual=packet,options=OPTIONS,source_entry_sha256='a'*64,before_optimizer=corrupt)
    assert tensor_digest(current.named_parameters())==before and not optimizer.state and all(p.grad is None for p in current.parameters())
    assert not torch.cuda.is_initialized()


def actual_live(trial):
    return tensor_digest(trial.view.named_parameters()),value_digest(trial.optimizer.state_dict()),value_digest(capture_rng('cpu'))


def compute(trial,stage,probes,index,hook):
    probes.append([random.random(),float(np.random.random()),float(torch.rand(()))])
    kind=['photo','word','native',None][index]
    def metric(record):
        losses=record['global_normalized_task_losses'];array=([losses['code']] if kind else [])+[losses['human']]
        if kind=='word':
            rows=record['actual_all_rank_task_records'];array.extend([math.fsum(r['tasks']['visual']['local_native_CE_sum'] for r in rows)/6,
                math.fsum(r['tasks']['visual']['local_byte_normalized_CTC_sum'] for r in rows)/2])
        elif kind:array.append(losses['visual'])
        return dict(step=index+1,entry_sha256=value_digest(trial.entries[index]),active_parameter_names=record['actual_active_parameter_names'],
            normalized_task_losses=array,gradient_norm_before_clip=record['gradient_norm_before_clip'],globally_reduced_task_gradients_before_combining=True,
            gradient_interaction=dict(actual_blend=record['gradient_interaction'],actual_task_transports=record['actual_task_transports'],actual_all_rank_tasks=record['actual_all_rank_task_records']))
    record=update(trial.view,trial.reference,trial.optimizer,stage.guard,
        code=dict(data=trial.data,positions=trial.positions,profile='balanced_sources') if kind else None,
        human=dict(data=trial.human,positions=[0,1],DPO_weight=1),visual=visual_packet(kind,stage.rank,stage.world) if kind else None,
        options=stage.options,source_entry_sha256=value_digest(trial.entries[index]),before_optimizer=lambda record:hook(metric(record)))
    return metric(record)


def worker(rank,folder):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and os.nice(0)>=19
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    folder=Path(folder)
    dist.init_process_group('gloo',init_method='file://'+str(folder/'rendezvous'),rank=rank,world_size=4,timeout=timedelta(minutes=15))
    try:
        data=MixedCodingSFTData(ROOT);trial=Trial(data,'bf16',rank,4);baseline=Trial(data,'bf16',rank,4)
        trial.reference=FrozenOwnTextReferenceV1(trial.view.actor.base,sources={'actual_fixture_data_sha256':data.identity_sha256},tokenizer_sha256=trial.sources['tokenizer_sha256'])
        stage=JointIndexedStageV1(trial.view,trial.reference,trial.optimizer,trial.entries,trial.sources,trial.options,
            history_root=folder/'history',history_budgets=budgets(),source_guard=data.guard.check,maximum_attempts=3)
        random.seed(994100+rank);np.random.seed(994100+rank);torch.manual_seed(994100+rank)
        stage.open_history();probes=[];calls={'stage':0,'fault':False}
        original=trial.optimizer.step
        def observed_step(*args,**kwargs):
            assert stage.pending is not None and stage.latest_metric is not None
            assert value_digest(stage.reader.compact(stage.pending['record']))==value_digest(stage.latest_metric)
            calls['stage']+=1;result=original(*args,**kwargs)
            if rank==0 and calls['fault']:raise RuntimeError('actual root failure after AdamW')
            return result
        trial.optimizer.step=observed_step
        run=lambda:stage.step(lambda index,hook:compute(trial,stage,probes,index,hook))
        def compare_baseline():
            rng=capture_rng('cpu');probe=baseline.step();expected=actual_live(baseline);record=copy.deepcopy(baseline.metrics[-1]);restore_rng(rng,'cpu')
            actual=run()
            assert probes[-1]==probe and actual_live(trial)==expected and actual==record
            return actual
        compare_baseline();stage.commit(folder/'state',maximum_bytes=8*1024**2,minimum_free_bytes=RESERVE)
        expected_metric=compare_baseline();expected_state=actual_live(trial);expected_probe=probes[-1]
        stage.restore(folder/'state');stage.open_history();calls['fault']=True
        with pytest.raises(ValueError,match='actual root failure after AdamW'):run()
        assert stage.failed and stage.completed==1 and stage.last_failure is not None and all(p.grad is None for p in trial.view.parameters())
        failed_location=folder/'history/attempt-000002'
        failed=read_failure(failed_location,stage.last_failure,maximum_bytes=1024**2)
        assert len(failed['original_observation']['all_actual_rank_observations'])==4
        failure_sha=value_digest(stage.last_failure)
        stage.restore(folder/'state');stage.open_history();calls['fault']=False
        assert run()==expected_metric and actual_live(trial)==expected_state and probes[-1]==expected_probe
        compare_baseline();compare_baseline();stage.commit(folder/'state',maximum_bytes=8*1024**2,minimum_free_bytes=RESERVE)
        shared,_,_=__import__('slm_lab.joint_indexed_durable_state_v1',fromlist=['read_latest']).read_latest(folder/'state')
        assert 'metrics' not in shared['common'] and len(shared['common']['history_receipts'])==4
        assert (failed_location/'failure-step-000000000002.pt').exists() and (folder/'history/attempt-000001/record-000000000002.json.gz').exists()
        native=None
        if rank==0:
            trial.metrics=baseline.metrics;native=check_native(trial,folder/'standalone')
        result=dict(rank=rank,precision='bf16',actual_small_model_counts=trial.view.actor.counts(),actual_stage_physical_AdamW_calls=calls['stage'],
            actual_original_v1_comparison_AdamW_calls=4,actual_logical_fixture_updates=4,all_original_v1_weights_Adam_private_RNG_full_metrics_exact=True,
            actual_post_Adam_root_failure_and_full_fresh_attempt_recovery=True,all_pre_Adam_records_independently_read_on_every_actual_peer=True,
            failed_and_uncommitted_actual_records_preserved=True,actual_failed_receipt_sha256=failure_sha,
            final_all_parameters_sha256=tensor_digest(trial.view.named_parameters()),final_Adam_sha256=value_digest(trial.optimizer.state_dict()),
            checkpoint_retains_receipts_not_original_metric_copies=True,actual_native_export=native,full_size_GPUcapacity_training_or_quality_gain=False)
        (folder/('actual-rank-'+str(rank)+'.json')).write_text(json.dumps(result,indent=2))
        dist.barrier()
    finally:dist.destroy_process_group()


def test_actual_four_replica_original_joint_math_pre_Adam_journal_sparse_state_failure_recovery_and_native_export(tmp_path):
    spawn(worker,args=(str(tmp_path),),nprocs=4,join=True)
    rows=[json.loads((tmp_path/('actual-rank-'+str(rank)+'.json')).read_text()) for rank in range(4)]
    assert len({r['final_all_parameters_sha256'] for r in rows})==len({r['final_Adam_sha256'] for r in rows})==1
    assert sum(r['actual_stage_physical_AdamW_calls'] for r in rows)==24
    assert sum(r['actual_original_v1_comparison_AdamW_calls'] for r in rows)==16
