"""Source admission and actual normal waiting; no production/model proof."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

from slm_lab.public_reasoned_code_owned_recipe_v1 import prepare,sha
from slm_lab.unfused_policy_assessment_binding_v1 import bind,SUITES

ROOT=Path(__file__).resolve().parents[1]
DRIVER=ROOT/'scripts/train_public_reasoned_code_stopping_owned_GPU_v1.py'


def emit(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))


def identity(pid):
    return dict(pid=pid,birth=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[19])


def envelopes(root):
    data=root/'source-plan';delivery=root/'delivery'
    data_owner=dict(pid=99999991,birth='0');delivery_owner=dict(pid=99999992,ticks='0')
    emit(data/'owner-plan.json',dict(owner=data_owner,controlled_metadata_fixture_only=True))
    emit(delivery/'waiter-plan.json',dict(owner=delivery_owner,controlled_metadata_fixture_only=True))
    budget=dict(updates=2,source_presentations=19,complete_context_tokens=200,response_EOS_targets=100,
        view_presentations=dict(code_only=10,reasoning=9))
    capacity=dict(format='own-public-reasoned-code-complete-capacity-source-selection-v1',complete_source_selection=True,
        mechanics_only=False,source_selection_only=True,full_size_capacity_passed=False,
        all_four_replicas_receive_each_selected_source_in_declared_plan=True,complete_training_budget=budget,
        epochs=1,width=16,seed=990231,complete_training_schedule_identity='c'*64,original_views=['code_only','reasoning'],
        dataset_manifest_sha256='a'*64,native_admission_sha256='b'*64)
    emit(data/'complete-source-capacity-plan.json',capacity)
    emit(data/'complete.json',dict(owner=data_owner,complete=True,all_original_train_code_reasoning_and_EOS_targets_preserved=True,
        original_tail_not_rounded=True,common_complete_source_schedule_for_both_profiles=True,reserved_optimization_forbidden=True,
        public_synthetic_data_not_human_RL_feedback=True,production_training_admitted=False,complete_original_budget=budget,
        complete_schedule_identity='c'*64,actual_complete_profiles=dict(plain_tokens=dict(targets=100,EOS=19,weighted_targets=100),
            eos_weight4=dict(targets=100,EOS=19,weighted_targets=157)),source_files_sha256={},
        output_files_sha256={str(data/'complete-source-capacity-plan.json'):sha(data/'complete-source-capacity-plan.json')}))
    publication=root/'checkpoints/native-policy-reward-standalone-v1';publication.mkdir(parents=True)
    (publication/'model.pt').write_bytes(b'Controlled metadata fixture. Not model weights; cannot pass real package inspection.')
    (publication/'tokenizer.json').write_bytes(b'Controlled metadata tokenizer fixture.')
    files={name:sha(publication/name) for name in ['model.pt','tokenizer.json']}
    emit(publication/'manifest.json',dict(format='own-standalone-policy-and-preserved-human-reward-delivery-v1',
        actual_registered_parameters=999706442,trained_reward_feature_basis_preserved=True,
        old_checkpoint_or_features_needed_for_inference=False,files_sha256=files,sources=dict(tokenizer_sha256=files['tokenizer.json'])))
    emit(delivery/'complete.json',dict(owner=delivery_owner,complete=True,actual_registered_parameters=999706442,
        deployed_policy_and_original_trained_shared_reward_preserved=True,old_checkpoints_or_training_features_not_needed_for_inference=True,
        original_four_L40PPO_and_full_assessment_artifacts_unchanged=True,published_folder=str(publication)))
    emit(delivery/'export-complete.json',dict(complete=True,actual_registered_parameters=999706442,
        all_policy_reward_and_auxiliary_values_and_original_full_outputs_CPUcold_exact=True,separate_actual_durable_CLI_completed=True,
        package_files_sha256=files))
    emit(delivery/'plan.json',dict(expected_completed_updates=10724,source_files_sha256={},
        PPO_plan=dict(cold_inputs={},cached_own_reference={},pulse_sources={})))
    kwargs=dict(data_plan=data,delivery=delivery,expected_data_owner=data_owner,expected_delivery_owner=delivery_owner,runtime_sources={})
    return kwargs,publication


def test_controlled_metadata_recipe_preserves_complete_profiles_budgets_counts_and_immutable_bytes(tmp_path):
    kwargs,publication=envelopes(tmp_path)
    recipe=prepare(tmp_path,**kwargs)
    assert recipe['profiles']==['plain_tokens','eos_weight4']
    assert recipe['complete_training_budget']['source_presentations']==19
    assert recipe['complete_training_budget']['response_EOS_targets']==100
    assert recipe['stage_options']['width']==16 and recipe['stage_options']['epochs']==1
    assert recipe['original_policy_adapters_critic_reward_and_multimodal_values_must_survive']
    assert recipe['actual_GPU_capacity_training_and_full_assessments_pending']
    (publication/'tokenizer.json').write_bytes(b'changed')
    with pytest.raises(ValueError,match='byte'):prepare(tmp_path,**kwargs)


def test_live_predecessor_and_terminal_partial_or_replaced_owner_cannot_admit_training(tmp_path):
    data=tmp_path/'data';delivery=tmp_path/'delivery';me=identity(os.getpid())
    emit(data/'owner-plan.json',dict(owner=me));emit(delivery/'waiter-plan.json',dict(owner=me))
    kwargs=dict(data_plan=data,delivery=delivery,expected_data_owner=me,expected_delivery_owner=me,runtime_sources={})
    assert prepare(tmp_path,**kwargs) is None
    wrong=dict(me,birth=str(int(me['birth'])+1))
    with pytest.raises(ValueError,match='exact'):prepare(tmp_path,**dict(kwargs,expected_data_owner=wrong))
    terminal=dict(pid=99999991,birth='0');emit(data/'owner-plan.json',dict(owner=terminal))
    with pytest.raises(ValueError,match='without complete'):prepare(tmp_path,**dict(kwargs,expected_data_owner=terminal))


def test_incomplete_original_scope_or_wrong_profile_denominator_is_rejected(tmp_path):
    kwargs,_=envelopes(tmp_path)
    path=kwargs['data_plan']/'complete.json';original=json.loads(path.read_text())
    for field in ['reserved_optimization_forbidden','original_tail_not_rounded']:
        changed=copy.deepcopy(original);changed[field]=False;emit(path,changed)
        with pytest.raises(ValueError):prepare(tmp_path,**kwargs)
    changed=copy.deepcopy(original);changed['actual_complete_profiles']['eos_weight4']['weighted_targets']=156;emit(path,changed)
    with pytest.raises(ValueError,match='budgets'):prepare(tmp_path,**kwargs)


def test_real_normal_waiting_process_has_no_Torch_model_memory_probe_GPU_lease_or_worker(tmp_path):
    data=tmp_path/'data';delivery=tmp_path/'delivery';me=identity(os.getpid())
    emit(data/'owner-plan.json',dict(owner=me));emit(delivery/'waiter-plan.json',dict(owner=me))
    workspace=tmp_path/'owned';forbidden=tmp_path/'forbidden';log=tmp_path/'observer.log'
    wrapper='\n'.join(['import argparse,importlib.util','from pathlib import Path',
        'spec=importlib.util.spec_from_file_location("new_owned_driver",'+repr(str(DRIVER))+')',
        'm=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)',
        'def forbidden(*args,**kwargs):',
        '    Path('+repr(str(forbidden))+').write_text("forbidden early resource or worker work")',
        '    raise RuntimeError("No neural/resource work before exact full source completion")',
        'm.memory_available=forbidden;m.subprocess.Popen=forbidden;m.configure=forbidden',
        'm.parent(argparse.Namespace(workspace=Path('+repr(str(workspace))+'),data_plan=Path('+repr(str(data))+'),delivery=Path('+repr(str(delivery))+')))'])
    with log.open('wb') as stream:
        child=subprocess.Popen([sys.executable,'-u','-c',wrapper],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,
            env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'})
        try:
            deadline=time.monotonic()+20
            while not (workspace/'reports/waiting-progress.json').is_file():
                if child.poll() is not None or time.monotonic()>deadline:pytest.fail(log.read_text())
                time.sleep(.05)
            plan=json.loads((workspace/'reports/waiter-plan.json').read_text())
            assert child.poll() is None and not forbidden.exists()
            assert plan['exact_data_plan_owner']==plan['exact_delivery_owner']==me
            assert plan['GPU_contexts']==plan['GPU_leases']==plan['new_training_updates']==0
            assert '/torch/' not in Path(f'/proc/{child.pid}/maps').read_text()
            assert os.getpriority(os.PRIO_PROCESS,child.pid)==0 and os.sched_getscheduler(child.pid)==os.SCHED_OTHER
            assert not (workspace/'checkpoints').exists() and not (workspace/'reports/children.json').exists()
            for fd in Path(f'/proc/{child.pid}/fd').iterdir():
                try:assert 'nvidia' not in os.readlink(fd).lower()
                except FileNotFoundError:pass
        finally:
            if child.poll() is None:child.terminate()
            child.wait(timeout=5)


def test_unfused_binding_preserves_original_worker_objects_and_restores_on_failure():
    old=object();loader=object();registered=object();worker=lambda:None;reconcile=lambda:None
    evaluator=SimpleNamespace(REPORTS=old,load_student=loader,legacy=SimpleNamespace(registered=registered),worker=worker,reconcile_suite=reconcile)
    plan=dict(suites=list(SUITES),expected_student_values=999706442,expected_native_base_values=984399690,
        policy_factors_not_merged=True,owner=dict(pid=1,birth='2'))
    new_loader=lambda:None
    with pytest.raises(RuntimeError):
        with bind(evaluator,reports='new-output',plan=plan,load_student=new_loader,register=lambda *a:plan) as functions:
            assert functions==(worker,reconcile) and evaluator.load_student is new_loader
            assert evaluator.legacy.registered('test',0) is plan
            raise RuntimeError('controlled assessment failure')
    assert evaluator.REPORTS is old and evaluator.load_student is loader and evaluator.legacy.registered is registered
    with pytest.raises(ValueError):
        with bind(evaluator,reports='bad',plan=dict(plan,expected_student_values=984399690),load_student=new_loader,register=lambda *a:plan):pass
