import ast
import importlib
import json
import os
from pathlib import Path
import sys

import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from slm_lab.public_pretraining_endpoints_v1 import ENDPOINTS,COMPARISONS,descriptors,expected_values

GPU_DRIVERS=['train_phase_human_preference_v8','train_broad_anchored_code_v7','assess_full_held_QA_pipeline_v5',
    'assess_full_AI2D_pipeline_v5','assess_full_original_MBPP_pipeline_v5','train_full_machine_preference_v2','assess_machine_standalone_full_v2']
CPU_DRIVERS=['finalize_public_pretraining_student_CPU_v1','finalize_machine_preference_student_CPU_v2']


@pytest.mark.parametrize('name',GPU_DRIVERS)
def test_live_exact_predecessor_prevents_new_gpu_stage(name,monkeypatch):
    module=importlib.import_module(name)
    monkeypatch.setattr(module.owner.resources,'same',lambda identity:True)
    assert module.previous_ready(dict(pid=100,ticks='200')) is False


@pytest.mark.parametrize('name',CPU_DRIVERS)
def test_missing_complete_source_cannot_be_exported_or_substituted(name,tmp_path,monkeypatch):
    module=importlib.import_module(name)
    monkeypatch.setattr(module,'SOURCE',tmp_path)
    monkeypatch.setattr(module,'CHECKPOINT',tmp_path/'absent.pt')
    monkeypatch.setattr(module,'same',lambda identity:True)
    assert not module.source_ready(dict(pid=100,ticks='200'))
    monkeypatch.setattr(module,'same',lambda identity:False)
    with pytest.raises(ValueError):module.source_ready(dict(pid=100,ticks='200'))


def test_nine_explicit_endpoints_preserve_old_eight_and_add_pre_alignment_public_student(tmp_path):
    from slm_lab.code_first_endpoints_v3 import ENDPOINTS as previous
    assert set(ENDPOINTS)==set(previous)|{'public_pretrained'} and len(ENDPOINTS)==9
    specs=descriptors(tmp_path)
    assert specs['public_pretrained']['checkpoint'].endswith('public-pretraining-v1/trained/model.pt')
    assert specs['code_first_trained']['checkpoint'].endswith('code-first-continuation-v3/trained/model.pt')
    assert specs['human_DPO']['checkpoint'].endswith('phase-human-preference-v8/human_DPO-actor.pt')
    assert specs['code_anchor_execution']['checkpoint'].endswith('broad-anchored-code-v7/anchor_execution-actor.pt')
    assert ('public_pretrained','human_DPO') in COMPARISONS and ('code_first_trained','public_pretrained') in COMPARISONS
    assert [expected_values(specs[n]) for n in ENDPOINTS].count(984399690)==5


def put(folder,name,value):
    path=folder/name;path.parent.mkdir(exist_ok=True,parents=True);path.write_text(json.dumps(value))


def test_human_alignment_requires_complete_public_raw_functions_retention_and_assessments(tmp_path,monkeypatch):
    module=importlib.import_module(GPU_DRIVERS[0]);monkeypatch.setattr(module,'PREVIOUS',tmp_path)
    monkeypatch.setattr(module.owner.resources,'same',lambda identity:False)
    plan=dict(planned_steps=100,all_original_raw_train_windows_once=True,all826verified_original_train_functions_once=True,
        all_previous_selected_source_presentations_preserved=True)
    trained=dict(actual_updates=100,all_new_admitted_raw_train_windows_once=True,all826verified_train_functions_once=True)
    done=dict(full_public_pretraining_source_epoch_complete=True,original_local_code_and_broad_assessments_complete=True)
    for name,value in [('plan.json',plan),('training-complete.json',trained),('complete.json',done),('restored.json',{}),('children.json',[])]:put(tmp_path,name,value)
    identity=dict(pid=100,ticks='200');assert module.previous_ready(identity)
    trained['actual_updates']=99;put(tmp_path,'training-complete.json',trained)
    with pytest.raises(ValueError):module.previous_ready(identity)


def test_all_nine_full_QA_endpoints_and_all36_shards_are_required_before_visual_scoring(tmp_path,monkeypatch):
    module=importlib.import_module('assess_full_AI2D_pipeline_v5');monkeypatch.setattr(module,'PREVIOUS',tmp_path)
    monkeypatch.setattr(module.owner.resources,'same',lambda identity:False)
    put(tmp_path,'plan.json',dict(full_QA_questions_each_endpoint=16961,endpoints=descriptors(tmp_path)))
    put(tmp_path,'complete.json',dict(all_nine_predeclared_endpoints_assessed=True,original_questions_each_endpoint=16961,
        total_original_question_endpoint_assessments=9*16961,all_raw_deliveries_and_question_verdicts_retained=True))
    for name in ['children.json','restored.json','paired-comparisons.json']:put(tmp_path,name,[] if name=='children.json' else {})
    for endpoint in ENDPOINTS:
        put(tmp_path,endpoint+'/full-QA.json',{})
        for rank in range(4):put(tmp_path,f'{endpoint}/QA-shard-{rank}-complete.json',dict(endpoint=endpoint,physical_GPU=rank,
            actual_original_questions=4241 if rank==0 else 4240,all_loaded_values_unchanged=True,model_updates=0))
    identity=dict(pid=100,ticks='200');assert module.previous_ready(identity)
    (tmp_path/'public_pretrained/full-QA.json').unlink()
    with pytest.raises(ValueError):module.previous_ready(identity)


def test_all_nine_visual_endpoints_all_conditions_and36_shards_are_required_before_MBPP(tmp_path,monkeypatch):
    module=importlib.import_module('assess_full_original_MBPP_pipeline_v5');monkeypatch.setattr(module,'PREVIOUS',tmp_path)
    monkeypatch.setattr(module.owner.resources,'same',lambda identity:False)
    put(tmp_path,'plan.json',dict(endpoints=descriptors(tmp_path)))
    put(tmp_path,'complete.json',dict(all_nine_predeclared_endpoints_assessed=True,original_questions_each_endpoint=3088,
        total_question_endpoint_assessments=9*3088,total_question_endpoint_condition_assessments=9*3088*3,
        all_actual_native_pixel_identities_and_delivered_tokens_and_likelihoods_retained=True))
    put(tmp_path,'children.json',[]);put(tmp_path,'restored.json',{})
    for endpoint in ENDPOINTS:
        put(tmp_path,endpoint+'/full-AI2D.json',{})
        for rank in range(4):put(tmp_path,f'{endpoint}/AI2D-shard-{rank}-complete.json',dict(endpoint=endpoint,physical_GPU=rank,
            actual_original_questions=772,actual_question_condition_assessments=2316,all_loaded_values_unchanged=True,model_updates=0))
    identity=dict(pid=100,ticks='200');assert module.previous_ready(identity)
    put(tmp_path,'public_pretrained/AI2D-shard-3-complete.json',dict(endpoint='public_pretrained',physical_GPU=3,actual_original_questions=771))
    with pytest.raises(ValueError):module.previous_ready(identity)


def test_all_nine_full_MBPP_endpoints_and5400_deliveries_precede_machine_feedback(tmp_path,monkeypatch):
    module=importlib.import_module('train_full_machine_preference_v2');monkeypatch.setattr(module,'PREVIOUS',tmp_path)
    monkeypatch.setattr(module.owner.resources,'same',lambda identity:False)
    identity=dict(pid=100,ticks='200')
    put(tmp_path,'plan.json',dict(owner=identity,endpoints=descriptors(tmp_path)))
    done=dict(all_nine_fixed_endpoints_assessed=True,original_problems_each_endpoint=600,primary_publisher_test_problems=500,
        total_actual_original_problem_endpoint_deliveries=5400,all_original_cases_and_generated_actions_and_raw_values_retained=True,
        every_raw_value_verdict_rederived_without_source_eligibility_exclusions=True)
    put(tmp_path,'complete.json',done);put(tmp_path,'children.json',[]);put(tmp_path,'restored.json',{})
    for endpoint in ENDPOINTS:
        put(tmp_path,endpoint+'/full-original-MBPP.json',{})
        for rank in range(4):put(tmp_path,f'{endpoint}/MBPP-shard-{rank}-complete.json',dict(endpoint=endpoint,physical_GPU=rank,actual_original_problems=150,model_updates=0))
    assert module.previous_ready(identity)
    done['total_actual_original_problem_endpoint_deliveries']=4800;put(tmp_path,'complete.json',done)
    with pytest.raises(ValueError):module.previous_ready(identity)


def test_real_original_full_benchmark_populations_have_not_been_reduced():
    from slm_lab.full_held_QA_native_v1 import FullHeldQA
    from slm_lab.ai2d_full_native_evaluation_v1 import FullAI2D
    from slm_lab.MBPP_original_population_v2 import FullOriginalMBPP
    assert len(FullHeldQA(ROOT).cases)==16961
    assert len(FullAI2D(ROOT).cases)==3088
    assert len(FullOriginalMBPP(ROOT).rows)==600


def test_all_ten_new_drivers_and_every_reachable_local_source_exist_before_queue_handoff():
    from slm_lab.local_source_closure_v2 import python_sources
    mapping=json.loads((ROOT/'reports/public-pretraining-lineage-file-map-v1.json').read_text())['drivers']
    assert len(mapping)==10
    seeds=[ROOT/'scripts'/name for _,name in mapping]
    paths=python_sources(ROOT,seeds)
    assert all(p.is_file() for p in paths)
    for p in seeds:
        source=p.read_text();ast.parse(source)
        assert 'public-pretraining-lineage-drivers-v1-final2-tests.log' in source
        assert 'bind_tests(files,ROOT)' in source
    assert ROOT/'src/slm_lab/phase_human_policy_v3.py' in paths
    assert ROOT/'src/slm_lab/broad_anchored_policy_v3.py' in paths
    assert ROOT/'src/slm_lab/machine_preference_policy_v2.py' in paths
