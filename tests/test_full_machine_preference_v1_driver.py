import importlib.util
import json
import os
from pathlib import Path
import sys

import pytest
import torch


def driver():
    root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'scripts'))
    spec=importlib.util.spec_from_file_location('full_machine_driver_unit',root/'scripts/train_full_machine_preference_v1.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def save(folder,name,value):(folder/name).write_text(json.dumps(value))


def test_same_live_owner_or_any_live_registered_child_waits_without_preemption(tmp_path,monkeypatch):
    m=driver();m.PREVIOUS=tmp_path;identity=dict(pid=101,ticks='202')
    monkeypatch.setattr(m.owner.resources,'same',lambda p:dict(state='R'))
    assert m.previous_ready(identity) is False
    save(tmp_path,'children.json',[dict(pid=303,ticks='404')])
    monkeypatch.setattr(m.owner.resources,'same',lambda p:dict(state='R') if p['pid']==303 else None)
    assert m.previous_ready(identity) is False
    assert m.PREVIOUS_QUEUE.name=='full-original-MBPP-pipeline-v4-queue.json'
    assert m.CODE_QUEUE.name=='broad-anchored-code-v6-queue.json'


def test_exact_full_eight_endpoint_predecessor_and_every_original_shard_required(tmp_path,monkeypatch):
    m=driver();m.PREVIOUS=tmp_path;identity=dict(pid=101,ticks='202')
    monkeypatch.setattr(m.owner.resources,'same',lambda p:None)
    with pytest.raises(ValueError,match='complete successfully'):m.previous_ready(identity)
    from slm_lab.code_first_endpoints_v3 import ENDPOINTS
    save(tmp_path,'plan.json',dict(owner=identity,endpoints={e:{} for e in ENDPOINTS}));save(tmp_path,'restored.json',{})
    save(tmp_path,'complete.json',dict(all_eight_fixed_endpoints_assessed=True,original_problems_each_endpoint=600,
        primary_publisher_test_problems=500,total_actual_original_problem_endpoint_deliveries=4800,
        all_original_cases_and_generated_actions_and_raw_values_retained=True,
        every_raw_value_verdict_rederived_without_source_eligibility_exclusions=True))
    for e in ENDPOINTS:
        folder=tmp_path/e;folder.mkdir();save(folder,'full-original-MBPP.json',{})
        for rank in range(4):save(folder,f'MBPP-shard-{rank}-complete.json',dict(endpoint=e,physical_GPU=rank,actual_original_problems=150,model_updates=0))
    assert m.previous_ready(identity)
    path=tmp_path/ENDPOINTS[-1]/'MBPP-shard-3-complete.json';r=json.loads(path.read_text());r['actual_original_problems']=149;path.write_text(json.dumps(r))
    with pytest.raises(ValueError,match='Every original four-worker'):m.previous_ready(identity)
    r['actual_original_problems']=150;path.write_text(json.dumps(r));save(tmp_path,'failure.json',{})
    with pytest.raises(ValueError,match='complete successfully'):m.previous_ready(identity)


def test_complete_own_code_RL_start_needs_both_full_source_epochs_and_four_private_states(tmp_path,monkeypatch):
    m=driver();m.CODE=tmp_path;identity=dict(pid=101,ticks='202');monkeypatch.setattr(m.owner.resources,'same',lambda p:None)
    save(tmp_path,'plan.json',dict(owner=identity,original_coding_sources=3721,planned_steps=3721));save(tmp_path,'restored.json',{})
    save(tmp_path,'complete.json',dict(all_3721_original_coding_sources_once_each_arm=True,
        matched_native_code_broad_human_preference_and_held_APPS_assessments_complete=True))
    for arm in ['anchor_CE','anchor_execution']:
        save(tmp_path,arm+'-training-complete.json',dict(actual_updates=3721,all_3721_original_coding_sources_once=True))
        save(tmp_path,'train-'+arm+'-resume-distributed-state.json',dict(completed=3721,
            all_four_private_states=[dict(rank=r,actual_completed=3721) for r in range(4)]))
    assert m.code_ready(identity)
    path=tmp_path/'train-anchor_execution-resume-distributed-state.json';r=json.loads(path.read_text());r['all_four_private_states'][3]['actual_completed']=3720
    path.write_text(json.dumps(r))
    with pytest.raises(ValueError,match='All four private'):m.code_ready(identity)
    save(tmp_path,'plan.json',dict(owner=dict(pid=999,ticks='202'),original_coding_sources=3721,planned_steps=3721))
    with pytest.raises(ValueError,match='exact predeclared owner'):m.code_ready(identity)


def test_actual1620pair_full_epoch_global16_tail_and_full_original_capacity_extremes():
    from slm_lab.execution_contrast_data_v1 import ExecutionContrastData
    from slm_lab.frozen_own_actor_reference_v1 import FrozenOwnActorReference
    from slm_lab.parallel_machine_preference_v1 import ParallelMachinePreferenceStage
    from test_machine_code_preference_stage_v1 import trained,SOURCES
    m=driver();root=Path(__file__).resolve().parents[1];data=ExecutionContrastData(root,purpose='optimization')
    model=trained();ref=FrozenOwnActorReference(model,sources=SOURCES);o=dict(m.training_options(),method='machine_DPO')
    opt=torch.optim.AdamW([p for _,p in model.named_selected(('actor',))],lr=o['learning_rate'],
        betas=tuple(o['betas']),weight_decay=o['weight_decay'],foreach=False)
    stage=ParallelMachinePreferenceStage(model,opt,data,ref,o,SOURCES)
    batches=[stage.positions(step) for step in range(stage.planned)]
    assert len(data)==1620 and stage.planned==102
    assert sorted(p for batch in batches for p in batch)==list(range(1620))
    assert [len(batches[-1][rank::4]) for rank in range(4)]==[1,1,1,1]
    assert stage.cursor(102)['actual_pair_draws']==1620 and stage.cursor(102)['repeated_rounding_pairs']==0
    extremes=m.extreme_batches(data)
    assert all(len(set(batch))==16 for batch in extremes)
    for field in ['chosen','rejected']:
        longest=max(range(len(data)),key=lambda p:(len(data.sequence(p,field)[0]),p))
        assert any(longest in batch for batch in extremes)
    assert m.ARMS==('chosen_CE','machine_DPO')
    assert set(m.SUITES)=={'code','broad','preferences','QA','AI2D','MBPP'}


@pytest.mark.parametrize('suite',['QA','AI2D','MBPP'])
def test_full_assessment_dispatch_preserves_native_population_rank_and_delivery_journal(tmp_path,monkeypatch,suite):
    from contextlib import nullcontext
    from types import SimpleNamespace
    from slm_lab import machine_preference_policy_v1,full_held_QA_native_v1,ai2d_full_native_evaluation_v1
    from slm_lab import MBPP_original_population_v2,MBPP_all_native_evaluation_v1,inference_rotary_cache_v1
    from test_machine_code_preference_stage_v1 import trained
    m=driver();m.REPORTS=tmp_path;m.OUTPUT=tmp_path/'outputs'
    model=trained().eval();plan=dict(source_files_sha256={'fixture':'a'*64});calls=[]
    monkeypatch.setattr(m.legacy,'registered',lambda *a:plan)
    monkeypatch.setattr(m.owner,'owned',lambda p:None)
    monkeypatch.setattr(m,'FileBindings',lambda p:SimpleNamespace(check=lambda **kw:None))
    monkeypatch.setattr(m,'file_hash',lambda p:'a'*64)
    monkeypatch.setattr(machine_preference_policy_v1,'load',lambda *a,**kw:(model,dict(tokenizer_sha256='b'*64)))
    monkeypatch.setattr(torch.cuda,'set_device',lambda *a:None)
    monkeypatch.setattr(torch.cuda,'set_per_process_memory_fraction',lambda *a:None)
    monkeypatch.setattr(torch.cuda,'get_device_name',lambda *a:'NVIDIA L40')
    monkeypatch.setattr(inference_rotary_cache_v1,'inference_cache',nullcontext)
    population=SimpleNamespace(tokenizer_sha256='b'*64,tokenizer=object())
    if suite=='QA':module=full_held_QA_native_v1;name='FullHeldQA'
    elif suite=='AI2D':module=ai2d_full_native_evaluation_v1;name='FullAI2D'
    else:module=MBPP_original_population_v2;name='FullOriginalMBPP'
    monkeypatch.setattr(module,name,lambda root:population)
    def evaluate(*args,**kwargs):
        calls.append((args,kwargs));kwargs['checked']();kwargs['record'](dict(actual_fixture_delivery=True));return dict(results=[{}])
    monkeypatch.setattr(MBPP_all_native_evaluation_v1 if suite=='MBPP' else module,'evaluate_shard',evaluate)
    m.assess('machine_DPO',2,suite)
    args,kwargs=calls[0];assert args[0] is model.policy and kwargs['rank']==2 and kwargs['world']==4
    assert args[1] is (population if suite=='MBPP' else population.tokenizer)
    assert (tmp_path/'machine_DPO'/f'{suite}-shard-2-actual-deliveries.jsonl').read_text().count('\n')==1
    receipt=json.loads((tmp_path/'machine_DPO'/f'assessment-{suite}-2-complete.json').read_text())
    assert receipt['all_loaded_values_unchanged'] and receipt['model_updates']==0
