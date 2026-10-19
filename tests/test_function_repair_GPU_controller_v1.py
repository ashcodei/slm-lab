"""Small actual neural controller checks; no full-size/GPU capacity claim."""
import ast
import copy
import json
from pathlib import Path

import pytest
import torch

from slm_lab import function_repair_GPU_contract_v1 as contract
from slm_lab.corpus import file_hash
from slm_lab.parallel_function_repair_stage_v1 import ParallelFunctionRepairStage
from slm_lab.parallel_repair_experiment_v1 import prove_next_update,train
from slm_lab.parallel_repair_durable_state_v1 import read_latest
from test_parallel_function_repair_stage_v1 import setup


@pytest.mark.parametrize('profile',contract.PROFILES)
def test_actual_small_neural_capacity_replay_rejections_then_real_global_tail(tmp_path,profile):
    stage=setup(profile=profile);proof=prove_next_update(stage,tmp_path/'capacity',maximum_bytes=128*1024**2,minimum_free_bytes=0)
    assert proof['physical_optimizer_calls']==3 and proof['logical_updates']==2
    assert proof['exact_next_draws_global_update_actor_Adam_and_private_RNG']
    assert proof['invalid_source_and_private_RNG_rejected_before_mutation']
    stage.update();assert stage.completed==3
    saved,receipt,_=read_latest(tmp_path/'capacity');assert saved['completed']==receipt['completed']==1
    assert not torch.cuda.is_initialized()


def test_actual_neural_production_complete_journal_and_all_durable_generations(tmp_path):
    capacity=setup();prove_next_update(capacity,tmp_path/'capacity',maximum_bytes=128*1024**2,minimum_free_bytes=0)
    fresh=setup();stage=ParallelFunctionRepairStage(fresh.model,fresh.optimizer,fresh.data,fresh.entries,fresh.profile,
        fresh.options,fresh.sources,fresh.weights,mode='train',precision='bf16')
    seen=[];preflights=[]
    result=train(stage,tmp_path/'production',tmp_path/'metrics.jsonl',maximum_bytes=128*1024**2,minimum_free_bytes=0,every=2,
        preflight=lambda:preflights.append(stage.completed),progress=lambda row,seconds:seen.append(copy.deepcopy(row)))
    assert result['actual_complete_updates']==3 and preflights==[0,1,2]
    assert json.loads((tmp_path/'production/LATEST.json').read_text())['generation']==3
    assert [json.loads(line) for line in (tmp_path/'metrics.jsonl').read_text().splitlines()]==seen==stage.metrics
    saved,receipt,_=read_latest(tmp_path/'production')
    assert saved['mode']=='train' and saved['completed']==receipt['completed']==3
    with pytest.raises(ValueError):train(stage,tmp_path/'production',tmp_path/'duplicate.jsonl',maximum_bytes=128*1024**2,
        minimum_free_bytes=0,every=2,preflight=lambda:None,progress=lambda *a:None)
    assert not (tmp_path/'duplicate.jsonl').exists() and not torch.cuda.is_initialized()


def write(root,path,value):
    p=root/path;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value));return p


def test_full_public_source_and_original_schedule_gates_before_GPU_factory(tmp_path,monkeypatch):
    birth={'pid':42,'ticks':'4242'};monkeypatch.setattr(contract,'terminal',lambda _:False)
    assert contract.public_ready(tmp_path,birth) is False
    monkeypatch.setattr(contract,'terminal',lambda _:True)
    with pytest.raises(ValueError):contract.public_ready(tmp_path,birth)
    public=Path(contract.PUBLIC);runtime=Path(contract.RUNTIME)
    plan=dict(planned_steps=4447,all_original_raw_train_windows_once=True,all826verified_original_train_functions_once=True,
        all_previous_selected_source_presentations_preserved=True)
    p=write(tmp_path,public/'plan.json',plan)
    write(tmp_path,runtime/'plan.json',{'owner':birth});write(tmp_path,runtime/'children.json',[])
    write(tmp_path,runtime/'complete.json',dict(original_scientific_job_complete=True,original_neural_plan_sha256=file_hash(p)))
    write(tmp_path,runtime/'restored.json',dict(all_registered_point_actors_terminal=True))
    write(tmp_path,public/'restored.json',dict(actual_recovery_owner=birth))
    write(tmp_path,public/'training-complete.json',dict(actual_updates=4447,all_new_admitted_raw_train_windows_once=True,all826verified_train_functions_once=True))
    write(tmp_path,public/'complete.json',dict(full_public_pretraining_source_epoch_complete=True,original_local_code_and_broad_assessments_complete=True))
    path=tmp_path/contract.MODEL;path.parent.mkdir(parents=True);path.write_bytes(b'fixture-only-not-a-model')
    assert contract.public_ready(tmp_path,birth)
    write(tmp_path,public/'training-complete.json',dict(actual_updates=4446,all_new_admitted_raw_train_windows_once=True,all826verified_train_functions_once=True))
    with pytest.raises(ValueError):contract.public_ready(tmp_path,birth)
    root=Path(__file__).resolve().parents[1]
    prepared=json.loads((root/'reports/full-function-repair-comparison-sources-CPU-v1/plan.json').read_text())
    entries=prepared['actual_complete_slot_batches'];selected=contract.capacity_entries(entries)
    assert selected[:2]==entries[:2] and len(selected[-1])==1
    with pytest.raises(ValueError):contract.capacity_entries(entries[:-1])
    tree=ast.parse((root/'scripts/train_full_function_repair_comparison_GPU_v1.py').read_text())
    parent=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='parent')
    loop=next(n for n in parent.body if isinstance(n,ast.While))
    assert 'handoff_admitted' in ast.unparse(loop) and 'public_ready' in ast.unparse(loop)
    assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='factory' for n in ast.walk(loop))
    assert next(n.lineno for n in ast.walk(parent) if isinstance(n,ast.With) and 'owner.LEASE' in ast.unparse(n.items[0]))>loop.end_lineno


def test_handoff_admission_requires_all_ten_actual_live_births_and_immutable_sources(tmp_path,monkeypatch):
    birth={'pid':42,'ticks':'4242'};monkeypatch.setattr(contract,'terminal',lambda _:False)
    assert not contract.handoff_admitted(tmp_path,birth)
    controllers=[]
    for i in range(10):
        owner={'pid':100+i,'ticks':str(1000+i)};queue='reports/new-'+str(i)+'.json'
        write(tmp_path,queue,dict(owner=owner));controllers.append(dict(owner=owner,queue=queue))
    source=tmp_path/'source.py';source.write_text('# exact tested operational source\n')
    done=dict(complete=True,GPU_study_owner=birth,old_ten_waiting_owners_retired=True,
        all_original_neural_functions_and_benchmark_populations_preserved=True,new_controllers=controllers,
        source_files_sha256={str(source):file_hash(source)})
    write(tmp_path,contract.ADMISSION,done);assert contract.handoff_admitted(tmp_path,birth)
    write(tmp_path,controllers[-1]['queue'],dict(owner={'pid':200,'ticks':'changed'}))
    with pytest.raises(ValueError):contract.handoff_admitted(tmp_path,birth)
    write(tmp_path,controllers[-1]['queue'],dict(owner=controllers[-1]['owner']))
    source.write_text('# changed\n')
    with pytest.raises(ValueError):contract.handoff_admitted(tmp_path,birth)


def test_optional_failure_releases_followers_only_after_exact_owner_and_GPU_children_terminal(tmp_path,monkeypatch):
    birth={'pid':42,'ticks':'4242'};child={'pid':43,'ticks':'4343'}
    monkeypatch.setattr(contract,'terminal',lambda _:False)
    assert not contract.study_released(tmp_path)
    write(tmp_path,contract.QUEUE,dict(owner=birth));assert not contract.study_released(tmp_path)
    monkeypatch.setattr(contract,'terminal',lambda _:True);assert contract.study_released(tmp_path)
    write(tmp_path,Path(contract.REPORT)/'plan.json',dict(owner=birth))
    write(tmp_path,Path(contract.REPORT)/'children.json',[child])
    write(tmp_path,Path(contract.REPORT)/'failure.json',dict(error='fixture failure'))
    with pytest.raises(ValueError):contract.study_released(tmp_path)
    write(tmp_path,Path(contract.REPORT)/'restored.json',dict(all_registered_point_actors_terminal=True))
    monkeypatch.setattr(contract,'terminal',lambda identity:identity==birth)
    assert not contract.study_released(tmp_path)
    monkeypatch.setattr(contract,'terminal',lambda _:True);assert contract.study_released(tmp_path)
