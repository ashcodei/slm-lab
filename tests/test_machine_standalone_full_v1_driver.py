import importlib.util
import json
import os
from pathlib import Path
import sys

import pytest
import torch


def driver():
    root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'scripts'))
    spec=importlib.util.spec_from_file_location('machine_standalone_full_unit',root/'scripts/assess_machine_standalone_full_v1.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def save(folder,name,value):(folder/name).write_text(json.dumps(value))


def test_live_original_owner_or_any_live_child_waits_without_preemption(tmp_path,monkeypatch):
    m=driver();m.PREVIOUS=tmp_path;identity=dict(pid=101,ticks='202')
    monkeypatch.setattr(m.owner.resources,'same',lambda p:dict(state='R'))
    assert not m.previous_ready(identity) and not m.finalizer_ready(identity)
    save(tmp_path,'children.json',[dict(pid=303,ticks='404')])
    monkeypatch.setattr(m.owner.resources,'same',lambda p:dict(state='R') if p['pid']==303 else None)
    assert not m.previous_ready(identity)


def test_both_source_epochs_all_original_populations_and_all_actor_shards_required(tmp_path,monkeypatch):
    m=driver();m.PREVIOUS=tmp_path;identity=dict(pid=101,ticks='202');monkeypatch.setattr(m.owner.resources,'same',lambda p:None)
    with pytest.raises(ValueError,match='exact complete'):m.previous_ready(identity)
    save(tmp_path,'plan.json',dict(owner=identity,original_machine_pairs=1620,planned_steps=102,arms=['chosen_CE','machine_DPO'],human_preferences=False))
    save(tmp_path,'restored.json',dict(all_registered_point_actors_terminal=True))
    save(tmp_path,'complete.json',dict(all1620original_machine_pairs_once_each_arm=True,
        full_original_QA16961_AI2D3088times3_MBPP600_and_human4642_each_endpoint_complete=True,
        matched_initial_and_both_training_arms_assessed=True,original_eight_endpoint_pipeline_preserved=True))
    for arm in ['initial','chosen_CE','machine_DPO']:
        folder=tmp_path/arm;folder.mkdir()
        for name,count in [('full-QA.json',16961),('full-AI2D.json',3088),('full-original-MBPP.json',600),('held-preferences.json',4642)]:save(folder,name,dict(results=[{}]*count))
        for suite in m.SOURCE_SUITES:
            for rank in range(4):save(folder,f'assessment-{suite}-{rank}-complete.json',dict(physical_GPU=rank,all_loaded_values_unchanged=True,model_updates=0,actual_counts=dict(actual_unique=999706442)))
    assert m.previous_ready(identity)
    save(tmp_path/'machine_DPO','full-original-MBPP.json',dict(results=[{}]*599))
    with pytest.raises(ValueError,match='All complete original'):m.previous_ready(identity)


def test_actual_standalone_completion_counts_and_text_photo_word_cold_receipt_required(tmp_path,monkeypatch):
    m=driver();m.FINALIZER=tmp_path;identity=dict(pid=101,ticks='202');monkeypatch.setattr(m.owner.resources,'same',lambda p:None)
    save(tmp_path,'plan.json',dict(owner=identity))
    done=dict(complete=True,completed_updates=102,actual_counts=dict(total_unique=984399690),
        all_actual_exported_parameters_and_complete_native_probes_cold_load_bit_exact=True,
        real_word_logits_exact_through_fold_and_cold_native=True,old_checkpoint_actor_optimizer_or_reference_not_needed_for_inference=True,
        fixed_machine_DPO_export_not_quality_winner=True,all_original_machine_source_updates_complete=True)
    save(tmp_path,'complete.json',done);assert m.finalizer_ready(identity)
    done['completed_updates']=101;save(tmp_path,'complete.json',done)
    with pytest.raises(ValueError,match='Every complete fixed'):m.finalizer_ready(identity)
    done['completed_updates']=102;done['real_word_logits_exact_through_fold_and_cold_native']=False;save(tmp_path,'complete.json',done)
    with pytest.raises(ValueError,match='Every complete fixed'):m.finalizer_ready(identity)


def student_fixture(tmp_path):
    from slm_lab.corpus import file_hash
    from slm_lab.own_dense_standalone_v1 import export
    from test_own_dense_standalone_v1 import fixture
    folder=tmp_path/'standalone';folder.mkdir();model,ids,_,_=fixture(folder)
    pixels=torch.linspace(-1,1,3*model.config.image_size**2).reshape(1,3,model.config.image_size,model.config.image_size)
    sources=dict(own_unit_source='a'*64);checkpoint=folder/'model.pt'
    torch.save(export(model,sources=sources,provenance=dict(small_synthetic_unit_fixture=True),completed_updates=102,
        tokenizer_sha256=file_hash(folder/'tokenizer.json')),checkpoint)
    receipt=tmp_path/'complete.json';save(tmp_path,'complete.json',dict(complete=True,completed_updates=102,
        checkpoint_sha256=file_hash(checkpoint),all_actual_exported_parameters_and_complete_native_probes_cold_load_bit_exact=True,standalone_sources=sources))
    plan=dict(standalone=dict(checkpoint=str(checkpoint),checkpoint_sha256=file_hash(checkpoint),receipt=str(receipt),receipt_sha256=file_hash(receipt)),
        expected_student_values=model.counts()['total_unique'],tokenizer_sha256=file_hash(folder/'tokenizer.json'))
    return model,ids,pixels,plan


def test_actual_small_standalone_cold_loading_needs_no_source_actor_and_counts_every_head(tmp_path):
    from slm_lab.whole_human_sft_evidence_v1 import tensor_digest
    m=driver();source,ids,pixels,plan=student_fixture(tmp_path)
    model,native,metadata=m.load_student(plan)
    assert tensor_digest(source.named_parameters())==tensor_digest(model.named_parameters())
    assert sum(p.numel() for p in native.parameters())==plan['expected_student_values']
    with torch.no_grad():assert torch.equal(source(ids,pixels)['logits'],native(ids,pixels)['logits'])
    assert metadata['no_old_base_actor_optimizer_or_training_reference_loaded']


def test_wrong_standalone_receipt_or_actual_parameter_inventory_cannot_load(tmp_path):
    m=driver();source,ids,pixels,plan=student_fixture(tmp_path)
    plan['expected_student_values']+=1
    with pytest.raises(ValueError,match='Every actual standalone'):m.load_student(plan)
    plan['expected_student_values']-=1;plan['standalone']['receipt_sha256']='0'*64
    with pytest.raises(ValueError,match='Exact fixed standalone'):m.load_student(plan)


def worker_fixture(tmp_path,monkeypatch):
    from contextlib import nullcontext
    from types import SimpleNamespace
    from slm_lab import inference_rotary_cache_v1
    from slm_lab.dense_foundation_inference_v1 import DenseNativeView
    from test_machine_code_preference_stage_v1 import trained
    m=driver();m.REPORTS=tmp_path;model=trained().base.eval();native=DenseNativeView(model).eval()
    plan=dict(source_files_sha256={'fixture':'a'*64},standalone=dict(checkpoint=str(tmp_path/'model.pt')),
        cached_own_reference=dict(report=str(tmp_path/'old.json'),report_sha256='c'*64,
            profile=dict(human_manifest_sha256='d'*64,frozen_dense_sha256='e'*64,DPO_beta=.1)))
    monkeypatch.setattr(m.legacy,'registered',lambda *a:plan);monkeypatch.setattr(m.owner,'owned',lambda p:None)
    monkeypatch.setattr(m,'FileBindings',lambda p:SimpleNamespace(check=lambda **kw:None));monkeypatch.setattr(m,'file_hash',lambda p:'a'*64)
    monkeypatch.setattr(m,'load_student',lambda *a:(model,native,dict(tokenizer_sha256='b'*64)))
    monkeypatch.setattr(torch.cuda,'set_device',lambda *a:None);monkeypatch.setattr(torch.cuda,'set_per_process_memory_fraction',lambda *a:None)
    monkeypatch.setattr(torch.cuda,'get_device_name',lambda *a:'NVIDIA L40');monkeypatch.setattr(inference_rotary_cache_v1,'inference_cache',nullcontext)
    return m,model,native,plan


@pytest.mark.parametrize('suite',['QA','AI2D','MBPP'])
def test_complete_native_population_and_real_delivery_journal_dispatch(tmp_path,monkeypatch,suite):
    from types import SimpleNamespace
    from slm_lab import full_held_QA_native_v1,ai2d_full_native_evaluation_v1,MBPP_original_population_v2,MBPP_all_native_evaluation_v1
    m,model,native,plan=worker_fixture(tmp_path,monkeypatch);calls=[]
    population=SimpleNamespace(tokenizer_sha256='b'*64,tokenizer=object())
    if suite=='QA':module=full_held_QA_native_v1;name='FullHeldQA'
    elif suite=='AI2D':module=ai2d_full_native_evaluation_v1;name='FullAI2D'
    else:module=MBPP_original_population_v2;name='FullOriginalMBPP'
    monkeypatch.setattr(module,name,lambda root:population)
    def evaluate(*args,**kw):calls.append((args,kw));kw['checked']();kw['record'](dict(unit_delivery=True));return dict(results=[{}])
    monkeypatch.setattr(MBPP_all_native_evaluation_v1 if suite=='MBPP' else module,'evaluate_shard',evaluate)
    m.worker(suite,2);args,kw=calls[0]
    assert args[0] is native and args[1] is (population if suite=='MBPP' else population.tokenizer)
    assert kw['rank']==2 and kw['world']==4
    assert (tmp_path/'standalone'/f'{suite}-shard-2-actual-deliveries.jsonl').read_text().count('\n')==1


def test_preferences_dispatches_actual_dense_student_and_explicit_cached_original_reference(tmp_path,monkeypatch):
    from slm_lab import standalone_held_human_v1,whole_human_data_v2
    from test_dense_human_preference_evaluation_v1 import held
    m,model,native,plan=worker_fixture(tmp_path,monkeypatch);data=held();cache=object();calls=[]
    monkeypatch.setattr(whole_human_data_v2,'WholeHumanData',lambda *a:data)
    monkeypatch.setattr(standalone_held_human_v1,'CachedHeldReference',lambda *a,**kw:cache)
    def evaluate(*args,**kwargs):calls.append((args,kwargs))
    monkeypatch.setattr(standalone_held_human_v1,'evaluate_shard',evaluate);m.worker('preferences',1)
    args,kw=calls[0];assert args[0] is model and args[1] is data and args[3] is cache
    assert kw['rank']==1 and kw['world']==4 and kw['provenance']['inference_precision']=='float32'


def test_all_original_code_cases_retain_actual_native_generation_actions(tmp_path,monkeypatch):
    from slm_lab import native_code_shards_v1
    m,model,native,plan=worker_fixture(tmp_path,monkeypatch)
    monkeypatch.setattr(model,'generate',lambda ids,**kwargs:torch.cat([ids,torch.tensor([[2]])],dim=1))
    def evaluate(*args,**kw):
        assert args[0] is native and kw['world']==4 and kw['rank']==3
        for _ in range(6):native.generate(torch.tensor([[1,4]]),max_new_tokens=512,eos_token_id=2,use_cache=True)
    monkeypatch.setattr(native_code_shards_v1,'evaluate_shard',evaluate);m.worker('code',3)
    for frame in ['completion','instruction']:
        rows=[json.loads(x) for x in (tmp_path/'standalone'/f'code-{frame}-shard-3-actual-actions.jsonl').read_text().splitlines()]
        assert len(rows)==6 and all(r['actual_generated_token_ids']==[2] and r['requested_native_action_budget']==512 for r in rows)


def test_all3429word_crop_controls_use_actual_student_and_fixed_global_shards(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from slm_lab import standalone_word_shards_v1,shared_word_data_v1
    from tokenizers import Tokenizer
    m,model,native,plan=worker_fixture(tmp_path,monkeypatch);calls=[]
    class Data:
        manifest=dict(tokenizer_sha256='b'*64)
        root=tmp_path
        def __len__(self):return 3429
    data=Data();tokenizer=object()
    monkeypatch.setattr(shared_word_data_v1,'SharedWordData',lambda *a:data)
    monkeypatch.setattr(Tokenizer,'from_file',lambda *a:tokenizer)
    def evaluate(*args,**kw):calls.append((args,kw))
    monkeypatch.setattr(standalone_word_shards_v1,'evaluate_shard',evaluate);m.worker('words',3)
    args,kw=calls[0];assert args[0] is model and args[1] is data and args[2] is tokenizer
    assert kw['rank']==3 and kw['world']==4 and kw['generation_budget']==64 and kw['batch_size']==16
