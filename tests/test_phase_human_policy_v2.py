import copy
import json

import pytest
import torch

from slm_lab import phase_human_policy_v2 as policy
from slm_lab.corpus import file_hash
from slm_lab.deterministic_positions_v1 import install
from slm_lab.fast_joint_inference_v2 import export as dense_export
from slm_lab.focused_code_continuation_data_v2 import CODE,EXECUTION
from slm_lab.own_dense_actor_v1 import OwnDenseActor
from slm_lab.parallel_human_preference_v2 import ParallelHumanPreferenceStage
from slm_lab.whole_human_sft_evidence_v1 import tensor_digest
from test_dense_human_preference_stage_v1 import pairs,options
from test_own_dense_actor_v1 import actor


def fixture(tmp_path):
    torch.set_num_threads(1);base=install(actor().base);initial=tmp_path/'main-initializer.pt'
    sources=dict(actual_small_unit_source_sha256='a'*64);torch.save(base.export(sources),initial)
    settings=dict(initializer=str(initial),initializer_sha256=file_hash(initial),initializer_sources=sources,
        actual_counts=base.counts(),tokenizer_sha256='b'*64)
    main=dict(initial_dense_settings=settings,actual_counts=base.counts(),planned_steps=1,
        all_selected_family_records_once=True,all_original_sources_once=False,later_reference_is_own_completed_foundation=True,
        tokenizer_sha256='b'*64,small_synthetic_unit_fixture=True)
    main_plan=tmp_path/'main-plan.json';main_plan.write_text(json.dumps(main))
    optimizer=torch.optim.AdamW(base.parameters(),lr=.001)
    def update(token):
        ids=torch.tensor([[1,4,token,2]]);base(ids,labels=ids)['loss'].backward();optimizer.step();optimizer.zero_grad(set_to_none=True)
    update(7);main_checkpoint=tmp_path/'main-complete.pt';torch.save(dense_export(base,main,file_hash(main_plan),'trained',1),main_checkpoint)
    inputs=dict(checkpoint=str(main_checkpoint),plan=str(main_plan),checkpoint_sha256=file_hash(main_checkpoint),plan_sha256=file_hash(main_plan))
    copied_sources=dict(own_complete_main_checkpoint_sha256=inputs['checkpoint_sha256'],own_complete_main_plan_sha256=inputs['plan_sha256'])
    copied=tmp_path/'code-initializer.pt';torch.save(base.export(copied_sources),copied)
    settings=dict(initializer=str(copied),initializer_sha256=file_hash(copied),initializer_sources=copied_sources,
        actual_counts=base.counts(),tokenizer_sha256='b'*64)
    code=dict(initial_dense_settings=settings,actual_counts=base.counts(),planned_steps=2,tokenizer_sha256='b'*64,
        starting_main_inputs=inputs,source_start_is_complete_main=True,complete_new_code_and_execution_source_epoch_once=True,
        all_original_replay_sources_once=False,complete_source_budget={CODE:dict(examples=1),EXECUTION:dict(examples=1)},
        selected_source_budget={CODE:dict(presentations=1),EXECUTION:dict(presentations=1)},small_synthetic_unit_fixture=True)
    source_plan=tmp_path/'code-first-plan.json';source_plan.write_text(json.dumps(code))
    update(8);update(9);checkpoint=tmp_path/'code-first-complete.pt';torch.save(dense_export(base,code,file_hash(source_plan),'trained',2),checkpoint)
    model=OwnDenseActor(base,rank=2,seed=958993);data=pairs();setting=options(data)
    setting.update(epochs=1,steps=1,accumulation=len(data));setting.pop('DPO_weight')
    plan=dict(foundation_inputs=dict(checkpoint=str(checkpoint),plan=str(source_plan),checkpoint_sha256=file_hash(checkpoint),plan_sha256=file_hash(source_plan)),
        actual_dense_counts=base.counts(),frozen_dense_sha256=tensor_digest(base.named_parameters()),actor_settings=model.settings,
        actual_counts=model.counts(),initial_actor_parts_sha256=tensor_digest(model.parts.named_parameters()),
        human_train_pairs=len(data),human_admission_sha256='c'*64,human_manifest_sha256='d'*64,training_options=setting,
        small_synthetic_unit_fixture=True)
    run_plan=tmp_path/'human-plan.json';run_plan.write_text(json.dumps(plan));return model,data,plan,run_plan


def test_actual_main_then_code_output_then_DPO_updates_and_native_text_image_cold_load(tmp_path):
    model,data,plan,run_plan=fixture(tmp_path);sha=file_hash(run_plan)
    loaded,_=policy.fresh(plan)
    assert tensor_digest(loaded.parts.named_parameters())==tensor_digest(model.parts.named_parameters())
    setting=policy.options(plan,'human_DPO');optimizer=torch.optim.AdamW([p for _,p in model.named_selected(('actor',))],
        lr=setting['learning_rate'],betas=tuple(setting['betas']),weight_decay=setting['weight_decay'],foreach=False)
    ParallelHumanPreferenceStage(model,optimizer,data,setting,policy.sources(plan,sha)).update();model.eval()
    path=tmp_path/'trained-human.pt';torch.save(policy.export(model,plan,sha,'human_DPO',1),path)
    cold,_=policy.load(path,run_plan,'human_DPO',expected_checkpoint_sha256=file_hash(path),expected_plan_sha256=sha)
    ids=torch.tensor([[1,4,7,2]]);pixels=torch.zeros(1,3,model.config.image_size,model.config.image_size)
    with torch.no_grad():
        assert torch.equal(model(ids)['logits'],cold(ids)['logits'])
        assert torch.equal(model(ids,pixels)['logits'],cold(ids,pixels)['logits'])


def test_incomplete_code_or_output_epoch_and_wrong_complete_main_linkage_rejected(tmp_path):
    _,_,plan,_=fixture(tmp_path);source=json.loads(open(plan['foundation_inputs']['plan']).read())
    raw=torch.load(plan['foundation_inputs']['checkpoint'],weights_only=True)
    for mutate in [lambda p:p.update(completed_updates=1),lambda p:p.update(whole_source_epochs_completed=0)]:
        bad=copy.deepcopy(raw);mutate(bad)
        with pytest.raises(ValueError,match='Complete code-first'):policy.validate_foundation(source,bad)
    bad=copy.deepcopy(source);bad['selected_source_budget'][EXECUTION]['presentations']=0
    with pytest.raises(ValueError):policy.validate_foundation(bad,raw)
    bad=copy.deepcopy(plan);bad['foundation_inputs']['checkpoint_sha256']='0'*64
    with pytest.raises(ValueError):policy.fresh(bad)
