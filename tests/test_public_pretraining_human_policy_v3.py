from copy import deepcopy
import json

import pytest
import torch

from slm_lab import phase_human_policy_v3 as policy
from slm_lab.corpus import file_hash
from slm_lab.own_dense_actor_v1 import OwnDenseActor
from slm_lab.parallel_human_preference_v2 import ParallelHumanPreferenceStage
from slm_lab.public_pretraining_data_v2 import RAW,FUNCTIONS
from slm_lab.public_pretraining_native_v1 import export as public_export
from slm_lab.whole_human_sft_evidence_v1 import tensor_digest
from test_own_dense_standalone_v1 import fixture as dense_fixture
from test_dense_human_preference_stage_v1 import pairs,options


def fixture(tmp_path):
    base,ids,old,_=dense_fixture(tmp_path)
    # Only the loading/gradient/format plumbing is exercised here. Declared
    # population counts are synthetic metadata, not a billion-token SGD proof.
    source=dict(complete_code_first_inputs=dict(checkpoint='absent-unit-ancestor',checkpoint_sha256='a'*64,plan='unit',plan_sha256='b'*64),
        actual_counts=base.counts(),tokenizer_sha256=file_hash(tmp_path/'tokenizer.json'),planned_steps=1,source_schedule_sha256='c'*64,
        public_raw_admission_sha256='d'*64,public_function_admission_sha256='e'*64,
        all_original_raw_train_windows_once=True,all826verified_original_train_functions_once=True,
        all_previous_selected_source_presentations_preserved=True,
        selected_source_budget={RAW:dict(response_EOS_targets=10**9),FUNCTIONS:dict(presentations=826)},small_synthetic_unit_fixture=True)
    source_plan=tmp_path/'public-plan.json';source_plan.write_text(json.dumps(source));checkpoint=tmp_path/'public.pt'
    torch.save(public_export(base,source,file_hash(source_plan),'trained',1),checkpoint);old.unlink()
    model=OwnDenseActor(base,rank=2,seed=958993);data=pairs();setting=options(data)
    setting.update(epochs=1,steps=1,accumulation=len(data));setting.pop('DPO_weight')
    plan=dict(foundation_inputs=dict(checkpoint=str(checkpoint),plan=str(source_plan),checkpoint_sha256=file_hash(checkpoint),plan_sha256=file_hash(source_plan)),
        actual_dense_counts=base.counts(),frozen_dense_sha256=tensor_digest(base.named_parameters()),actor_settings=model.settings,
        actual_counts=model.counts(),initial_actor_parts_sha256=tensor_digest(model.parts.named_parameters()),human_train_pairs=len(data),
        human_admission_sha256='f'*64,human_manifest_sha256='0'*64,training_options=setting,small_synthetic_unit_fixture=True)
    path=tmp_path/'human-plan.json';path.write_text(json.dumps(plan));return model,data,plan,path


def test_real_DPO_from_self_contained_public_student_cold_loads_exact_text_and_image_without_ancestor(tmp_path):
    model,data,plan,path=fixture(tmp_path);digest=file_hash(path)
    cold,_=policy.fresh(plan)
    assert tensor_digest(cold.base.named_parameters())==plan['frozen_dense_sha256']
    setting=policy.options(plan,'human_DPO');opt=torch.optim.AdamW([p for _,p in model.named_selected(('actor',))],
        lr=setting['learning_rate'],betas=tuple(setting['betas']),weight_decay=setting['weight_decay'],foreach=False)
    ParallelHumanPreferenceStage(model,opt,data,setting,policy.sources(plan,digest)).update();model.eval()
    checkpoint=tmp_path/'human.pt';torch.save(policy.export(model,plan,digest,'human_DPO',1),checkpoint)
    cold,_=policy.load(checkpoint,path,'human_DPO',expected_checkpoint_sha256=file_hash(checkpoint),expected_plan_sha256=digest)
    ids=torch.tensor([[1,4,7,2]]);pixels=torch.zeros(1,3,16,16)
    with torch.no_grad():
        assert torch.equal(model(ids)['logits'],cold(ids)['logits'])
        assert torch.equal(model(ids,pixels)['logits'],cold(ids,pixels)['logits'])


def test_incomplete_or_wrong_public_epoch_function_population_or_digest_rejected(tmp_path):
    _,_,plan,_=fixture(tmp_path)
    source=json.loads(open(plan['foundation_inputs']['plan']).read());raw=torch.load(plan['foundation_inputs']['checkpoint'],weights_only=True)
    for kind in ['partial','function','retention']:
        p,r=deepcopy(source),deepcopy(raw)
        if kind=='partial':r['completed_updates']=0
        elif kind=='function':p['selected_source_budget'][FUNCTIONS]['presentations']=825
        else:p['all_previous_selected_source_presentations_preserved']=False
        with pytest.raises(ValueError):policy.validate_foundation(p,r)
    bad=deepcopy(plan);bad['foundation_inputs']['checkpoint_sha256']='0'*64
    with pytest.raises(ValueError):policy.fresh(bad)
