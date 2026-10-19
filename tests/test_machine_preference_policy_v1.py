from copy import deepcopy
import json

import pytest
import torch

from slm_lab import broad_anchored_policy_v2 as code
from slm_lab import machine_preference_policy_v1 as policy
from slm_lab.corpus import file_hash
from slm_lab.frozen_own_actor_reference_v1 import FrozenOwnActorReference
from slm_lab.parallel_machine_preference_v1 import ParallelMachinePreferenceStage
from slm_lab.whole_human_sft_evidence_v1 import tensor_digest
from test_broad_anchored_policy_v2 import trained_start
from test_machine_code_preference_stage_v1 import data,settings


def fixture(tmp_path):
    m,p,source_plan=trained_start(tmp_path)
    ids=torch.tensor([[1,5,7,2]]);parameters=[q for _,q in m.named_selected(('actor',))]
    opt=torch.optim.AdamW(parameters,lr=.001,foreach=False)
    m(ids,labels=ids)['loss'].backward(inputs=parameters);opt.step();opt.zero_grad(set_to_none=True)
    checkpoint=tmp_path/'complete-code.pt';torch.save(code.export(m,p,file_hash(source_plan),'anchor_execution',1),checkpoint)
    inputs=dict(checkpoint=str(checkpoint),plan=str(source_plan),checkpoint_sha256=file_hash(checkpoint),plan_sha256=file_hash(source_plan))
    o=settings();del o['method']
    plan=dict(starting_policy_inputs=inputs,actual_counts=m.counts(),
        frozen_dense_sha256=tensor_digest(m.base.named_parameters()),starting_actor_parts_sha256=tensor_digest(m.parts.named_parameters()),
        original_machine_pairs=5,planned_steps=3,training_options=o,machine_admission_sha256='d'*64,
        machine_manifest_sha256='e'*64,small_synthetic_unit_fixture=True)
    plan['own_reference_binding']=FrozenOwnActorReference(m,sources=policy.reference_sources(plan)).binding()
    path=tmp_path/'machine-plan.json';path.write_text(json.dumps(plan));return m,plan,path


def test_actual_nonzero_complete_start_full_epoch_cold_text_and_image_outputs_and_counts(tmp_path):
    m,plan,path=fixture(tmp_path);fresh,_=policy.fresh(plan);ref=policy.make_reference(m,plan)
    assert tensor_digest(fresh.parts.named_parameters())==plan['starting_actor_parts_sha256']
    o=policy.options(plan,'machine_DPO');opt=torch.optim.AdamW([p for _,p in m.named_selected(('actor',))],
        lr=o['learning_rate'],betas=tuple(o['betas']),weight_decay=o['weight_decay'],foreach=False)
    stage=ParallelMachinePreferenceStage(m,opt,data(),ref,o,policy.sources(plan,file_hash(path)))
    for _ in range(stage.planned):stage.update()
    checkpoint=tmp_path/'machine-DPO.pt';torch.save(policy.export(m,plan,file_hash(path),'machine_DPO',stage.completed),checkpoint)
    loaded,_=policy.load(checkpoint,path,'machine_DPO',expected_checkpoint_sha256=file_hash(checkpoint),expected_plan_sha256=file_hash(path))
    ids=torch.tensor([[1,5,7,2]]);pixels=torch.zeros(1,3,m.config.image_size,m.config.image_size)
    with torch.no_grad():
        assert torch.equal(m(ids)['logits'],loaded(ids)['logits'])
        assert torch.equal(m(ids,pixels)['logits'],loaded(ids,pixels)['logits'])
    assert loaded.counts()==plan['actual_counts']


def test_partial_wrong_source_human_flag_critic_and_reference_reject(tmp_path):
    m,plan,path=fixture(tmp_path)
    with pytest.raises(ValueError,match='complete unrepeated'):policy.export(m,plan,file_hash(path),'chosen_CE',2)
    raw=policy.export(m,plan,file_hash(path),'chosen_CE',3);checkpoint=tmp_path/'bad.pt'
    for mutate in [lambda r:r.update(completed_updates=2),lambda r:r.update(human_preferences=True),
                   lambda r:r.update(original_machine_pairs_consumed=4),
                   lambda r:r['policy']['parts']['critic.weight'].add_(1),
                   lambda r:r['policy'].update(frozen_dense_sha256='0'*64)]:
        bad=deepcopy(raw);mutate(bad);torch.save(bad,checkpoint)
        with pytest.raises(ValueError):policy.load(checkpoint,path,'chosen_CE',
            expected_checkpoint_sha256=file_hash(checkpoint),expected_plan_sha256=file_hash(path))
    wrong=deepcopy(plan);wrong['own_reference_binding']['frozen_actor_parts_sha256']='0'*64
    with pytest.raises(ValueError,match='nonzero starting'):policy.make_reference(m,wrong)
