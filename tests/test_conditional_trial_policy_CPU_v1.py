import copy
import json
from pathlib import Path

import pytest
import torch

from slm_lab import conditional_trial_policy_CPU_v1 as policy
from slm_lab.corpus import file_hash
from slm_lab.model import ModelConfig
from slm_lab.own_dense_actor_v1 import OwnDenseActor
from slm_lab.own_dense_growth_v1 import OwnDenseAssistant,new_backbone
from slm_lab.whole_human_sft_evidence_v1 import tensor_digest


def test_student_only_cpu_trial_roundtrip_and_incomplete_or_changed_identity_rejection(tmp_path,monkeypatch):
    torch.set_num_threads(1)
    config=ModelConfig(vocab_size=64,hidden_size=32,intermediate_size=64,num_layers=2,num_heads=4,num_kv_heads=2,
        max_text_length=64,vision_hidden_size=32,vision_intermediate_size=64,vision_layers=1,vision_heads=4,image_size=32,patch_size=16)
    base=OwnDenseAssistant(new_backbone(config,966117),reward_hidden=8,word_temporal=4,pointing_rank=8,
        router_hidden=8,detector_hidden=4,word_expansion=1).eval()
    actor=OwnDenseActor(base,rank=2,seed=966217).eval()
    plan=dict(base_inputs=dict(arm='step-2048',checkpoint='unit-mocked-source',plan='unit-mocked-source-plan',
        checkpoint_sha256='a'*64,plan_sha256='b'*64),actual_base_updates=2048,
        is_complete_human_DPO_or_source_epoch=False,actor_settings=actor.settings,actual_counts=actor.counts(),
        initial_parts_sha256=tensor_digest(actor.parts.named_parameters()),planned_updates_each_arm=1,qualified_episode_binding_sha256='c'*64)
    # Only the outer native-policy IO contract is tested here. No claim is made
    # that this tiny mock base physically completed2048 foundation updates.
    monkeypatch.setattr(policy,'load_base',lambda *a,**k:(copy.deepcopy(base),dict(unit_mock_not_actual_full_base=True)))
    path=tmp_path/'plan.json';path.write_text(json.dumps(plan));digest=file_hash(path)
    optimizer=torch.optim.AdamW([p for _,p in actor.named_selected(('actor',))],lr=1e-3)
    ids=torch.tensor([[1,5,6,2]]);actor(ids,labels=ids)['loss'].backward(inputs=[p for _,p in actor.named_selected(('actor',))]);optimizer.step();optimizer.zero_grad(set_to_none=True)
    checkpoint=tmp_path/'student.pt';torch.save(policy.export(actor,plan,digest,'anchor_execution',1),checkpoint)
    cold,meta=policy.load(checkpoint,path,'anchor_execution',expected_checkpoint_sha256=file_hash(checkpoint),expected_plan_sha256=digest)
    with torch.no_grad():assert torch.equal(actor(ids)['logits'],cold(ids)['logits'])
    assert meta['training_reference_not_loaded'] and not torch.load(checkpoint,weights_only=True)['private_training_reference_included']
    with pytest.raises(ValueError):policy.export(actor,plan,digest,'anchor_execution',0)
    raw=torch.load(checkpoint,weights_only=True);raw['complete_human_alignment_or_source_epoch']=True;torch.save(raw,checkpoint)
    with pytest.raises(ValueError):policy.load(checkpoint,path,'anchor_execution',expected_checkpoint_sha256=file_hash(checkpoint),expected_plan_sha256=digest)
