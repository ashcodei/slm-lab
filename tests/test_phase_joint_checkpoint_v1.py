import copy

import pytest
import torch

from slm_lab.codenet_checkpoint_v1 import capture_rng
from slm_lab.corpus import file_hash
from slm_lab.deterministic_positions_v1 import install
from slm_lab.fast_joint_stage_v1 import make_optimizer
from slm_lab.model import ModelConfig
from slm_lab.own_dense_growth_v1 import OwnDenseAssistant,new_backbone
from slm_lab.phase_joint_checkpoint_v1 import make_state,restore_state
from slm_lab.phase_joint_update_v1 import update
from slm_lab.whole_human_sft_evidence_v1 import tensor_digest


def test_phase_checkpoint_binds_new_reference_and_rejects_false_clocks_without_mutation(tmp_path):
    torch.set_num_threads(1)
    c=ModelConfig(vocab_size=96,hidden_size=32,intermediate_size=64,num_layers=2,num_heads=4,num_kv_heads=2,
        max_text_length=64,vision_hidden_size=8,vision_intermediate_size=16,vision_layers=1,vision_heads=2,image_size=16,patch_size=4,pool_size=2)
    model=install(OwnDenseAssistant(new_backbone(c,958411),reward_hidden=8,word_temporal=4,pointing_rank=8,router_hidden=8,detector_hidden=4))
    rates=dict(language=.001,vision=.001,word=.001)
    options=dict(learning_rates=rates,phase_learning_rates=dict(foundation=rates,instruction_and_vision=rates),betas=[.9,.95],weight_decay=.01,
        gradient_clip=1.,warmup_fraction=.02,cosine_floor=.1,maximum_microbatch=1,maximum_microbatch_tokens=8192,
        ctc_weight=1.,word_native_CE_weight=1.,gradient_bucket_values=1024,image_size=16,reference_KL_weight=.01)
    class Data:
        role='train';purpose='optimization';kinds=dict(raw='text',qa='text',photo='image')
        lengths=dict(raw=[5],qa=[5],photo=[5]);targets=dict(raw=[4],qa=[4],photo=[4])
        def batch(self,name,positions,device):
            batch=dict(input_ids=torch.tensor([[1,11,12,14,2]]),labels=torch.tensor([[1,11,12,14,2]]),attention_mask=torch.ones(1,5,dtype=torch.bool))
            if name=='photo':batch['pixel_values']=torch.ones(1,3,16,16)
            return batch
    class Schedule:
        phase_for_step=['foundation','instruction_and_vision','instruction_and_vision']
        phase_bounds=dict(foundation=[0,1],instruction_and_vision=[1,3])
        def __len__(self):return 3
        def at(self,i):return ['raw','qa','photo'][i],[0]
    schedule=Schedule();data=Data();optimizer=make_optimizer(model,options)
    plan=dict(actual_counts=model.counts(),initial_dense_settings=dict(initializer_sha256='b'*64),planned_steps=3,training_options=options,
        phase_bounds=schedule.phase_bounds,complete_source_budget={n:dict(kind=k) for n,k in data.kinds.items()})
    records=[update(model,optimizer,data,schedule,0,options)]
    reference=copy.deepcopy(model).eval().requires_grad_(False);path=tmp_path/'reference.pt'
    torch.save(reference.export(dict(plan_sha256='a'*64)),path)
    binding=dict(path=str(path),checkpoint_sha256=file_hash(path),parameter_sha256=tensor_digest(reference.named_parameters()),foundation_completed_updates=1)
    records.append(update(model,optimizer,data,schedule,1,options,reference=reference))
    snapshot=make_state(model,optimizer,plan,'a'*64,2,records,[capture_rng('cpu')],'train',reference_binding=binding,reference=reference)
    expected=update(model,optimizer,data,schedule,2,options,reference=reference);weights=tensor_digest(model.named_parameters())
    bad=copy.deepcopy(snapshot);state=next(iter(bad['core']['optimizer']['state'].values()));state['step'].add_(1)
    with pytest.raises(ValueError,match='Adam clock'):
        restore_state(model,optimizer,bad,plan,'a'*64,'cpu',0,1,'train',reference_binding=binding,reference=reference)
    assert tensor_digest(model.named_parameters())==weights
    invalid=dict(binding,parameter_sha256='f'*64)
    with pytest.raises(ValueError,match='profile'):
        restore_state(model,optimizer,snapshot,plan,'a'*64,'cpu',0,1,'train',reference_binding=invalid,reference=reference)
    assert tensor_digest(model.named_parameters())==weights
    completed,rows=restore_state(model,optimizer,snapshot,plan,'a'*64,'cpu',0,1,'train',reference_binding=binding,reference=reference)
    assert completed==2 and len(rows)==2
    assert update(model,optimizer,data,schedule,2,options,reference=reference)==expected
    assert tensor_digest(model.named_parameters())==weights
