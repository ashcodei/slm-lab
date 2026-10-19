import copy

import pytest
import torch

from slm_lab.codenet_checkpoint_v1 import capture_rng
from slm_lab.corpus import file_hash
from slm_lab.deterministic_positions_v1 import install
from slm_lab.fast_joint_stage_v1 import make_optimizer
from slm_lab.focused_continuation_run_contract_v1 import make_state,restore_state
from slm_lab.model import ModelConfig
from slm_lab.own_dense_growth_v1 import OwnDenseAssistant,new_backbone
from slm_lab.phase_joint_update_v1 import update
from slm_lab.whole_human_sft_evidence_v1 import tensor_digest


def test_real_zero_foundation_continuation_recovers_and_rejects_changed_original_source_before_mutation(tmp_path):
    torch.set_num_threads(1)
    c=ModelConfig(vocab_size=96,hidden_size=32,intermediate_size=64,num_layers=2,num_heads=4,num_kv_heads=2,
        max_text_length=64,vision_hidden_size=8,vision_intermediate_size=16,vision_layers=1,vision_heads=2,
        image_size=16,patch_size=4,pool_size=2)
    model=install(OwnDenseAssistant(new_backbone(c,972417),reward_hidden=8,word_temporal=4,pointing_rank=8,router_hidden=8,detector_hidden=4))
    reference=copy.deepcopy(model).eval().requires_grad_(False)
    path=tmp_path/'own-completed-start.pt';torch.save(reference.export(dict(own_source='f'*64)),path)
    binding=dict(path=str(path),checkpoint_sha256=file_hash(path),parameter_sha256=tensor_digest(reference.named_parameters()),foundation_completed_updates=0)
    rates=dict(language=.001,vision=.001,word=.001)
    options=dict(learning_rates=rates,phase_learning_rates=dict(instruction_and_vision=rates),betas=[.9,.95],weight_decay=.01,
        gradient_clip=1.,warmup_fraction=.02,cosine_floor=.1,maximum_microbatch=1,maximum_microbatch_tokens=8192,
        ctc_weight=1.,word_native_CE_weight=1.,gradient_bucket_values=1024,image_size=16,reference_KL_weight=.01)
    class Data:
        role='train';purpose='optimization';kinds=dict(code='text',public_output='text',photo='image')
        lengths=dict(code=[5],public_output=[5],photo=[5]);targets=dict(code=[3],public_output=[3],photo=[3])
        def batch(self,name,positions,device):
            token=17 if name=='public_output' else 11
            batch=dict(input_ids=torch.tensor([[1,token,12,14,2]],device=device),
                labels=torch.tensor([[-100,-100,12,14,2]],device=device),attention_mask=torch.ones(1,5,dtype=torch.bool,device=device))
            if name=='photo':batch['pixel_values']=torch.ones(1,3,16,16,device=device)
            return batch
    class Schedule:
        phase_for_step=['instruction_and_vision']*3
        phase_bounds=dict(foundation=[0,0],instruction_and_vision=[0,3])
        def __len__(self):return 3
        def at(self,i):return ['code','public_output','photo'][i],[0]
    schedule=Schedule();data=Data();optimizer=make_optimizer(model,options)
    plan=dict(actual_counts=model.counts(),initial_dense_settings=dict(initializer_sha256='b'*64),planned_steps=3,
        training_options=options,phase_bounds=schedule.phase_bounds,
        complete_source_budget={n:dict(kind=k) for n,k in data.kinds.items()})
    rows=[update(model,optimizer,data,schedule,0,options,reference=reference)]
    rows.append(update(model,optimizer,data,schedule,1,options,reference=reference))
    assert rows[1]['full_context_reference_KL_sum']>0
    snapshot=make_state(model,optimizer,plan,'a'*64,2,rows,[capture_rng('cpu')],'train',
        schedule=schedule,reference_binding=binding,reference=reference)
    language=tensor_digest(model.language_named_parameters())
    expected=update(model,optimizer,data,schedule,2,options,reference=reference)
    assert tensor_digest(model.language_named_parameters())==language
    weights=tensor_digest(model.named_parameters());moments=copy.deepcopy(optimizer.state_dict())
    for field,value in [('family','different-code-source'),('global_original_positions',[1])]:
        bad=copy.deepcopy(snapshot);bad['core']['metrics'][1][field]=value
        with pytest.raises(ValueError,match='source-family cursor'):
            restore_state(model,optimizer,bad,plan,'a'*64,'cpu',0,1,'train',
                schedule=schedule,reference_binding=binding,reference=reference)
        assert tensor_digest(model.named_parameters())==weights
        actual=optimizer.state_dict()
        for key,state in moments['state'].items():
            for name,value in state.items():assert torch.equal(value,actual['state'][key][name])
    count,restored=restore_state(model,optimizer,snapshot,plan,'a'*64,'cpu',0,1,'train',
        schedule=schedule,reference_binding=binding,reference=reference)
    assert count==2 and len(restored)==2
    assert update(model,optimizer,data,schedule,2,options,reference=reference)==expected
    assert tensor_digest(model.named_parameters())==weights
