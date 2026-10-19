import copy

import torch

from slm_lab.corpus import file_hash
from slm_lab.deterministic_positions_v1 import install
from slm_lab.model import ModelConfig
from slm_lab.own_dense_growth_v1 import OwnDenseAssistant,new_backbone
from slm_lab.state_backward_rehearsal_v1 import rehearse
from slm_lab.whole_human_sft_evidence_v1 import tensor_digest


def test_actual_state_backwards_all_dense_leaves_recovery_and_private_RNG_rejection(tmp_path):
    torch.set_num_threads(1)
    config=ModelConfig(vocab_size=96,hidden_size=32,intermediate_size=64,num_layers=2,num_heads=4,num_kv_heads=2,
        max_text_length=64,vision_hidden_size=8,vision_intermediate_size=16,vision_layers=1,vision_heads=2,image_size=16,patch_size=4,pool_size=2)
    model=install(OwnDenseAssistant(new_backbone(config,978117),reward_hidden=8,word_temporal=4,pointing_rank=8,router_hidden=8,detector_hidden=4))
    reference=copy.deepcopy(model).eval().requires_grad_(False);path=tmp_path/'source.pt';torch.save(reference.export({'unit':'b'*64}),path)
    class Data:
        role='train';purpose='optimization';kinds={'state':'text'};lengths={'state':[5,7]};targets={'state':[3,5]}
        def batch(self,name,positions,device):
            p=positions[0];ids=[1,11,12,14,2] if p==0 else [1,17,18,19,21,22,2]
            return dict(input_ids=torch.tensor([ids],device=device),labels=torch.tensor([[-100,-100]+ids[2:]],device=device),attention_mask=torch.ones(1,len(ids),dtype=torch.bool,device=device),pixel_values=None)
    class Schedule:
        phase_for_step=['instruction_and_vision']*2;phase_bounds={'foundation':[0,0],'instruction_and_vision':[0,2]}
        def __len__(self):return 2
        def at(self,i):return 'state',[i]
    rates=dict(language=.001,vision=.001,word=.001)
    options=dict(learning_rates=rates,phase_learning_rates=dict(instruction_and_vision=rates),betas=[.9,.95],weight_decay=.01,
        gradient_clip=1.,warmup_fraction=.02,cosine_floor=.1,maximum_microbatch=1,maximum_microbatch_tokens=8192,
        ctc_weight=1.,word_native_CE_weight=1.,gradient_bucket_values=1024,image_size=16,reference_KL_weight=.01)
    schedule=Schedule();plan=dict(actual_counts=model.counts(),initial_dense_settings=dict(initializer_sha256='b'*64),planned_steps=2,
        training_options=options,phase_bounds=schedule.phase_bounds,complete_source_budget={'state':{'kind':'text'}})
    binding=dict(path=str(path),checkpoint_sha256=file_hash(path),parameter_sha256=tensor_digest(reference.named_parameters()),foundation_completed_updates=0)
    result=rehearse(model,Data(),schedule,plan,'a'*64,reference=reference,reference_binding=binding)
    assert result['actual_logical_updates']==2 and result['physical_optimizer_calls']==3
    assert result['actual_in_memory_next_update_weights_Adam_and_private_CPU_RNG_exact']
    assert len(result['invalid_state_rejections'])==2 and result['all_nonlanguage_heads_and_reference_parameters_exact']
    assert len(result['actual_changed_language_tensor_names'])==result['actual_active_language_tensors']
