import copy
import torch

from slm_lab.fast_joint_stage_v1 import make_optimizer,parameter_groups
from slm_lab.protected_joint_update_v1 import update
from slm_lab.own_dense_growth_v1 import OwnDenseAssistant,new_backbone
from slm_lab.model import ModelConfig


def test_actual_global_updates_and_prepared_batches_keep_inactive_Adam_clocks_fixed():
    torch.set_num_threads(1)
    c=ModelConfig(vocab_size=96,hidden_size=32,intermediate_size=64,num_layers=2,num_heads=4,num_kv_heads=2,
        max_text_length=64,vision_hidden_size=8,vision_intermediate_size=16,vision_layers=1,vision_heads=2,
        image_size=16,patch_size=4,pool_size=2)
    model=OwnDenseAssistant(new_backbone(c,957611),reward_hidden=8,word_temporal=4,pointing_rank=8,router_hidden=8,detector_hidden=4)
    ref=copy.deepcopy(model).eval().requires_grad_(False)
    with torch.no_grad():ref.network.embedding.weight.add_(.05)
    options=dict(learning_rates=dict(language=.001,vision=.001,word=.001),betas=[.9,.95],weight_decay=.1,
        ctc_weight=1.,word_native_CE_weight=1.,reference_KL_weight=.1,warmup_fraction=.02,cosine_floor=.1,
        maximum_microbatch=2,maximum_microbatch_tokens=64,image_size=16,gradient_bucket_values=1024,gradient_clip=1.)
    optimizer=make_optimizer(model,options);groups=parameter_groups(model)
    class Data:
        role='train';purpose='optimization';kinds=dict(image='image',text='text')
        lengths=dict(image=[4,4],text=[4,4]);targets=dict(image=[3,3],text=[3,3])
        def batch(self,name,positions,device):
            ids=torch.tensor([[1,5,6,2]]*len(positions),device=device)
            b=dict(input_ids=ids,labels=ids.clone(),attention_mask=torch.ones_like(ids,dtype=torch.bool));b['labels'][:,0]=-100
            if name=='image':b['pixel_values']=torch.ones(len(positions),3,16,16,device=device)
            return b
    class Schedule:
        def __len__(self):return 2
        def at(self,step):return ['image','text'][step],[0,1]
    data=Data();schedule=Schedule();old={n:p.detach().clone() for n,p in groups['language']}
    image=update(model,optimizer,data,schedule,0,options,reference=ref)
    assert image['actual_active_groups']==['vision'] and image['response_EOS_targets']==6
    assert all(torch.equal(p,old[n]) and p not in optimizer.state for n,p in groups['language'])
    vision_before={n:p.detach().clone() for n,p in groups['vision']}
    text=update(model,optimizer,data,schedule,1,options,reference=ref,prepared=[data.batch('text',[0,1],'cpu')])
    assert text['full_context_reference_KL_sum']>0 and text['actual_active_groups']==['language']
    assert all(torch.equal(p,vision_before[n]) and int(optimizer.state[p]['step'])==1 for n,p in groups['vision'])
    assert any(not torch.equal(p,old[n]) for n,p in groups['language'])
