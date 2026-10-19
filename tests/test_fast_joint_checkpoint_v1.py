import copy
import pytest
import torch

from slm_lab.codenet_checkpoint_v1 import capture_rng
from slm_lab.fast_joint_checkpoint_v1 import bundle,restore
from slm_lab.fast_joint_data_v1 import CompleteBucketSchedule
from slm_lab.fast_joint_stage_v1 import make_optimizer,update
from test_fast_joint_v1 import options,batch
from test_own_dense_growth_v1 import model as make_model
from test_own_alignment_checkpoint_v1 import assert_nested_equal
from types import SimpleNamespace


def test_joint_checkpoint_recovers_actual_weights_Adam_RNG_and_rejects_corruption_before_mutation():
    m,_=make_model();setting=options();opt=make_optimizer(m,setting);b=batch()
    d=SimpleNamespace(role='train',purpose='optimization',kinds={'text':'text'},lengths={'text':[5,4,5,4]},targets={'text':[3]*4})
    d.batch=lambda name,positions,device:{k:v[[p%2 for p in positions]].to(device) for k,v in b.items()}
    s=CompleteBucketSchedule(d.lengths,dict(text=2),3);first=update(m,opt,d,s,0,setting)
    plan=dict(initial_dense_settings=dict(initializer_sha256='a'*64),actual_counts=m.counts(),planned_steps=2,training_options=setting)
    payload=bundle(m,opt,plan,'b'*64,1,[first],[capture_rng('cpu')],'capacity')
    expected=update(m,opt,d,s,1,setting);weights={n:p.detach().clone() for n,p in m.named_parameters()};moments=copy.deepcopy(opt.state_dict())
    completed,metrics=restore(m,opt,payload,plan,'b'*64,'cpu',0,1,'capacity')
    assert completed==1 and metrics==[first] and update(m,opt,d,s,1,setting)==expected
    assert all(torch.equal(p,weights[n]) for n,p in m.named_parameters());assert_nested_equal(opt.state_dict(),moments)
    for mutate in [lambda x:x['rngs'][0].update(torch=torch.zeros(4,dtype=torch.uint8)),
        lambda x:x['optimizer']['state'][0]['exp_avg'].fill_(float('nan'))]:
        bad=copy.deepcopy(payload);mutate(bad);rng=torch.get_rng_state().clone()
        with pytest.raises((ValueError,RuntimeError)):restore(m,opt,bad,plan,'b'*64,'cpu',0,1,'capacity')
        assert all(torch.equal(p,weights[n]) for n,p in m.named_parameters()) and torch.equal(rng,torch.get_rng_state())
        assert_nested_equal(opt.state_dict(),moments)
