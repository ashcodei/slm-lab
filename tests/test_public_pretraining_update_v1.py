import copy
from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F

from slm_lab.fast_joint_stage_v1 import make_optimizer, parameter_groups
from slm_lab.public_pretraining_update_v1 import backward_public, update
from test_fast_joint_v1 import batch, options
from test_own_dense_growth_v1 import model as make_model


def raw_batch():
    return dict(input_ids=torch.tensor([[1,4,5,2,1,6,7,2],[1,10,2,0,0,0,0,0]]),
        labels=torch.tensor([[-100,4,5,2,-100,6,7,2],[-100,10,2,-100,-100,-100,-100,-100]]),
        attention_mask=torch.tensor([[1]*8,[1,1,1,0,0,0,0,0]],dtype=torch.bool))


def test_raw_native_internal_BOS_mask_and_global_normalizer_match_an_independent_loss_without_reference():
    model,_ = make_model();oracle = copy.deepcopy(model);setting = options()
    make_optimizer(model,setting);make_optimizer(oracle,setting)
    b = raw_batch();calls = []
    handle = model.register_forward_hook(lambda *a:calls.append(1))
    result = backward_public(model,b,kind='text',global_targets=8,global_context_targets=9)
    handle.remove()
    logits = oracle(input_ids=b['input_ids'],attention_mask=b['attention_mask'])['logits'][:,:-1].float()
    mask = b['labels'][:,1:] != -100
    loss = F.cross_entropy(logits[mask],b['labels'][:,1:][mask],reduction='sum')/8
    loss.backward(inputs=[p for g in parameter_groups(oracle).values() for _,p in g])
    assert len(calls) == 1 and result['CE_sum'] == pytest.approx(float(loss.detach())*8)
    for (name,p),(_,q) in zip(model.named_parameters(),oracle.named_parameters()):
        if p.grad is None:assert q.grad is None
        else:assert torch.allclose(p.grad,q.grad,atol=3e-7,rtol=3e-5),name


@pytest.mark.parametrize('kind',['text','image','word'])
def test_actual_full_group_update_changes_only_the_intended_modality_without_reference_model(kind):
    model,_ = make_model();setting = options();optimizer = make_optimizer(model,setting)
    b = raw_batch() if kind == 'text' else batch(image=True,word=kind=='word')
    lengths = b['attention_mask'].sum(1).tolist();targets = (b['labels'][:,1:] != -100).sum(1).tolist()
    d = SimpleNamespace(role='train',purpose='optimization',kinds={'source':kind},lengths={'source':lengths},targets={'source':targets})
    d.batch = lambda name,p,device:{k:(v[p].to(device) if isinstance(v,torch.Tensor) else [v[i] for i in p]) for k,v in b.items()}
    class Schedule:
        def __len__(self):return 1
        def at(self,i):return 'source',[0,1]
    before = {n:p.detach().clone() for n,p in model.named_parameters()}
    report = update(model,optimizer,d,Schedule(),0,setting)
    wanted = ['language'] if kind == 'text' else ['vision']+(['word'] if kind=='word' else [])
    assert report['actual_active_groups'] == wanted
    assert report['reference_forward_queried'] is False and report['resident_training_only_reference_parameters'] == 0
    for group,rows in parameter_groups(model).items():
        if group in wanted:assert any(not torch.equal(p,before[n]) for n,p in rows)
        else:assert all(torch.equal(p,before[n]) and p.grad is None for n,p in rows)
    assert all(torch.equal(p,before[n]) for n,p in model.named_parameters() if n.startswith('heads.') and not n.startswith('heads.word.'))
    d.role = 'valid'
    with pytest.raises(ValueError,match='train-only'):update(model,optimizer,d,Schedule(),0,setting)
