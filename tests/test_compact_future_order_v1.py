import copy
import math

import pytest
import torch
from torch.nn import functional as F

from slm_lab.compact_future_order_v1 import CompactFutureOrder,targets
from test_own_dense_actor_v1 import actor
from test_own_aligned_assistant_v1 import original_signature


def batch():
    return dict(input_ids=torch.tensor([[1,4,4,5,4,2],[1,6,7,2,0,0]]),
        labels=torch.tensor([[-100,-100,4,5,4,2],[-100,-100,7,2,-100,-100]]),
        attention_mask=torch.tensor([[1,1,1,1,1,1],[1,1,1,1,0,0]],dtype=torch.bool))


def dense_gold(batch,window,vocab):
    rows=[]
    for b in range(len(batch['input_ids'])):
        for t in range(batch['input_ids'].shape[1]-1):
            if int(batch['labels'][b,t+1])==-100:continue
            gold=torch.zeros(vocab);seen=set()
            for j in range(t+1,min(t+1+window,batch['input_ids'].shape[1])):
                if int(batch['labels'][b,j])==-100:break
                token=int(batch['input_ids'][b,j])
                if token not in seen:gold[token]=math.exp(-(j-t));seen.add(token)
                if token==2:break
            rows.append((b,t,gold/gold.sum()))
    return rows


def test_first_future_occurrence_including_repeated_current_token_matches_independent_dense_oracle():
    b=batch();positions,ids,probabilities,first=targets(**b,window=4,vocab_size=32);oracle=dense_gold(b,4,32)
    assert len(positions)==6 and first[0].tolist()==[True,True,False,True]
    assert int(b['input_ids'][0,positions[0,1]])==int(ids[0,0])==4
    for i,(row,t,gold) in enumerate(oracle):
        actual=torch.zeros(32).scatter_add(0,ids[i],probabilities[i])
        assert positions[i].tolist()==[row,t];torch.testing.assert_close(actual,gold,atol=1e-7,rtol=1e-6)


def test_chunked_recomputed_full_vocab_loss_and_all_gradients_match_dense_ListNet_oracle():
    m=CompactFutureOrder(actor(),rank=2,window=4);b=batch();hidden=torch.linspace(-1,1,2*6*16).reshape(2,6,16).requires_grad_()
    actual,record=m.auxiliary_sum(hidden,**b,chunk_positions=2,recompute_chunks=True)
    parts=[hidden,*m.future.parameters()];actual_grad=torch.autograd.grad(actual,parts)
    independent=hidden.detach().clone().requires_grad_();oracle=independent.sum()*0
    for row,t,gold in dense_gold(b,4,32):
        state=independent[row,t];projected=m.future['norm'](state+m.future['up'](m.future['down'](state)))
        logits=F.linear(projected,m.actor.base.network.embedding.weight.detach())
        oracle=oracle-(gold*logits.log_softmax(-1)).sum()
    expected_grad=torch.autograd.grad(oracle,[independent,*m.future.parameters()])
    torch.testing.assert_close(actual,oracle,atol=2e-6,rtol=1e-6)
    for a,e in zip(actual_grad,expected_grad):torch.testing.assert_close(a,e,atol=2e-6,rtol=2e-5)
    assert actual_grad[0][0,0].count_nonzero()==0 and actual_grad[0][1,3:].count_nonzero()==0
    assert record['original_auxiliary_positions']==record['original_next_token_targets']==6
    assert not record['dense_vocabulary_gold_targets_allocated']


def test_own_native_outputs_cache_RNG_and_single_backbone_inventory_are_preserved_by_attachment():
    a=actor();ids=torch.tensor([[1,4,7]]);pixels=torch.zeros(1,3,a.config.image_size,a.config.image_size)
    with torch.no_grad():text=a(ids)['logits'].clone();image=a(ids,pixels)['logits'].clone();generated=a.generate(ids,max_new_tokens=4,eos_token_id=2)
    before=original_signature(a);rng=torch.get_rng_state().clone();m=CompactFutureOrder(a,rank=2,window=4)
    assert original_signature(a)==before and torch.equal(rng,torch.get_rng_state()) and m.actor is a
    assert m.counts()['auxiliary_values']==2*16*2+16 and m.counts()['total_unique']==a.counts()['actual_unique']+80
    with torch.no_grad():
        assert torch.equal(text,m(ids)['logits']) and torch.equal(image,m(ids,pixels)['logits'])
        assert torch.equal(generated,m.generate(ids,max_new_tokens=4,eos_token_id=2))
    with pytest.raises(ValueError,match='strictly under'):
        CompactFutureOrder(a,rank=2,window=4,parameter_limit=m.counts()['total_unique'])


def test_real_auxiliary_learning_reaches_actor_keeps_base_and_critic_fixed_and_CE_control_has_no_auxiliary_gradient():
    m=CompactFutureOrder(actor(),rank=2,window=4);b=batch();fixed=original_signature(m.actor.base);critic=m.actor.parts['critic'].weight.detach().clone()
    before={n:p.detach().clone() for n,p in m.named_selected(auxiliary=True)}
    for auxiliary in (False,True):
        m.zero_grad(set_to_none=True);loss,record=m.objective(b,auxiliary_weight=.1 if auxiliary else 0.,chunk_positions=2)
        params=[p for _,p in m.named_selected(auxiliary=auxiliary)];loss.backward(inputs=params)
        assert all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in params)
        assert all(p.grad is None for p in m.actor.base.parameters()) and m.actor.parts['critic'].weight.grad is None
        if not auxiliary:assert all(p.grad is None for p in m.future.parameters())
        else:
            assert m.future['up'].weight.grad.abs().sum()>0
            opt=torch.optim.AdamW(params,lr=.001,foreach=False);opt.step();opt.zero_grad(set_to_none=True)
    assert fixed==original_signature(m.actor.base) and torch.equal(critic,m.actor.parts['critic'].weight)
    assert any(not torch.equal(before[n],p) for n,p in m.named_selected(auxiliary=True) if n.startswith('actor.'))
    assert any(not torch.equal(before[n],p) for n,p in m.named_selected(auxiliary=True) if n.startswith('future.'))


@pytest.mark.parametrize('damage',['missing_EOS','interior_EOS','hidden_gold','wrong_gold'])
def test_damaged_original_response_boundaries_reject_before_any_model_mutation(tmp_path,damage):
    m=CompactFutureOrder(actor(),rank=2,window=4);b=batch()
    if damage=='missing_EOS':b['input_ids'][0,-1]=b['labels'][0,-1]=4
    if damage=='interior_EOS':b['input_ids'][0,3]=b['labels'][0,3]=2
    if damage=='hidden_gold':b['attention_mask'][0,-1]=False
    if damage=='wrong_gold':b['labels'][0,3]=6
    before=original_signature(m);rng=torch.get_rng_state().clone()
    with pytest.raises(ValueError):m.objective(b)
    assert before==original_signature(m) and torch.equal(rng,torch.get_rng_state())
