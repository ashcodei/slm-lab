import copy

import pytest
import torch

from slm_lab.own_dense_actor_v1 import OwnDenseActor,budget
from slm_lab.own_dense_actor_checkpoint_v1 import make_state,restore_state
from test_own_dense_growth_v1 import model as make_model
from test_own_alignment_checkpoint_v1 import assert_nested_equal
from test_own_aligned_assistant_v1 import original_signature

SOURCES=dict(dense_source='a'*64,human_source='b'*64)


def actor():return OwnDenseActor(make_model()[0],rank=2).eval()


def test_one_counted_backbone_initial_text_image_cache_parity_and_frozen_metadata_and_RNG():
    base,_=make_model();signature=original_signature(base);rng=torch.get_rng_state().clone()
    model=OwnDenseActor(base,rank=2).eval();ids=torch.tensor([[1,4,7,2]]);pixels=torch.linspace(-1,1,3*16*16).reshape(1,3,16,16)
    assert torch.equal(rng,torch.get_rng_state()) and signature==original_signature(base)
    assert model.counts()['actual_unique']==budget(base,2)['total_unique']
    assert sum(p.numel() for p in model.policy.parameters())==model.counts()['actual_unique']
    with torch.no_grad():
        for image in (None,pixels):assert torch.equal(base(ids,image)['logits'],model(ids,image)['logits'])
        cache=model.policy.prefill(ids)['cache'];actual=model.policy.decode(torch.tensor([[5]]),cache)['logits']
        assert torch.allclose(actual,model(torch.tensor([[1,4,7,2,5]]))['logits'][:,-1:],atol=1e-7,rtol=2e-5)
        rollout=model.generate(ids,max_new_tokens=3,temperature=.7,return_rollout=True)
        assert rollout['logps'].shape==rollout['values'].shape==rollout['response_mask'].shape==(1,3)
        assert rollout['values'].count_nonzero()==0 and bool(torch.isfinite(rollout['logps']).all())
    assert signature==original_signature(base)


def test_actor_and_separate_critic_gradients_leave_every_base_value_flag_pointer_and_existing_cache_exact():
    model=actor();base=model.base;ids=torch.tensor([[1,4,7,8,2]]);signature=original_signature(base)
    with torch.no_grad():
        base_cache=base.network.prefill(ids)['cache'];before=base.network.decode(torch.tensor([[6]]),base_cache)['logits'].clone()
        actor_cache=model.policy.prefill(ids)['cache']
    opt=torch.optim.AdamW([p for _,p in model.named_selected(('actor','critic'))],lr=.001,foreach=False)
    model.train();model.policy.gradient_checkpointing=True
    out=model(ids,labels=ids);loss=out['loss']+(out['values']-1).square().mean();loss.backward()
    assert all(p.grad is None for p in base.parameters())
    assert all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for _,p in model.named_selected(('actor','critic')))
    assert model.parts['critic'].weight.grad.abs().sum()>0
    opt.step();opt.zero_grad(set_to_none=True);model.eval()
    assert signature==original_signature(base)
    with torch.no_grad():assert torch.equal(before,base.network.decode(torch.tensor([[6]]),base_cache)['logits'])
    with pytest.raises(ValueError,match='unchanged'):model.policy.decode(torch.tensor([[6]]),actor_cache)


def test_selected_Adam_recovery_preserves_payload_unused_critic_and_source_graph_with_corrupt_rejection():
    model=actor();ids=torch.tensor([[1,4,7,2]]);params=[p for _,p in model.named_selected(('actor',))]
    optimizer=torch.optim.AdamW(params,lr=.001,betas=(.9,.95),weight_decay=.01,foreach=False)
    critic=(model.parts['critic'].weight.clone(),model.parts['critic'].weight._version)
    def update():
        optimizer.zero_grad(set_to_none=True);loss=model(ids,labels=ids)['loss'];loss.backward();optimizer.step()
        optimizer.zero_grad(set_to_none=True);return dict(loss=float(loss.detach()))
    first=update();state=make_state(model,optimizer,groups=('actor',),sources=SOURCES,completed=1,planned=3,cursor=dict(draws=1),metrics=[first],device='cpu')
    pristine=copy.deepcopy(state);expected=update();weights=model.export(SOURCES);adam=copy.deepcopy(optimizer.state_dict());rng=torch.get_rng_state().clone()
    restore_state(model,optimizer,state,groups=('actor',),expected_sources=SOURCES,planned=3,expected_cursor=dict(draws=1),device='cpu')
    assert update()==expected;assert_nested_equal(weights,model.export(SOURCES));assert_nested_equal(adam,optimizer.state_dict())
    assert_nested_equal(pristine,state);assert torch.equal(rng,torch.get_rng_state())
    assert torch.equal(critic[0],model.parts['critic'].weight) and critic[1]==model.parts['critic'].weight._version
    for mutate in [lambda s:s['model'].update(frozen_dense_sha256='c'*64),
                   lambda s:s['model']['parts']['critic.weight'].add_(1),lambda s:s['cursor'].update(draws=9),
                   lambda s:s['optimizer']['state'][0]['exp_avg'].fill_(float('nan')),
                   lambda s:s['rng'].update(torch=torch.zeros(4,dtype=torch.uint8))]:
        bad=copy.deepcopy(state);mutate(bad);before=original_signature(model)
        with pytest.raises(ValueError):restore_state(model,optimizer,bad,groups=('actor',),expected_sources=SOURCES,planned=3,expected_cursor=dict(draws=1),device='cpu')
        assert before==original_signature(model);assert_nested_equal(weights,model.export(SOURCES))
        assert_nested_equal(adam,optimizer.state_dict());assert torch.equal(rng,torch.get_rng_state())
    cold=actor();cold.import_export(weights,expected_sources=SOURCES)
    with torch.no_grad():assert torch.equal(model(ids)['logits'],cold(ids)['logits'])
