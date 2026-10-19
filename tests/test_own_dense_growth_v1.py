import copy

import pytest
import torch

from slm_lab.own_dense_growth_v1 import DenseBackbone,OwnDenseAssistant,depth_layout,fold_code,grow
from slm_lab.own_dense_checkpoint_v1 import make_state,restore_state
from slm_lab.own_aligned_assistant_v1 import byte_ctc_loss
from test_own_aligned_assistant_v1 import fixture,original_signature
from test_own_alignment_checkpoint_v1 import assert_nested_equal

SOURCES=dict(own_original='a'*64)
HEADS=dict(reward_hidden=8,word_temporal=4,pointing_rank=8,router_hidden=8,detector_hidden=4)


def model():
    original=fixture();folded=fold_code(original.language);network,layout=grow(folded,5)
    return OwnDenseAssistant(network,**HEADS).eval(),layout


def test_own_code_fold_and_interleaved_identity_growth_preserve_initial_computation_without_source_aliases():
    source=fixture();signatures=original_signature(source);rng=torch.get_rng_state().clone()
    folded=fold_code(source.language);grown,layout=grow(folded,5);candidate=OwnDenseAssistant(grown,**HEADS).eval()
    assert torch.equal(rng,torch.get_rng_state()) and signatures==original_signature(source)
    assert layout==depth_layout(2,5) and [i for i in layout if i>=0]==[0,1] and layout.count(-1)==3
    ids=torch.tensor([[1,4,7,2]]);pixels=torch.linspace(-1,1,3*32*32).reshape(1,3,32,32)
    with torch.no_grad():
        for image in (None,pixels):
            old=source(ids,image,route='code')['logits'];dense=folded(ids,image)['logits'];expanded=candidate(ids,image)['logits']
            assert torch.allclose(old,dense,atol=8e-7,rtol=2e-6)
            assert torch.equal(dense,expanded)
    assert not {p.data_ptr() for p in source.parameters()}&{p.data_ptr() for p in candidate.parameters()}
    assert candidate.network.embedding.weight is candidate.network.lm_head.weight
    assert candidate.counts()['total_unique']==sum(p.numel() for p in candidate.parameters())
    proposal=candidate.future_actor_parameter_budget(2)
    assert proposal['proposed_total']==proposal['actual_dense_model']+proposal['proposed_actor_values']
    with pytest.raises(ValueError,match='no shrinking'):depth_layout(5,2)


def test_new_identity_blocks_receive_output_gradients_then_inner_gradients_and_cache_binds_actual_weights():
    m,layout=model();ids=torch.tensor([[1,4,7,8,2]])
    with torch.no_grad():
        cache=m.network.prefill(ids)['cache']
        assert torch.allclose(m.network.decode(torch.tensor([[6]]),cache)['logits'],m(torch.tensor([[1,4,7,8,2,6]]))['logits'][:,-1:],atol=1e-7,rtol=1e-5)
    params=[p for _,p in m.language_named_parameters()];optimizer=torch.optim.AdamW(params,lr=.003,foreach=False)
    m.train();m.network.gradient_checkpointing=True
    for step in range(2):
        optimizer.zero_grad(set_to_none=True);m(ids,labels=ids)['loss'].backward(inputs=params)
        assert all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in params)
        for position,old in enumerate(layout):
            if old<0:
                b=m.network.blocks[position]
                assert b.attn.o_proj.weight.grad.abs().sum()>0 and b.down.weight.grad.abs().sum()>0
                if step==0:assert b.attn.q_proj.weight.grad.count_nonzero()==0 and b.gate.weight.grad.count_nonzero()==0
                else:assert b.attn.q_proj.weight.grad.abs().sum()>0 and b.gate.weight.grad.abs().sum()>0
        optimizer.step()
    optimizer.zero_grad(set_to_none=True);m.eval()
    with pytest.raises(ValueError,match='unchanged'):m.network.decode(torch.tensor([[6]]),cache)


def test_full_dense_language_Adam_RNG_recovery_and_all_fixed_versions_with_corrupt_rejection_before_mutation():
    m,_=model();named=m.language_named_parameters();params=[p for _,p in named];selected={id(p) for p in params}
    fixed=[(p,p.detach().clone(),p._version,p.data_ptr(),p.requires_grad) for p in m.parameters() if id(p) not in selected]
    optimizer=torch.optim.AdamW(params,lr=.001,betas=(.9,.95),weight_decay=.01,foreach=False)
    ids=torch.tensor([[1,4,7,2]])
    def update():
        optimizer.zero_grad(set_to_none=True);loss=m(ids,labels=ids)['loss'];loss.backward(inputs=params)
        optimizer.step();optimizer.zero_grad(set_to_none=True);return dict(loss=float(loss.detach()),actual_full_targets=3)
    first=update();state=make_state(m,optimizer,sources=SOURCES,completed=1,planned=3,cursor=dict(draws=1),metrics=[first],device='cpu')
    pristine=copy.deepcopy(state)
    expected=update();weights=m.export(SOURCES);adam=copy.deepcopy(optimizer.state_dict());rng=torch.get_rng_state().clone()
    restore_state(m,optimizer,state,expected_sources=SOURCES,planned=3,expected_cursor=dict(draws=1),device='cpu')
    assert update()==expected;assert_nested_equal(weights,m.export(SOURCES));assert_nested_equal(adam,optimizer.state_dict())
    assert_nested_equal(pristine,state)
    assert torch.equal(rng,torch.get_rng_state())
    assert all(torch.equal(p,t) and (p._version,p.data_ptr(),p.requires_grad)==(v,ptr,flag) for p,t,v,ptr,flag in fixed)
    def broken_tie(s):s['model']['network']['lm_head.weight']=s['model']['network']['lm_head.weight'].clone()+1.
    for mutate in [lambda s:s['cursor'].update(draws=99),lambda s:s['model']['heads']['reward.0.weight'].add_(1.),
                   lambda s:s['optimizer']['state'][0]['exp_avg'].fill_(float('nan')),
                   lambda s:s['rng'].update(torch=torch.zeros(4,dtype=torch.uint8)),
                   broken_tie]:
        bad=copy.deepcopy(state);mutate(bad);before=original_signature(m)
        with pytest.raises(ValueError):restore_state(m,optimizer,bad,expected_sources=SOURCES,planned=3,expected_cursor=dict(draws=1),device='cpu')
        assert before==original_signature(m);assert_nested_equal(weights,m.export(SOURCES));assert_nested_equal(adam,optimizer.state_dict())
        assert torch.equal(rng,torch.get_rng_state())
    cold,_=model();cold.import_export(weights,expected_sources=SOURCES)
    with torch.no_grad():assert torch.equal(m(ids)['logits'],cold(ids)['logits'])


def test_complete_heads_share_the_same_real_network_and_have_operational_byte_reward_point_and_detector_paths():
    m,_=model();ids=torch.tensor([[1,4,7,2]]);pixels=torch.linspace(-1,1,3*16*16).reshape(1,3,16,16)
    scores=m.preference_score(ids);scores.sum().backward()
    assert all(p.grad is not None for p in m.heads['reward'].parameters())
    assert all(p.grad is None for p in m.network.parameters());m.zero_grad(set_to_none=True)
    logits=m.word_logits(torch.ones(1,3,8,32));assert logits.shape==(1,40,257)
    byte_ctc_loss(logits,['aaé']).backward();assert all(p.grad is not None for p in m.heads['word'].parameters())
    m.zero_grad(set_to_none=True)
    assert m.point_logits(ids,pixels).shape==(1,4)
    assert m.detection_logits(pixels).shape==(1,5,4,4)
    hidden=m(ids,return_hidden=True,return_logits=False)['hidden'][:,-1]
    assert m.heads['router'](hidden).shape==(1,3)
    with pytest.raises(ValueError,match='RGB'):m.word_logits(torch.zeros(1,3,7,32))
