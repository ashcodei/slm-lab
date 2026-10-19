import copy

import pytest
import torch

from slm_lab.own_aligned_assistant_v2 import OwnAlignedAssistantV2
from slm_lab.own_aligned_assistant_v3 import OwnAlignedAssistantV3
from slm_lab.own_alignment_checkpoint_v1 import make_state, restore_state
from slm_lab.whole_human_sft_stage_v1 import _LanguageRestoreView
from test_own_aligned_assistant_v1 import fixture, original_signature
from test_own_alignment_checkpoint_v1 import assert_nested_equal


SOURCES = dict(original_source='a'*64)


def model():
    return OwnAlignedAssistantV3(fixture(),rank=2,reward_hidden=8,word_temporal=4).eval()


def test_actual_same_count_initial_code_parity_and_route_cache_isolation():
    original = fixture(); before = original_signature(original)
    v2 = OwnAlignedAssistantV2(original,rank=2,reward_hidden=8,word_temporal=4).eval()
    v3 = OwnAlignedAssistantV3(original,rank=2,reward_hidden=8,word_temporal=4).eval()
    assert v3.counts() == v2.counts() and original_signature(original) == before
    assert v3.aligned.embedding.weight is original.language.code.embedding.weight
    assert v3.aligned.lm_head.weight is original.language.code.embedding.weight
    ids = torch.tensor([[1,4,7,2]]); pixels = torch.linspace(-1,1,3*32*32).reshape(1,3,32,32)
    with torch.no_grad():
        assert torch.equal(v2(ids,pixels)['logits'],original(ids,pixels,route='general')['logits'])
        assert torch.equal(v3(ids,pixels)['logits'],original(ids,pixels,route='code')['logits'])
        assert not torch.equal(v2(ids,pixels)['logits'],v3(ids,pixels)['logits'])
        cache = v3.aligned.prefill(ids,pixels)['cache']
        expected = v3(torch.tensor([[1,4,7,2,5]]),pixels)['logits'][:,-1:]
        assert torch.allclose(expected,v3.aligned.decode(torch.tensor([[5]]),cache)['logits'],atol=2e-7,rtol=2e-6)
        with pytest.raises(ValueError,match='route'):
            original.language.code.decode(torch.tensor([[5]]),cache)


def test_real_code_initialized_new_language_gradients_and_exact_recovery_leave_original_code_fixed():
    v3 = model(); fixed_ids = {id(p) for p in v3.extension_parameters('language')}
    fixed = [(p,p.detach().clone(),p._version,p.data_ptr()) for p in v3.parameters() if id(p) not in fixed_ids]
    optimizer = torch.optim.AdamW(v3.extension_parameters('language'),lr=.003,foreach=False)
    ids = torch.tensor([[1,4,7,5,2]])
    with torch.no_grad():
        code_cache = v3.original.language.code.prefill(ids)['cache']
        original_next = v3.original.language.code.decode(torch.tensor([[6]]),code_cache)['logits'].clone()
    def update():
        optimizer.zero_grad(set_to_none=True); loss = v3(ids,labels=ids)['loss']; loss.backward()
        assert all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in v3.extension_parameters('language'))
        assert all(p.grad is None for p in v3.original.parameters())
        assert all(p.grad is None for p in v3.extension_parameters('vision','reward','critic','word'))
        optimizer.step(); optimizer.zero_grad(set_to_none=True); return float(loss.detach())
    first = update()
    state = make_state(v3,optimizer,groups=('language',),sources=SOURCES,completed_steps=1,planned_steps=3,
        sampler_state=dict(draws=1),metrics=[dict(loss=first)],device='cpu')
    expected = update(); weights = v3.extension_state(); adam = copy.deepcopy(optimizer.state_dict())
    restore_state(_LanguageRestoreView(v3),optimizer,state,groups=('language',),expected_sources=SOURCES,
        planned_steps=3,expected_sampler_state=dict(draws=1),device='cpu')
    assert update() == expected
    assert_nested_equal(weights,v3.extension_state()); assert_nested_equal(adam,optimizer.state_dict())
    assert all(torch.equal(p,t) and p._version==version and p.data_ptr()==pointer for p,t,version,pointer in fixed)
    with torch.no_grad():
        assert torch.equal(original_next,v3.original.language.code.decode(torch.tensor([[6]]),code_cache)['logits'])


def test_versioned_trained_code_initialized_roundtrip_cannot_be_misread_as_general_initialization():
    v3 = model(); ids = torch.tensor([[1,4,6,2]])
    optimizer = torch.optim.AdamW(v3.extension_parameters('language'),lr=.003,foreach=False)
    v3(ids,labels=ids)['loss'].backward(); optimizer.step(); optimizer.zero_grad(set_to_none=True)
    payload = v3.export(SOURCES)
    cold = OwnAlignedAssistantV3(v3.original,rank=2,reward_hidden=8,word_temporal=4).eval()
    cold.import_export(payload,expected_sources=SOURCES)
    with torch.no_grad(): assert torch.equal(v3(ids)['logits'],cold(ids)['logits'])
    v2 = OwnAlignedAssistantV2(v3.original,rank=2,reward_hidden=8,word_temporal=4).eval()
    with pytest.raises(ValueError): v2.import_export(payload,expected_sources=SOURCES)
    with pytest.raises(ValueError,match='v3 computation'):
        cold.import_export(v2.export(SOURCES),expected_sources=SOURCES)
