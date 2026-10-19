import copy

import pytest
import torch
from torch.nn import functional as F

from slm_lab.frozen_base_pointer_stage_v1 import make_head
from slm_lab.own_aligned_assistant_v1 import OwnAlignedAssistant, parameter_budget
from slm_lab.own_assistant_v1 import OwnAssistant
from test_own_code_delta_v1 import own_models, components
from slm_lab.own_code_delta_v1 import OwnCodeDelta


def fixture():
    base, target = own_models()
    language = OwnCodeDelta(base, components(base, target, 2), reserved_parameters=256).eval()
    return OwnAssistant(language, make_head(base, rank=8)).eval()


def original_signature(model):
    return [(id(p), p._version, p.data_ptr(), p.requires_grad) for p in model.parameters()]


def test_actual_shared_inventory_initial_equivalence_caches_and_new_gradients():
    original = fixture()
    ids = torch.tensor([[1, 4, 7, 2]])
    pixels = torch.linspace(-1, 1, 3*32*32).reshape(1, 3, 32, 32)
    with torch.no_grad():
        expected = {r: original(ids, pixels, route=r)['logits'].clone() for r in ('general', 'code', 'grounding')}
        code_cache = original.language.code.prefill(ids, pixels)['cache']
        next_code = original.language.code.decode(torch.tensor([[8]]), code_cache)['logits'].clone()
    signatures = original_signature(original)
    state = {n: t.clone() for n, t in original.state_dict().items()}
    rng = torch.get_rng_state().clone()
    model = OwnAlignedAssistant(original, rank=2, reward_hidden=8, word_temporal=4).eval()
    assert torch.equal(rng, torch.get_rng_state()) and signatures == original_signature(original)
    model.assert_shared()
    assert model.counts()['total_unique'] == parameter_budget(original, rank=2, reward_hidden=8, word_temporal=4)['total_unique']
    assert model.aligned.embedding is original.language.base.embedding
    assert sum(p.numel() for p in model.aligned.parameters()) == model.counts()['total_unique']
    with torch.no_grad():
        assert torch.allclose(model(ids, pixels)['logits'], expected['general'], atol=2e-7, rtol=2e-6)
        cached = model.aligned.prefill(ids, pixels)['cache']
        decoded = model.aligned.decode(torch.tensor([[8]]), cached)['logits']
        full = model(torch.tensor([[1, 4, 7, 2, 8]]), pixels)['logits'][:, -1:]
        assert torch.allclose(decoded, full, atol=2e-7, rtol=2e-6)
        with pytest.raises(ValueError):
            model.aligned.decode(torch.tensor([[8]]), code_cache)
        with pytest.raises(ValueError):
            original.language.code.decode(torch.tensor([[8]]), cached)
    parameters = model.extension_parameters('language', 'vision')
    assert not {id(p) for p in parameters} & {id(p) for p in original.parameters()}
    optimizer = torch.optim.AdamW(parameters, lr=.003, foreach=False)
    model.aligned.train(); model.aligned.gradient_checkpointing = True
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        model(ids, pixels, labels=ids)['loss'].backward()
        assert all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in parameters)
        assert all(p.grad is None for p in original.parameters())
        optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    model.eval()
    with pytest.raises(ValueError, match='unchanged actual parameters'):
        model.aligned.decode(torch.tensor([[8]]), cached)
    assert all(torch.equal(t, original.state_dict()[n]) for n, t in state.items())
    assert signatures == original_signature(original)
    with torch.no_grad():
        for route in expected:
            assert torch.equal(expected[route], model(ids, pixels, route=route)['logits'])
        assert torch.equal(next_code, original.language.code.decode(torch.tensor([[8]]), code_cache)['logits'])
        assert not torch.equal(expected['general'], model(ids, pixels)['logits'])


def test_actual_reward_critic_and_word_CTC_gradients_and_input_contracts():
    model = OwnAlignedAssistant(fixture(), rank=2, reward_hidden=8, word_temporal=4).eval()
    ids = torch.tensor([[1, 4, 7, 2], [1, 6, 2, 0]])
    mask = torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]], dtype=torch.bool)
    scores = model.preference_score(ids, attention_mask=mask)
    F.softplus(-(scores[0]-scores[1])).backward()
    assert all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.extension_parameters('reward'))
    assert all(p.grad is None for p in model.extension_parameters('language', 'vision', 'critic', 'word'))
    model.zero_grad(set_to_none=True)
    model(ids, attention_mask=mask)['values'].square().mean().add(model(ids)['values'].mean()).backward()
    assert all(p.grad is not None for p in model.extension_parameters('critic'))
    model.zero_grad(set_to_none=True)
    # Rectangular word images use the actual unpooled patch width, not an
    # invented full-photo OCR interface. No source bytes are truncated here.
    logits = model.word_logits(torch.randn(2, 3, 8, 32))
    assert logits.shape == (2, 8, 257)
    loss = F.ctc_loss(logits.log_softmax(-1).transpose(0, 1), torch.tensor([99, 97, 116, 97, 98]),
        torch.tensor([8, 8]), torch.tensor([3, 2]), blank=256)
    loss.backward()
    for p in model.extension_parameters('vision', 'word'):
        # Vision projection serves the multimodal prefix; CTC uses its input.
        if id(p) not in {id(q) for q in model.parts['vision_projection'].parameters()}:
            assert p.grad is not None and bool(torch.isfinite(p.grad).all())
    assert all(p.grad is None for p in model.original.parameters())
    with torch.no_grad():
        rollout = model.generate(ids[:1], max_new_tokens=2, temperature=1., return_rollout=True)
        assert rollout['logps'].shape == rollout['values'].shape == rollout['response_mask'].shape == (1, 2)
        assert torch.equal(rollout['values'], torch.zeros_like(rollout['values']))
    with pytest.raises(ValueError, match='right-padded'):
        model.preference_score(ids, attention_mask=torch.tensor([[1, 0, 1, 1], [1, 1, 1, 0]]))
    with pytest.raises(ValueError, match='patch multiples'):
        model.word_logits(torch.randn(1, 3, 7, 32))
    with pytest.raises(ValueError, match='square'):
        model(ids[:1], torch.randn(1, 3, 16, 32))


def test_new_only_roundtrip_corruption_rejection_preserves_original_and_rng():
    original = fixture()
    model = OwnAlignedAssistant(original, rank=2, reward_hidden=8, word_temporal=4).eval()
    sources = dict(original_policy_sha256='a'*64, tokenizer_sha256='b'*64)
    with torch.no_grad():
        for p in model.parts.parameters():
            p.add_(.001)
    payload = model.export(sources)
    assert not any('original' in n or 'embedding' in n for n in payload['extensions'])
    cold = OwnAlignedAssistant(original, rank=2, reward_hidden=8, word_temporal=4).eval()
    signature = original_signature(original)
    rng = torch.get_rng_state().clone()
    cold.import_export(payload, expected_sources=sources)
    assert signature == original_signature(original) and torch.equal(rng, torch.get_rng_state())
    ids = torch.tensor([[1, 4, 7, 2]])
    pixels = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        assert torch.equal(model(ids, pixels)['logits'], cold(ids, pixels)['logits'])
        assert torch.equal(model.word_logits(pixels), cold.word_logits(pixels))
        assert torch.equal(model.preference_score(ids), cold.preference_score(ids))
    expected = cold.extension_state()
    for mutate in [lambda s: s['sources'].update(tokenizer_sha256='c'*64),
                   lambda s: s['extensions'][next(iter(s['extensions']))].fill_(float('nan')),
                   lambda s: s['extensions'].pop(next(iter(s['extensions']))),
                   lambda s: s['actual_counts'].update(total_unique=1)]:
        bad = copy.deepcopy(payload); mutate(bad)
        before = original_signature(cold)
        with pytest.raises(ValueError):
            cold.import_export(bad, expected_sources=sources)
        assert before == original_signature(cold)
        assert all(torch.equal(t, cold.extension_state()[n]) for n, t in expected.items())
    with pytest.raises(ValueError, match='strictly below1B'):
        OwnAlignedAssistant(original, rank=2, reward_hidden=8, word_temporal=4,
                            parameter_limit=model.counts()['total_unique'])
    with pytest.raises(ValueError, match='rank'):
        OwnAlignedAssistant(original, rank=64)
    with pytest.raises(ValueError, match='Explicit distinct'):
        model.extension_parameters('language', 'language')
