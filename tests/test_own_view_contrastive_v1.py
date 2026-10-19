import math

import pytest
import torch

from slm_lab.own_code_delta_v1 import OwnCodeDelta
from slm_lab.own_view_contrastive_v1 import generate, scores
from test_own_code_delta_v1 import own_models, components


def fixture():
    torch.set_num_threads(1)
    general, code = own_models()
    return OwnCodeDelta(general, components(general, code, 2), reserved_parameters=0).eval()


def test_score_is_explicit_likelihood_ratio_with_expert_plausibility_and_rejects_invalid_inputs():
    code = torch.tensor([[2., 1., -20.]])
    base = torch.tensor([[4., -4., -40.]])
    expected = (code.log_softmax(-1) - .5 * base.log_softmax(-1)).masked_fill(code < code.max() + math.log(.1), -torch.inf)
    assert torch.equal(scores(code, base, strength=.5, plausibility=.1), expected)
    assert expected.argmax(-1).item() == 1  # Reject the implausible token2 despite its large ratio.
    for strength, plausibility in [(-.1, .1), (float('nan'), .1), (.5, 0), (.5, 1.1)]:
        with pytest.raises(ValueError):
            scores(code, base, strength=strength, plausibility=plausibility)
    with pytest.raises(ValueError):
        scores(code, base.fill_(float('inf')), strength=.5, plausibility=.1)


def test_cached_two_view_generation_matches_independent_full_prefix_math_and_preserves_shared_weights():
    model = fixture(); ids = torch.tensor([[1, 4, 7], [1, 8, 5]])
    pixels = torch.linspace(-1, 1, 2 * 3 * 32 * 32).reshape(2, 3, 32, 32)
    before = {n: p.clone() for n, p in model.named_parameters()}
    versions = [(id(p), p._version, p.data_ptr()) for p in model.parameters()]
    rng = torch.get_rng_state().clone(); calls = []
    with torch.no_grad():
        for strength in [0., .5, 1.]:
            actual = generate(model, ids, pixels, strength=strength, max_new_tokens=4, checked=lambda: calls.append(1))
            expected = ids.clone(); done = torch.zeros(2, dtype=torch.bool)
            for _ in range(4):
                code = model.code(expected, pixels)['logits'][:, -1].float().log_softmax(-1)
                if strength:
                    base = model.base(expected, pixels)['logits'][:, -1].float().log_softmax(-1)
                    objective = (code - strength * base).masked_fill(code < code.max(-1, keepdim=True).values + math.log(.1), -torch.inf)
                else:
                    objective = code
                token = objective.argmax(-1, keepdim=True)
                token = torch.where(done[:, None], 2, token); done |= token[:, 0] == 2
                expected = torch.cat((expected, token), dim=1)
                if done.all():
                    break
            assert torch.equal(actual['tokens'], expected)
            assert actual['inference_views'] == (2 if strength else 1)
            assert len(actual['trace']) == expected.shape[1] - ids.shape[1]
        assert torch.equal(generate(model, ids, pixels, strength=0., max_new_tokens=4)['tokens'],
                           model.code.generate(ids, pixels, max_new_tokens=4, eos_token_id=2))
    assert calls and torch.equal(rng, torch.get_rng_state())
    assert versions == [(id(p), p._version, p.data_ptr()) for p in model.parameters()]
    assert all(torch.equal(before[n], p) for n, p in model.named_parameters())
    assert model.code.vision is model.base.vision


def test_whole_context_and_checked_owner_rejection_precede_generation(monkeypatch):
    model = fixture(); calls = []
    monkeypatch.setattr(model.code, 'prefill', lambda *a: calls.append(1))
    with pytest.raises(ValueError, match='Whole original'):
        generate(model, torch.ones(1, model.config.max_text_length, dtype=torch.long), strength=.5, max_new_tokens=1)
    def reject():
        raise ValueError('owner ended')
    with pytest.raises(ValueError, match='owner ended'):
        generate(model, torch.tensor([[1, 4]]), strength=.5, max_new_tokens=2, checked=reject)
    assert calls == []
