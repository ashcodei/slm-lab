import copy

import pytest
import torch

from slm_lab.own_task_experts_v1 import OwnTaskExperts, EXPERTS
from test_own_dense_growth_v1 import model as make_model
from test_own_aligned_assistant_v1 import original_signature
from test_own_alignment_checkpoint_v1 import assert_nested_equal

SOURCES = dict(own_base='a' * 64, training_sources='b' * 64)


def model():
    return OwnTaskExperts(make_model()[0], ranks=dict(code=2, assistant=1, reasoning=1)).eval()


def test_real_code_updates_leave_base_other_experts_and_their_existing_caches_exact():
    base, _ = make_model()
    rng = torch.get_rng_state().clone(); signature = original_signature(base)
    m = OwnTaskExperts(base, ranks=dict(code=2, assistant=1, reasoning=1)).eval()
    assert torch.equal(rng, torch.get_rng_state()) and signature == original_signature(base)
    ids = torch.tensor([[1, 4, 7, 8, 2]])
    pixels = torch.linspace(-1, 1, 3 * 16 * 16).reshape(1, 3, 16, 16)
    caches, before = {}, {}
    with torch.no_grad():
        for expert in EXPERTS:
            view = m.native_view(expert)
            assert sum(p.numel() for p in view.parameters()) == m.counts()['actual_unique']
            for image in [None, pixels]:
                assert torch.equal(m(ids, image, expert=expert)['logits'], base(ids, image)['logits'])
            before[expert] = m(ids, expert=expert)['logits'].clone()
            caches[expert] = view.prefill(ids)['cache']
        assistant_next = m.native_view('assistant').decode(torch.tensor([[6]]), caches['assistant'])['logits'].clone()
    initial = m.export(SOURCES)
    parameters = [p for _, p in m.named_selected('code')]
    optimizer = torch.optim.AdamW(parameters, lr=.003, betas=(.9, .95), weight_decay=.01, foreach=False)
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        m.train(); m(ids, labels=ids, expert='code')['loss'].backward()
        assert all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in parameters)
        assert all(p.grad is None for p in base.parameters())
        assert m.critic.weight.grad is None
        assert all(p.grad is None for name in ['assistant', 'reasoning'] for _, p in m.named_selected(name))
        optimizer.step()
    optimizer.zero_grad(set_to_none=True); m.eval()
    after = m.export(SOURCES)
    assert signature == original_signature(base)
    assert_nested_equal(initial['critic'], after['critic'])
    with torch.no_grad():
        assert not torch.equal(before['code'], m(ids, expert='code')['logits'])
        for name in ['assistant', 'reasoning']:
            assert_nested_equal(initial['adapters'][name], after['adapters'][name])
            assert torch.equal(before[name], m(ids, expert=name)['logits'])
        assert torch.equal(assistant_next, m.native_view('assistant').decode(torch.tensor([[6]]), caches['assistant'])['logits'])
        with pytest.raises(ValueError, match='unchanged'):
            m.native_view('code').decode(torch.tensor([[6]]), caches['code'])
    with pytest.raises(ValueError, match='automatic routing'):
        m(ids, expert='auto')


def test_cold_export_counts_all_experts_once_and_rejects_bad_last_expert_before_mutation():
    m = model(); ids = torch.tensor([[1, 6, 7, 2]])
    with torch.no_grad():
        # Give each adapter and shared critic distinct nonzero values before export.
        for index, name in enumerate(EXPERTS, 1):
            for tensor in m.experts[name].parts['linears']:
                tensor.up.add_(index * .001)
        m.critic.weight.fill_(.03)
    payload = m.export(SOURCES); cold = model()
    cold.import_export(payload, expected_sources=SOURCES)
    assert_nested_equal(payload, cold.export(SOURCES))
    assert len({id(cold.experts[n].parts['critic']) for n in EXPERTS}) == 1
    with torch.no_grad():
        for name in EXPERTS:
            assert torch.equal(m(ids, expert=name)['logits'], cold(ids, expert=name)['logits'])
    for mutate in [lambda p: p['adapters']['reasoning']['linears.0.up'].fill_(float('nan')),
                   lambda p: p.update(frozen_dense_sha256='c' * 64),
                   lambda p: p['actual_counts'].update(actual_unique=1)]:
        bad = copy.deepcopy(payload); mutate(bad); signature = original_signature(cold)
        with pytest.raises(ValueError):
            cold.import_export(bad, expected_sources=SOURCES)
        assert signature == original_signature(cold)
        assert_nested_equal(payload, cold.export(SOURCES))
    signature = original_signature(m.base); rng = torch.get_rng_state().clone()
    with pytest.raises(ValueError, match='strictly below'):
        OwnTaskExperts(m.base, ranks=dict(code=2, assistant=1, reasoning=1), parameter_limit=1000)
    assert signature == original_signature(m.base) and torch.equal(rng, torch.get_rng_state())
