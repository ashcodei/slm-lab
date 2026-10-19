import copy

import pytest
import torch

from slm_lab.frozen_own_actor_reference_v1 import FrozenOwnActorReference, signature
from test_own_dense_actor_v1 import actor


def test_snapshot_retains_actual_trained_actor_while_new_policy_learns_without_copying_or_freezing_shared_base():
    m = actor(); ids = torch.tensor([[1, 5, 7, 2]])
    parameters = [p for _, p in m.named_selected(('actor',))]
    optimizer = torch.optim.AdamW(parameters, lr=.002, foreach=False)
    m(ids, labels=ids)['loss'].backward(inputs=parameters); optimizer.step(); optimizer.zero_grad(set_to_none=True)
    base_signature = signature(m.base.parameters()); rng = torch.get_rng_state().clone()
    reference = FrozenOwnActorReference(m, sources=dict(own_checkpoint='a' * 64))
    assert torch.equal(rng, torch.get_rng_state()) and signature(m.base.parameters()) == base_signature
    saved = {k: t.clone() for k, t in reference.parts.state_dict().items()}
    with torch.no_grad():
        before = reference(ids)['logits'].clone()
        assert torch.equal(before, m(ids)['logits'])
        assert not torch.equal(before, m.base(ids)['logits'])
        pixels = torch.linspace(-1, 1, 3 * 16 * 16).reshape(1, 3, 16, 16)
        assert torch.equal(reference(ids, pixels)['logits'], m(ids, pixels)['logits'])
    m(ids, labels=ids)['loss'].backward(inputs=parameters); optimizer.step(); optimizer.zero_grad(set_to_none=True)
    reference.assert_bound(m)
    with torch.no_grad():
        assert torch.equal(before, reference(ids)['logits'])
        assert not torch.equal(before, m(ids)['logits'])
    assert all(torch.equal(t, saved[k]) for k, t in reference.parts.state_dict().items())
    assert signature(m.base.parameters()) == base_signature
    inventory = reference.resident_inventory(m)
    assert inventory['exported_student_values'] == m.counts()['actual_unique']
    assert inventory['total_unique_resident_values'] == m.counts()['actual_unique'] + sum(p.numel() for p in m.parts.parameters())
    binding = reference.binding(); binding['sources']['own_checkpoint'] = 'b' * 64
    assert reference.binding()['sources']['own_checkpoint'] == 'a' * 64


def test_changed_snapshot_wrong_base_and_invalid_source_reject():
    m = actor(); reference = FrozenOwnActorReference(m, sources=dict(own_checkpoint='a' * 64))
    with pytest.raises(ValueError, match='snapshot'):
        reference.assert_bound(actor())
    with torch.no_grad():
        next(reference.parts.parameters()).add_(.1)
    with pytest.raises(ValueError, match='snapshot'):
        reference.assert_bound(m)
    before = signature(m.parameters()); rng = torch.get_rng_state().clone()
    with pytest.raises(ValueError, match='SHA'):
        FrozenOwnActorReference(m, sources=dict(own_checkpoint='invalid'))
    assert signature(m.parameters()) == before and torch.equal(rng, torch.get_rng_state())
