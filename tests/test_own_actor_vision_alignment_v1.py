import copy

import pytest
import torch

from slm_lab.model import ModelConfig
from slm_lab.own_actor_vision_alignment_v1 import OwnActorVisionAlignment, pair_positives, symmetric_pair_loss
from slm_lab.own_dense_actor_v1 import OwnDenseActor
from slm_lab.own_dense_growth_v1 import OwnDenseAssistant, new_backbone


def test_pair_identity_is_not_transitive_and_loss_matches_pair_probabilities():
    torch.set_num_threads(1)
    positive = pair_positives(['a', 'a', 'b', 'b'], ['cat', 'generic', 'generic', 'dog'])
    assert positive[0, 1] and positive[1, 2] and positive[2, 3]
    assert positive[0, 2] and positive[3, 1]
    assert not positive[0, 3] and not positive[2, 0]
    images = torch.tensor([[1., .1], [.2, 1.], [1., .4], [.8, 1.]], requires_grad=True)
    texts = torch.tensor([[.3, 1.], [1., .2], [.5, 1.], [.1, 1.]], requires_grad=True)
    loss = symmetric_pair_loss(images, texts, positive, temperature=.7)
    scores = torch.nn.functional.normalize(images, dim=-1) @ torch.nn.functional.normalize(texts, dim=-1).T / .7
    expected = (-(scores.softmax(1) * positive).sum(1).log().mean()
                -(scores.softmax(0) * positive).sum(0).log().mean()) / 2
    assert torch.allclose(loss, expected, atol=1e-6)
    loss.backward()
    assert bool(torch.isfinite(images.grad).all()) and bool(torch.isfinite(texts.grad).all())
    with pytest.raises(ValueError):
        symmetric_pair_loss(images, texts, torch.ones(4, 4, dtype=torch.bool))
    with pytest.raises(ValueError):
        symmetric_pair_loss(images, texts, positive, temperature=float('nan'))


def test_native_image_caption_gradient_reaches_vision_without_changing_text():
    torch.set_num_threads(1)
    config = ModelConfig(vocab_size=64, hidden_size=32, intermediate_size=64, num_layers=2,
        num_heads=4, num_kv_heads=2, max_text_length=64, vision_hidden_size=32,
        vision_intermediate_size=64, vision_layers=1, vision_heads=4, image_size=32, patch_size=16)
    actor = OwnDenseActor(OwnDenseAssistant(new_backbone(config, 968117), reward_hidden=8,
        word_temporal=4, pointing_rank=8, router_hidden=8, detector_hidden=4, word_expansion=1),
        rank=2, seed=968217).eval()
    view = OwnActorVisionAlignment(actor).eval()
    ids = torch.tensor([[1, 9, 10, 2], [1, 11, 12, 2]])
    pixels = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        before_text = actor(ids)['logits'].clone()
        assert torch.equal(view.native(ids, pixels)['logits'], actor(ids, pixels)['logits'])
    original = {n: p.detach().clone() for n, p in actor.named_parameters()}
    selected = [p for _, p in view.named_selected()]
    adam = torch.optim.AdamW(selected, lr=1e-3, weight_decay=.01, foreach=False)
    batch = dict(input_ids=ids, labels=ids, pixel_values=pixels)
    objective = view.objective(batch, ids, torch.ones_like(ids, dtype=torch.bool), torch.eye(2, dtype=torch.bool))
    view.backward(objective)
    assert actor.base.network.vision.position.grad is not None
    assert bool(actor.base.network.vision.position.grad.abs().sum() > 0)
    torch.nn.utils.clip_grad_norm_(selected, 1.)
    adam.step(); adam.zero_grad(set_to_none=True)
    with torch.no_grad():
        assert torch.equal(before_text, actor(ids)['logits'])
    changed = [n for n, p in actor.named_parameters() if not torch.equal(original[n], p)]
    assert changed and all(n.startswith(('base.network.vision.', 'base.heads.pointing.')) for n in changed)
    assert view.actor.counts()['actual_unique'] == sum(p.numel() for p in view.parameters())
    assert len(adam.state) == len(selected)
    assert all(float(state['step']) == 1 for state in adam.state.values())
    bad = copy.deepcopy(batch); bad['labels'][:, -1] = -100
    with pytest.raises(ValueError):
        view.objective(bad, ids, torch.ones_like(ids, dtype=torch.bool), torch.eye(2, dtype=torch.bool))
