from contextlib import nullcontext
from copy import deepcopy

import pytest
import torch

from slm_lab.anchored_preference_gradients_v1 import gradients, score
from slm_lab.model import ModelConfig, TinyMultimodalLM
from slm_lab.objectives import dpo_loss


def setup():
    torch.manual_seed(718)
    model = TinyMultimodalLM(ModelConfig(vocab_size=24, hidden_size=16, intermediate_size=32,
        num_layers=2, num_heads=4, num_kv_heads=2, max_text_length=32,
        vision_hidden_size=16, vision_intermediate_size=32, vision_layers=1,
        vision_heads=4, image_size=16, patch_size=8, pool_size=2))
    def batch(ids, start):
        ids = torch.tensor([ids]); labels = ids.clone(); labels[:, :start] = -100
        return {'input_ids': ids, 'labels': labels, 'attention_mask': torch.ones_like(ids, dtype=torch.bool),
                'pixel_values': None}
    return model, batch([1, 2, 3, 4, 5, 6, 7], 3), batch([1, 2, 3, 8, 9], 3)


@pytest.mark.parametrize('weight', [0., .1, 1.])
@pytest.mark.parametrize('checkpointed', [False, True])
def test_separate_response_graphs_match_full_joint_objective_gradients(weight, checkpointed):
    model, chosen, rejected = setup(); model.gradient_checkpointing = checkpointed
    ordinary = deepcopy(model)
    with torch.no_grad():
        ref_good = score(model, chosen) - .7
        ref_bad = score(model, rejected) + .3
    good, bad = score(ordinary, chosen), score(ordinary, rejected)
    expected = -good.mean() / 4 + weight * dpo_loss(good, bad, ref_good, ref_bad, .2)
    expected.backward()
    actual = gradients(model, chosen, rejected, ref_good, ref_bad, nullcontext, .2, weight)
    assert actual['loss'] == pytest.approx(float(expected.detach()))
    assert actual['chosen_target_tokens'] == 4 and actual['rejected_target_tokens'] == 2
    for (name, original), (_, sequential) in zip(ordinary.named_parameters(), model.named_parameters()):
        if original.grad is None:
            assert sequential.grad is None, name
        else:
            torch.testing.assert_close(sequential.grad, original.grad, rtol=3e-5, atol=3e-7, msg=name)


def test_live_reference_scores_and_nonfinite_configuration_are_rejected():
    model, chosen, rejected = setup()
    with torch.no_grad(): good, bad = score(model, chosen), score(model, rejected)
    with pytest.raises(ValueError, match='frozen reference'):
        gradients(model, chosen, rejected, good.requires_grad_(), bad, nullcontext)
    for kwargs in [{'beta': float('nan')}, {'preference_weight': -1.}, {'preference_weight': float('inf')}]:
        with pytest.raises(ValueError, match='Finite positive'):
            gradients(model, chosen, rejected, good.detach(), bad, nullcontext, **kwargs)
