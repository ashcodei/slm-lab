import pytest
import torch

from slm_lab.code_residual_branch_v1 import RoutedCodeResidual
from slm_lab.code_source_generation_diagnostic_v1 import OriginalBaseView, reference_agreement
from slm_lab.model import ModelConfig, TinyMultimodalLM
from slm_lab.recurrent_ablation import state_digest


def test_shared_base_is_exact_after_actual_residual_updates():
    torch.manual_seed(67)
    config = ModelConfig(vocab_size=32, hidden_size=16, intermediate_size=32, num_layers=2,
        num_heads=4, num_kv_heads=2, max_text_length=32, vision_hidden_size=8,
        vision_intermediate_size=16, vision_layers=1, vision_heads=2,
        image_size=8, patch_size=4, pool_size=2)
    original = TinyMultimodalLM(config)
    routed = RoutedCodeResidual(original, width=12)
    ids = torch.tensor([[1, 5, 6, 7, 2]])
    opt = torch.optim.AdamW(routed.adapter_parameters(), lr=.003, foreach=False)
    before = state_digest(original.state_dict())
    for _ in range(3):
        opt.zero_grad(set_to_none=True)
        routed(ids, labels=ids, route='code')['loss'].backward()
        opt.step()
    routed.eval()
    view = OriginalBaseView(routed.code).eval()
    assert state_digest(view.state_dict()) == state_digest(original.state_dict()) == before
    assert {id(p) for p in view.parameters()} == {id(p) for p in original.parameters()}
    image = torch.randn(1, 3, 8, 8)
    with torch.no_grad():
        for pixels in [None, image]:
            expected = original(ids, pixel_values=pixels)['logits']
            actual = view(ids, pixel_values=pixels)['logits']
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            a = view.prefill(ids[:, :3], pixel_values=pixels)
            b = original.prefill(ids[:, :3], pixel_values=pixels)
            torch.testing.assert_close(a['logits'], b['logits'], rtol=0, atol=0)
            a = view.decode(ids[:, 3:4], a['cache'])
            b = original.decode(ids[:, 3:4], b['cache'])
            torch.testing.assert_close(a['logits'], b['logits'], rtol=0, atol=0)
        assert torch.equal(view.generate(ids, max_new_tokens=5), original.generate(ids, max_new_tokens=5))


def test_base_view_rejects_an_unrelated_model():
    with pytest.raises(ValueError):
        OriginalBaseView(torch.nn.Linear(2, 2))


@pytest.mark.parametrize('actual,expected', [
    ([5, 6, 2], (True, 3, True, True)),
    ([5, 7, 2], (True, 1, False, True)),
    ([7, 7, 7], (False, 0, False, False)),
    ([], (False, 0, False, False)),
])
def test_reference_agreement_preserves_terminator_and_prefix_meaning(actual, expected):
    result = reference_agreement(actual, [5, 6, 2], 2)
    assert tuple(result[k] for k in ['first_reference_token_match', 'matching_reference_prefix_tokens',
        'exact_complete_reference_token_match', 'terminal_EOS']) == expected
