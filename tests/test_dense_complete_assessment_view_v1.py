import pytest
import torch

from slm_lab.dense_native_view_v2 import CompleteDenseNativeView
from slm_lab.inference_rotary_cache_v1 import inference_cache
from slm_lab.native_greedy_observer_v1 import signature
from test_own_dense_growth_v1 import model as own_model


@pytest.mark.parametrize('bf16',[False,True])
def test_complete_view_preserves_all_parameters_forward_vision_cached_knowledge_and_actions(bf16):
    torch.set_num_threads(1);model,_=own_model();model.eval();view=CompleteDenseNativeView(model).eval()
    before=signature(model);rng=torch.get_rng_state().clone();ids=torch.tensor([[1,4,7,8]])
    pixels=torch.linspace(-1,1,3*32*32).reshape(1,3,32,32)
    assert signature(view)==before
    assert sum(p.numel() for p in view.parameters())==model.counts()['total_unique']
    assert view.vision is model.network.vision and view.config is model.config
    with torch.no_grad(),torch.autocast('cpu',dtype=torch.bfloat16,enabled=bf16),inference_cache():
        for image in [None,pixels]:
            expected=model(ids,pixel_values=image);actual=view(ids,pixel_values=image)
            assert torch.equal(expected['logits'],actual['logits'])
            initial=view.prefill(ids,pixel_values=image);original=model.network.prefill(ids,pixel_values=image)
            assert torch.equal(initial['logits'],original['logits'])
            token=torch.tensor([[6]])
            assert torch.equal(view.decode(token,initial['cache'])['logits'],model.network.decode(token,original['cache'])['logits'])
            assert torch.equal(model.generate(ids,image,max_new_tokens=3,eos_token_id=2,use_cache=True),
                view.generate(ids,image,max_new_tokens=3,eos_token_id=2,use_cache=True))
        assert torch.equal(view.vision(pixels),model.network.vision(pixels))
    assert signature(model)==signature(view)==before and torch.equal(rng,torch.get_rng_state())
