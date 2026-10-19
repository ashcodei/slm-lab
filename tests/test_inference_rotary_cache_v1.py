import copy
import torch

from slm_lab import model as core
from slm_lab.inference_rotary_cache_v1 import ORIGINAL,cached_rotary,inference_cache,TABLES,COUNTS,MAXIMUM_TABLES
from test_own_dense_growth_v1 import model as make_model


def test_original_inference_values_match_bit_for_bit_for_multiple_shapes_precisions_and_offsets():
    TABLES.clear();COUNTS.update(hits=0,misses=0)
    for dtype in [torch.float32,torch.bfloat16]:
        for offset in [0,17,319]:
            x=torch.linspace(-1,1,2*4*7*8).reshape(2,4,7,8).to(dtype)
            with torch.no_grad():
                expected=ORIGINAL(x,10000.,offset);actual=cached_rotary(x,10000.,offset)
                assert torch.equal(expected,actual) and torch.equal(actual,cached_rotary(x,10000.,offset))
    assert COUNTS['hits']==6 and COUNTS['misses']==6


def test_native_model_prefill_decode_and_greedy_outputs_remain_exact_and_original_is_restored():
    m,_=make_model();ids=torch.tensor([[1,4,5]])
    with torch.no_grad():expected=m.generate(ids,max_new_tokens=7,eos_token_id=2)
    before=core.rotary
    with inference_cache():
        with torch.no_grad():actual=m.generate(ids,max_new_tokens=7,eos_token_id=2)
        assert torch.equal(expected,actual)
    assert core.rotary is before and COUNTS['hits']>0


def test_training_gradients_use_original_computation_and_cache_has_a_strict_table_bound():
    x=torch.randn(1,2,5,8,requires_grad=True);reference=x.detach().clone().requires_grad_()
    with inference_cache():cached_rotary(x,10000.,3).square().sum().backward()
    ORIGINAL(reference,10000.,3).square().sum().backward()
    assert torch.equal(x.grad,reference.grad)
    with torch.no_grad():
        for i in range(MAXIMUM_TABLES+3):cached_rotary(x.detach(),10000.,i)
    assert len(TABLES)<=MAXIMUM_TABLES
