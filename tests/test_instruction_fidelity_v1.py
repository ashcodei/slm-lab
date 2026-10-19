from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F

from slm_lab.instruction_fidelity_v1 import token_metrics, control_sequence, prefix_metrics, padding_cache_diagnostic
from slm_lab.model import ModelConfig, TinyMultimodalLM


def test_response_boundary_next_token_shift_and_eos_are_preserved():
    ids=torch.tensor([[1,6,5,4,2]])
    logits=torch.randn(1,5,8,generator=torch.Generator().manual_seed(19))
    result=token_metrics(logits,ids,3)
    expected=F.cross_entropy(logits[0,2:4],ids[0,3:],reduction='sum')
    assert result['tokens']==2 and result['nll_sum']==pytest.approx(float(expected))
    assert result['first_token_gold_id']==4
    altered=logits.clone();altered[0,:2]=1000
    assert token_metrics(altered,ids,3)==result
    assert result['first_8']['tokens']==2 and result['after_128']['tokens']==0


def test_control_retains_whole_other_prompt_and_only_declared_target_window():
    row={'probe_id':'a','language':'Python','actual_training_ids':[1,10,11,40,41,42,2],'actual_training_target_start':3}
    other={'probe_id':'b','language':'Python','actual_training_ids':[1,21,22,23,99,2],'actual_training_target_start':4}
    assert control_sequence(row,other,8,target_budget=3)==([1,21,22,23,40,41,42],4)
    with pytest.raises(ValueError,match='never clip'):
        control_sequence(row,other,6,target_budget=3)
    with pytest.raises(ValueError,match='Distinct'):
        control_sequence(row,row,8)


def test_missing_generated_tokens_remain_in_prefix_denominator():
    result=prefix_metrics([4],[4,5,2],3)
    assert result['compared_gold_prefix_tokens']==3 and result['position_matched_tokens']==1
    assert not result['exact_full_training_response_tokens'] and not result['exact_bounded_gold_prefix']
    assert prefix_metrics([4,5,2],[4,5,2],3)['exact_full_training_response_tokens']
    assert prefix_metrics([4,5],[4,5,6,2],2)['exact_full_training_response_tokens'] is None


def test_metric_rejects_invalid_boundaries_and_nonfinite_target_probabilities():
    ids=torch.tensor([[1,3,2]]);logits=torch.zeros(1,3,5)
    with pytest.raises(ValueError):token_metrics(logits,ids,0)
    logits[0,1,2]=float('nan')
    with pytest.raises(ValueError,match='Finite'):token_metrics(logits,ids,1)


def test_real_own_architecture_padding_and_first_response_cache_match():
    torch.set_num_threads(1);torch.manual_seed(123)
    config=ModelConfig(vocab_size=32,hidden_size=16,intermediate_size=32,num_layers=2,
        num_heads=4,num_kv_heads=2,max_text_length=32,vision_hidden_size=16,
        vision_intermediate_size=32,vision_layers=1,vision_heads=4,image_size=16,patch_size=4,pool_size=2)
    model=TinyMultimodalLM(config).eval();ids=torch.tensor([[1,7,8,9,10,2]])
    before={k:v.clone() for k,v in model.state_dict().items()}
    result=padding_cache_diagnostic(model,ids,3)
    assert result['padded_vs_unpadded_max_abs']<1e-5
    assert result['first_response_full_vs_cached_max_abs']<1e-5
    assert result['first_response_argmax_agrees']
    assert all(torch.equal(v,before[k]) for k,v in model.state_dict().items())


def test_a_future_leaking_model_is_detected_by_first_response_comparison():
    class Leaking:
        config=SimpleNamespace(max_text_length=32)
        def __call__(self,ids,**kwargs):
            logits=torch.zeros(ids.shape[0],ids.shape[1],16)
            logits[:,:,int(ids[0,-1])]=10
            return {'logits':logits}
        def prefill(self,ids):return self(ids)
    result=padding_cache_diagnostic(Leaking(),torch.tensor([[1,4,5,7,2]]),3)
    assert result['first_response_full_vs_cached_max_abs']==10
    assert not result['first_response_argmax_agrees']
