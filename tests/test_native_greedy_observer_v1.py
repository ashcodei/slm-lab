import pytest
import torch

from slm_lab.model import ModelConfig,TinyMultimodalLM
from slm_lab.native_greedy_observer_v1 import observe,signature


def model():
    return TinyMultimodalLM(ModelConfig(vocab_size=32,hidden_size=16,num_layers=1,num_heads=2,num_kv_heads=1,
        intermediate_size=32,max_text_length=32,image_size=16,patch_size=8,pool_size=1,
        vision_hidden_size=8,vision_layers=1,vision_heads=1,vision_intermediate_size=16)).eval()


def test_real_small_neural_generation_is_identical_and_actions_are_actual_returned_tensor_values():
    m=model();ids=torch.tensor([[1,4,5]]);before=signature(m)
    with torch.no_grad():expected=m.generate(ids,max_new_tokens=3,eos_token_id=2,use_cache=True)
    seen=[]
    with observe(m,seen.append):actual=m.generate(ids,max_new_tokens=3,eos_token_id=2,use_cache=True)
    assert torch.equal(expected,actual) and before==signature(m)
    assert seen[0]['original_input_token_ids']==[1,4,5]
    assert seen[0]['actual_generated_token_ids']==actual[0,3:].tolist()
    assert seen[0]['actual_native_action_count']==actual.shape[1]-3
    assert seen[0]['selected_action_log_probabilities_not_retained']


def test_sampling_or_nested_capture_rejects_and_original_method_is_restored_after_failure():
    m=model();original=m.generate;ids=torch.tensor([[1,4]])
    with pytest.raises(ValueError):
        with observe(m,lambda x:None):m.generate(ids,max_new_tokens=3,temperature=.7,eos_token_id=2,use_cache=True)
    assert m.generate==original and not m._native_greedy_observer_active_v1
    with observe(m,lambda x:None):
        with pytest.raises(ValueError):
            with observe(m,lambda x:None):pass
