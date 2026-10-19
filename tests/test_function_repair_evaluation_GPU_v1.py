"""CPU fixtures verify device-aware observation; no GPU capacity is implied."""
from contextlib import nullcontext
import json

import pytest
import torch

from slm_lab.native_generation_observer_device_v1 import observe
from slm_lab.function_repair_evaluation_GPU_v1 import source_positions,validate_source_cohort
from test_own_dense_actor_v1 import actor


@pytest.mark.parametrize('precision',['fp32','bf16'])
def test_actual_native_text_and_positional_image_actions_return_unchanged_with_real_precision(precision):
    torch.set_num_threads(1);model=actor().policy.eval();ids=torch.tensor([[1,4,7]]);pixels=torch.linspace(-1,1,3*16*16).reshape(1,3,16,16)
    amp=(lambda:torch.autocast('cpu',dtype=torch.bfloat16)) if precision=='bf16' else nullcontext
    events=[];original=model.generate;rng=torch.get_rng_state().clone()
    for image in [None,pixels]:
        with torch.no_grad(),amp():
            wanted=original(ids,image,max_new_tokens=3,eos_token_id=2,use_cache=True)
            with observe(model,events.append,device='cpu',precision=precision):actual=model.generate(ids,image,max_new_tokens=3,eos_token_id=2,use_cache=True)
        assert torch.equal(wanted,actual) and events[-1]['actual_generated_token_ids']==actual[0,ids.shape[1]:].tolist()
        assert events[-1]['inference_device_type']=='cpu' and events[-1]['inference_precision']==precision
        assert (events[-1]['input_pixels_sha256'] is None)==(image is None)
    assert model.generate==original and torch.equal(rng,torch.get_rng_state()) and not torch.cuda.is_initialized()
    with pytest.raises(ValueError,match='precision'):
        with observe(model,events.append,device='cpu',precision='bf16'):model.generate(ids,max_new_tokens=1)
    assert model.generate==original


def test_all15source_validation_shards_preserve_original_roles_and_fixed_budgets(tmp_path):
    source=tmp_path/'cohort.json';rows=[dict(position=i,source_role='validation',optimization_forbidden=True,
        native_generation_budget=1024,original_prompt_ids=[1,5+i]) for i in range(15)]
    source.write_text(json.dumps(rows));assert validate_source_cohort(source)==rows
    assert sorted(p for r in range(4) for p in source_positions(r,4))==list(range(15))
    assert [len(source_positions(r,4)) for r in range(4)]==[4,4,4,3]
    for mode in ['dropped','role','budget']:
        altered=json.loads(json.dumps(rows))
        if mode=='dropped':altered.pop()
        elif mode=='role':altered[0]['source_role']='train'
        else:altered[0]['native_generation_budget']=128
        source.write_text(json.dumps(altered))
        with pytest.raises(ValueError):validate_source_cohort(source)
