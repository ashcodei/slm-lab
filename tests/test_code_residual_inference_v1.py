from dataclasses import asdict
import copy

import pytest
import torch

from slm_lab.code_residual_branch_v1 import RoutedCodeResidual
from slm_lab.code_residual_checkpoint_v1 import make_training_state
from slm_lab.code_residual_inference_v1 import load_residual_policy
from slm_lab.code_residual_pilot_v1 import PROFILES
from slm_lab.corpus import file_hash
from slm_lab.model import ModelConfig,TinyMultimodalLM


def artifacts(tmp_path):
    torch.manual_seed(73)
    config=ModelConfig(vocab_size=32,hidden_size=16,intermediate_size=32,num_layers=3,
        num_heads=4,num_kv_heads=2,max_text_length=32,vision_hidden_size=8,vision_intermediate_size=16,
        vision_layers=1,vision_heads=2,image_size=8,patch_size=4,pool_size=2)
    base=TinyMultimodalLM(config);token=tmp_path/'tokenizer.json';token.write_bytes(b'own-tokenizer-hash-control')
    source=tmp_path/'base.pt';torch.save(dict(config=asdict(config),model=base.state_dict(),tokenizer_sha256=file_hash(token)),source)
    model=RoutedCodeResidual(base,width=12);opt=torch.optim.AdamW(model.adapter_parameters(),lr=.003,foreach=False)
    ids=torch.tensor([[1,5,6,7,2]])
    for _ in range(2):
        opt.zero_grad(set_to_none=True);model(ids,labels=ids,route='code')['loss'].backward();opt.step()
    state=make_training_state(model,opt,base_checkpoint_sha256=file_hash(source),tokenizer_sha256=file_hash(token),
        run_plan_sha256='c'*64,completed_steps=2,planned_steps=2,device='cpu',metrics=[])
    state.update(stage='code_residual_lr_pilot_v1',pilot_profile=PROFILES[1],source_manifest_sha256='d'*64)
    path=tmp_path/'best.pt';torch.save(state,path)
    kwargs=dict(expected_checkpoint_sha256=file_hash(path),expected_base_sha256=file_hash(source),expected_plan_sha256='c'*64)
    return model.eval(),ids,path,source,token,state,kwargs


def test_complete_own_checkpoint_inference_matches_trained_code_and_cached_generation(tmp_path):
    original,ids,path,source,token,state,kwargs=artifacts(tmp_path)
    policy,metadata=load_residual_policy(path,source,token,**kwargs)
    assert metadata['optimizer_loaded_on_GPU'] is False and metadata['route']=='code'
    assert metadata['model_counts']==original.parameter_counts()
    with torch.no_grad():
        torch.testing.assert_close(policy(ids)['logits'],original(ids,route='code')['logits'],rtol=0,atol=0)
    cached=policy.decode(ids[:,3:],policy.prefill(ids[:,:3])['cache'])
    torch.testing.assert_close(cached['logits'],policy(ids)['logits'][:,3:],rtol=1e-5,atol=1e-6)
    assert torch.equal(policy.generate(ids[:,:2],max_new_tokens=3,temperature=0.,use_cache=True),
                       original.code.generate(ids[:,:2],max_new_tokens=3,temperature=0.,use_cache=True))


@pytest.mark.parametrize('fault',['plan','base_hash','tokenizer','count','order','frozen_base','adapter','incomplete'])
def test_wrong_or_incomplete_policy_binding_rejects(tmp_path,fault):
    original,ids,path,source,token,state,kwargs=artifacts(tmp_path);state=copy.deepcopy(state)
    if fault=='plan':state['run_plan_sha256']='e'*64
    elif fault=='base_hash':kwargs['expected_base_sha256']='e'*64
    elif fault=='tokenizer':token.write_bytes(b'changed own tokenizer')
    elif fault=='count':state['model_counts']['total_unique']+=1
    elif fault=='order':state['residual_parameter_names'].reverse()
    elif fault=='frozen_base':state['frozen_backbone_state_sha256']='e'*64
    elif fault=='adapter':next(iter(state['residuals'].values())).flatten()[0]=float('nan')
    elif fault=='incomplete':state['steps']=1
    torch.save(state,path);kwargs['expected_checkpoint_sha256']=file_hash(path)
    with pytest.raises(ValueError):load_residual_policy(path,source,token,**kwargs)
