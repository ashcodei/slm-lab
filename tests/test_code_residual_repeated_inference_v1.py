"""Small real-update inference fixtures; never capability-evaluation records."""
import copy
from dataclasses import asdict

import pytest
import torch

from slm_lab.code_residual_branch_v1 import RoutedCodeResidual
from slm_lab.code_residual_checkpoint_v1 import make_training_state
from slm_lab.code_residual_repeated_inference_v1 import load_repeated_source_policy
from slm_lab.corpus import file_hash
from slm_lab.model import ModelConfig,TinyMultimodalLM


@pytest.fixture(scope='module')
def fixture_state():
    torch.manual_seed(914)
    c=ModelConfig(vocab_size=32,hidden_size=16,intermediate_size=32,num_layers=2,num_heads=4,
        num_kv_heads=2,max_text_length=32,vision_hidden_size=8,vision_intermediate_size=16,
        vision_layers=1,vision_heads=2,image_size=8,patch_size=4,pool_size=2)
    model=RoutedCodeResidual(TinyMultimodalLM(c),width=12)
    opt=torch.optim.AdamW(model.adapter_parameters(),lr=.0003,foreach=False)
    ids=torch.tensor([[1,5,6,7,2]])
    for _ in range(512):
        opt.zero_grad(set_to_none=True);model(ids,labels=ids,route='code')['loss'].backward();opt.step()
    model.eval()
    return model,opt,ids


def artifacts(tmp_path,fixture_state):
    model,opt,ids=fixture_state
    token=tmp_path/'tokenizer.json';token.write_bytes(b'own tiny hash fixture')
    source=tmp_path/'base.pt'
    torch.save(dict(config=asdict(model.config),model=model.base.state_dict(),tokenizer_sha256=file_hash(token)),source)
    state=make_training_state(model,opt,base_checkpoint_sha256=file_hash(source),tokenizer_sha256=file_hash(token),
        run_plan_sha256='c'*64,completed_steps=512,planned_steps=512,device='cpu',metrics=[])
    state.update(stage='code_residual_repeated_source_pilot_v1',training_sources=512,additional_complete_passes=4,
        additional_training_exposures=2048,fresh_optimizer=True,original_selection_sha256='d'*64,
        starting_policy_checkpoint_sha256='e'*64,source_manifest_sha256='f'*64)
    path=tmp_path/'final.pt';torch.save(state,path)
    kwargs=dict(expected_checkpoint_sha256=file_hash(path),expected_base_sha256=file_hash(source),
        expected_plan_sha256='c'*64,expected_selection_sha256='d'*64,expected_starting_policy_sha256='e'*64)
    return model,ids,path,source,token,state,kwargs


def test_actual512_update_fixture_has_exact_raw_and_cached_inference(tmp_path,fixture_state):
    model,ids,path,base,token,state,kwargs=artifacts(tmp_path,fixture_state)
    policy,metadata=load_repeated_source_policy(path,base,token,**kwargs)
    assert metadata['steps']==512 and metadata['optimizer_loaded_on_GPU'] is False
    with torch.no_grad():
        torch.testing.assert_close(policy(ids)['logits'],model.code(ids)['logits'],rtol=0,atol=0)
    assert torch.equal(policy.generate(ids[:,:2],max_new_tokens=4,use_cache=True),
                       model.code.generate(ids[:,:2],max_new_tokens=4,use_cache=True))


@pytest.mark.parametrize('fault',['old_stage','incomplete','selection','starting_policy','base','count','order','nan'])
def test_mismatched_or_incomplete_experiment_rejects(tmp_path,fixture_state,fault):
    model,ids,path,base,token,state,kwargs=artifacts(tmp_path,fixture_state)
    state=copy.deepcopy(state)
    if fault=='old_stage':state['stage']='code_residual_lr_pilot_v1'
    elif fault=='incomplete':state['steps']=511
    elif fault=='selection':state['original_selection_sha256']='a'*64
    elif fault=='starting_policy':state['starting_policy_checkpoint_sha256']='a'*64
    elif fault=='base':state['frozen_backbone_state_sha256']='a'*64
    elif fault=='count':state['model_counts']['total_unique']+=1
    elif fault=='order':state['residual_parameter_names'].reverse()
    elif fault=='nan':next(iter(state['residuals'].values())).flatten()[0]=float('nan')
    torch.save(state,path);kwargs['expected_checkpoint_sha256']=file_hash(path)
    with pytest.raises(ValueError):load_repeated_source_policy(path,base,token,**kwargs)
