"""Actual tiny updates and strict endpoint loading, never model capability records."""
import copy
from dataclasses import asdict

import pytest
import torch

from slm_lab.code_repetition_inference_v1 import load_objective_policy,PROFILES
from slm_lab.code_repetition_objective_v1 import sequence_unlikelihood
from slm_lab.code_residual_branch_v1 import RoutedCodeResidual
from slm_lab.code_residual_checkpoint_v1 import make_training_state
from slm_lab.corpus import file_hash
from slm_lab.model import ModelConfig,TinyMultimodalLM


@pytest.fixture(scope='module',params=PROFILES)
def trained_fixture(request):
    profile=request.param;torch.manual_seed(617)
    c=ModelConfig(vocab_size=32,hidden_size=16,intermediate_size=32,num_layers=2,num_heads=4,num_kv_heads=2,
        max_text_length=32,vision_hidden_size=8,vision_intermediate_size=16,vision_layers=1,vision_heads=2,
        image_size=8,patch_size=4,pool_size=2)
    model=RoutedCodeResidual(TinyMultimodalLM(c),width=12)
    optimizer=torch.optim.AdamW(model.adapter_parameters(),lr=.0003,foreach=False)
    ids=torch.tensor([[1,3,4,3,4,2]])
    for _ in range(128):
        optimizer.zero_grad(set_to_none=True);outputs=model.code(ids,labels=ids)
        loss=outputs['loss']+profile['alpha']*sequence_unlikelihood(outputs['logits'],ids,2,[2])
        loss.backward();optimizer.step()
    return model.eval(),optimizer,ids,profile


def artifacts(tmp_path,trained_fixture):
    model,opt,ids,profile=trained_fixture
    token=tmp_path/'tokenizer.json';token.write_bytes(b'own tiny tokenizer binding fixture')
    base=tmp_path/'base.pt';torch.save(dict(config=asdict(model.config),model=model.base.state_dict(),tokenizer_sha256=file_hash(token)),base)
    state=make_training_state(model,opt,base_checkpoint_sha256=file_hash(base),tokenizer_sha256=file_hash(token),
        run_plan_sha256='c'*64,completed_steps=128,planned_steps=128,device='cpu',metrics=[])
    # Explicit unit metadata; no fixture rows/updates contribute to research scores.
    state.update(stage='code_repetition_objective_pilot_v1',objective_profile=profile,fresh_optimizer=True,
        training_sources=512,training_rollout_sources=128,rollout_generation_budget=128,regularizer_ngram=16,
        reference_repeated_ngrams_exempt=True,starting_policy_checkpoint_sha256='d'*64,
        original_selection_sha256='e'*64,source_manifest_sha256='f'*64)
    checkpoint=tmp_path/'final.pt';torch.save(state,checkpoint)
    kwargs=dict(expected_checkpoint_sha256=file_hash(checkpoint),expected_base_sha256=file_hash(base),
        expected_plan_sha256='c'*64,expected_starting_policy_sha256='d'*64,expected_selection_sha256='e'*64,
        expected_profile=profile)
    return model,ids,checkpoint,base,token,state,kwargs


def test_real_trained_endpoint_has_identical_logits_and_cached_generation(tmp_path,trained_fixture):
    model,ids,path,base,token,state,kwargs=artifacts(tmp_path,trained_fixture)
    policy,metadata=load_objective_policy(path,base,token,**kwargs)
    with torch.no_grad():torch.testing.assert_close(policy(ids)['logits'],model.code(ids)['logits'],rtol=0,atol=0)
    assert metadata['objective_profile']==trained_fixture[3] and metadata['optimizer_loaded_on_GPU'] is False
    assert torch.equal(policy.generate(ids[:,:2],max_new_tokens=3,use_cache=True),
                       model.code.generate(ids[:,:2],max_new_tokens=3,use_cache=True))


@pytest.mark.parametrize('fault',['old_stage','wrong_objective','wrong_alpha','wrong_plan','incomplete','selection','starting','mask_contract'])
def test_wrong_endpoint_or_objective_contract_rejects(tmp_path,trained_fixture,fault):
    model,ids,path,base,token,state,kwargs=artifacts(tmp_path,trained_fixture);state=copy.deepcopy(state)
    if fault=='old_stage':state['stage']='code_residual_lr_pilot_v1'
    if fault=='wrong_objective':state['objective_profile']['name']='unregistered'
    if fault=='wrong_alpha':state['objective_profile']['alpha']=.2
    if fault=='wrong_plan':state['run_plan_sha256']='a'*64
    if fault=='incomplete':state['steps']=127
    if fault=='selection':state['original_selection_sha256']='a'*64
    if fault=='starting':state['starting_policy_checkpoint_sha256']='a'*64
    if fault=='mask_contract':state['reference_repeated_ngrams_exempt']=False
    torch.save(state,path);kwargs['expected_checkpoint_sha256']=file_hash(path)
    with pytest.raises(ValueError):load_objective_policy(path,base,token,**kwargs)
