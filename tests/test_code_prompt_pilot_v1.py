"""Actual own-model gradients, complete checkpoints and unchanged evaluator lease."""
from contextlib import nullcontext
from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F

from slm_lab.code_prompt_pilot_v1 import update, verify_saved_state, FrozenPilotLoader, evaluator_loader
from slm_lab.code_residual_branch_v1 import RoutedCodeResidual
from slm_lab.code_residual_checkpoint_v1 import make_training_state
from slm_lab.codenet_checkpoint_v1 import atomic_checkpoint
from slm_lab.corpus import file_hash
from slm_lab.model import ModelConfig, TinyMultimodalLM


def own_model():
    torch.manual_seed(381)
    config = ModelConfig(vocab_size=32, hidden_size=8, intermediate_size=16, num_layers=2,
        num_heads=2, num_kv_heads=1, max_text_length=32, vision_hidden_size=8,
        vision_intermediate_size=16, vision_layers=1, vision_heads=2, image_size=8,
        patch_size=2, pool_size=2)
    return RoutedCodeResidual(TinyMultimodalLM(config), width=5)


def examples():
    return [dict(id='a', role='train', negative_source_id='b',
        positive=([1,3,4,7,8,2],[-100]*3+[7,8,2]),
        negative=([1,6,7,8,2],[-100]*2+[7,8,2])),
        dict(id='b', role='train', negative_source_id='a',
            positive=([1,6,9,10,11,12,2],[-100]*2+[9,10,11,12,2]),
            negative=([1,3,4,9,10,11,12,2],[-100]*3+[9,10,11,12,2]))]


class OriginalReplayFixture:
    truncated = 0
    def batch(self, rows, field, device):
        assert len(rows) == 1 and field == 'response'
        return dict(input_ids=torch.tensor([[1,3,6,2]],device=device),
            labels=torch.tensor([[-100,-100,6,2]],device=device),
            pixel_values=torch.linspace(-1,1,3*8*8,device=device).reshape(1,3,8,8))


@pytest.mark.parametrize('alpha',[0.,.1])
def test_actual_own_model_complete_code_plus_image_replay_matches_direct_combined_objective(alpha):
    model = own_model(); reference = deepcopy(model)
    optimizer = torch.optim.AdamW(model.adapter_parameters(),lr=3e-5,weight_decay=.01,foreach=False)
    expected_optimizer = torch.optim.AdamW(reference.adapter_parameters(),lr=3e-5/16,weight_decay=.01,foreach=False)
    rows = examples(); encoder = OriginalReplayFixture()
    for step in range(2):
        expected_optimizer.zero_grad(set_to_none=True)
        expected_optimizer.param_groups[0]['lr'] = 3e-5*(step+1)/16
        objective = 0.
        for example in rows:
            def loss(sequence):
                ids,labels=sequence
                return reference.code(torch.tensor([ids]),labels=torch.tensor([labels]))['loss']
            correct,wrong = loss(example['positive']),loss(example['negative'])
            weight = sum(y!=-100 for y in example['positive'][1])/8
            objective = objective + weight*(correct + alpha*F.softplus(10*(correct-wrong)+.1))
        objective = objective + .5*reference.code(**encoder.batch([{}],'response','cpu'))['loss']
        objective.backward()
        torch.nn.utils.clip_grad_norm_(reference.adapter_parameters(),1.,error_if_nonfinite=True)
        expected_optimizer.step()
        result = update(model,optimizer,rows,encoder,{},'generic_visual',step,alpha,'cpu',nullcontext)
        assert result['source_matching']['positive_source_target_tokens']==8
        assert result['replay']['objectives']['generic_visual']['global_target_tokens']==2
        assert not any(p.grad is not None for p in model.parameters())
        for actual,expected in zip(model.residuals.parameters(),reference.residuals.parameters()):
            assert torch.allclose(actual,expected,atol=2e-8,rtol=1e-5)
        for actual,expected in zip(model.base.parameters(),reference.base.parameters()):
            assert torch.equal(actual,expected)
    for a,e in zip(optimizer.state.values(),expected_optimizer.state.values()):
        assert torch.allclose(a['exp_avg'],e['exp_avg'],atol=1e-7,rtol=2e-5)


def saved(tmp_path):
    model = own_model()
    optimizer=torch.optim.AdamW(model.adapter_parameters(),lr=3e-5,foreach=False)
    update(model,optimizer,examples(),OriginalReplayFixture(),{},'generic_visual',0,.1,'cpu',nullcontext)
    bundle=make_training_state(model,optimizer,base_checkpoint_sha256='a'*64,tokenizer_sha256='b'*64,
        run_plan_sha256='c'*64,completed_steps=1,planned_steps=128,device='cpu',metrics=[{'step':1}])
    path=tmp_path/'latest.pt'
    atomic_checkpoint(path,bundle,maximum_bytes=1024**2,minimum_free_bytes=0)
    return model,path,bundle


def test_full_saved_weights_moments_rng_are_read_back_and_missing_moment_rejected(tmp_path):
    model,path,bundle=saved(tmp_path)
    kw=dict(device='cpu',plan_sha256='c'*64,base_sha256='a'*64,tokenizer_sha256='b'*64,completed=1)
    receipt=verify_saved_state(model,path,**kw)
    assert receipt['steps']==1 and receipt['optimizer_and_RNG_complete']
    assert not receipt['optimizer_loaded_on_GPU']
    del bundle['optimizer']['state'][0]['exp_avg']
    torch.save(bundle,path)
    with pytest.raises(ValueError,match='both Adam moments'):
        verify_saved_state(model,path,**kw)


def test_evaluator_receives_exact_saved_live_code_route_and_lease_restores_on_failure(tmp_path):
    model,path,bundle=saved(tmp_path)
    receipt=verify_saved_state(model,path,device='cpu',plan_sha256='c'*64,base_sha256='a'*64,
        tokenizer_sha256='b'*64,completed=1)
    token=tmp_path/'tokenizer.json';token.write_text('own fixture')
    loader=FrozenPilotLoader(model,path,{'tokenizer_sha256':file_hash(token)},receipt,'cpu',lambda:None)
    original=lambda *a,**k:None
    module=SimpleNamespace(load_checkpoint=original)
    ids=torch.tensor([[1,4,6]])
    expected=model.code(ids)['logits'].detach()
    with pytest.raises(RuntimeError,match='fixture interrupted'):
        with evaluator_loader(module,loader):
            view,meta=module.load_checkpoint(path,'cpu',training_state=False)
            assert view is model.code and torch.equal(view(ids)['logits'],expected)
            with pytest.raises(ValueError,match='inference path/device'):
                module.load_checkpoint(path,'cpu',training_state=True)
            raise RuntimeError('fixture interrupted')
    assert module.load_checkpoint is original and loader.calls==1


def test_unregistered_new_pilot_process_with_stale_owner_never_reaches_cuda(tmp_path,monkeypatch):
    import sys
    root=Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root/'scripts'))
    spec=importlib.util.spec_from_file_location('source_pilot_driver',root/'scripts/train_code_prompt_matching_pilot_v1.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    with pytest.raises(ValueError,match='before CUDA worker registration'):
        module.await_registration(dict(owner=dict(pid=os.getpid(),ticks='0')),'CE_control')
