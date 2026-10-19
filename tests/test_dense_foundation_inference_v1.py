import copy
import json

import pytest
import torch

from slm_lab import dense_foundation_inference_v1 as inference
from slm_lab.corpus import file_hash
from slm_lab.whole_human_sft_evidence_v1 import tensor_digest
from test_dense_foundation_stage_v1 import data,options,stage
from test_own_dense_growth_v1 import model as make_model


def test_actual_trained_dense_reload_native_cache_and_fixed_head_source_budget_rejection(monkeypatch,tmp_path):
    model,_=make_model();d=data();setting=options(d,0.);setting.update(epochs=1,steps=1,accumulation=7)
    rate=setting.pop('learning_rate');setting.pop('reference_KL_alpha')
    plan=dict(initializer_sha256='a'*64,education_admission_sha256='b'*64,code_admission_sha256='c'*64,
        human_admission_sha256='d'*64,program_admission_sha256='e'*64,actual_counts=model.counts(),
        initial_model_digest=tensor_digest(model.named_parameters()),source_exposure_budget=dict(original_items=7,repeats=0),
        training_options=setting,arms=dict(control=dict(learning_rate=rate,reference_KL_alpha=0.)))
    path=tmp_path/'plan.json';path.write_text(json.dumps(plan));sha=file_hash(path)
    monkeypatch.setattr(inference,'fresh',lambda p,device='cpu':(make_model()[0],dict(synthetic_graph_fixture=True)))
    initial=inference.export(model,plan,sha,'initial',0);initial_path=tmp_path/'initial.pt';torch.save(initial,initial_path)
    cold,_=inference.load(initial_path,path,'initial',expected_checkpoint_sha256=file_hash(initial_path),expected_plan_sha256=sha)
    assert tensor_digest(cold.named_parameters())==plan['initial_model_digest']
    learner=stage(model,None,d,{**setting,'learning_rate':rate,'reference_KL_alpha':0.});learner.update();model.eval()
    value=inference.export(model,plan,sha,'control',1);checkpoint=tmp_path/'trained.pt';torch.save(value,checkpoint)
    cold,_=inference.load(checkpoint,path,'control',expected_checkpoint_sha256=file_hash(checkpoint),expected_plan_sha256=sha)
    view=inference.DenseNativeView(cold).eval();ids=torch.tensor([[1,4,7,2]])
    with torch.no_grad():
        assert torch.equal(model(ids)['logits'],view(ids)['logits'])
        cache=view.prefill(ids)['cache'];actual=view.decode(torch.tensor([[6]]),cache)['logits']
        expected=view(torch.tensor([[1,4,7,2,6]]))['logits'][:,-1:]
        assert torch.allclose(actual,expected,atol=1e-7,rtol=2e-5)
    assert sum(p.numel() for p in view.parameters())==cold.counts()['total_unique']
    for mutate in [lambda s:s['model']['heads']['reward.0.weight'].add_(1.),
                   lambda s:s.update(actual_dense_foundation_updates=0),
                   lambda s:s['source_exposure_budget'].update(original_items=6),
                   lambda s:s['model']['sources'].update(education_admission_sha256='f'*64)]:
        bad=copy.deepcopy(value);mutate(bad);torch.save(bad,checkpoint)
        with pytest.raises(ValueError):inference.load(checkpoint,path,'control',expected_checkpoint_sha256=file_hash(checkpoint),expected_plan_sha256=sha)
