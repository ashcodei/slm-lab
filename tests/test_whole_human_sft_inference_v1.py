import copy
import json

import pytest
import torch

from slm_lab.corpus import file_hash
from slm_lab import whole_human_sft_inference_v1 as inference
from test_whole_human_sft_stage_v1 import setup


def test_completed_and_initial_source_bound_full_graph_exports_roundtrip_without_optimizer(monkeypatch,tmp_path):
    stage=setup();model=stage.model
    fresh=copy.deepcopy(model).eval()
    plan=dict(arms=dict(small=dict(learning_rate=stage.options['learning_rate'])),
        training_options={k:v for k,v in stage.options.items() if k!='learning_rate'},
        actual_counts=model.counts(),source_admission_sha256='a'*64,source_manifest_sha256='b'*64,
        own_code_checkpoint_sha256='c'*64,source_exposure_budget=dict(source_exposures=8))
    path=tmp_path/'plan.json';path.write_text(json.dumps(plan));sha=file_hash(path)
    initial=tmp_path/'initial.pt';torch.save(inference.export_initial(model,plan,sha),initial)
    for _ in range(stage.options['steps']):stage.update()
    trained=tmp_path/'trained.pt';torch.save(inference.export(model,plan,sha,'small',stage.completed),trained)
    calls=[]
    def factory(actual_plan,device):
        calls.append(device);return copy.deepcopy(fresh),dict(tokenizer_sha256='d'*64,original_backbones_loaded=1)
    monkeypatch.setattr(inference,'fresh',factory)
    ids=torch.tensor([[1,4,7]])
    for checkpoint,arm,expected in ((initial,'initial',fresh),(trained,'small',model)):
        loaded,metadata=inference.load(checkpoint,path,arm,expected_checkpoint_sha256=file_hash(checkpoint),expected_plan_sha256=sha)
        expected.eval()
        with torch.no_grad():
            for route in ('aligned','general','code'):
                assert torch.equal(expected(ids,route=route)['logits'],loaded(ids,route=route)['logits'])
        assert metadata['original_backbones_loaded']==1
    assert calls==['cpu','cpu']
    before=len(calls)
    with pytest.raises(ValueError,match='completed whole-human'):
        inference.load(trained,path,'initial',expected_checkpoint_sha256=file_hash(trained),expected_plan_sha256=sha)
    assert len(calls)==before
