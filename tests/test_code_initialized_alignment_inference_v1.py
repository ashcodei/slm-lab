import copy
import json

import pytest
import torch

from slm_lab.corpus import file_hash
from slm_lab import code_initialized_alignment_inference_v1 as inference
from slm_lab.whole_human_sft_evidence_v1 import tensor_digest
from test_code_initialized_alignment_stage_v1 import setup


def test_actual_completed_source_model_roundtrip_and_strict_initial_or_unused_tensors(monkeypatch,tmp_path):
    stage = setup(); original = copy.deepcopy(stage.model)
    plan = dict(actual_counts=original.counts(),initial_model_plan=dict(own_code_checkpoint_sha256='a'*64),
        human_source_admission_sha256='b'*64,program_source_admission_sha256='c'*64,
        initial_parts_digest=tensor_digest(original.parts.state_dict().items()),
        training_options={k:v for k,v in stage.options.items() if k!='reference_KL_alpha'},
        arms=dict(mixed=dict(reference_KL_alpha=.1)),source_exposure_budget=dict(human=12,generation=3))
    path = tmp_path/'plan.json'; path.write_text(json.dumps(plan)); plan_sha = file_hash(path)
    initial = tmp_path/'initial.pt'; torch.save(inference.export(stage.model,plan,plan_sha,'initial',0),initial)
    for _ in range(stage.options['steps']): stage.update()
    trained = tmp_path/'trained.pt'; torch.save(inference.export(stage.model,plan,plan_sha,'mixed',stage.completed),trained)
    calls = []
    def factory(p,device): calls.append(device); return copy.deepcopy(original),dict(original_backbones_loaded=1)
    monkeypatch.setattr(inference,'fresh',factory)
    ids = torch.tensor([[1,4,7,2]])
    for checkpoint,arm,expected in [(initial,'initial',original),(trained,'mixed',stage.model)]:
        restored,_ = inference.load(checkpoint,path,arm,expected_checkpoint_sha256=file_hash(checkpoint),expected_plan_sha256=plan_sha)
        with torch.no_grad():
            for route in ('aligned','general','code'):
                assert torch.equal(expected(ids,route=route)['logits'],restored(ids,route=route)['logits'])
    for checkpoint,arm in [(initial,'initial'),(trained,'mixed')]:
        bad = torch.load(checkpoint,map_location='cpu',weights_only=True)
        key = 'language_norms.0.weight' if arm=='initial' else 'critic.weight'
        bad['aligned_model']['extensions'][key].add_(1)
        corrupt = tmp_path/(arm+'-bad.pt'); torch.save(bad,corrupt)
        with pytest.raises(ValueError,match='exact own code initialization'):
            inference.load(corrupt,path,arm,expected_checkpoint_sha256=file_hash(corrupt),expected_plan_sha256=plan_sha)
