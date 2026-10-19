from dataclasses import asdict
import copy
import json

import pytest
import torch
from tokenizers import Tokenizer, models

from slm_lab.corpus import file_hash
from slm_lab.own_code_training_stage_v1 import make_optimizer
from slm_lab.own_onpolicy_stage_v1 import OwnOnPolicyStage
from slm_lab.own_onpolicy_inference_v1 import export_payload, load
from test_own_onpolicy_stage_v1 import fixtures


def test_complete_native_export_loads_single_base_without_prior_weights_and_rejects_false_arm(tmp_path):
    model, streams, pairs, reference, opts, sources = fixtures()
    tokenizer = Tokenizer(models.WordLevel({str(i): i for i in range(32)}, unk_token='3'))
    token = tmp_path/'tokenizer.json'; tokenizer.save(str(token))
    base = tmp_path/'base.pt'; torch.save(dict(config=asdict(model.config), model=model.base.state_dict(), tokenizer_sha256=file_hash(token)), base)
    reference_path = tmp_path/'reference.json'; reference_path.write_text(json.dumps(reference))
    plan = dict(training_options={k: v for k, v in opts.items() if k != 'DPO_alpha'},
        arms={'execution_DPO': dict(DPO_alpha=.1), 'CE_control': dict(DPO_alpha=0.)},
        initial_manifest_sha256='a'*64, initial_code_checkpoint_sha256='b'*64,
        base_checkpoint=str(base), base_checkpoint_sha256=file_hash(base), tokenizer=str(token), tokenizer_sha256=file_hash(token),
        data_admissions_sha256=dict(generation='c'*64, variety='d'*64, pairs='e'*64), actual_model_counts=model.counts())
    path = tmp_path/'plan.json'; path.write_text(json.dumps(plan))
    stage = OwnOnPolicyStage(model, make_optimizer(model, opts), streams, pairs, reference, opts, sources, 'cpu')
    for _ in range(stage.steps):
        stage.update()
    output = tmp_path/'model.pt'
    torch.save(export_payload(model, plan, file_hash(path), 'execution_DPO', file_hash(reference_path), stage.completed), output)
    loaded, metadata = load(output, path, 'execution_DPO', reference_path, expected_checkpoint_sha256=file_hash(output),
        expected_plan_sha256=file_hash(path), expected_reference_sha256=file_hash(reference_path))
    assert metadata['original_backbones_loaded'] == 1 and not metadata['full_dense_code_source_or_initial_component_weights_loaded']
    model.eval(); loaded.eval(); ids = torch.tensor([[1, 4, 7]])
    pixels = torch.zeros(1, 3, 32, 32)
    with torch.no_grad():
        for route in ['general', 'code']:
            assert torch.equal(model(ids, pixels, route=route)['logits'], loaded(ids, pixels, route=route)['logits'])
            assert torch.equal(model.generate(ids, pixels, route=route, max_new_tokens=3), loaded.generate(ids, pixels, route=route, max_new_tokens=3))
    with pytest.raises(ValueError, match='completed own arm'):
        load(output, path, 'CE_control', reference_path, expected_checkpoint_sha256=file_hash(output),
            expected_plan_sha256=file_hash(path), expected_reference_sha256=file_hash(reference_path))
