from dataclasses import asdict
import copy
import json

import pytest
import torch
from tokenizers import Tokenizer, models

from slm_lab.corpus import file_hash
from slm_lab.own_code_training_inference_v1 import export_payload, load
from slm_lab.own_code_training_stage_v1 import OwnCodeTrainingStage, make_optimizer, components
from test_own_code_training_stage_v1 import fixtures
from test_own_code_delta_v1 import own_models


def test_actual_trained_component_reload_matches_both_routes_and_rejects_false_training_provenance(tmp_path):
    torch.set_num_threads(1)
    model, streams, options = fixtures()
    tokenizer = Tokenizer(models.WordLevel({'<pad>': 0, '<bos>': 1, '<eos>': 2, '<unk>': 3,
        **{str(i): i for i in range(4, 32)}}, unk_token='<unk>'))
    token = tmp_path / 'tokenizer.json'; tokenizer.save(str(token))
    base = tmp_path / 'base.pt'; target_path = tmp_path / 'target.pt'; _, target = own_models()
    for path, native in [(base, model.base), (target_path, target)]:
        torch.save(dict(model=native.state_dict(), config=asdict(native.config), tokenizer_sha256=file_hash(token)), path)
    initial_plan = tmp_path / 'initial-plan.json'; initial_plan.write_text(json.dumps(dict(own_base=file_hash(base), own_target=file_hash(target_path))))
    counts = model.counts()
    initial = dict(format='own-conditional-code-delta-v1', components=components(model), config=asdict(model.config),
        base_checkpoint_sha256=file_hash(base), own_code_source_checkpoint_sha256=file_hash(target_path),
        tokenizer_sha256=file_hash(token), source_plan_sha256=file_hash(initial_plan), reserved_point_head_parameters=256,
        base_parameters=counts['frozen_general'], additional_code_parameters=counts['trainable_code'],
        total_unique_parameters=counts['total_unique'], total_with_reserved_point_head=counts['total_with_reserved_head'],
        candidate_promoted=False, goal_achieved=False)
    initial_path = tmp_path / 'initial.pt'; torch.save(initial, initial_path)
    plan = dict(training_options=options, actual_model_counts=counts,
        initial_component_checkpoint_sha256=file_hash(initial_path), base_checkpoint_sha256=file_hash(base),
        initial_plan_sha256=file_hash(initial_plan), own_code_target_sha256=file_hash(target_path),
        data_admissions_sha256={name: str(i) * 64 for i, name in enumerate(streams)})
    training_plan = tmp_path / 'training-plan.json'; training_plan.write_text(json.dumps(plan))
    kernel = OwnCodeTrainingStage(model, make_optimizer(model, options), streams, options, {'plan': file_hash(training_plan)}, 'cpu')
    for _ in range(3):
        kernel.update()
    payload = export_payload(model, initial, plan, file_hash(training_plan), kernel.completed)
    checkpoint = tmp_path / 'trained.pt'; torch.save(payload, checkpoint)
    kwargs = dict(expected_checkpoint_sha256=file_hash(checkpoint), expected_base_sha256=file_hash(base),
        expected_target_sha256=file_hash(target_path), expected_initial_plan_sha256=file_hash(initial_plan),
        expected_training_plan_sha256=file_hash(training_plan))
    loaded, metadata = load(checkpoint, base, token, initial_plan, training_plan, **kwargs)
    ids = torch.tensor([[1, 4, 7]]); pixels = torch.linspace(-1, 1, 3 * 16 * 16).reshape(1, 3, 16, 16)
    model.eval(); loaded.eval()
    with torch.no_grad():
        for route in ['general', 'code']:
            assert torch.equal(model(ids, pixels, route=route)['logits'], loaded(ids, pixels, route=route)['logits'])
            assert torch.equal(model.generate(ids, pixels, route=route, max_new_tokens=3), loaded.generate(ids, pixels, route=route, max_new_tokens=3))
    assert metadata['own_code_training']['actual_steps'] == 3
    assert not metadata['full_code_source_loaded_at_inference']
    bad = copy.deepcopy(payload); bad['own_code_training']['actual_steps'] = 2
    bad_path = tmp_path / 'bad.pt'; torch.save(bad, bad_path)
    with pytest.raises(ValueError, match='completed own training'):
        load(bad_path, base, token, initial_plan, training_plan, **{**kwargs, 'expected_checkpoint_sha256': file_hash(bad_path)})
