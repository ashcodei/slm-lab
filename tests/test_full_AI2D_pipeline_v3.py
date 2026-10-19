import copy
import importlib.util
import json
from pathlib import Path
import sys

import pytest


def driver():
    root = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(root / 'scripts'))
    spec = importlib.util.spec_from_file_location('full_AI2D_pipeline_unit', root / 'scripts/assess_full_AI2D_pipeline_v3.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module


def test_exact_original_eight_endpoints_remain_fixed_before_scores(tmp_path):
    module = driver(); specs = module.descriptors(tmp_path)
    assert tuple(specs) == module.ENDPOINTS and len(specs) == 8
    assert specs['human_DPO']['arm'] == 'human_DPO' and specs['human_chosen_CE']['arm'] == 'chosen_CE'
    assert specs['code_anchor_execution']['checkpoint'] == str(tmp_path / 'checkpoints/broad-anchored-code-v5/anchor_execution-actor.pt')
    assert module.PREVIOUS.name == 'full-held-QA-pipeline-v3'


def test_live_exact_predecessor_or_worker_blocks_acquisition(tmp_path, monkeypatch):
    module = driver(); module.PREVIOUS = tmp_path; identity = dict(pid=101, ticks='201')
    monkeypatch.setattr(module.owner.resources, 'same', lambda p: dict(state='R'))
    assert not module.previous_ready(identity)
    (tmp_path / 'children.json').write_text(json.dumps([dict(pid=303, ticks='403')]))
    monkeypatch.setattr(module.owner.resources, 'same', lambda p: dict(state='R') if p['pid'] == 303 else None)
    assert not module.previous_ready(identity)


def test_terminal_text_QA_requires_all32_original_shards_complete_and_restored(tmp_path, monkeypatch):
    module = driver(); module.PREVIOUS = tmp_path; identity = dict(pid=101, ticks='201')
    monkeypatch.setattr(module.owner.resources, 'same', lambda p: None)
    with pytest.raises(ValueError): module.previous_ready(identity)
    (tmp_path / 'plan.json').write_text(json.dumps(dict(full_QA_questions_each_endpoint=16961, endpoints=module.descriptors(tmp_path))))
    (tmp_path / 'complete.json').write_text(json.dumps(dict(all_eight_predeclared_endpoints_assessed=True,
        original_questions_each_endpoint=16961, total_original_question_endpoint_assessments=135688,
        all_raw_deliveries_and_question_verdicts_retained=True)))
    (tmp_path / 'restored.json').write_text('{}'); (tmp_path / 'paired-comparisons.json').write_text('{}')
    for endpoint in module.ENDPOINTS:
        folder = tmp_path / endpoint; folder.mkdir(); (folder / 'full-QA.json').write_text('{}')
        for gpu in range(4):
            (folder / f'QA-shard-{gpu}-complete.json').write_text(json.dumps(dict(endpoint=endpoint, physical_GPU=gpu,
                actual_original_questions=4241 if gpu == 0 else 4240, all_loaded_values_unchanged=True, model_updates=0)))
    assert module.previous_ready(identity)
    receipt = tmp_path / module.ENDPOINTS[-1] / 'QA-shard-3-complete.json'
    row = json.loads(receipt.read_text()); row['actual_original_questions'] = 100; receipt.write_text(json.dumps(row))
    with pytest.raises(ValueError): module.previous_ready(identity)


def test_paired_full_population_preserves_actual_visual_controls_and_prompts():
    module = driver(); conditions = ['proper_image', 'gray_image', 'wrong_image']
    # Synthetic reconciliation arithmetic only, no source/NN/quality claim.
    rows = [dict(id='unit:' + str(i), original_native_prompt_ids=[1, 7], generation_budget=32,
        wrong_source_question_id='wrong:' + str(i), original_ABC_labels=False,
        conditions={name: dict(actual_model_input_tensor_sha256_448=name, supplied_image_id=i,
            verdict=dict(strict_correct=False)) for name in conditions}) for i in range(3088)]
    before = dict(protocol='unit-only', original_population_sha256='a' * 64, generation_budget=32, conditions=conditions,
        inference_device_type='cuda', inference_precision='BF16_autocast', results=rows)
    after = copy.deepcopy(before); after['results'][3]['conditions']['proper_image']['verdict']['strict_correct'] = True
    result = module.paired(before, after)['by_image_condition']
    assert result['proper_image']['questions'] == 3088 and result['proper_image']['gained'] == ['unit:3']
    assert result['gray_image']['after_strict_correct'] == 0
    after['results'][4]['conditions']['wrong_image']['actual_model_input_tensor_sha256_448'] = 'changed'
    with pytest.raises(ValueError): module.paired(before, after)
    after = copy.deepcopy(before); after['results'][4]['original_native_prompt_ids'] = [1, 8]
    with pytest.raises(ValueError): module.paired(before, after)
