import importlib.util
import json
from pathlib import Path
import sys

import pytest


def driver():
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0,str(root/'scripts'))
    spec = importlib.util.spec_from_file_location('completed_execution_assessment_test',
        root/'scripts/assess_completed_execution_v4.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_live_exact_predecessor_or_worker_prevents_evaluation_ownership(tmp_path,monkeypatch):
    module = driver()
    module.PREVIOUS = tmp_path
    prior = dict(owner=dict(pid=101,ticks='202'))
    monkeypatch.setattr(module.owner.resources,'same',lambda actor:dict(state='R'))
    assert not module.previous_ready(prior)
    (tmp_path/'children.json').write_text(json.dumps([dict(pid=303,ticks='404')]))
    monkeypatch.setattr(module.owner.resources,'same',lambda actor:dict(state='R') if actor['pid'] == 303 else None)
    assert not module.previous_ready(prior)


def test_only_completed173_training_can_be_reused_after_inference_failure(tmp_path,monkeypatch):
    module = driver()
    module.PREVIOUS = tmp_path
    prior = dict(owner=dict(pid=101,ticks='202'))
    monkeypatch.setattr(module.owner.resources,'same',lambda actor:None)
    (tmp_path/'children.json').write_text('[]')
    with pytest.raises(ValueError,match='restored'):
        module.previous_ready(prior)
    (tmp_path/'restored.json').write_text('{}')
    with pytest.raises(ValueError,match='all173'):
        module.previous_ready(prior)
    for arm in ['verified_solution','execution_hybrid']:
        (tmp_path/(arm+'-training-complete.json')).write_text(json.dumps(dict(actual_updates=173)))
    (tmp_path/'failure.json').write_text(json.dumps(dict(error='native evaluation missing tokenizer')))
    assert module.previous_ready(prior)
    (tmp_path/'execution_hybrid-training-complete.json').write_text(json.dumps(dict(actual_updates=172)))
    with pytest.raises(ValueError,match='all173'):
        module.previous_ready(prior)
