import importlib.util
import json
from pathlib import Path
import sys

import pytest


def driver():
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0,str(root/'scripts'))
    spec = importlib.util.spec_from_file_location('execution_host_recovery_driver_test',
        root/'scripts/train_execution_host_recovery_v2.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_live_exact_owner_or_child_cannot_be_restarted(tmp_path,monkeypatch):
    module = driver()
    module.PREVIOUS = tmp_path
    prior = dict(owner=dict(pid=101,ticks='202'))
    monkeypatch.setattr(module.owner.resources,'same',lambda a:dict(state='R'))
    assert not module.previous_ready(prior)
    (tmp_path/'children.json').write_text(json.dumps([dict(pid=303,ticks='404')]))
    monkeypatch.setattr(module.owner.resources,'same',lambda a:dict(state='R') if a['pid'] == 303 else None)
    assert not module.previous_ready(prior)


def test_terminal_owner_needs_restoration_and_explicit_failure_not_just_slow_progress(tmp_path,monkeypatch):
    module = driver()
    module.PREVIOUS = tmp_path
    prior = dict(owner=dict(pid=101,ticks='202'))
    monkeypatch.setattr(module.owner.resources,'same',lambda a:None)
    (tmp_path/'children.json').write_text('[]')
    with pytest.raises(ValueError,match='restored'):
        module.previous_ready(prior)
    (tmp_path/'restored.json').write_text('{}')
    with pytest.raises(ValueError,match='terminal failure'):
        module.previous_ready(prior)
    (tmp_path/'failure.json').write_text(json.dumps(dict(error='actual collective failure')))
    assert module.previous_ready(prior)
