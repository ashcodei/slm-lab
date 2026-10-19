import importlib.util
import json
from pathlib import Path
import sys

import pytest


def driver():
    scripts = Path(__file__).resolve().parents[1] / 'scripts'
    sys.path.insert(0, str(scripts))
    spec = importlib.util.spec_from_file_location('ownpolicy_pilot_driver_test', scripts / 'train_own_onpolicy_pilot_v1.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def test_live_predecessor_or_registered_rank_rejects_before_resource_queries_or_output(monkeypatch, tmp_path):
    module = driver(); monkeypatch.setattr(module, 'ROOT', tmp_path)
    monkeypatch.setattr(module, 'REPORTS', tmp_path / 'pilot')
    prior = tmp_path / 'reports/own-code-generation-repair-v1'; prior.mkdir(parents=True)
    (prior / 'plan.json').write_text(json.dumps({'owner': {'pid': 11, 'ticks': '21'}}))
    (prior / 'children.json').write_text(json.dumps([{'pid': 12, 'ticks': '22'}]))
    monkeypatch.setattr(module.subprocess, 'check_output', lambda *a, **k: pytest.fail('No GPU queries with live predecessor'))
    monkeypatch.setattr(module.owner, 'idle_original', lambda: pytest.fail('Never hold original waiter with live predecessor'))
    for live in [11, 12]:
        monkeypatch.setattr(module.owner.resources, 'same', lambda actor: actor if actor['pid'] == live else None)
        with pytest.raises(ValueError, match='actually end'):
            module.parent()
        assert not module.REPORTS.exists()


def test_active_original_waiter_rejects_before_source_preparation_or_resource_queries(monkeypatch, tmp_path):
    module = driver(); monkeypatch.setattr(module, 'REPORTS', tmp_path / 'pilot')
    monkeypatch.setattr(module, 'previous_ended', lambda: None)
    monkeypatch.setattr(module.owner.resources, 'same', lambda actor: dict(**actor, state='R'))
    monkeypatch.setattr(module.subprocess, 'check_output', lambda *a, **k: pytest.fail('No GPU queries with active waiter'))
    with pytest.raises(ValueError, match='Never preempt original work'):
        module.parent()
    assert not module.REPORTS.exists()
