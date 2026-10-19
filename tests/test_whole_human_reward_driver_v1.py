import importlib.util
import json
from pathlib import Path

import pytest


def load():
    path = Path(__file__).resolve().parents[1]/'scripts/train_whole_human_reward_v1.py'
    import sys
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location('reward_driver_test', path)
    driver = importlib.util.module_from_spec(spec); spec.loader.exec_module(driver)
    return driver


def test_exact_prior_owner_or_registered_worker_prevents_new_GPU_acquisition(monkeypatch, tmp_path):
    driver = load(); monkeypatch.setattr(driver, 'PREVIOUS', tmp_path)
    old = dict(pid=121, ticks='31'); child = dict(pid=122, ticks='32')
    (tmp_path/'plan.json').write_text(json.dumps(dict(owner=old)))
    (tmp_path/'children.json').write_text(json.dumps([child]))
    monkeypatch.setattr(driver.subprocess, 'check_output', lambda *a, **k:pytest.fail('No GPU query while previous work is live'))
    for live in (old, child):
        monkeypatch.setattr(driver.owner.resources, 'same', lambda actor:actor if actor == live else None)
        assert driver.predecessor_ended() is False
    monkeypatch.setattr(driver.owner.resources, 'same', lambda actor:None)
    with pytest.raises(ValueError, match='restore'):
        driver.predecessor_ended()
    (tmp_path/'restored.json').write_text('{}')
    with pytest.raises(ValueError, match='did not complete'):
        driver.predecessor_ended()
    (tmp_path/'complete.json').write_text('{}')
    assert driver.predecessor_ended() is True
    (tmp_path/'failure.json').write_text('{}')
    with pytest.raises(ValueError, match='did not complete'):
        driver.predecessor_ended()


def test_GPU_worker_rejects_nonisolated_device_before_any_CUDA_work(monkeypatch, tmp_path):
    driver = load(); monkeypatch.setattr(driver, 'REPORTS', tmp_path)
    (tmp_path/'plan.json').write_text('{}')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    import torch
    monkeypatch.setattr(torch.cuda, 'set_device', lambda *a:pytest.fail('No CUDA initialization before exact resource admission'))
    with pytest.raises(ValueError, match='isolated physical GPU'):
        driver.worker(0)


def test_parent_requires_full_size_proof_before_waiting_or_querying_devices(monkeypatch, tmp_path):
    driver = load(); monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    for key in ('REPORTS', 'OUTPUT', 'FEATURES', 'PROBE'):
        monkeypatch.setattr(driver, key, tmp_path/key)
    monkeypatch.setattr(driver, 'QUEUE', tmp_path/'queue.json')
    monkeypatch.setattr(driver.subprocess, 'check_output', lambda *a, **k:pytest.fail('No GPU query without full-size neural proof'))
    with pytest.raises(ValueError, match='full-size CPU'):
        driver.parent()
