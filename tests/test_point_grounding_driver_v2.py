import importlib.util
import json
from pathlib import Path
import sys

import pytest
import torch

from slm_lab.model import ModelConfig, TinyMultimodalLM
from slm_lab.point_grounding_head_v1 import PointGroundingHead


def load_driver():
    path = Path('scripts/train_point_grounding_joint_v2.py').resolve()
    spec = importlib.util.spec_from_file_location('checked_point_driver_v2', path)
    driver = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(path.parent))
    try:
        spec.loader.exec_module(driver)
    finally:
        sys.path.pop(0)
    return driver


def test_live_evaluation_owner_or_registered_worker_rejects_before_resource_changes(monkeypatch, tmp_path):
    driver = load_driver()
    reports = tmp_path / 'no-report'; output = tmp_path / 'no-output'
    monkeypatch.setattr(driver, 'REPORTS', reports); monkeypatch.setattr(driver, 'OUTPUT', output)
    actions = []
    monkeypatch.setattr(driver.subprocess, 'run', lambda *a, **k: actions.append('GPU query'))
    monkeypatch.setattr(driver.os, 'kill', lambda *a, **k: actions.append('signal'))
    predecessor = dict(pid=123, ticks='456')
    monkeypatch.setattr(driver.original.resources, 'same', lambda actor: dict(**actor, state='R'))
    with pytest.raises(ValueError, match='coding-evaluation owner must actually end'):
        driver.parent({}, predecessor)
    previous = tmp_path / 'previous'; previous.mkdir()
    (previous / 'plan.json').write_text(json.dumps(dict(owner=predecessor)))
    (previous / 'children.json').write_text(json.dumps([dict(pid=124, ticks='worker')]))
    monkeypatch.setattr(driver, 'PREDECESSOR_REPORTS', previous)
    monkeypatch.setattr(driver.original.resources, 'same', lambda actor: None if actor['pid'] == 123 else dict(**actor, state='R'))
    with pytest.raises(ValueError, match='workers must actually end'):
        driver.parent({}, predecessor)
    assert not actions and not reports.exists() and not output.exists()


def test_seeded_head_and_complete_native_state_are_unchanged_by_bridge_install(monkeypatch):
    driver = load_driver()
    torch.set_num_threads(1); torch.manual_seed(940)
    config = ModelConfig(vocab_size=32, hidden_size=16, intermediate_size=32, num_layers=1,
        num_heads=4, num_kv_heads=2, max_text_length=64, vision_hidden_size=8,
        vision_intermediate_size=16, vision_layers=1, vision_heads=2, image_size=16, patch_size=4, pool_size=2)
    head = PointGroundingHead(TinyMultimodalLM(config), rank=8)
    before = {n: t.clone() for n, t in head.state_dict().items()}
    rng = torch.get_rng_state().clone(); inventory = head.trainable_inventory()
    metadata = {'own-native': True}
    monkeypatch.setattr(driver, '_native_load_head', lambda plan, device, phase: (head, metadata))
    actual, actual_metadata = driver.load_head({}, 'cpu')
    assert actual is head and actual_metadata is metadata
    assert inventory == actual.trainable_inventory()
    assert torch.equal(rng, torch.get_rng_state())
    assert all(torch.equal(t, actual.state_dict()[n]) for n, t in before.items())
