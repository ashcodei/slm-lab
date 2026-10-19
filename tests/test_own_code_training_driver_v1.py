import importlib.util
import json
from pathlib import Path
import sys

import pytest

from slm_lab.own_code_training_stage_v1 import LanguageCycleOrder


def test_existing_vision_owner_or_registered_actor_rejects_before_resource_changes(monkeypatch, tmp_path):
    path = Path('scripts/train_own_code_generation_repair_v1.py').resolve()
    spec = importlib.util.spec_from_file_location('checked_own_code_training_driver_v1', path)
    driver = importlib.util.module_from_spec(spec); sys.path.insert(0, str(path.parent))
    try:
        spec.loader.exec_module(driver)
    finally:
        sys.path.pop(0)
    monkeypatch.setattr(driver, 'REPORTS', tmp_path / 'no-report')
    monkeypatch.setattr(driver, 'OUTPUT', tmp_path / 'no-output')
    actions = []
    monkeypatch.setattr(driver.subprocess, 'run', lambda *a, **k: actions.append('GPU query'))
    monkeypatch.setattr(driver.os, 'kill', lambda *a, **k: actions.append('signal'))
    predecessor = dict(pid=123, ticks='one-owner-birth')
    monkeypatch.setattr(driver.owner.resources, 'same', lambda a: dict(**a, state='R'))
    with pytest.raises(ValueError, match='vision owner must actually end'):
        driver.parent({}, {}, predecessor, {})
    previous = tmp_path / 'previous'; previous.mkdir()
    (previous / 'plan.json').write_text(json.dumps(dict(owner=predecessor)))
    (previous / 'children.json').write_text(json.dumps([dict(pid=124, ticks='rank-birth')]))
    monkeypatch.setattr(driver, 'PREVIOUS', previous)
    monkeypatch.setattr(driver.owner.resources, 'same', lambda a: None if a['pid'] == 123 else dict(**a, state='R'))
    with pytest.raises(ValueError, match='vision worker must actually end'):
        driver.parent({}, {}, predecessor, {})
    assert not actions and not driver.REPORTS.exists() and not driver.OUTPUT.exists()


def test_exact_completed_predecessor_artifacts_are_bound_and_missing_is_not_assumed_failure(tmp_path):
    path = Path('scripts/train_own_code_generation_repair_v1.py').resolve()
    spec = importlib.util.spec_from_file_location('checked_own_code_terminal_artifacts_v1', path)
    driver = importlib.util.module_from_spec(spec); sys.path.insert(0, str(path.parent))
    try:
        spec.loader.exec_module(driver)
    finally:
        sys.path.pop(0)
    with pytest.raises(ValueError, match='without complete actual'):
        driver.predecessor_artifacts(tmp_path)
    names = ['complete-neural-source-readback.json', 'matched-attention-and-native-comparison.json']
    for name in names:
        (tmp_path / name).write_text('{}')
    assert driver.predecessor_artifacts(tmp_path) == [tmp_path / name for name in names]


def test_language_cycles_cover_rare_source_language_without_claiming_full_corpus_pass():
    groups = {'Rare': [0], 'Common': list(range(1, 32))}
    order = LanguageCycleOrder(groups, 819)
    actual = [order.at(p) for p in range(10)]
    assert actual.count(0) == 5
    assert len(set(p for p in actual if p != 0)) == 5
    for offset in range(0, 10, 2):
        assert sum(p == 0 for p in actual[offset:offset + 2]) == 1
    cursor = order.cursor(10)
    assert cursor['by_language']['Rare']['actual_draws'] == 5
    assert cursor['by_language']['Rare']['unique_original_positions'] == 1
    assert cursor['by_language']['Common']['actual_draws'] == 5
    assert not cursor['full_original_source_pass']
    for p in [0, 5, 1, 9]:
        assert order.at(p) == actual[p]
