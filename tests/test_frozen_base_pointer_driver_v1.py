import importlib.util
import json
from pathlib import Path
import sys

import pytest
import torch

from test_own_code_delta_v1 import own_models


def driver():
    path = Path('scripts/train_frozen_base_pointer_v1.py').resolve()
    spec = importlib.util.spec_from_file_location('checked_frozen_pointer_driver', path)
    module = importlib.util.module_from_spec(spec); sys.path.insert(0, str(path.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


def test_live_coding_owner_and_every_registered_actor_reject_before_resource_changes(monkeypatch, tmp_path):
    module = driver()
    monkeypatch.setattr(module, 'REPORTS', tmp_path / 'no-report'); monkeypatch.setattr(module, 'OUTPUT', tmp_path / 'no-output')
    events = []
    monkeypatch.setattr(module.subprocess, 'run', lambda *a, **k: events.append('GPU query'))
    monkeypatch.setattr(module.os, 'kill', lambda *a, **k: events.append('signal'))
    predecessor = dict(pid=123, ticks='one-exact-code-owner')
    monkeypatch.setattr(module.owner.resources, 'same', lambda a: dict(**a, state='R'))
    with pytest.raises(ValueError, match='coding owner must actually end'):
        module.parent({}, predecessor, {})
    previous = tmp_path / 'previous'; previous.mkdir()
    (previous / 'plan.json').write_text(json.dumps(dict(owner=predecessor)))
    (previous / 'children.json').write_text(json.dumps([dict(pid=124, ticks='one-exact-code-rank')]))
    monkeypatch.setattr(module, 'PREVIOUS', previous)
    monkeypatch.setattr(module.owner.resources, 'same', lambda a: None if a['pid'] == 123 else dict(**a, state='R'))
    with pytest.raises(ValueError, match='coding worker must actually end'):
        module.parent({}, predecessor, {})
    assert not events and not module.REPORTS.exists() and not module.OUTPUT.exists()


def test_terminal_handoff_requires_actual_complete_code_artifact_or_explicit_failure(tmp_path):
    module = driver()
    with pytest.raises(ValueError, match='without complete actual terminal'):
        module.predecessor_artifacts(tmp_path)
    failure = tmp_path / 'failure.json'; failure.write_text('{}')
    assert module.predecessor_artifacts(tmp_path) == [failure]
    success = tmp_path / 'independent-comparison.json'; success.write_text('{}')
    assert module.predecessor_artifacts(tmp_path) == [success]


def test_baseline_loader_freezes_the_entire_original_base_before_head_use(monkeypatch):
    module = driver()
    from slm_lab import training
    base, _ = own_models(); original = {n: t.clone() for n, t in base.state_dict().items()}
    monkeypatch.setattr(training, 'load_checkpoint', lambda *a, **k: (base, {'tokenizer_sha256': 'a' * 64}))
    head, metadata = module.load_head(dict(training_options=dict(seed=839032), head_rank=8), 'cpu')
    assert head.base is base and all(not p.requires_grad for p in base.parameters())
    assert len([p for p in head.parameters() if p.requires_grad]) == 2
    assert all(torch.equal(t, head.base.state_dict()[n]) for n, t in original.items())
    assert metadata['actual_inventory']['actual_trainable_parameters'] == 256
