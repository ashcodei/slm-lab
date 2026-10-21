import importlib.util
import json
from pathlib import Path
import sys

import pytest


def driver():
    root = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(root/'scripts'))
    spec = importlib.util.spec_from_file_location('broad_handoff_v3_test', root/'scripts/resume_broad_anchored_code_GPU_v3.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def test_exact_live_parent_or_worker_blocks_handoff_without_relying_on_queue_files(tmp_path, monkeypatch):
    module = driver(); monkeypatch.setattr(module, 'PRIOR', tmp_path)
    parent = dict(pid=100, birth='200'); worker = dict(pid=300, birth='400')
    monkeypatch.setattr(module, 'live', lambda actor:actor == parent)
    assert not module.predecessor_ready([parent, worker])
    monkeypatch.setattr(module, 'live', lambda actor:actor == worker)
    assert not module.predecessor_ready([parent, worker])
    assert module.normalize(dict(pid=300, ticks='400', state='R', subject='old worker')) == worker


def test_dead_predecessor_needs_complete_original_scope_and_all_registered_contexts_terminal(tmp_path, monkeypatch):
    module = driver(); monkeypatch.setattr(module, 'PRIOR', tmp_path)
    monkeypatch.setattr(module, 'live', lambda actor:False)
    actors = [dict(pid=100, birth='200'), dict(pid=300, birth='400')]
    with pytest.raises(ValueError, match='full four-rate'):
        module.predecessor_ready(actors)
    (tmp_path/'complete.json').write_text(json.dumps(dict(all_four_rates_all_230updates_and_bounded_assessments=True)))
    (tmp_path/'attempts/2').mkdir(parents=True)
    (tmp_path/'attempts/2/restored.json').write_text(json.dumps(dict(own_children_terminal=False)))
    with pytest.raises(ValueError, match='registered GPU'):
        module.predecessor_ready(actors)
    (tmp_path/'attempts/2/restored.json').write_text(json.dumps(dict(own_children_terminal=True)))
    assert module.predecessor_ready(actors)
    (tmp_path/'complete.json').write_text(json.dumps(dict(all_four_rates_all_230updates_and_bounded_assessments=False)))
    with pytest.raises(ValueError, match='full four-rate'):
        module.predecessor_ready(actors)
