import importlib.util
import json
from pathlib import Path
import sys

import pytest


def driver():
    root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'scripts'))
    spec=importlib.util.spec_from_file_location('phase_joint_driver_test',root/'scripts/train_phase_joint_v2.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def test_exact_live_predecessor_or_worker_always_prevents_new_GPU_acquisition(tmp_path,monkeypatch):
    m=driver();m.PREVIOUS=tmp_path;prior=dict(owner=dict(pid=101,ticks='202'))
    monkeypatch.setattr(m.owner.resources,'same',lambda actor:dict(state='R'))
    assert not m.previous_ready(prior)
    (tmp_path/'children.json').write_text(json.dumps([dict(pid=303,ticks='404')]))
    monkeypatch.setattr(m.owner.resources,'same',lambda actor:dict(state='R') if actor['pid']==303 else None)
    assert not m.previous_ready(prior)


def test_ended_predecessor_needs_restoration_and_actual_complete_native_assessments(tmp_path,monkeypatch):
    m=driver();m.PREVIOUS=tmp_path;prior=dict(owner=dict(pid=101,ticks='202'))
    monkeypatch.setattr(m.owner.resources,'same',lambda actor:None);(tmp_path/'children.json').write_text('[]')
    with pytest.raises(ValueError,match='restore'):m.previous_ready(prior)
    (tmp_path/'restored.json').write_text('{}')
    with pytest.raises(ValueError,match='actually complete'):m.previous_ready(prior)
    (tmp_path/'training-complete.json').write_text(json.dumps(dict(matched_native_development_assessments_finished=True)))
    assert m.previous_ready(prior)
    (tmp_path/'failure.json').write_text('{}')
    with pytest.raises(ValueError,match='actually complete'):m.previous_ready(prior)
