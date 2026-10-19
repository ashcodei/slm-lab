import importlib.util
from pathlib import Path
import json

import pytest

ROOT=Path(__file__).resolve().parents[1]


def driver():
    import sys
    sys.path.insert(0,str(ROOT/'scripts'))
    spec=importlib.util.spec_from_file_location('fast_joint_driver_test',ROOT/'scripts/train_fast_joint_v1.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def test_previous_still_live_or_live_registered_worker_cannot_handover(tmp_path,monkeypatch):
    d=driver();d.PREVIOUS=tmp_path;d.HANDOVER=tmp_path/'handover';identity=dict(pid=111,ticks='222')
    monkeypatch.setattr(d.owner.resources,'same',lambda actor:dict(state='R'))
    assert not d.previous_ended(identity)
    monkeypatch.setattr(d.owner.resources,'same',lambda actor:dict(state='R') if actor['pid']==333 else None)
    (tmp_path/'children.json').write_text(json.dumps([dict(pid=333,ticks='444')]))
    assert not d.previous_ended(identity)


def test_missing_or_failed_predecessor_requires_explicit_completed_training_retirement(tmp_path,monkeypatch):
    d=driver();d.PREVIOUS=tmp_path;d.HANDOVER=tmp_path/'handover';d.HANDOVER.mkdir();identity=dict(pid=111,ticks='222')
    monkeypatch.setattr(d.owner.resources,'same',lambda actor:None)
    with pytest.raises(ValueError,match='restored'):d.previous_ended(identity)
    (tmp_path/'restored.json').write_text('{}')
    with pytest.raises(ValueError,match='explicit'):d.previous_ended(identity)
    receipt=dict(retired_owner=identity,both_training_arms_completed_before_signal=True,remaining_evaluations_explicitly_incomplete=True,
        retained_terminal_training_receipts_sha256={})
    (d.HANDOVER/'code-evaluation-retirement.json').write_text(json.dumps(receipt))
    assert d.previous_ended(identity)
    receipt['both_training_arms_completed_before_signal']=False
    (d.HANDOVER/'code-evaluation-retirement.json').write_text(json.dumps(receipt))
    with pytest.raises(ValueError,match='receipt'):d.previous_ended(identity)
