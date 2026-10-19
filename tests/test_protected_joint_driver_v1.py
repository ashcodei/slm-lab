import importlib.util
import json
from pathlib import Path
import sys
import pytest


def driver():
    root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'scripts'))
    spec=importlib.util.spec_from_file_location('protected_driver_test',root/'scripts/train_protected_joint_v1.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def test_live_exact_owner_or_worker_always_prevents_protected_acquisition(tmp_path,monkeypatch):
    m=driver();prior=dict(owner=dict(pid=101,ticks='202'))
    monkeypatch.setattr(m.owner.resources,'same',lambda actor:dict(state='R'))
    assert not m.previous_ready(tmp_path,prior)
    (tmp_path/'children.json').write_text(json.dumps([dict(pid=303,ticks='404')]))
    monkeypatch.setattr(m.owner.resources,'same',lambda actor:dict(state='R') if actor['pid']==303 else None)
    assert not m.previous_ready(tmp_path,prior)


def test_partial_epoch_needs_restoration_and_exact_preserved_coding_priority_receipt(tmp_path,monkeypatch):
    m=driver();m.ROOT=tmp_path;previous=tmp_path/'previous';previous.mkdir();(tmp_path/'reports').mkdir()
    (previous/'children.json').write_text('[]');prior=dict(owner=dict(pid=101,ticks='202'))
    monkeypatch.setattr(m.owner.resources,'same',lambda actor:None)
    with pytest.raises(ValueError,match='restore'):m.previous_ready(previous,prior)
    (previous/'restored.json').write_text('{}')
    with pytest.raises(ValueError,match='finish'):m.previous_ready(previous,prior)
    receipt=dict(retired_exact_owner=prior['owner'],retained_completed_updates=2048,
        partial_epoch_and_full_endpoints_explicitly_incomplete=True,initial_and_all_learned_checkpoints_preserved=True,matched_reports_sha256={})
    path=tmp_path/'reports/protected-joint-v1-handover.json';path.write_text(json.dumps(receipt))
    import monitor_fast_joint_regression_v1 as monitor
    monkeypatch.setattr(monitor,'scores',lambda folder:dict(protocol='same',syntax=20 if folder.name=='initial' else 12,correct=int(folder.name=='initial')))
    assert m.previous_ready(previous,prior)
    receipt['retired_exact_owner']=dict(pid=101,ticks='999');path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError,match='handover'):m.previous_ready(previous,prior)
