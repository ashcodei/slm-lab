import importlib.util
import json
import sys
from pathlib import Path

import pytest


def driver():
    path=Path(__file__).resolve().parents[1]/'scripts/train_dense_foundation_v1.py'
    sys.path.insert(0,str(path.parent));spec=importlib.util.spec_from_file_location('dense_foundation_driver_test',path)
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result);return result


def test_exact_live_prior_owner_and_workers_block_despite_stale_completion_and_failures_reject(monkeypatch,tmp_path):
    d=driver();monkeypatch.setattr(d,'PREVIOUS',tmp_path)
    prior=dict(pid=11,ticks='101');child=dict(pid=12,ticks='102')
    (tmp_path/'children.json').write_text(json.dumps([child]));(tmp_path/'complete.json').write_text('{}')
    monkeypatch.setattr(d.owner.resources,'same',lambda a:a if a==prior else None)
    assert d.previous_ended(prior) is False
    monkeypatch.setattr(d.owner.resources,'same',lambda a:a if a==child else None)
    assert d.previous_ended(prior) is False
    monkeypatch.setattr(d.owner.resources,'same',lambda a:None)
    with pytest.raises(ValueError,match='successfully'):d.previous_ended(prior)
    for name in ('restored.json','independent-comparison.json'):(tmp_path/name).write_text('{}')
    assert d.previous_ended(prior)
    (tmp_path/'failure.json').write_text('{}')
    with pytest.raises(ValueError,match='successfully'):d.previous_ended(prior)


def test_active_original_work_rejects_before_GPU_query_and_signals(monkeypatch):
    d=driver()
    def active():raise ValueError('Original work active')
    monkeypatch.setattr(d.owner,'idle_original',active)
    monkeypatch.setattr(d.subprocess,'check_output',lambda *a,**k:pytest.fail('No GPU query before original idle admission'))
    monkeypatch.setattr(d.os,'kill',lambda *a,**k:pytest.fail('No signals to active original work'))
    with pytest.raises(ValueError,match='active'):d.require_idle_devices()


def test_full_mixed_source_epochs_preserve_every_source_language_and_unrepeated_partial_tail():
    from test_dense_foundation_stage_v1 import data,options
    d=driver();sources=data();setting=options(sources);budget=d.exposure(sources,setting)
    assert budget['actual_exposures']==14 and budget['actual_unique_sources']==7
    assert budget['complete_passes']==2 and budget['actual_last_update_sources']==1
    assert budget['repeated_rounding_exposures']==0
    assert budget['complete_source_budget']['education']['response_EOS_targets']==7
    assert budget['complete_source_budget']['program']['response_EOS_targets']==3
    assert set(budget['complete_source_budget'])=={'education','variety','human','program'}
    assert d.source_extremes(sources)==[0,2,3,4,6]
