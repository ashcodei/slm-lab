import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


def load():
    import sys
    path=Path(__file__).resolve().parents[1]/'scripts/train_code_initialized_alignment_v1.py'
    sys.path.insert(0,str(path.parent))
    spec=importlib.util.spec_from_file_location('code_initialized_driver_test',path)
    driver=importlib.util.module_from_spec(spec);spec.loader.exec_module(driver);return driver


def test_exact_prior_owner_and_workers_block_even_after_a_stale_completion_file(monkeypatch,tmp_path):
    driver=load();monkeypatch.setattr(driver,'PREVIOUS',tmp_path)
    actor=dict(pid=201,ticks='51');child=dict(pid=202,ticks='52')
    (tmp_path/'children.json').write_text(json.dumps([child]));(tmp_path/'complete.json').write_text('{}')
    monkeypatch.setattr(driver.owner.resources,'same',lambda x:x if x==actor else None)
    assert driver.previous_ended(actor) is False
    monkeypatch.setattr(driver.owner.resources,'same',lambda x:x if x==child else None)
    assert driver.previous_ended(actor) is False
    monkeypatch.setattr(driver.owner.resources,'same',lambda x:None)
    with pytest.raises(ValueError,match='restore'):driver.previous_ended(actor)
    (tmp_path/'restored.json').write_text('{}');assert driver.previous_ended(actor)


def test_active_original_work_rejected_before_GPU_queries_or_signals(monkeypatch):
    driver=load()
    def active():raise ValueError('Original active work is not an idle data waiter')
    monkeypatch.setattr(driver.owner,'idle_original',active)
    monkeypatch.setattr(driver.subprocess,'check_output',lambda *a,**k:pytest.fail('Never query/acquire devices before original idle admission'))
    monkeypatch.setattr(driver.os,'kill',lambda *a,**k:pytest.fail('Never signal active original work'))
    with pytest.raises(ValueError,match='active work'):driver.require_idle_devices()


def test_whole_mixed_budget_covers_all_sources_and_records_shorter_stream_repetition():
    from slm_lab.code_initialized_alignment_evidence_v1 import budget
    class Source:
        def __init__(self,name):
            self.name=name
            self.rows=[dict(native=dict(response=dict(length=5+i,target_start=2))) for i in range(7)]
            self.pack=SimpleNamespace(index=[dict(length=7+i,target_start=3) for i in range(3)],positions=[0,1,2])
        def __len__(self):return 7 if self.name=='human' else 3
    streams=dict(human=Source('human'),generation=Source('generation'))
    actual=budget(streams,dict(seed=113,steps=3))
    assert actual['human']['source_exposures']==12 and actual['human']['unique_original_positions']==7
    assert actual['human']['repeated_exposures']==5
    assert actual['generation']['source_exposures']==actual['generation']['unique_original_positions']==3
    assert actual['generation']['complete_context_tokens']==24 and actual['generation']['response_EOS_targets']==15
