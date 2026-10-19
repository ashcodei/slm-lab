import importlib.util
import json
from pathlib import Path
import sys

import pytest


def driver():
    scripts=Path(__file__).resolve().parents[1]/'scripts';sys.path.insert(0,str(scripts))
    spec=importlib.util.spec_from_file_location('alignment_four_L40_driver_v2_test',scripts/'probe_own_alignment_four_L40_v2.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def test_actual_live_previous_GPU_birth_rejects_before_queries_or_acquisition(monkeypatch,tmp_path):
    module=driver();monkeypatch.setattr(module,'ROOT',tmp_path);monkeypatch.setattr(module,'REPORTS',tmp_path/'capacity')
    prior=tmp_path/'reports/own-onpolicy-pilot-v2';prior.mkdir(parents=True)
    (prior/'plan.json').write_text(json.dumps(dict(owner=dict(pid=11,ticks='21'))))
    (prior/'children.json').write_text(json.dumps([dict(pid=12,ticks='22')]))
    (prior/'restored.json').write_text('{}')
    monkeypatch.setattr(module.subprocess,'check_output',lambda *a,**k:pytest.fail('No GPU query while prior actors are live'))
    monkeypatch.setattr(module.owner,'idle_original',lambda:pytest.fail('No acquisition with a live previous actor'))
    for live in (11,12):
        monkeypatch.setattr(module.owner.resources,'same',lambda actor:actor if actor['pid']==live else None)
        with pytest.raises(ValueError,match='actually end'):
            module.parent()
        assert not module.REPORTS.exists()


def test_original_active_work_rejects_before_source_loading_or_GPU_queries(monkeypatch,tmp_path):
    module=driver();monkeypatch.setattr(module,'REPORTS',tmp_path/'capacity')
    monkeypatch.setattr(module,'previous_ended',lambda:None)
    monkeypatch.setattr(module.owner.resources,'same',lambda actor:dict(**actor,state='R'))
    monkeypatch.setattr(module.subprocess,'check_output',lambda *a,**k:pytest.fail('No GPU query with active original work'))
    with pytest.raises(ValueError,match='Never preempt original work'):
        module.parent()
    assert not module.REPORTS.exists()


def test_extremes_keep_every_whole_context_and_response_maximum_for_both_sides():
    module=driver()
    class Source:
        fields=('chosen','rejected')
        rows=[dict(native=dict(chosen=dict(length=4096,target_start=4000),rejected=dict(length=200,target_start=100))),
              dict(native=dict(chosen=dict(length=4000,target_start=20),rejected=dict(length=3500,target_start=20))),
              dict(native=dict(chosen=dict(length=200,target_start=100),rejected=dict(length=4090,target_start=3900))),
              dict(native=dict(chosen=dict(length=200,target_start=100),rejected=dict(length=4000,target_start=10)))]
        def __len__(self):return len(self.rows)
    assert module.extrema(Source())==[0,1,2,3]
