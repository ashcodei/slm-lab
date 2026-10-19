import importlib.util
import json
from pathlib import Path
import sys

import pytest


def driver():
    scripts=Path(__file__).resolve().parents[1]/'scripts';sys.path.insert(0,str(scripts))
    spec=importlib.util.spec_from_file_location('whole_human_SFT_driver_v1_test',scripts/'train_whole_human_SFT_v1.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def test_live_previous_GPU_owner_or_rank_rejects_before_queries_or_acquisition(monkeypatch,tmp_path):
    module=driver();monkeypatch.setattr(module,'ROOT',tmp_path);monkeypatch.setattr(module,'REPORTS',tmp_path/'new')
    monkeypatch.setattr(module,'OUTPUT',tmp_path/'weights')
    previous=tmp_path/'reports/own-alignment-four-L40-v2';previous.mkdir(parents=True)
    (previous/'plan.json').write_text(json.dumps(dict(owner=dict(pid=11,ticks='21'))))
    (previous/'children.json').write_text(json.dumps([dict(pid=12,ticks='22')]))
    (previous/'restored.json').write_text('{}')
    monkeypatch.setattr(module.subprocess,'check_output',lambda *a,**k:pytest.fail('No GPU query with live previous work'))
    monkeypatch.setattr(module.owner,'idle_original',lambda:pytest.fail('No waiter hold with live previous work'))
    for live in (11,12):
        monkeypatch.setattr(module.owner.resources,'same',lambda actor:actor if actor['pid']==live else None)
        with pytest.raises(ValueError,match='must end'):module.parent()
        assert not module.REPORTS.exists() and not module.OUTPUT.exists()


def test_active_original_work_rejects_before_source_reads_and_GPU_queries(monkeypatch,tmp_path):
    module=driver();monkeypatch.setattr(module,'REPORTS',tmp_path/'new');monkeypatch.setattr(module,'OUTPUT',tmp_path/'weights')
    monkeypatch.setattr(module,'previous_ended',lambda:None)
    monkeypatch.setattr(module.owner.resources,'same',lambda actor:dict(**actor,state='R'))
    monkeypatch.setattr(module.subprocess,'check_output',lambda *a,**k:pytest.fail('No GPU query with active original work'))
    with pytest.raises(ValueError,match='Never preempt original work'):module.parent()
    assert not module.REPORTS.exists() and not module.OUTPUT.exists()


def test_actual_full_source_budget_covers_every_position_and_explicit_rounding():
    from slm_lab.whole_human_sft_evidence_v1 import budget
    from test_whole_human_sft_stage_v1 import Source
    result=budget(Source(),dict(seed=71,steps=2,microbatch=1,accumulation=4))
    assert result['source_population']==result['unique_original_positions']==5
    assert result['source_exposures']==8 and result['rounding_exposures']==3
    assert sum(n-1 for n in result['repeated_source_positions'].values())==3
