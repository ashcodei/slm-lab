import importlib.util
import json
import os
from pathlib import Path
import sys

import pytest


def driver():
    root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'scripts'))
    spec=importlib.util.spec_from_file_location('code_first_continuation_unit',root/'scripts/train_code_first_continuation_v3.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def test_explicit_handoff_and_actual_live_original_main_prevent_early_GPU_takeover(tmp_path):
    m=driver();m.HANDOFF=tmp_path/'handoff.json';identity=m.owner.resources.identity(os.getpid())
    deps=dict(main=identity,auxiliary=identity,states=identity)
    assert m.dependencies_ready(deps) is False
    m.HANDOFF.write_text(json.dumps(dict(old_failed_production_controllers_all_terminal=True,original_main_owner=identity)))
    assert m.dependencies_ready(deps) is False


def test_complete_main_admitted_auxiliary_and_original_worker_restoration_required(tmp_path,monkeypatch):
    m=driver();m.HANDOFF=tmp_path/'handoff.json';m.MAIN=tmp_path/'main';m.AUXILIARY=tmp_path/'auxiliary';m.STATES=tmp_path/'states'
    m.MAIN.mkdir();m.AUXILIARY.mkdir();m.STATES.mkdir();deps=dict(main=dict(pid=101,ticks='1'),auxiliary=dict(pid=102,ticks='2'),states=dict(pid=103,ticks='3'))
    monkeypatch.setattr(m.owner.resources,'same',lambda identity:None)
    def save(folder,name,value):(folder/name).write_text(json.dumps(value))
    m.HANDOFF.write_text(json.dumps(dict(old_failed_production_controllers_all_terminal=True,original_main_owner=deps['main'])))
    save(m.MAIN,'children.json',[]);save(m.MAIN,'restored.json',{})
    save(m.MAIN,'plan.json',dict(planned_steps=35284,all_selected_family_records_once=True))
    save(m.MAIN,'training-complete.json',dict(actual_updates=35284))
    save(m.MAIN,'complete.json',dict(native_development_assessments_complete=True))
    save(m.AUXILIARY,'complete.json',dict(every_original_reference_disposition_and_candidate_actual_native_target_rebuilt=True,
        dataset_admitted_to_auxiliary_CE_training=True))
    save(m.STATES,'complete.json',dict(every_original_source_and_every_state_target_and_mask_rebuilt=True,dataset_admitted_to_auxiliary_CE_training=True))
    assert m.dependencies_ready(deps)
    save(m.STATES,'complete.json',dict(every_original_source_and_every_state_target_and_mask_rebuilt=True,dataset_admitted_to_auxiliary_CE_training=False))
    with pytest.raises(ValueError):m.dependencies_ready(deps)
    save(m.STATES,'complete.json',dict(every_original_source_and_every_state_target_and_mask_rebuilt=True,dataset_admitted_to_auxiliary_CE_training=True))
    save(m.MAIN,'training-complete.json',dict(actual_updates=12853))
    with pytest.raises(ValueError,match='main35284'):m.dependencies_ready(deps)
    save(m.MAIN,'training-complete.json',dict(actual_updates=35284));(m.MAIN/'restored.json').unlink()
    with pytest.raises(ValueError,match='restore'):m.dependencies_ready(deps)
