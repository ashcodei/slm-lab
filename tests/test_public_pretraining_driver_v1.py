import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import train_public_pretraining_v1 as driver


def fixture(tmp_path,monkeypatch):
    previous=tmp_path/'previous';previous.mkdir()
    preparation=tmp_path/'preparation';preparation.mkdir()
    handoff=tmp_path/'handoff.json'
    monkeypatch.setattr(driver,'PREVIOUS',previous);monkeypatch.setattr(driver,'PREPARATION',preparation);monkeypatch.setattr(driver,'HANDOFF',handoff)
    identities=dict(code_first=dict(pid=100,ticks='200'),source=dict(pid=101,ticks='201'),self=dict(pid=102,ticks='202'))
    monkeypatch.setattr(driver.owner.resources,'same',lambda identity:False)
    for key,path in [('code_first',previous),('source',preparation)]:
        (path/'plan.json').write_text(json.dumps(dict(owner=identities[key],planned_steps=7558)))
    (previous/'children.json').write_text('[]');(previous/'restored.json').write_text('{}')
    (previous/'training-complete.json').write_text(json.dumps(dict(actual_updates=7558)))
    (previous/'complete.json').write_text(json.dumps(dict(original_local_code_and_broad_assessments_complete=True,
        all58488new_code_and_all_admitted_execution_examples_once=True,all_admitted_public_state_examples_once=True)))
    (preparation/'complete.json').write_text(json.dumps(dict(complete=True,all_new_admitted_raw_train_windows_once=True,
        all826independently_admitted_source_train_functions_once=True,all_previous_selected_source_presentations_exactly_preserved=True)))
    evidence=tmp_path/'tested-replacements.txt';evidence.write_text('unit handoff evidence')
    record=dict(complete=True,old_unstarted_controllers_all_terminal=True,new_public_pretraining_owner=identities['self'],
        original_code_first_owner=identities['code_first'],replacement_full_pipeline_tested_and_queued=True,
        retired_unstarted_controllers=[dict(pid=99,ticks='199')],source_files_sha256={str(evidence):driver.file_hash(evidence)})
    return identities,handoff,record


def test_no_handoff_or_live_registered_original_worker_cannot_release_new_training(tmp_path,monkeypatch):
    identities,handoff,record=fixture(tmp_path,monkeypatch)
    assert not driver.dependencies_ready(identities)
    handoff.write_text(json.dumps(record));assert driver.dependencies_ready(identities)
    (driver.PREVIOUS/'children.json').write_text(json.dumps([dict(pid=98,ticks='198')]))
    monkeypatch.setattr(driver.owner.resources,'same',lambda identity:identity['pid']==98)
    assert not driver.dependencies_ready(identities)


def test_short_original_epoch_and_missing_function_or_retention_sources_are_rejected(tmp_path,monkeypatch):
    identities,handoff,record=fixture(tmp_path,monkeypatch);handoff.write_text(json.dumps(record))
    path=driver.PREVIOUS/'training-complete.json';path.write_text(json.dumps(dict(actual_updates=7557)))
    with pytest.raises(ValueError,match='Every complete'):driver.dependencies_ready(identities)
    path.write_text(json.dumps(dict(actual_updates=7558)))
    path=driver.PREPARATION/'complete.json';value=json.loads(path.read_text())
    value['all826independently_admitted_source_train_functions_once']=False;path.write_text(json.dumps(value))
    with pytest.raises(ValueError,match='Every complete'):driver.dependencies_ready(identities)


def test_exact_handoff_owner_and_terminal_retired_process_births_and_evidence_are_required(tmp_path,monkeypatch):
    identities,handoff,record=fixture(tmp_path,monkeypatch)
    original=record['new_public_pretraining_owner'];record['new_public_pretraining_owner']=dict(pid=102,ticks='old')
    handoff.write_text(json.dumps(record))
    with pytest.raises(ValueError,match='Exact tested'):driver.dependencies_ready(identities)
    record['new_public_pretraining_owner']=original;handoff.write_text(json.dumps(record))
    monkeypatch.setattr(driver.owner.resources,'same',lambda identity:identity['pid']==99)
    with pytest.raises(ValueError,match='still own'):driver.dependencies_ready(identities)
    monkeypatch.setattr(driver.owner.resources,'same',lambda identity:False)
    Path(next(iter(record['source_files_sha256']))).write_text('changed after handoff')
    with pytest.raises(ValueError):driver.dependencies_ready(identities)
