import importlib.util
import json
import os
from pathlib import Path
import sys

import pytest


def driver():
    root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'scripts'))
    spec=importlib.util.spec_from_file_location('focused_continuation_driver_unit',root/'scripts/train_focused_code_continuation_v1.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def test_actual_live_birth_identity_waits_and_reused_pid_cannot_replace_missing_completed_dependency(tmp_path):
    module=driver();module.VISUAL_READER=tmp_path
    identity=module.owner.resources.identity(os.getpid())
    assert module.dependencies_ready(dict(visual_reader=identity,auxiliary=identity)) is False
    with pytest.raises(ValueError,match='must complete'):
        module.dependencies_ready(dict(visual_reader=dict(identity,ticks='0'),auxiliary=identity))


def test_terminal_predecessors_require_complete_epoch_admission_visual_readback_and_restoration(tmp_path,monkeypatch):
    module=driver()
    for name in ['VISUAL_READER','VISUAL','AUXILIARY','CODE_SOURCE']:
        folder=tmp_path/name;folder.mkdir();setattr(module,name,folder)
    deps=dict(visual_reader=dict(pid=11,ticks='1'),auxiliary=dict(pid=12,ticks='2'))
    monkeypatch.setattr(module.owner.resources,'same',lambda identity:None)
    def save(folder,name,value):(folder/name).write_text(json.dumps(value))
    save(module.VISUAL_READER,'complete.json',dict(actual_question_endpoint_image_condition_verdicts_rederived=55584,
        all_original_primary_populations_and_native_budgets_preserved=True))
    save(module.AUXILIARY,'complete.json',dict(every_original_reference_disposition_and_candidate_actual_native_target_rebuilt=True,
        dataset_admitted_to_auxiliary_CE_training=True))
    save(module.VISUAL,'plan.json',dict(owner=dict(pid=13,ticks='3')))
    save(module.VISUAL,'children.json',[dict(pid=14,ticks='4')])
    save(module.VISUAL,'complete.json',dict(total_question_endpoint_condition_assessments=55584))
    save(module.VISUAL,'restored.json',dict(all_registered_point_actors_terminal=True))
    save(module.CODE_SOURCE,'complete.json',dict(all_3721_original_coding_sources_once_each_arm=True,
        matched_native_code_broad_human_preference_and_held_APPS_assessments_complete=True))
    assert module.dependencies_ready(deps)
    monkeypatch.setattr(module.owner.resources,'same',lambda identity:dict(state='R') if identity['pid']==14 else None)
    assert module.dependencies_ready(deps) is False
    monkeypatch.setattr(module.owner.resources,'same',lambda identity:None)
    (module.VISUAL/'restored.json').unlink()
    with pytest.raises(ValueError,match='restore'):module.dependencies_ready(deps)
    save(module.VISUAL,'restored.json',{})
    save(module.AUXILIARY,'complete.json',dict(every_original_reference_disposition_and_candidate_actual_native_target_rebuilt=True,
        dataset_admitted_to_auxiliary_CE_training=False))
    with pytest.raises(ValueError,match='independently admitted'):module.dependencies_ready(deps)
    save(module.AUXILIARY,'complete.json',dict(every_original_reference_disposition_and_candidate_actual_native_target_rebuilt=True,
        dataset_admitted_to_auxiliary_CE_training=True))
    save(module.CODE_SOURCE,'complete.json',dict(all_3721_original_coding_sources_once_each_arm=False,
        matched_native_code_broad_human_preference_and_held_APPS_assessments_complete=True))
    with pytest.raises(ValueError,match='full code epoch'):module.dependencies_ready(deps)
