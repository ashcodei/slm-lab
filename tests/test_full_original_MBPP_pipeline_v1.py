import importlib.util
import json
import os
from pathlib import Path
import sys

import pytest


def driver():
    root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'scripts'))
    spec=importlib.util.spec_from_file_location('full_original_MBPP_pipeline_unit',root/'scripts/assess_full_original_MBPP_pipeline_v1.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def test_eight_endpoints_fixed_and_actual_live_predecessor_waits(tmp_path):
    module=driver();specs=module.descriptors(tmp_path)
    assert tuple(specs)==module.ENDPOINTS and len(specs)==8
    assert specs['focused_trained']['arm']=='trained'
    assert specs['code_anchor_execution']['arm']=='anchor_execution'
    assert module.previous_ready(module.owner.resources.identity(os.getpid())) is False


def test_complete_all_source_epochs_assessments_and_four_cold_receipts_required(tmp_path,monkeypatch):
    module=driver();module.PREVIOUS=tmp_path;identity=dict(pid=101,ticks='1')
    monkeypatch.setattr(module.owner.resources,'same',lambda identity:None)
    with pytest.raises(ValueError):module.previous_ready(identity)
    def save(name,value):(tmp_path/name).write_text(json.dumps(value))
    save('children.json',[]);save('plan.json',dict(planned_steps=7500));save('restored.json',{})
    save('training-complete.json',dict(actual_updates=7500,all58488new_code_and_all_admitted_public_execution_targets_once=True))
    save('complete.json',dict(all58488new_code_and_all_admitted_execution_examples_once=True,
        actual_original_QA_question_endpoint_assessments=33922,actual_original_AI2D_question_endpoint_condition_assessments=18528,
        standalone_model_and_tokenizer_with_no_older_checkpoint_dependency=True))
    for rank in range(4):save(f'standalone-cold-reload-{rank}.json',dict(physical_GPU=rank,actual_completed_updates=7500,every_exported_parameter_bit_exact=True))
    assert module.previous_ready(identity)
    save('training-complete.json',dict(actual_updates=7499,all58488new_code_and_all_admitted_public_execution_targets_once=True))
    with pytest.raises(ValueError,match='Every original source'):module.previous_ready(identity)
    save('training-complete.json',dict(actual_updates=7500,all58488new_code_and_all_admitted_public_execution_targets_once=True))
    save('standalone-cold-reload-3.json',dict(physical_GPU=3,actual_completed_updates=7500,every_exported_parameter_bit_exact=False))
    with pytest.raises(ValueError,match='cold-load'):module.previous_ready(identity)
