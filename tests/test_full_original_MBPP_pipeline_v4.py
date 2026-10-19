import importlib.util
import json
import os
from pathlib import Path
import sys
import pytest


def driver():
    root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'scripts'))
    spec=importlib.util.spec_from_file_location('MBPP_code_first_unit',root/'scripts/assess_full_original_MBPP_pipeline_v4.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def test_fixed_eight_lineage_endpoints_and_actual_live_predecessor_waits(tmp_path):
    module=driver();specs=module.descriptors(tmp_path)
    assert tuple(specs)==module.ENDPOINTS and len(specs)==8
    assert specs['code_first_trained']['arm']=='trained'
    assert specs['standalone_final']['kind']=='standalone'
    assert specs['code_anchor_execution']['arm']=='anchor_execution'
    assert module.PREVIOUS.name=='full-AI2D-pipeline-v4'
    assert module.previous_ready(module.owner.resources.identity(os.getpid())) is False


def test_all_original_visual_shards_and_restoration_required(tmp_path,monkeypatch):
    module=driver();module.PREVIOUS=tmp_path;identity=dict(pid=101,ticks='1')
    monkeypatch.setattr(module.owner.resources,'same',lambda identity:None)
    with pytest.raises(ValueError):module.previous_ready(identity)
    def save(name,value):(tmp_path/name).write_text(json.dumps(value))
    save('children.json',[]);save('plan.json',dict(endpoints=module.descriptors(tmp_path)));save('restored.json',{})
    save('complete.json',dict(all_eight_predeclared_endpoints_assessed=True,original_questions_each_endpoint=3088,
        total_question_endpoint_assessments=24704,total_question_endpoint_condition_assessments=74112,
        all_actual_native_pixel_identities_and_delivered_tokens_and_likelihoods_retained=True))
    for endpoint in module.ENDPOINTS:
        folder=tmp_path/endpoint;folder.mkdir();(folder/'full-AI2D.json').write_text('{}')
        for rank in range(4):
            (folder/f'AI2D-shard-{rank}-complete.json').write_text(json.dumps(dict(physical_GPU=rank,endpoint=endpoint,
                actual_original_questions=772,actual_question_condition_assessments=2316,all_loaded_values_unchanged=True,model_updates=0)))
    assert module.previous_ready(identity)
    receipt=tmp_path/module.ENDPOINTS[-1]/'AI2D-shard-3-complete.json'
    row=json.loads(receipt.read_text());row['actual_original_questions']=771;receipt.write_text(json.dumps(row))
    with pytest.raises(ValueError,match='All32'):module.previous_ready(identity)
