import json
import shutil

import pytest
import torch

from slm_lab.corpus import file_hash
from slm_lab.public_pretraining_native_v1 import export,load,validate_start
from slm_lab.focused_code_continuation_data_v3 import CODE,EXECUTION,STATE
from test_own_dense_standalone_v1 import fixture


def test_partial_code_epoch_or_missing_original_code_output_state_exposure_cannot_initialize():
    source=dict(source_start_is_complete_main=True,complete_new_code_and_execution_source_epoch_once=True,
        complete_public_state_source_epoch_once=True,planned_steps=3,
        complete_source_budget={n:dict(examples=1) for n in [CODE,EXECUTION,STATE]},
        selected_source_budget={n:dict(presentations=1) for n in [CODE,EXECUTION,STATE]})
    validate_start(source,dict(completed_updates=3))
    with pytest.raises(ValueError,match='complete own code/output/state'):
        validate_start(source,dict(completed_updates=2))
    source['selected_source_budget'][STATE]['presentations']=0
    with pytest.raises(ValueError):validate_start(source,dict(completed_updates=3))


def test_actual_trained_model_all_values_and_image_outputs_cold_load_without_starting_checkpoint(tmp_path):
    model,ids,old,sources=fixture(tmp_path)
    start=tmp_path/'ancestor.pt';old.rename(start)
    plan=dict(complete_code_first_inputs=dict(checkpoint=str(start),checkpoint_sha256=file_hash(start),plan='unit-source-plan',plan_sha256='a'*64),
        actual_counts=model.counts(),tokenizer_sha256=file_hash(tmp_path/'tokenizer.json'),planned_steps=1,
        source_schedule_sha256='b'*64,public_raw_admission_sha256='c'*64,public_function_admission_sha256='d'*64)
    path=tmp_path/'run-plan.json';path.write_text(json.dumps(plan));digest=file_hash(path)
    final=tmp_path/'model.pt';torch.save(export(model,plan,digest,'trained',1),final)
    start.unlink() # The independently packaged final model must not open its ancestor.
    cold,metadata=load(final,path,'trained',expected_checkpoint_sha256=file_hash(final),expected_plan_sha256=digest)
    assert metadata['no_starting_model_loaded_for_final_inference']
    assert all(torch.equal(a,b) for a,b in zip(model.parameters(),cold.parameters()))
    pixels=torch.linspace(-1,1,3*16*16).reshape(1,3,16,16)
    with torch.no_grad():assert torch.equal(model(ids,pixel_values=pixels)['logits'],cold(ids,pixel_values=pixels)['logits'])
    with pytest.raises(ValueError):
        load(final,path,'initial',expected_checkpoint_sha256=file_hash(final),expected_plan_sha256=digest)
    with pytest.raises(ValueError,match='declared'):
        export(model,plan,digest,'unknown-arm',1)
