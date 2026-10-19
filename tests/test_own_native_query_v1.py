import json
from pathlib import Path
import subprocess
import sys

from PIL import Image
import pytest
import torch
from tokenizers import Tokenizer

from slm_lab.corpus import file_hash
from slm_lab.own_dense_actor_v1 import OwnDenseActor
from slm_lab.own_actor_dense_fold_v1 import fold,export as export_folded
from slm_lab.own_native_query_v1 import inspect_package,load_package,prompt_ids,query,image_pixels
from test_own_dense_standalone_v1 import fixture


def test_complete_package_external_identity_and_actual_cold_registered_count(tmp_path):
    original,ids,path,sources=fixture(tmp_path)
    package=inspect_package(path,expected_sha256=file_hash(path),expected_sources=sources)
    assert package['inspect_only'] and package['registered_parameter_count_and_tensor_values_not_checked_by_inspection']
    assert package['source_map_verified_against_external_input']
    cold,metadata=load_package(package)
    assert metadata['actual_registered_parameters']==sum(p.numel() for p in original.parameters())<10**9
    with torch.no_grad():assert torch.equal(original(ids)['logits'],cold(ids)['logits'])
    with pytest.raises(ValueError,match='checksum'):inspect_package(path,expected_sha256='0'*64)
    with pytest.raises(ValueError,match='source identities'):inspect_package(path,expected_sources={'wrong':'0'*64})
    with pytest.raises(ValueError,match='tokenizer checksum'):inspect_package(path,expected_tokenizer_sha256='0'*64)


@pytest.mark.parametrize('with_image',[False,True])
def test_native_complete_text_and_image_outputs_match_direct_FP32_greedy_generation(tmp_path,with_image):
    model,_,path,_=fixture(tmp_path);tok=Tokenizer.from_file(str(tmp_path/'tokenizer.json'))
    image=tmp_path/'image.png' if with_image else None
    if image is not None:Image.new('RGB',(23,11),(60,170,30)).save(image)
    ids,text=prompt_ids(tok,model.config,'hi',raw_prompt=False,max_new_tokens=3)
    pixels,metadata=image_pixels(image,model.config,16)
    rng=torch.get_rng_state().clone()
    with torch.no_grad():wanted=model.generate(ids,pixels,max_new_tokens=3,eos_token_id=2,use_cache=True)
    result=query(model,tok,'hi',max_new_tokens=3,image=image,image_size=16,precision='fp32')
    assert result['complete_prompt']==text and result['complete_prompt_token_ids']==ids[0].tolist()
    assert result['actual_generated_token_ids']==wanted[0,ids.shape[1]:].tolist()
    assert result['response']==tok.decode(wanted[0,ids.shape[1]:].tolist())
    assert result['image']==metadata and result['generated_text_returned_without_repair']
    assert torch.equal(rng,torch.get_rng_state())


def test_overlong_complete_prompt_rejected_before_generation_and_invalid_image_size_rejected(tmp_path,monkeypatch):
    model,_,_,_=fixture(tmp_path);tok=Tokenizer.from_file(str(tmp_path/'tokenizer.json'))
    def forbidden(*args,**kwargs):raise AssertionError('Invalid input must never reach neural generation')
    monkeypatch.setattr(model,'generate',forbidden)
    with pytest.raises(ValueError,match='context'):query(model,tok,'a complete prompt',max_new_tokens=64)
    image=tmp_path/'image.png';Image.new('RGB',(16,16)).save(image)
    with pytest.raises(ValueError,match='resolution'):query(model,tok,'hi',max_new_tokens=1,image=image,image_size=17)


def test_folded_final_student_is_loadable_without_old_actor_or_base_checkpoint(tmp_path):
    model,_,_,sources=fixture(tmp_path);actor=OwnDenseActor(model,rank=2).eval();dense,receipt=fold(actor)
    path=tmp_path/'folded.pt';torch.save(export_folded(dense,sources=sources,fold_receipt=receipt,completed_updates=0),path)
    package=inspect_package(path,expected_sources=sources);cold,metadata=load_package(package)
    assert metadata['no_old_model_or_actor_or_training_reference_loaded']
    assert cold.counts()==model.counts() and metadata['actual_registered_parameters']==sum(p.numel() for p in dense.parameters())


def test_CLI_actual_tiny_neural_query_JSON_records_tokens_identity_and_CPU_precision(tmp_path):
    _,_,path,_=fixture(tmp_path)
    result=subprocess.run([sys.executable,'-m','slm_lab.own_native_query_v1','--checkpoint',str(path),
        '--expected-sha256',file_hash(path),'--prompt','hi','--raw-prompt','--max-new-tokens','1','--threads','1','--json'],
        text=True,capture_output=True,timeout=30,check=True)
    report=json.loads(result.stdout)
    assert report['actual_native_actions']==1 and len(report['actual_generated_token_ids'])==1
    assert report['inference_device']=='cpu' and report['inference_precision']=='bf16'
    assert report['completed_checkpoint_updates']==1 and report['checkpoint_sha256']==file_hash(path)
