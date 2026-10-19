from dataclasses import asdict
import json
import os
import subprocess
import sys

import pytest
import torch

from slm_lab.corpus import file_hash
from slm_lab.inference_snapshot_archive_v1 import archive_snapshot,restore_snapshot
from slm_lab.model import ModelConfig,TinyMultimodalLM
from slm_lab.recurrent_ablation import state_digest


def files(tmp_path):
    torch.manual_seed(81);root=tmp_path/'checkpoints';root.mkdir();folder=root/'intermediate';folder.mkdir()
    anchor_folder=root/'base';anchor_folder.mkdir();scratch=tmp_path/'scratch';scratch.mkdir()
    config=ModelConfig(vocab_size=32,hidden_size=16,intermediate_size=32,num_layers=2,num_heads=4,num_kv_heads=2,
        max_text_length=32,vision_hidden_size=8,vision_intermediate_size=16,vision_layers=1,vision_heads=2,
        image_size=8,patch_size=4,pool_size=2)
    model=TinyMultimodalLM(config)
    common=dict(config=asdict(config),tokenizer_sha256='b'*64,world_size=4,data_sha256='c'*64)
    anchor=anchor_folder/'model.pt';torch.save(dict(**common,model=model.state_dict(),stage='pretrain',steps=7,planned_steps=7),anchor)
    with torch.no_grad():
        for name,p in model.named_parameters():
            if not name.startswith('vision.'):p.add_(1e-5)
    expected=dict(stage='sft',steps=2,planned_steps=8,world_size=4,data_sha256='c'*64,tokenizer_sha256='b'*64)
    source=folder/'model.pt';torch.save(dict(config=asdict(config),model=model.state_dict(),**expected),source)
    receipt=folder/'extraction.json';record=dict(inference_checkpoint=str(source),inference_checkpoint_sha256=file_hash(source),
        inference_checkpoint_bytes=source.stat().st_size,expected_metadata=expected,all_model_state_tensors_identical=True,
        original_and_restored_model_state_sha256=state_digest(model.state_dict()),source_checkpoint_modified=False,excluded_fields=['optimizer'])
    receipt.write_text(json.dumps(record));archive=folder/'model.pt.inference-v1.xor'
    options=dict(receipt=receipt,checkpoint_root=root,expected_anchor_sha256=file_hash(anchor),maximum_payload_bytes=2*1024**2,
        minimum_free_bytes=0,scratch_directory=scratch,minimum_scratch_free_bytes=0,maximum_capacity_wait_seconds=0)
    return source,anchor,archive,record,options


def test_actual_intermediate_weights_preserved_by_physical_restore_without_completed_budget_claim(tmp_path):
    source,anchor,archive,record,options=files(tmp_path);original=source.read_bytes();base=anchor.read_bytes()
    result=archive_snapshot(source,anchor,archive,**options)
    assert not source.exists() and result['original_removed']
    assert result['source_training_budget_completed'] is False
    assert result['expected_original_metadata']['steps']==2 and result['expected_original_metadata']['planned_steps']==8
    assert result['verification']['whole_file_verified_exact']
    assert result['physical_restore_verification']['independent_file_sha256']==record['inference_checkpoint_sha256']
    assert not (source.parent/'training.json').exists() and anchor.read_bytes()==base
    restored=source.parent/'restored.pt';restore_snapshot(archive,restored)
    assert restored.read_bytes()==original
    assert list(options['scratch_directory'].iterdir())==[]


@pytest.mark.parametrize('fault',['receipt','shared_link','protected_anchor','payload_budget','reserve','nonfinite','optimizer'])
def test_any_failed_precondition_or_capacity_check_keeps_original(tmp_path,fault):
    source,anchor,archive,record,options=files(tmp_path)
    if fault=='receipt':
        record['inference_checkpoint_sha256']='f'*64;options['receipt'].write_text(json.dumps(record))
    elif fault=='shared_link':
        os.link(source,source.parent/'shared.pt')
    elif fault=='protected_anchor':
        old=options['checkpoint_root']/'other.xor';old.mkdir();(old/'manifest.json').write_text(json.dumps(dict(anchor=str(source))))
    elif fault=='payload_budget':options['maximum_payload_bytes']=1
    elif fault=='reserve':options['minimum_free_bytes']=10**18
    elif fault in ['nonfinite','optimizer']:
        snapshot=torch.load(source,weights_only=True)
        if fault=='nonfinite':next(iter(snapshot['model'].values())).flatten()[0]=float('nan')
        else:snapshot['optimizer']={}
        torch.save(snapshot,source);record['inference_checkpoint_sha256']=file_hash(source);record['inference_checkpoint_bytes']=source.stat().st_size
        record['original_and_restored_model_state_sha256']=state_digest(snapshot['model']);options['receipt'].write_text(json.dumps(record))
    original=source.read_bytes()
    with pytest.raises(ValueError):archive_snapshot(source,anchor,archive,**options)
    assert source.exists() and source.read_bytes()==original


def test_actual_live_reader_prevents_archival(tmp_path):
    source,anchor,archive,record,options=files(tmp_path)
    child=subprocess.Popen([sys.executable,'-c','import sys,time; f=open(sys.argv[1],"rb"); print("ready",flush=True); time.sleep(30)',str(source)],stdout=subprocess.PIPE,text=True)
    try:
        assert child.stdout.readline().strip()=='ready'
        with pytest.raises(ValueError,match='live'):archive_snapshot(source,anchor,archive,**options)
        assert source.exists()
    finally:child.terminate();child.wait(timeout=10)


def test_corrupt_payload_and_existing_output_cannot_produce_or_overwrite_restored_file(tmp_path):
    source,anchor,archive,record,options=files(tmp_path);archive_snapshot(source,anchor,archive,**options)
    existing=tmp_path/'existing.pt';existing.write_bytes(b'keep original output')
    with pytest.raises(ValueError,match='Fresh'):restore_snapshot(archive,existing)
    assert existing.read_bytes()==b'keep original output'
    payload=archive/'payload.bin';data=bytearray(payload.read_bytes());data[0]^=1;payload.write_bytes(data)
    output=tmp_path/'must-not-exist.pt'
    with pytest.raises(ValueError,match='anchor/payload'):restore_snapshot(archive,output)
    assert not output.exists()
