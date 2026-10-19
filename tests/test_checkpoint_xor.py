import hashlib
import json
from pathlib import Path

import pytest
import torch
from slm_lab.checkpoint_xor import archive_checkpoint,restore


def files(tmp_path):
    source=tmp_path/'source';target=tmp_path/'target';source.mkdir();target.mkdir()
    values=torch.arange(4096,dtype=torch.float32)
    torch.save({'model':{'weight':values,'other':torch.zeros(32)},'phase':'anchor'},source/'model.pt')
    torch.save({'model':{'weight':values+.001,'other':torch.zeros(32)},'phase':'completed'},target/'model.pt')
    (target/'training.json').write_text(json.dumps({'status':'training run completed','steps':2,'planned_steps':2}))
    return source/'model.pt',target/'model.pt'


def test_lossless_xor_archive_restores_original_torch_file_and_retains_exact_anchor(tmp_path):
    anchor,checkpoint=files(tmp_path);original=checkpoint.read_bytes();source=anchor.read_bytes()
    archive=tmp_path/'archive';report=archive_checkpoint(checkpoint,anchor,archive,True)
    assert not checkpoint.exists() and anchor.read_bytes()==source
    output=tmp_path/'restored.pt';verified=restore(archive,output)
    assert output.read_bytes()==original and verified['restored_sha256']==hashlib.sha256(original).hexdigest()
    assert report['payload_bytes']<len(original)


@pytest.mark.parametrize('changed',['anchor','payload'])
def test_archive_refuses_changed_anchor_or_payload_before_reconstructing(changed,tmp_path):
    anchor,checkpoint=files(tmp_path);archive=tmp_path/'archive';archive_checkpoint(checkpoint,anchor,archive)
    path=anchor if changed=='anchor' else archive/'payload.bin';data=path.read_bytes();path.write_bytes(data[:-1]+bytes([data[-1]^1]))
    with pytest.raises(ValueError):restore(archive)
    assert checkpoint.exists()


def test_incomplete_training_cannot_be_removed_for_storage(tmp_path):
    anchor,checkpoint=files(tmp_path)
    checkpoint.with_name('training.json').write_text(json.dumps({'status':'training','steps':1,'planned_steps':2}))
    with pytest.raises(ValueError):archive_checkpoint(checkpoint,anchor,tmp_path/'archive',True)
    assert checkpoint.exists()


def test_large_highly_compressible_tensor_segments_are_streamed_and_restored_exactly(tmp_path):
    anchor,checkpoint=files(tmp_path)
    torch.save({'model':{'weight':torch.zeros(3_000_000)}},anchor)
    torch.save({'model':{'weight':torch.ones(3_000_000)*.001}},checkpoint)
    original=checkpoint.read_bytes();archive=tmp_path/'archive';archive_checkpoint(checkpoint,anchor,archive,True)
    output=tmp_path/'restored.pt';restore(archive,output)
    assert output.read_bytes()==original
