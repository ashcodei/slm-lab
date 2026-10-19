import json
from pathlib import Path

import pytest
from PIL import Image
from tokenizers import Tokenizer
import torch

from slm_lab.corpus import file_hash
from slm_lab.diverse_images import pixel_hash
from slm_lab.point_grounding_data_v1 import PointerData
from slm_lab.model import ModelConfig
from slm_lab.training import Encoder


TOKEN=Path('checkpoints/dialogue-retention-v1/systems-variety-code/tokenizer.json')


def write(path,value):
    path.write_text(json.dumps(value)+'\n')


def fixture(root):
    root.mkdir(exist_ok=True)
    path=root/'source.png';image=Image.new('RGB',(400,200),(20,90,170));image.save(path)
    tokenizer=Tokenizer.from_file(str(TOKEN))
    photos=[]
    for role,query in [('train','red flowers'),('validation','a green tree')]:
        key='original-'+role
        photo=dict(image_id=key,split=role,image=str(path),stored_dimensions=[400,200],
            stored_image_sha256=file_hash(path),model_pixel_sha256_448=pixel_hash(image,448))
        photos.append(photo)
        row=dict(id=key+':query',image_id=key,split=role,optimization_forbidden=role!='train',query=query,
            query_input_ids=[1,*tokenizer.encode(query).ids,2],image_size=448,grid_side=14,
            attention_grid_histogram=[[57,4]],source_point_count=4)
        write(root/(role+'.jsonl'),row)
    (root/'photos.jsonl').write_text(''.join(json.dumps(p)+'\n' for p in photos))
    manifest=dict(files_sha256={name:file_hash(root/name) for name in ['train.jsonl','validation.jsonl','photos.jsonl']},
        tokenizer_sha256=file_hash(TOKEN),complete_query_rows=dict(train=1,validation=1))
    write(root/'manifest.json',manifest)
    proof=root/'unit-fixture-proof.json'
    write(proof,dict(candidate_manifest_sha256=file_hash(root/'manifest.json'),
        every_whole_original_query_source_record_offset_sha_times_and_point_index_reconstructed=True,
        every_actual_photo_header_EXIF_dimensions_pixels_creator_and_role_rechecked=True,
        every_query_BPE_BOS_content_EOS_and_448px_14x14_grid_count_reconstructed=True))
    return proof,path


def test_whole_native_queries_pixels_and_targets_and_separate_train_admission(tmp_path):
    proof,path=fixture(tmp_path)
    data=PointerData(tmp_path,'validation','evaluation',proof=proof,tokenizer=TOKEN)
    batch=data.batch([0]);assert batch['input_ids'].shape[1]==len(data.rows[0]['query_input_ids'])
    assert batch['attention_mask'].all() and batch['targets'].sum()==4 and batch['targets'][0,57]==4
    assert batch['valid_cells'][0,57] and not batch['valid_cells'][0,0]
    tokenizer=Tokenizer.from_file(str(TOKEN))
    encoder=Encoder(tokenizer,ModelConfig(vocab_size=tokenizer.get_vocab_size()),tmp_path,image_size=448)
    assert torch.equal(batch['pixel_values'][0],encoder.pixels(dict(image=path.name)))
    with pytest.raises(FileNotFoundError):PointerData(tmp_path,'train','optimization',proof=proof,tokenizer=TOKEN)
    write(tmp_path/'admission.json',dict(training_admitted=True,scope='original-human-attention-distribution-supervision-only',
        independent_source_readback_sha256=file_hash(proof),candidate_manifest_sha256=file_hash(tmp_path/'manifest.json'),
        original_validation_optimization_forbidden=True))
    train=PointerData(tmp_path,'train','optimization',proof=proof,tokenizer=TOKEN)
    assert train.batch([0])['pixel_values'].shape==(1,3,448,448)
    for purpose in ['optimization','prior']:
        with pytest.raises(ValueError,match='Validation never supplies gradients'):PointerData(tmp_path,'validation',purpose)


def test_original_file_and_actual_model_pixel_mutation_rejects(tmp_path):
    proof,path=fixture(tmp_path)
    data=PointerData(tmp_path,'validation','evaluation',proof=proof,tokenizer=TOKEN)
    Image.new('RGB',(400,200),(170,90,20)).save(path)
    with pytest.raises(ValueError,match='original photo bytes changed'):data.batch([0])
    with (tmp_path/'validation.jsonl').open('a') as out:out.write('{}\n')
    with pytest.raises(ValueError,match='manifest bytes changed'):
        PointerData(tmp_path,'validation','evaluation',proof=proof,tokenizer=TOKEN)
