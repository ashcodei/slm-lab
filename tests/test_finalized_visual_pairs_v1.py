"""Admitted-source boundary integration fixtures; no real VQA admission claim."""
from copy import deepcopy
from dataclasses import asdict,replace
import json
from pathlib import Path
import shutil

from PIL import Image
import pytest
from tokenizers import Tokenizer

from slm_lab.corpus import file_hash
from slm_lab.diverse_images import pixel_hash
from slm_lab.finalized_visual_pairs_v1 import FinalizedVisualPairs
import slm_lab.finalized_visual_pairs_v1 as source
from slm_lab.paired_visual_contrast_v1 import validate,eligible
from test_codenet_runner_v1 import small_config

ROOT=Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('size',[224,448])
def test_original_golden_pixels_full_masks_and_source_mutation_boundary(tmp_path,monkeypatch,size):
    data=tmp_path/'data/vqa-real-pairs-v1';data.mkdir(parents=True)
    token=tmp_path/'data/tokenizer-corpus/tokenizer.json';token.parent.mkdir()
    shutil.copy2(ROOT/'data/tokenizer-corpus/tokenizer.json',token)
    cfg=replace(small_config(),vocab_size=32768,max_text_length=4096,image_size=224,patch_size=16,pool_size=2)
    (tmp_path/'configs').mkdir();(tmp_path/'configs/model.json').write_text(json.dumps(asdict(cfg)))
    rows=[]
    for i,(answer,color) in enumerate([('yes',(5,45,200)),('no',(190,40,10))]):
        image=data/f'image{i}.png';Image.new('RGB',(120,100),color).save(image)
        page=data/f'fixture{i}.html';page.write_text('Explicit synthetic admitted-component fixture; not a publisher page')
        with Image.open(image) as opened:gold={size:pixel_hash(opened,size) for size in [224,448]}
        rows.append(dict(id='q'+str(i),split='train',image=str(image),task='visual_qa',prompt='Same question?\nAnswer briefly.',
            response=answer,group_id='fixture-photo:'+str(i),stored_image_sha256=file_hash(image),attribution_page=page.name,
            attribution_page_sha256=file_hash(page),model_pixel_sha256=gold[224],model_pixel_sha256_448=gold[448],
            exact_human_agreement=10,original_human_annotation=dict(answer_type='yes/no')))
    pair=data/'pairs.train.jsonl';pair.write_text(json.dumps(dict(id='fixture-pair',question_ids=['q0','q1'],canonical_question='same question?'))+'\n')
    manifest=dict(fixture=True,real_source_admission=False,tokenizer_sha256=file_hash(token),sources_sha256={},
        files_sha256={'pairs.train.jsonl':file_hash(pair)})
    (data/'manifest.json').write_text(json.dumps(manifest))
    receipt=dict(passed=True,validation_rows_not_returned_for_training=True,
        all_original_question_answer_photo_pair_rows_reconstructed=True,fixture=True,real_source_admission=False)
    # Only the independently audited admission dependency is replaced here;
    # actual encoding, golden hashes, contracts and source guards execute.
    monkeypatch.setattr(source,'admit',lambda root:(deepcopy(rows),deepcopy(receipt)))
    reader=FinalizedVisualPairs(tmp_path,Tokenizer.from_file(str(token)),cfg,size)
    example=reader.example(0)
    assert all(tuple(image.shape)==(3,size,size) for image in example['images'])
    assert validate([example],cfg,reader.contract_sha256)[0]['contrast_eligible'] and eligible(example)
    assert all(ids[-1]==2 and labels[-1]==2 for ids,labels in example['sequences'])
    reader.rows['q0']['response']='CHANGED'
    with pytest.raises(ValueError,match='source role/body'):reader.example(0)
