import json

import numpy as np
import pytest
import torch

from slm_lab.admitted_education_v1 import ADMISSION_FORMAT,EducationWindows
from slm_lab.corpus import file_hash
from slm_lab.open_education_pack_v1 import FORMAT


def corpus(tmp_path):
    shards={}
    for role,seq in [('train',[1,8,9,2,1,7,5,2]),('valid',[1,6,2])]:
        pieces=[seq[:4],seq[4:]] if len(seq)>4 else [seq]
        shards[role]=[]
        for i,piece in enumerate(pieces):
            path=tmp_path/f'{role}-{i}.bin';path.write_bytes(np.asarray(piece,dtype='<u2').tobytes())
            shards[role].append(dict(path=path.name,tokens=len(piece),sha256=file_hash(path)))
    for name in ['original-pages.jsonl','independent-raw-source-and-token-readback.json','cross-corpus-hash-index-review.json','publisher-attribution-review.json']:
        (tmp_path/name).write_text('{}\n')
    manifest=dict(format=FORMAT,tokenizer_sha256='a'*64,splits={r:dict(shards=v) for r,v in shards.items()},training_admitted=False)
    (tmp_path/'corpus.json').write_text(json.dumps(manifest))
    admission=dict(format=ADMISSION_FORMAT,training_admitted=True,admitted_role='train',held_original_gradients_forbidden=True)
    for key,name in {'corpus_sha256':'corpus.json','original_page_index_sha256':'original-pages.jsonl',
        'full_raw_token_readback_sha256':'independent-raw-source-and-token-readback.json',
        'retained_index_overlap_review_sha256':'cross-corpus-hash-index-review.json',
        'publisher_attribution_review_sha256':'publisher-attribution-review.json'}.items():admission[key]=file_hash(tmp_path/name)
    (tmp_path/'admission.json').write_text(json.dumps(admission));return tmp_path


def test_every_original_next_token_once_across_shards_and_unrepeated_padded_tail(tmp_path):
    source=EducationWindows(corpus(tmp_path),'a'*64,4,'train','optimization')
    assert len(source)==3
    raw=[1,8,9,2,1,7,5,2];targets=[];positions=[]
    for p in range(len(source)):
        row=source.window(p);targets.extend(v for v in row['labels'][1:] if v!=-100)
        positions.extend(range(row['original_first_target_position'],row['original_last_target_position_exclusive']))
    assert targets==raw[1:] and positions==list(range(1,len(raw)))
    last=source.window(2)
    assert last['input_ids']==[5,2,0,0] and last['labels']==[5,2,-100,-100] and last['attention_mask']==[True,True,False,False]
    batch=source.batch([0,2]);assert batch['attention_mask'].dtype is torch.bool
    assert int((batch['labels'][:,1:]!=-100).sum())==4
    assert source.source_counts([0,1,2])==dict(examples=3,complete_context_tokens=10,response_EOS_targets=7)


def test_held_books_reject_training_and_capacity_before_accessing_any_source(tmp_path):
    for purpose in ('optimization','capacity'):
        with pytest.raises(ValueError,match='Held'):EducationWindows(tmp_path/'missing','a'*64,4,'valid',purpose)
    root=corpus(tmp_path);held=EducationWindows(root,'a'*64,4,'valid','evaluation')
    assert len(held)==1 and held.source_counts([0])['response_EOS_targets']==2


def test_admission_metadata_proof_or_token_corruption_rejects_even_with_frozen_preparer_pending_flag(tmp_path):
    root=corpus(tmp_path);EducationWindows(root,'a'*64,4,'train','optimization')
    # A later separate admission is authoritative; the immutable preparer's
    # historical pending flag must not be rewritten to pretend it was admitted.
    assert json.loads((root/'corpus.json').read_text())['training_admitted'] is False
    (root/'publisher-attribution-review.json').write_text('{"changed":true}')
    with pytest.raises(ValueError,match='admission'):EducationWindows(root,'a'*64,4,'train','optimization')
    a=json.loads((root/'admission.json').read_text());a['publisher_attribution_review_sha256']=file_hash(root/'publisher-attribution-review.json');(root/'admission.json').write_text(json.dumps(a))
    (root/'train-1.bin').write_bytes(np.array([1,7,4,2],dtype='<u2').tobytes())
    with pytest.raises(ValueError,match='token shard'):EducationWindows(root,'a'*64,4,'train','optimization')
