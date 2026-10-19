import hashlib
import json

import numpy as np
from PIL import Image, ImageOps
import pytest
import torch

from slm_lab.corpus import file_hash
from slm_lab.diverse_images import word_crops
from slm_lab.own_aligned_assistant_v1 import decode_bytes
from slm_lab.shared_word_data_v1 import (
    FORMAT, SharedWordData, array_digest, image_arrays, native_bytes, original_crop,
)


def test_complete_UTF8_native_alphabet_uses_blank256_without_losing_byte_zero_or_repeats():
    assert native_bytes('\x00')[0] == [0]
    values, required = native_bytes('aaéaa')
    assert values == [97,97,195,169,97,97] and required == 8
    assert decode_bytes([97,256,97,195,169,97,256,97]) == ('aaéaa', True)
    assert native_bytes('a'*64)[1] == 127
    with pytest.raises(ValueError, match='Complete'):
        native_bytes('a'*65)


def test_original_human_crop_reconstruction_matches_source_geometry_and_native_image_pixels():
    array = np.arange(900*1000*3, dtype=np.uint32).reshape(900,1000,3).astype(np.uint8)
    parent = Image.fromarray(array)
    metadata = dict(id='original', width=1000, height=900)
    annotation = dict(id='word', image_id='original', bbox=[120.5, 140.2, 700.5, 600.2], utf8_string='whole')
    old = word_crops(parent, 1000, 900, [annotation], 1)[0][2]
    for family, maximum in [('real_word',768),('additional_real_word',448)]:
        expected = ImageOps.contain(old.convert('RGB'), (maximum, maximum), Image.Resampling.LANCZOS)
        actual = original_crop(parent, metadata, annotation, family)
        assert actual.size == expected.size and actual.tobytes() == expected.tobytes()
        word, native = image_arrays(actual)
        assert word.shape == (3,64,448) and native.shape == (3,448,448)
        assert word.dtype == native.dtype == np.float32
        padded = ImageOps.pad(actual.convert('RGB'), (448,448), method=Image.Resampling.BICUBIC, color=(128,128,128))
        independent = torch.from_numpy((np.asarray(padded,dtype=np.float32)/127.5-1).copy()).permute(2,0,1)
        assert torch.equal(independent, torch.from_numpy(native))
    with pytest.raises(ValueError, match='identity'):
        original_crop(parent, metadata, {**annotation,'image_id':'other'}, 'real_word')


def test_actual_image_and_native_response_batches_keep_whole_labels_and_reject_held_gradients(tmp_path):
    path = tmp_path/'source.png'; Image.new('RGB',(70,30),(19,20,21)).save(path)
    with Image.open(path) as image: word, native = image_arrays(image)
    records = []
    sequences = [[1,8,9,4,5,2],[1,8,9,6,7,10,2]]
    cursor = 0
    for i,text in enumerate(['aa','é']):
        target, required = native_bytes(text); seq = sequences[i]
        records.append(dict(id=str(i), split='train', response=text, image=str(path), stored_image_sha256=file_hash(path),
            native_byte_targets=target, minimum_CTC_steps=required, word_input_pixels_sha256=array_digest(word),
            native_input_pixels_sha256=array_digest(native), native=dict(offset=cursor,length=len(seq),target_start=3)))
        cursor += len(seq)
    index = tmp_path/'train.jsonl'; index.write_text(''.join(json.dumps(r)+'\n' for r in records))
    tokens = tmp_path/'train.tokens.u16'; tokens.write_bytes(np.asarray(sum(sequences,[]),dtype='<u2').tobytes())
    manifest = dict(format=FORMAT,input_geometry=dict(word=[3,64,448],native=[3,448,448]),ctc_blank=256,ctc_frames=140,
        files_sha256={p.name:file_hash(p) for p in [index,tokens]},splits=dict(train=dict(examples=2)))
    (tmp_path/'manifest.json').write_text(json.dumps(manifest))
    data = SharedWordData(tmp_path,'train','audit'); batch = data.batch([0,1])
    assert batch['texts'] == ['aa','é']
    for i,seq in enumerate(sequences):
        assert batch['input_ids'][i,:len(seq)].tolist() == seq
        assert batch['labels'][i,:3].tolist() == [-100]*3
        assert batch['labels'][i,3:len(seq)].tolist() == seq[3:]
        assert batch['attention_mask'][i].sum() == len(seq)
    assert batch['labels'][0,-1] == -100 and not batch['attention_mask'][0,-1]
    assert batch['pixel_values'].shape == (2,3,448,448) and batch['word_pixels'].shape == (2,3,64,448)
    with pytest.raises(ValueError, match='Held'):
        SharedWordData(tmp_path/'does-not-exist','valid','optimization')
    Image.new('RGB',(70,30),(200,20,21)).save(path)
    with pytest.raises(ValueError, match='image bytes changed'):
        data.batch([0])
