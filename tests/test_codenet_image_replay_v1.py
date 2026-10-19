import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from slm_lab.codenet_image_replay_v1 import FAMILIES,TASKS,ARTIFACTS,ImageReplayCorpus,image_role,canonical_image
from slm_lab.codenet_replay_v1 import INDEX_DTYPE,FullReplayEncoder
from slm_lab.corpus import file_hash


class Tokenizer:
    def token_to_id(self,s):return {'<pad>':0,'<bos>':1,'<eos>':2}[s]
    def get_vocab_size(self):return 32768
    def encode(self,s):return SimpleNamespace(ids=list(map(ord,s)))


def fixture(tmp_path):
    folder=tmp_path/'replay';folder.mkdir();encoder=FullReplayEncoder(Tokenizer(),
        SimpleNamespace(max_text_length=128,image_size=448,vocab_size=32768,image_token_count=lambda _:196),tmp_path,448)
    entries=[];sources=[];images=[];rows=[]
    for i,name in enumerate(FAMILIES):
        source=tmp_path/f'train-{i}.jsonl';image=tmp_path/f'photo-{i}.png';Image.new('RGB',(13,7),(i*30,22,77)).save(image)
        row=dict(id=f'original-{i}',group_id=f'parent-{i}',image=image.name,stored_image_sha256=file_hash(image),
            model_pixel_sha256_448='',prompt='Original request?',response='Original answer',task=TASKS[name],
            split='train',source_subset='train',own_predicted_context_supplied=True,feedback_source='human',
            visual_source='additional_everyday_photos' if i==0 else 'commons',original_metadata={'preserved':True})
        tensor=encoder.pixels(row);rgb=np.rint((tensor.permute(1,2,0).numpy()+1)*127.5).clip(0,255).astype(np.uint8)
        row['model_pixel_sha256_448']=hashlib.sha256(rgb.tobytes()).hexdigest();raw=(json.dumps(row)+'\n').encode();source.write_bytes(raw)
        ids,labels=encoder.sequence(row,'response');entries.append((0,len(raw),i,len(ids),sum(v!=-100 for v in labels),
            np.void(hashlib.sha256(raw).digest()),np.void(hashlib.sha256(np.asarray(ids,dtype='<u2').tobytes()).digest())))
        original=tmp_path/f'original-manifest-{i}.json';original.write_text(json.dumps({'files_sha256':{'train.jsonl':file_hash(source)}}))
        sources.append(dict(family=name,path=str(source),sha256=file_hash(source),original_manifest=str(original),
            original_manifest_sha256=file_hash(original),original_file_key='train.jsonl',original_file_path=str(source),
            original_file_sha256=file_hash(source),selected_rows=1,review_rows=0))
        images.append(dict(path=str(image),sha256=file_hash(image),model_pixel_sha256_448=row['model_pixel_sha256_448']));rows.append(row)
    np.save(folder/'index.npy',np.asarray(entries,dtype=INDEX_DTYPE),allow_pickle=False)
    (folder/'image-files.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in images))
    for name in ['records.jsonl','review.jsonl','own-context-original-rows.jsonl']:(folder/name).write_text('')
    (folder/'source-admission.json').write_text(json.dumps(dict(passed=True,held_roles_excluded=True,
        source_manifest_sha256={s['family']:s['original_manifest_sha256'] for s in sources})))
    manifest=dict(format='slm-complete-later-image-replay-v1',project_root=str(tmp_path),family_names=list(FAMILIES),
        sources=sources,complete_fit_rows=6,family_counts={n:1 for n in FAMILIES},unique_original_image_files=6,
        artifact_files_sha256={n:file_hash(folder/n) for n in ARTIFACTS})
    (folder/'manifest.json').write_text(json.dumps(manifest));return folder,rows,encoder


def test_every_original_source_family_whole_mask_and_actual_pixels_survive_reader(tmp_path):
    folder,rows,encoder=fixture(tmp_path);reader=ImageReplayCorpus(folder)
    for name,row in zip(FAMILIES,rows):
        actual=reader.row(name,0)
        assert {k:actual[k] for k in row if k!='image'}=={k:v for k,v in row.items() if k!='image'}
        assert Path(actual['image'])==tmp_path/row['image']
        assert encoder.sequence(actual,'response')==encoder.sequence(row,'response')
        assert np.array_equal(encoder.pixels(actual).numpy(),encoder.pixels(row).numpy())
        assert reader.size(name)==1


def test_changed_image_with_unchanged_size_and_timestamp_is_rejected(tmp_path):
    folder,rows,encoder=fixture(tmp_path);reader=ImageReplayCorpus(folder);image=tmp_path/rows[0]['image'];s=image.stat()
    raw=bytearray(image.read_bytes());raw[-5]^=1;image.write_bytes(raw);os.utime(image,ns=(s.st_atime_ns,s.st_mtime_ns))
    with pytest.raises(ValueError,match='image bytes/pixel identity changed'):reader.row(FAMILIES[0],0)


def test_source_row_same_size_rewrite_with_unchanged_timestamp_is_rejected(tmp_path):
    folder,rows,encoder=fixture(tmp_path);reader=ImageReplayCorpus(folder);source=tmp_path/'train-0.jsonl';s=source.stat()
    raw=source.read_bytes().replace(b'Original request',b'Modified request');assert len(raw)==s.st_size
    source.write_bytes(raw);os.utime(source,ns=(s.st_atime_ns,s.st_mtime_ns))
    with pytest.raises(ValueError,match='row bytes differ'):reader.row(FAMILIES[0],0)


def test_resealed_wrong_target_span_is_rejected_by_actual_encoder(tmp_path):
    folder,rows,encoder=fixture(tmp_path);index=np.load(folder/'index.npy');index[0]['target_tokens']+=1
    np.save(folder/'index.npy',index,allow_pickle=False);m=json.loads((folder/'manifest.json').read_text())
    m['artifact_files_sha256']['index.npy']=file_hash(folder/'index.npy');(folder/'manifest.json').write_text(json.dumps(m))
    with pytest.raises(ValueError,match='text/mask differs'):encoder.sequence(ImageReplayCorpus(folder).row(FAMILIES[0],0),'response')


@pytest.mark.parametrize('change',[{'split':'valid'},{'source_split':'test'},{'source_subset':'validation'}])
def test_explicit_held_source_roles_never_become_training_images(tmp_path,change):
    folder,rows,encoder=fixture(tmp_path)
    with pytest.raises(ValueError):image_role({**rows[0],**change},FAMILIES[0])


def test_image_path_outside_project_is_rejected(tmp_path):
    image=tmp_path/'outside.png';image.write_bytes(b'outside');allowed=tmp_path/'allowed';allowed.mkdir()
    with pytest.raises(ValueError,match='escapes'):canonical_image({'image':str(image)},allowed/'train.jsonl',allowed)
