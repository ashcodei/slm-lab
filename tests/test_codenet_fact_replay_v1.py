import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from slm_lab.codenet_fact_replay_v1 import FAMILY,FactReplayCorpus,CombinedReplay
from slm_lab.codenet_replay_v1 import FullReplayEncoder,INDEX_DTYPE
from slm_lab.corpus import file_hash


class Tokenizer:
    def token_to_id(self,s):return {'<pad>':0,'<bos>':1,'<eos>':2}[s]
    def get_vocab_size(self):return 32768
    def encode(self,s):return SimpleNamespace(ids=list(map(ord,s)))


def fixture(tmp_path):
    folder=tmp_path/'component';folder.mkdir();source=tmp_path/'original-train.jsonl'
    config=SimpleNamespace(max_text_length=128,image_size=224,vocab_size=32768,image_token_count=lambda _:0)
    encoder=FullReplayEncoder(Tokenizer(),config,tmp_path)
    rows=[dict(id=f'original-{i}',group_id=f'question-{i}',split='train',source_split='train',
        task='nq_open_question',language='English',prompt=f'Question {i}?\nAnswer briefly.',
        response=f'gold{i}',reference_answers=[f'gold{i}','a preserved alias'],extra_metadata={'original':True}) for i in range(2)]
    entries=[]
    with source.open('wb') as f:
        for row in rows:
            offset=f.tell();raw=(json.dumps(row)+'\n').encode();f.write(raw);ids,labels=encoder.sequence(row,'response')
            entries.append((offset,len(raw),0,len(ids),sum(v!=-100 for v in labels),
                np.void(hashlib.sha256(raw).digest()),np.void(hashlib.sha256(np.asarray(ids,dtype='<u2').tobytes()).digest())))
    np.save(folder/'index.npy',np.asarray(entries,dtype=INDEX_DTYPE),allow_pickle=False)
    (folder/'records.jsonl').write_text('')
    (folder/'source-admission.json').write_text(json.dumps(dict(passed=True,validation_rows_not_returned_for_training=True,
        original_training_questions=2,manifest_sha256='a'*64)))
    manifest=dict(format='slm-complete-human-fact-replay-v1',family_names=[FAMILY],complete_fit_rows=2,
        original_training_source=str(source),original_training_source_sha256=file_hash(source),
        original_human_fact_manifest_sha256='a'*64,
        artifact_files_sha256={n:file_hash(folder/n) for n in ['index.npy','records.jsonl','source-admission.json']})
    (folder/'manifest.json').write_text(json.dumps(manifest))
    return folder,source,rows,encoder


def test_aliases_metadata_whole_target_and_source_family_are_preserved(tmp_path):
    folder,source,rows,encoder=fixture(tmp_path);replay=FactReplayCorpus(folder)
    assert replay.families==[FAMILY] and replay.size(FAMILY)==2
    actual=replay.row(FAMILY,0)
    assert {k:actual[k] for k in rows[0]}==rows[0]
    assert encoder.sequence(actual,'response')==encoder.sequence(rows[0],'response')
    assert actual['reference_answers']==['gold0','a preserved alias']
    with pytest.raises(KeyError):replay.row('code:English',0)


def test_same_length_rewrite_with_original_mtime_still_rejects_changed_source(tmp_path):
    folder,source,rows,encoder=fixture(tmp_path);replay=FactReplayCorpus(folder);s=source.stat()
    raw=source.read_bytes().replace(b'Question 0',b'Changed! 0');assert len(raw)==s.st_size
    source.write_bytes(raw);os.utime(source,ns=(s.st_atime_ns,s.st_mtime_ns))
    with pytest.raises(ValueError,match='row bytes differ'):replay.row(FAMILY,0)


def test_held_source_role_is_rejected_even_if_artifact_checksums_are_resealed(tmp_path):
    folder,source,rows,encoder=fixture(tmp_path)
    raw=source.read_bytes().replace(b'"split": "train"',b'"split": "valid"',1);source.write_bytes(raw)
    index=np.load(folder/'index.npy',allow_pickle=False);first=raw[:int(index[0]['bytes'])]
    index[0]['row_sha256']=np.void(hashlib.sha256(first).digest());np.save(folder/'index.npy',index,allow_pickle=False)
    manifest=json.loads((folder/'manifest.json').read_text());manifest['original_training_source_sha256']=file_hash(source)
    manifest['artifact_files_sha256']['index.npy']=file_hash(folder/'index.npy');(folder/'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='training-only'):FactReplayCorpus(folder).row(FAMILY,0)


def test_wrong_contiguous_target_count_cannot_bypass_actual_encoder_mask_check(tmp_path):
    folder,source,rows,encoder=fixture(tmp_path);index=np.load(folder/'index.npy',allow_pickle=False)
    index[0]['target_tokens']+=1;np.save(folder/'index.npy',index,allow_pickle=False)
    manifest=json.loads((folder/'manifest.json').read_text());manifest['artifact_files_sha256']['index.npy']=file_hash(folder/'index.npy')
    (folder/'manifest.json').write_text(json.dumps(manifest))
    row=FactReplayCorpus(folder).row(FAMILY,0)
    with pytest.raises(ValueError,match='text/mask differs'):encoder.sequence(row,'response')


def test_combined_reader_routes_each_complete_family_and_rejects_collisions(tmp_path):
    folder,source,rows,encoder=fixture(tmp_path);facts=FactReplayCorpus(folder)
    class Prior:
        families=['math']
        def size(self,name):assert name=='math';return 3
        def row(self,name,position):return {'original_family':name,'position':position}
    combined=CombinedReplay(Prior(),facts)
    assert combined.families==['math',FAMILY] and combined.size(FAMILY)==2
    assert combined.row('math',2)=={'original_family':'math','position':2}
    assert combined.row(FAMILY,1)['id']==rows[1]['id']
    with pytest.raises(ValueError,match='distinct source families'):CombinedReplay(facts,facts)
