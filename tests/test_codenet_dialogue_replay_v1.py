import hashlib
import json
import os
from types import SimpleNamespace

import numpy as np
import pytest

from slm_lab.codenet_dialogue_replay_v1 import FAMILY,SOURCE,DialogueReplayCorpus,dialogue_role
from slm_lab.codenet_replay_v1 import FullReplayEncoder,INDEX_DTYPE
from slm_lab.corpus import file_hash


class Tokenizer:
    def token_to_id(self,s):return {'<pad>':0,'<bos>':1,'<eos>':2}[s]
    def get_vocab_size(self):return 32768
    def encode(self,s):return SimpleNamespace(ids=list(map(ord,s)))


def fixture(tmp_path):
    folder=tmp_path/'component';folder.mkdir();source=tmp_path/'train.jsonl';source_manifest=tmp_path/'source-manifest.json'
    encoder=FullReplayEncoder(Tokenizer(),SimpleNamespace(max_text_length=128,image_size=224,vocab_size=32768,image_token_count=lambda _:0),tmp_path)
    rows=[dict(id=f'oasst2-message{i}',group_id=f'oasst2:tree{i}',source_message_id=f'message{i}',source_message_tree_id=f'tree{i}',
        split='train',source=SOURCE,synthetic=False,human_quality={'count':2,'value':.8},
        prompt=f'User: Earlier {i}\nAssistant: Reply\nUser: Continue\nAssistant:',response=f'Answer {i}',
        language='en',source_text_sha256=hashlib.sha256(f'Answer {i}'.encode()).hexdigest(),extra={'preserved':True}) for i in range(2)]
    entries=[]
    with source.open('wb') as f:
        for row in rows:
            offset=f.tell();raw=(json.dumps(row)+'\n').encode();f.write(raw);ids,labels=encoder.sequence(row,'response')
            entries.append((offset,len(raw),0,len(ids),sum(v!=-100 for v in labels),
                np.void(hashlib.sha256(raw).digest()),np.void(hashlib.sha256(np.asarray(ids,dtype='<u2').tobytes()).digest())))
    source_manifest.write_text(json.dumps(dict(splits={'train':{'examples':2}},files_sha256={'train.jsonl':file_hash(source)})))
    np.save(folder/'index.npy',np.asarray(entries,dtype=INDEX_DTYPE),allow_pickle=False)
    (folder/'records.jsonl').write_text('')
    (folder/'source-admission.json').write_text(json.dumps(dict(passed=True,validation_rows_not_returned_for_training=True,
        original_training_examples=2,manifest_sha256=file_hash(source_manifest))))
    manifest=dict(format='slm-complete-reviewed-dialogue-replay-v1',family_names=[FAMILY],complete_fit_rows=2,
        original_training_source=str(source),original_training_source_sha256=file_hash(source),
        original_source_manifest=str(source_manifest),original_source_manifest_sha256=file_hash(source_manifest),
        artifact_files_sha256={n:file_hash(folder/n) for n in ['index.npy','records.jsonl','source-admission.json']})
    (folder/'manifest.json').write_text(json.dumps(manifest))
    return folder,source,rows,encoder


def test_whole_multiturn_text_metadata_and_explicit_family_survive_actual_reader(tmp_path):
    folder,source,rows,encoder=fixture(tmp_path);replay=DialogueReplayCorpus(folder)
    assert replay.families==[FAMILY] and replay.size(FAMILY)==2
    actual=replay.row(FAMILY,1)
    assert {k:actual[k] for k in rows[1]}==rows[1]
    ids,labels=encoder.sequence(actual,'response')
    assert ids==[1]+list(map(ord,rows[1]['prompt']+rows[1]['response']))+[2]
    assert labels==[-100]*(1+len(rows[1]['prompt']))+list(map(ord,rows[1]['response']))+[2]
    with pytest.raises(KeyError):replay.row('code:en',0)


@pytest.mark.parametrize('field,value',[('split','valid'),('source_split','test'),('synthetic',True),('human_quality',{'count':0,'value':1.}),('response','Changed')])
def test_held_or_unreviewed_or_changed_original_answer_is_rejected(tmp_path,field,value):
    folder,source,rows,encoder=fixture(tmp_path)
    with pytest.raises(ValueError,match='training-only'):dialogue_role({**rows[0],field:value})


def test_changed_source_with_original_timestamp_is_rejected(tmp_path):
    folder,source,rows,encoder=fixture(tmp_path);reader=DialogueReplayCorpus(folder);s=source.stat()
    raw=source.read_bytes().replace(b'Earlier 0',b'Changed 0');assert len(raw)==s.st_size
    source.write_bytes(raw);os.utime(source,ns=(s.st_atime_ns,s.st_mtime_ns))
    with pytest.raises(ValueError,match='row bytes differ'):reader.row(FAMILY,0)


def test_resealed_wrong_target_count_reaches_actual_encoder_rejection(tmp_path):
    folder,source,rows,encoder=fixture(tmp_path);index=np.load(folder/'index.npy');index[0]['target_tokens']+=1
    np.save(folder/'index.npy',index,allow_pickle=False)
    m=json.loads((folder/'manifest.json').read_text());m['artifact_files_sha256']['index.npy']=file_hash(folder/'index.npy')
    (folder/'manifest.json').write_text(json.dumps(m));row=DialogueReplayCorpus(folder).row(FAMILY,0)
    with pytest.raises(ValueError,match='text/mask differs'):encoder.sequence(row,'response')
