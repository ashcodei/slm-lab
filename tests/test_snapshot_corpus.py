import json
from pathlib import Path
import pytest

from slm_lab.corpus import file_hash
from slm_lab.snapshot_corpus import freeze


def source(tmp_path):
    root = tmp_path/'source'; root.mkdir(); (root/'parts').mkdir()
    (root/'collection-config.json').write_text(json.dumps({'date':'20261001', 'extraction_version':'wiki-clean-v2',
                                                          'tokenizer_sha256':'tokenizer', 'questions_sha256':'questions'}))
    return root


def commit(root, name):
    part = root/'parts'/name; part.mkdir(); (part/'documents.jsonl').write_text('{"id":"article"}\n')
    splits = {}
    for split, token in [('train',1), ('valid',2)]:
        shard = part/(split+'.bin'); shard.write_bytes(bytes([token,0])*4096)
        splits[split] = {'tokens':4096, 'shards':[{'path':shard.name,'tokens':4096,'sha256':file_hash(shard)}]}
    manifest = {'dtype':'<u2','tokenizer_sha256':'tokenizer','extraction_version':'wiki-clean-v2',
                'source':{'snapshot':{'date':'20261001'},'official_sha1':'a'*40}, 'documents':1,
                'document_provenance_sha256':file_hash(part/'documents.jsonl'),
                'domain_tokens':{'math':{'train':4096,'valid':4096}},'splits':splits}
    (part/'corpus.json').write_text(json.dumps(manifest)); return part


def test_snapshot_ignores_uncommitted_parts_and_stays_frozen_as_collection_grows(tmp_path):
    root = source(tmp_path); part = commit(root,'first')
    pending = root/'parts'/'unfinished'; pending.mkdir(); (pending/'train.bin').write_bytes(b'incomplete')
    output = tmp_path/'frozen'; first = freeze(root,output)
    assert len(first['parts']) == 1 and first['total_tokens'] == 8192
    for split, token in [('train',1), ('valid',2)]:
        path = output/first['splits'][split]['shards'][0]['path']
        assert path.is_symlink() and path.read_bytes() == bytes([token,0])*4096
        assert path.resolve().parent == part
    commit(root,'later')
    assert freeze(root,output) == first


def test_snapshot_rejects_changed_committed_tokens(tmp_path):
    root = source(tmp_path); part = commit(root,'first'); output = tmp_path/'frozen'
    freeze(root,output); (part/'train.bin').write_bytes(b'\x03\x00'*4096)
    with pytest.raises(ValueError,match='Frozen token shard changed'): freeze(root,output)


def test_snapshot_rejects_provenance_changes(tmp_path):
    root = source(tmp_path); part = commit(root,'first')
    (part/'documents.jsonl').write_text('changed')
    with pytest.raises(ValueError,match='provenance changed'): freeze(root,tmp_path/'frozen')
