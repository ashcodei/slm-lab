from pathlib import Path

import pytest

from slm_lab.corpus import file_hash
from slm_lab.own_tokenizer_package_v1 import ensure

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'data/tokenizer-corpus/tokenizer.json'


def test_missing_checkpoint_tokenizer_is_completed_once_with_exact_trained_vocabulary(tmp_path):
    digest = file_hash(SOURCE)
    first = ensure(SOURCE,tmp_path,expected_sha256=digest)
    assert not first['already_present']
    destination = tmp_path/'tokenizer.json'
    assert file_hash(destination) == digest
    metadata = destination.stat()
    second = ensure(SOURCE,tmp_path,expected_sha256=digest)
    assert second['already_present'] and destination.stat().st_mtime_ns == metadata.st_mtime_ns
    assert sorted(p.name for p in tmp_path.iterdir()) == ['tokenizer.json']


def test_wrong_source_vocabulary_or_existing_package_rejects_without_overwrite(tmp_path):
    digest = file_hash(SOURCE)
    with pytest.raises(ValueError,match='source bytes'):
        ensure(SOURCE,tmp_path,expected_sha256='0'*64)
    with pytest.raises(ValueError,match='vocabulary'):
        ensure(SOURCE,tmp_path,expected_sha256=digest,expected_vocab_size=10)
    destination = tmp_path/'tokenizer.json'
    destination.write_text('{}')
    with pytest.raises(ValueError,match='not be overwritten'):
        ensure(SOURCE,tmp_path,expected_sha256=digest)
    assert destination.read_text() == '{}'
