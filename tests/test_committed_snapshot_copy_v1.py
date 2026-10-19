import hashlib

import pytest

from slm_lab.committed_snapshot_copy_v1 import copy_committed


def sha(raw):return hashlib.sha256(raw).hexdigest()


def test_atomically_replaced_live_path_keeps_exact_old_committed_bytes_without_blocking_producer(tmp_path):
    original=b'a'*(1024**2+17);source=tmp_path/'rotating';source.write_bytes(original)
    replacement=tmp_path/'new';replacement.write_bytes(b'next complete checkpoint');calls=[0]
    def race():
        calls[0]+=1
        if calls[0]==3:replacement.replace(source)
    target=tmp_path/'captured';proof=copy_committed(source,target,expected_sha256=sha(original),maximum_bytes=2*1024**2,checked=race)
    assert target.read_bytes()==original and source.read_bytes()==b'next complete checkpoint'
    assert proof['original_producer_path_not_bound_for_future_immutability']


def test_growing_journal_captures_only_exact_external_prefix_and_retains_all_later_source_bytes(tmp_path):
    prefix=b'{"step":1}\n';source=tmp_path/'journal';source.write_bytes(prefix+b'{"step":2}\n')
    def append():
        with source.open('ab') as stream:stream.write(b'{"step":3}\n')
    target=tmp_path/'prefix';copy_committed(source,target,expected_sha256=sha(prefix),prefix_bytes=len(prefix),maximum_bytes=1024,checked=append)
    assert target.read_bytes()==prefix and source.read_bytes().startswith(prefix+b'{"step":2}\n')


def test_corrupt_or_oversized_capture_never_mutates_source_or_publishes_a_success(tmp_path):
    source=tmp_path/'source';source.write_bytes(b'original');before=source.read_bytes()
    target=tmp_path/'bad'
    with pytest.raises(ValueError,match='match'):copy_committed(source,target,expected_sha256='0'*64,maximum_bytes=32)
    assert target.read_bytes()==before and source.read_bytes()==before
    with pytest.raises(ValueError,match='budget'):
        copy_committed(source,tmp_path/'too-large',expected_sha256=sha(before),maximum_bytes=2)
    assert not (tmp_path/'too-large').exists() and source.read_bytes()==before
