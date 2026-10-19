import os
import pytest
from slm_lab.corpus import file_hash
from slm_lab.codenet_checkpoint_v1 import FileBindings
from slm_lab.identical_checkpoint_copy_sharing_v1 import consolidate,metadata


def test_identical_archived_copies_keep_all_paths_bytes_and_live_canonical_inode_guard(tmp_path):
    a=tmp_path/'a';b=tmp_path/'b';content=os.urandom(1024**2+13);a.write_bytes(content);b.write_bytes(content)
    digest=file_hash(a);guard=FileBindings({str(a):digest});before=metadata(a)
    result=consolidate(a,b,expected_sha256=digest,live_bound_paths={str(a)})
    guard.check(full=True)
    assert a.read_bytes()==b.read_bytes()==content and metadata(a)==before and a.stat().st_ino==b.stat().st_ino
    assert result['original_destination_metadata']['inode']!=result['actual_shared_metadata']['inode']
    assert list(tmp_path.iterdir())==[a,b]


def test_live_destination_or_nonidentical_bytes_cannot_be_replaced(tmp_path):
    a=tmp_path/'a';b=tmp_path/'b';a.write_bytes(b'same content');b.write_bytes(b'same content');before=metadata(b)
    with pytest.raises(ValueError,match='live-bound'):consolidate(a,b,expected_sha256=file_hash(a),live_bound_paths={str(b)})
    assert metadata(b)==before
    b.write_bytes(b'other bytes!');before=metadata(b)
    with pytest.raises(ValueError,match='byte streams'):consolidate(a,b,expected_sha256=file_hash(a),live_bound_paths=set())
    assert metadata(b)==before and a.read_bytes()==b'same content' and b.read_bytes()==b'other bytes!'
