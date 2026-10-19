import os

import pytest

from slm_lab.checkpoint_storage_sharing_v1 import share_identical,compare_share_range
from slm_lab.codenet_checkpoint_v1 import FileBindings
from slm_lab.corpus import file_hash


def test_real_filesystem_sharing_preserves_original_bytes_inodes_mtimes_guards_and_copy_on_write(tmp_path):
    # Use this workspace filesystem, not pytest's separate root overlay.
    from pathlib import Path
    import tempfile
    root=Path(__file__).resolve().parents[1]/'reports'
    with tempfile.TemporaryDirectory(prefix='storage-sharing-control-',dir=root) as temporary:
        folder=Path(temporary);a=folder/'source.bin';b=folder/'destination.bin'
        content=os.urandom(1024**2+37);a.write_bytes(content);b.write_bytes(content);digest=file_hash(a)
        guard=FileBindings({str(a):digest,str(b):digest})
        result=share_identical(a,b,expected_sha256=digest)
        assert result['logical_identical_bytes_compared_and_shared']==1024**2
        assert result['unchanged_unshared_final_partial_block_bytes']==37
        guard.check(full=True)
        with b.open('r+b') as stream:stream.write(b'changed only this test destination')
        assert file_hash(a)==digest and a.read_bytes()==content and file_hash(b)!=digest


def test_different_ranges_are_rejected_by_kernel_without_replacing_destination(tmp_path):
    from pathlib import Path
    import tempfile
    root=Path(__file__).resolve().parents[1]/'reports'
    with tempfile.TemporaryDirectory(prefix='storage-sharing-different-control-',dir=root) as temporary:
        folder=Path(temporary);a=folder/'a';b=folder/'b';a.write_bytes(b'a'*4096);b.write_bytes(b'b'*4096)
        before=[file_hash(a),file_hash(b)]
        with a.open('rb') as source,b.open('r+b') as destination:
            result=compare_share_range(source.fileno(),destination.fileno(),0,4096)
        assert result['kernel_status']==1 and result['actual_logical_bytes_compared_and_shared']==0
        assert [file_hash(a),file_hash(b)]==before
        with pytest.raises(ValueError,match='entire original'):share_identical(a,b,expected_sha256=before[0])
