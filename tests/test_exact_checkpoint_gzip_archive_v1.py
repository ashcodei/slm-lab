"""Actual project filesystem, opaque bytes and separate physical restoration."""
import json
import os
from pathlib import Path
import tempfile

import pytest

from slm_lab.exact_checkpoint_gzip_archive_v1 import archive, restore, sha

ROOT = Path(__file__).resolve().parents[1]


def test_actual_exact_archive_physically_restores_before_retiring_opaque_checkpoint_bytes():
    with tempfile.TemporaryDirectory(prefix='exact-archive-test-', dir=ROOT / 'reports') as work, tempfile.TemporaryDirectory(prefix='exact-archive-scratch-', dir='/tmp') as scratch:
        work, scratch = Path(work), Path(scratch)
        source = work / 'opaque-original.pt'
        content = (b'opaque weights optimizer moments private RNG and source cursor\n' * 16000) + os.urandom(8192)
        source.write_bytes(content); digest = sha(source)
        row = archive(source, work / 'archive', scratch, expected_sha256=digest, maximum_payload_bytes=2*1024**2,
            minimum_free_bytes=0, minimum_reclaimed_bytes=1, guard=lambda: None, retire_original=True)
        assert not source.exists() and row['original_retired'] and row['physical_restore_checked']
        assert row['physical_restore']['independent_physical_file_rehash_passed']
        restored = scratch / 'separate-reader-restored.pt'
        check = restore(work / 'archive', restored)
        assert check['actual_restored_sha256'] == digest and restored.read_bytes() == content
        assert row['tensor_bytes_and_optimizer_RNG_metadata_unchanged']


def test_corrupt_archive_and_existing_destination_reject_while_original_checkpoint_is_retained():
    with tempfile.TemporaryDirectory(prefix='exact-archive-negative-', dir=ROOT / 'reports') as work, tempfile.TemporaryDirectory(prefix='exact-archive-scratch-', dir='/tmp') as scratch:
        work, scratch = Path(work), Path(scratch)
        source = work / 'original.pt'; source.write_bytes(b'checkpoint' * 20000); digest = sha(source)
        row = archive(source, work / 'archive', scratch, expected_sha256=digest, maximum_payload_bytes=1024**2,
            minimum_free_bytes=0, minimum_reclaimed_bytes=1, guard=lambda: None)
        assert sha(source) == digest and not row['original_retired']
        target = scratch / 'existing.pt'; target.write_bytes(b'keep this existing file')
        with pytest.raises(ValueError): restore(work / 'archive', target)
        assert target.read_bytes() == b'keep this existing file'
        payload = work / 'archive/payload.gz'; data = bytearray(payload.read_bytes()); data[len(data)//2] ^= 1; payload.write_bytes(data)
        with pytest.raises(ValueError): restore(work / 'archive', scratch / 'must-not-exist.pt')
        assert not (scratch / 'must-not-exist.pt').exists() and sha(source) == digest


def test_payload_overflow_or_shared_scratch_never_retires_original_bytes():
    with tempfile.TemporaryDirectory(prefix='exact-archive-budget-', dir=ROOT / 'reports') as work, tempfile.TemporaryDirectory(prefix='exact-archive-scratch-', dir='/tmp') as scratch:
        work, scratch = Path(work), Path(scratch)
        source = work / 'original.pt'; source.write_bytes(os.urandom(8192)); digest = sha(source)
        with pytest.raises(ValueError, match='reserve'):
            archive(source, work / 'overflow', scratch, expected_sha256=digest, maximum_payload_bytes=1,
                minimum_free_bytes=0, minimum_reclaimed_bytes=1, guard=lambda: None, retire_original=True)
        with pytest.raises(ValueError, match='independent scratch'):
            archive(source, work / 'same-filesystem', work, expected_sha256=digest, maximum_payload_bytes=1024**2,
                minimum_free_bytes=0, minimum_reclaimed_bytes=1, guard=lambda: None, retire_original=True)
        assert sha(source) == digest
