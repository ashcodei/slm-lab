"""Actual copied source bytes, real SHA reads, and same-metadata corruption."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path

import pytest

from slm_lab import corpus
from slm_lab.closed_source_recovery_guard_v1 import SourceOnlyRecoveryGuard

ROOT = Path(__file__).resolve().parents[1]


def actual_files(tmp_path):
    files = {}
    for name in ['independent_token_suffix_readback_v1.py', 'independent_suffix_reader_recovery_v1.py']:
        p = tmp_path / name
        p.write_bytes((ROOT / 'src/slm_lab' / name).read_bytes())
        files[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
    return files


def test_actual_hash_reads_are_complete_at_both_boundaries_without_repeated_payload_reads(tmp_path, monkeypatch):
    files = actual_files(tmp_path)
    original = corpus.file_hash
    reads = []
    def observed(path):
        reads.append(str(path)); return original(path)
    monkeypatch.setattr(corpus, 'file_hash', observed)
    guard = SourceOnlyRecoveryGuard(files, lambda: None)
    assert set(reads) == set(files) and len(reads) == len(files)
    reads.clear()
    for _ in range(129): guard.check()
    assert not reads
    guard.verify_final()
    assert set(reads) == set(files) and len(reads) == len(files)
    assert guard.verified
    with pytest.raises(ValueError): guard.check()
    (tmp_path / 'actual-source-only-recovery-boundaries.json').write_text(json.dumps(dict(
        actual_source_files=list(files), real_complete_SHA_reads_per_boundary=len(files),
        intermediate_resource_checks=129, intermediate_payload_hash_reads=0,
        full_final_hashes_rechecked_before_publication=True, neural_forwards=0, optimizer_updates=0), indent=2))


def test_actual_same_size_same_timestamp_source_change_rejects_before_final_publication(tmp_path):
    files = actual_files(tmp_path)
    guard = SourceOnlyRecoveryGuard(files, lambda: None)
    guard.check()
    p = Path(next(iter(files)))
    before = p.stat(); raw = p.read_bytes()
    p.write_bytes(bytes([raw[0] ^ 1]) + raw[1:])
    os.utime(p, ns=(before.st_atime_ns, before.st_mtime_ns))
    for _ in range(3): guard.check()
    assert not guard.verified
    with pytest.raises(ValueError, match='changed'): guard.verify_final()
    assert not guard.verified


def test_actual_resource_failure_changed_file_and_foreign_thread_cannot_bypass_the_guard(tmp_path):
    files = actual_files(tmp_path)
    allowed = [True]
    def resources():
        if not allowed[0]: raise ValueError('source-only resource context expired')
    guard = SourceOnlyRecoveryGuard(files, resources, check_every=1)
    with ThreadPoolExecutor(max_workers=1) as executor:
        with pytest.raises(ValueError, match='owning'): executor.submit(guard.check).result()
    allowed[0] = False
    with pytest.raises(ValueError, match='resource'): guard.check()
    allowed[0] = True
    Path(next(iter(files))).write_bytes(b'changed actual source bytes\n')
    with pytest.raises(ValueError, match='changed'): guard.check()
    assert not guard.verified
