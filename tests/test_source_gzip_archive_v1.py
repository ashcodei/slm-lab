import hashlib
import json
import os

import pytest

from slm_lab.source_gzip_archive_v1 import archive,restore,verify


def original(tmp_path):
    path=tmp_path/'source.csv';raw=b'ImageID,LabelName,XMin,XMax\n'+b'000123,/m/123,0.1,0.4\n'*2000
    path.write_bytes(raw)
    return path,raw,hashlib.sha256(raw).hexdigest(),tmp_path/'archive.json'


def test_actual_archive_restore_exact_and_existing_destination_preserved(tmp_path):
    path,raw,digest,manifest=original(tmp_path)
    record=archive(path,manifest,digest,10000,0)
    assert not path.exists() and record['reclaimed_allocated_bytes']>0
    assert verify(manifest)['original_sha256']==digest
    restore(manifest)
    assert path.read_bytes()==raw
    with pytest.raises(ValueError,match='Fresh restore'):restore(manifest)
    assert path.read_bytes()==raw


def test_wrong_original_digest_and_output_overflow_leave_original(tmp_path):
    path,raw,digest,manifest=original(tmp_path)
    with pytest.raises(ValueError,match='checksum'):archive(path,manifest,'0'*64,10000,0)
    with pytest.raises(ValueError,match='reserved storage|round trip'):archive(path,manifest,digest,10,0)
    assert path.read_bytes()==raw and not manifest.exists()
    assert not list(tmp_path.glob('*.compressing-*'))


def test_corrupt_archive_cannot_restore_or_overwrite_anything(tmp_path):
    path,raw,digest,manifest=original(tmp_path);record=archive(path,manifest,digest,10000,0)
    from pathlib import Path
    compressed=Path(record['archive_path']);damaged=bytearray(compressed.read_bytes());damaged[-5]^=1;compressed.write_bytes(damaged)
    with pytest.raises(ValueError,match='Compressed archive'):restore(manifest)
    assert not path.exists()


def test_open_reader_and_shared_hard_link_reject_removal(tmp_path):
    path,raw,digest,manifest=original(tmp_path)
    with path.open('rb') as stream:
        # no_readers deliberately ignores its own process; exercise a real other reader.
        import subprocess,sys
        process=subprocess.Popen([sys.executable,'-c','import sys,time; f=open(sys.argv[1],"rb"); print("ready",flush=True); time.sleep(20)',str(path)],stdout=subprocess.PIPE,text=True)
        try:
            assert process.stdout.readline().strip()=='ready'
            with pytest.raises(ValueError,match='live'):archive(path,manifest,digest,10000,0)
        finally:process.terminate();process.wait(timeout=5)
    os.link(path,tmp_path/'other-link')
    with pytest.raises(ValueError,match='shared hard link'):archive(path,manifest,digest,10000,0)
    assert path.read_bytes()==raw
