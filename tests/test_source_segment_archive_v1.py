import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from slm_lab.source_segment_archive_v1 import archive_duplicate,no_readers,restore,verify


def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def fixture(tmp_path):
    anchor=tmp_path/'retained.jsonl';target=tmp_path/'duplicate.jsonl';manifest=tmp_path/'duplicate.segment.json'
    raw=b'{"text":"original public source"}\n'*9
    anchor.write_bytes(b'prefix\n'+raw+b'other-source\n');target.write_bytes(raw)
    return target,anchor,manifest,raw


def test_exact_segment_archive_and_real_original_path_restore(tmp_path):
    target,anchor,manifest,raw=fixture(tmp_path)
    result=archive_duplicate(target,anchor,7,manifest,digest(target),digest(anchor))
    assert result['original_removed'] and not target.exists()
    assert verify(manifest)['original_sha256']==hashlib.sha256(raw).hexdigest()
    restore(manifest)
    assert target.read_bytes()==raw and anchor.read_bytes().startswith(b'prefix\n')


def test_wrong_expected_anchor_or_segment_keeps_original(tmp_path):
    target,anchor,manifest,raw=fixture(tmp_path)
    with pytest.raises(ValueError,match='checksums'):
        archive_duplicate(target,anchor,7,manifest,digest(target),'0'*64)
    assert target.read_bytes()==raw and not manifest.exists()
    with pytest.raises(ValueError,match='not exactly identical'):
        archive_duplicate(target,anchor,0,manifest,digest(target),digest(anchor))
    assert target.read_bytes()==raw


def test_changed_retained_anchor_prevents_restore_and_existing_output_is_preserved(tmp_path):
    target,anchor,manifest,raw=fixture(tmp_path)
    archive_duplicate(target,anchor,7,manifest,digest(target),digest(anchor))
    target.write_bytes(b'new user data')
    with pytest.raises(ValueError,match='never overwrite'):restore(manifest)
    assert target.read_bytes()==b'new user data'
    target.unlink();anchor.write_bytes(b'changed')
    with pytest.raises(ValueError,match='anchor changed'):restore(manifest)
    assert not target.exists()


def test_a_live_file_descriptor_prevents_source_removal(tmp_path):
    target,anchor,manifest,raw=fixture(tmp_path)
    code='import sys; f=open(sys.argv[1],"rb"); print("ready",flush=True); sys.stdin.read()'
    child=subprocess.Popen([sys.executable,'-c',code,str(target)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True)
    try:
        assert child.stdout.readline().strip()=='ready'
        with pytest.raises(ValueError,match='live process|file descriptor'):
            archive_duplicate(target,anchor,7,manifest,digest(target),digest(anchor))
        assert target.read_bytes()==raw
    finally:
        child.stdin.close();child.wait(timeout=10)


def test_shared_hard_link_and_invalid_range_are_rejected(tmp_path):
    target,anchor,manifest,raw=fixture(tmp_path)
    with pytest.raises(ValueError,match='fit the retained'):
        archive_duplicate(target,anchor,-1,manifest,digest(target),digest(anchor))
    import os
    os.link(target,tmp_path/'another-link')
    with pytest.raises(ValueError,match='shared hard link'):
        archive_duplicate(target,anchor,7,manifest,digest(target),digest(anchor))
    assert target.read_bytes()==raw


def private_process(seconds):
    code='import ctypes,time; assert ctypes.CDLL(None).prctl(4,0,0,0,0)==0; print("ready",flush=True); time.sleep('+str(seconds)+')'
    child=subprocess.Popen([sys.executable,'-c',code],stdout=subprocess.PIPE,text=True)
    assert child.stdout.readline().strip()=='ready'
    return child


def test_transient_private_exec_is_waited_until_terminal(tmp_path):
    target,anchor,manifest,raw=fixture(tmp_path);child=private_process(.2)
    try:
        archive_duplicate(target,anchor,7,manifest,digest(target),digest(anchor))
        assert not target.exists() and verify(manifest)['original_bytes']==len(raw)
    finally:child.wait(timeout=10)


def test_persistent_uninspectable_process_fails_closed(tmp_path):
    target,anchor,manifest,raw=fixture(tmp_path);child=private_process(1)
    try:
        with pytest.raises(ValueError,match='Cannot inspect same-owner process'):
            no_readers(target,maximum_wait_seconds=.03)
        assert target.read_bytes()==raw and not manifest.exists()
    finally:child.wait(timeout=10)
