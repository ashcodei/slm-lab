import json
from pathlib import Path

import pytest

from slm_lab.reserved_atomic_metadata_v1 import write_json


def test_transient_space_shortfall_waits_and_preserves_full_reserve_and_complete_report(tmp_path):
    current=[0.];calls=[]
    def free():
        calls.append(current[0]);return 39*1024**3 if current[0]<10 else 41*1024**3
    def pause(seconds):current[0]+=seconds
    guards=[]
    path=tmp_path/'progress.json';value=dict(all_sources=list(range(3721)),full_scope=True)
    report=write_json(path,value,root=tmp_path,minimum_free_bytes=40*1024**3,maximum_metadata_bytes=1024**2,
        guard=lambda:guards.append(current[0]),maximum_wait_seconds=30,free_bytes=free,now=lambda:current[0],pause=pause)
    assert report['waited_seconds']==10 and report['failed_space_observations']==2
    assert report['unchanged_minimum_free_bytes']==40*1024**3 and json.loads(path.read_text())==value
    assert guards and not path.with_suffix('.json.pending').exists()


def test_timeout_leaves_previous_complete_report_and_no_partial_publication(tmp_path):
    current=[0.];path=tmp_path/'progress.json';path.write_text('{"previous":true}\n')
    def pause(seconds):current[0]+=seconds
    with pytest.raises(ValueError,match='bounded wait'):
        write_json(path,dict(next=True),root=tmp_path,minimum_free_bytes=40*1024**3,maximum_metadata_bytes=1024,
            guard=lambda:None,maximum_wait_seconds=10,free_bytes=lambda:39*1024**3,now=lambda:current[0],pause=pause)
    assert json.loads(path.read_text())==dict(previous=True) and not path.with_suffix('.json.pending').exists()


def test_source_change_while_waiting_cannot_publish_progress_or_readiness(tmp_path):
    current=[0.];path=tmp_path/'ready.json'
    def pause(seconds):current[0]+=seconds
    def guard():
        if current[0]>=5:raise ValueError('An immutable source changed during the actual reserve wait')
    with pytest.raises(ValueError,match='immutable source changed'):
        write_json(path,dict(ready=True),root=tmp_path,minimum_free_bytes=40*1024**3,maximum_metadata_bytes=1024,
            guard=guard,maximum_wait_seconds=10,free_bytes=lambda:39*1024**3,now=lambda:current[0],pause=pause)
    assert not path.exists() and not path.with_suffix('.json.pending').exists()


def test_existing_pending_and_oversized_report_are_preserved_without_waiting(tmp_path):
    path=tmp_path/'progress.json';pending=path.with_suffix('.json.pending');pending.write_bytes(b'old incomplete evidence')
    with pytest.raises(ValueError,match='pending metadata'):
        write_json(path,{},root=tmp_path,minimum_free_bytes=40*1024**3,maximum_metadata_bytes=1024,guard=lambda:None)
    assert pending.read_bytes()==b'old incomplete evidence'
    pending.unlink()
    with pytest.raises(ValueError,match='cannot be truncated'):
        write_json(path,dict(full='complete'*100),root=tmp_path,minimum_free_bytes=40*1024**3,maximum_metadata_bytes=16,guard=lambda:None)
    assert not path.exists() and not pending.exists()
