"""Lossless transport stress using real closed records and source bindings.

The neural records are from a previously completed small CPU fixture. Adding
the actual production-source map and explicit transport padding is a wire-size
stress only; it does not create production-model samples or quality evidence.
"""
import copy
import gzip
import hashlib
import json
from pathlib import Path

import pytest

from slm_lab.factored_native_observation_stream_v1 import Writer,Reader,canonical,digest,kernels,MAX_WIRE

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'reports/full-native-token-suffix-qualification-gzip-recovery-CPU-v3/closed-source-bindings.json'
JOURNAL=ROOT/'reports/native-suffix-training-controller-v1-full-pytest/test_actual_four_replica_capac0/anchor_execution/complete/attempt.jsonl.gz'


def fixture():
    with gzip.open(JOURNAL,'rt') as stream:row=json.loads(stream.readline())
    observed=list(kernels(row));assert len(observed)==4 and all(len(k['trajectories'])==8 for k in observed)
    reference=copy.deepcopy(observed[0]['own_reference_binding'])
    reference['sources']=json.loads(SOURCE.read_text())['source_files_sha256']
    for kernel in observed:kernel['own_reference_binding']=copy.deepcopy(reference)
    row['source_only_transport_stress_not_new_neural_samples']=True
    return row,reference


def writer(path,reference,**kwargs):
    options=dict(root=path.parent,maximum_logical_bytes=256*1024**2,maximum_physical_bytes=32*1024**2,
        minimum_free_bytes=0,guard=lambda:None)
    options.update(kwargs)
    return Writer(path,reference_binding=reference,**options)


def test_actual_closed_four_rank_attempts_and_full_source_maps_roundtrip_beyond_old_line_limit(tmp_path):
    row,reference=fixture()
    padding='explicit source-only transport padding\n'
    row['source_only_transport_padding']=(padding*((9*1024**2)//len(padding)+1))[:9*1024**2]
    expected=[copy.deepcopy(row) for _ in range(3)]
    for index,value in enumerate(expected):value['transport_fixture_record_position']=index
    raw=[canonical(r) for r in expected]
    assert all(len(r)>8*1024**2 for r in raw)
    path=tmp_path/'complete.nativeobs.gz';out=writer(path,reference)
    for row in expected:out.append(row);out.fsync()
    out.close();receipt=out.receipt()
    reader=Reader(path,maximum_logical_bytes=256*1024**2,expected_reference_sha256=digest(canonical(reference)))
    actual=list(reader)
    assert reader.complete and [r for r,_ in actual]==raw and [x for _,x in actual]==expected
    assert reader.records==receipt['records']==3
    assert receipt['expanded_update_bytes']==sum(map(len,raw))
    assert receipt['logical_budget_used']==sum(map(len,raw))+len(canonical(reference))
    assert receipt['expanded_update_stream_sha256']==hashlib.sha256(b''.join(raw)).hexdigest()
    assert receipt['compressed_sha256']==hashlib.sha256(path.read_bytes()).hexdigest()
    assert receipt['actual_compressed_bytes']==path.stat().st_size
    assert receipt['actual_compressed_bytes']<len(canonical(reference))
    assert all(len(k['trajectories'])==8 for r in expected for k in kernels(r))
    proof=dict(actual_closed_input_binding_count=len(reference['sources']),
        actual_closed_four_rank_records_used=3,actual_attempts_per_source=8,
        original_small_CPU_fixture_only=True,production_source_map_used_for_transport_stress_only=True,
        explicit_per_record_non_neural_padding_bytes=9*1024**2,
        complete_original_expanded_record_bytes=[len(r) for r in raw],
        every_original_expanded_byte_action_probability_verdict_and_timing_preserved=True,
        actual_logical_quota_counts_all_repeated_original_bindings=True,
        actual_transport_receipt=receipt,new_neural_updates=0,full_size_GPUcapacity_or_quality_gain=False,goal_achieved=False)
    (tmp_path/'actual-production-source-map-transport-stress.json').write_text(json.dumps(proof,indent=2))


def test_changed_binding_rejects_before_appending_and_original_prefix_remains_complete(tmp_path):
    row,reference=fixture();path=tmp_path/'prefix.nativeobs.gz';out=writer(path,reference)
    out.append(row);changed=copy.deepcopy(row)
    next(kernels(changed))['own_reference_binding']['sources']['changed-binding']='a'*64
    with pytest.raises(ValueError,match='byte-exact'):out.append(changed)
    assert out.records==1;out.close()
    reader=Reader(path,maximum_logical_bytes=256*1024**2)
    assert list(reader)==[(canonical(row),row)] and reader.complete


def test_original_expanded_quota_cannot_be_bypassed_by_smaller_wire_data(tmp_path):
    row,reference=fixture();expanded=len(canonical(row));header=len(canonical(reference))
    path=tmp_path/'logical-limit.nativeobs.gz'
    out=writer(path,reference,maximum_logical_bytes=header+expanded-1)
    with pytest.raises(ValueError,match='expanded observation quota'):out.append(row)
    assert out.records==0 and out.logical_used==header
    out.close();reader=Reader(path,maximum_logical_bytes=header+expanded-1)
    assert list(reader)==[] and reader.complete


def test_corrupt_truncated_or_different_reference_streams_never_finish_readback(tmp_path):
    row,reference=fixture();path=tmp_path/'good.nativeobs.gz';out=writer(path,reference)
    out.append(row);out.close();raw=path.read_bytes()
    reader=Reader(path,maximum_logical_bytes=256*1024**2,expected_reference_sha256='0'*64)
    with pytest.raises(ValueError,match='reference header'):list(reader)
    assert not reader.complete
    for name,data in [('truncated',raw[:-8]),('corrupt',raw[:-8]+bytes([raw[-8]^1])+raw[-7:])]:
        bad=tmp_path/(name+'.gz');bad.write_bytes(data);reader=Reader(bad,maximum_logical_bytes=256*1024**2)
        with pytest.raises((ValueError,EOFError,gzip.BadGzipFile)):list(reader)
        assert not reader.complete


def test_actual_physical_quota_and_existing_output_are_protected(tmp_path):
    row,reference=fixture();existing=tmp_path/'existing.gz';existing.write_bytes(b'keep original evidence')
    with pytest.raises(FileExistsError):writer(existing,reference)
    assert existing.read_bytes()==b'keep original evidence'
    with pytest.raises(ValueError,match='physical quota'):
        writer(tmp_path/'too-small.gz',reference,maximum_physical_bytes=1)
