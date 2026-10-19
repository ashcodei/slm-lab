import gzip
import hashlib
import json

import pytest

from slm_lab.bounded_gzip_observation_stream_v1 import Writer, RawBudget, lines, NEXT_RECORD


def writer(path, budget, guard=lambda: None, reserve=lambda *a, **kw: kw['guard']()):
    return Writer(path, root=path.parent, budget=budget, reserve_bytes=40 * 1024**3, guard=guard, reserve=reserve)


def test_exact_original_byte_records_roundtrip_preserving_Unicode_and_all_dispositions(tmp_path):
    raw = b'{"failed":true,"text":"a\xe2\x80\xa8b","reward":null}\n' + b'{"trivial":true,"reward":1.0}\n'
    source = tmp_path / 'original.jsonl'
    source.write_bytes(raw)
    budget = RawBudget(1024**3, retained_source_bytes=1234)
    observed = []
    def reserve(root, *, minimum_bytes, guard):
        observed.append(minimum_bytes)
        guard()
    with writer(tmp_path / 'closed.jsonl.gz', budget, reserve=reserve) as out:
        for record, _ in lines(source): out.append_raw(record)
    assert gzip.decompress(out.path.read_bytes()) == raw
    assert [row for _, row in lines(out.path)] == [json.loads(v) for v in raw.split(b'\n') if v]
    receipt = out.receipt()
    assert receipt['complete'] and receipt['records'] == 2 and budget.used == 1234 + len(raw)
    assert receipt['actual_uncompressed_sha256'] == hashlib.sha256(raw).hexdigest()
    assert observed == [40 * 1024**3 + NEXT_RECORD] * 2


def test_aggregate_full_raw_quota_is_shared_across_all_outputs_and_no_failed_record_is_dropped_silently(tmp_path):
    raw = b'{"case":0}\n'
    budget = RawBudget(2 * len(raw))
    with writer(tmp_path / 'first.gz', budget) as first, writer(tmp_path / 'second.gz', budget) as second:
        first.append_raw(raw)
        second.append_raw(raw)
        with pytest.raises(ValueError): second.append_raw(raw)
        assert first.records == second.records == 1 and budget.used == 2 * len(raw)
    assert list(lines(first.path))[0][0] == raw and list(lines(second.path))[0][0] == raw


def test_changed_source_and_actual_reserve_failure_prevent_any_new_record(tmp_path):
    def fail(): raise ValueError('changed immutable source')
    with writer(tmp_path / 'guard.gz', RawBudget(10000), guard=fail) as out:
        with pytest.raises(ValueError): out.append(dict(case=0))
        assert out.records == out.raw_bytes == out.budget.used == 0
    def fail_reserve(*args, **kwargs): raise ValueError('actual other-writer floor unavailable')
    with writer(tmp_path / 'space.gz', RawBudget(10000), reserve=fail_reserve) as out:
        with pytest.raises(ValueError): out.append(dict(case=0))
        assert out.records == out.raw_bytes == out.budget.used == 0


def test_torn_original_LF_invalid_JSON_and_nonfinite_values_are_never_rewritten(tmp_path):
    path = tmp_path / 'torn.jsonl'
    path.write_bytes(b'{"case":1}')
    with pytest.raises(ValueError): list(lines(path))
    with writer(tmp_path / 'invalid.gz', RawBudget(10000)) as out:
        for bad in [b'{bad}\n', b'{}', b'\xff\n']:
            with pytest.raises(ValueError): out.append_raw(bad)
        with pytest.raises(ValueError): out.append(dict(reward=float('nan')))
        assert out.records == 0


def test_interrupted_container_and_attempted_overwrite_cannot_be_relabelled_complete(tmp_path):
    path = tmp_path / 'closed.gz'
    with writer(path, RawBudget(10000)) as out: out.append(dict(case=1))
    with pytest.raises(FileExistsError): writer(path, RawBudget(10000))
    path.write_bytes(path.read_bytes()[:-4])
    with pytest.raises(EOFError): list(lines(path))
