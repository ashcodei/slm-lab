import hashlib
import json

import pytest

from slm_lab.broad_resume_journal_v1 import copy_prefix


def test_committed_prefix_is_exact_and_never_promotes_or_truncates_uncommitted_tail(tmp_path):
    prefix = b''.join((json.dumps(dict(step=i, tokens=[i, 2]))+'\n').encode() for i in [1, 2])
    original = tmp_path/'original'; tail = b'{"step":3,"unfinished":'
    original.write_bytes(prefix+tail); output = tmp_path/'fork'
    result = copy_prefix(original, output, length=len(prefix), expected_sha256=hashlib.sha256(prefix).hexdigest(),
                         completed=2, minimum_free_bytes=0)
    assert original.read_bytes() == prefix+tail and output.read_bytes() == prefix
    assert result['original_uncommitted_tail_bytes_preserved'] == len(tail)
    with pytest.raises(ValueError):
        copy_prefix(original, output, length=len(prefix), expected_sha256=result['sha256'], completed=2, minimum_free_bytes=0)


@pytest.mark.parametrize('damage', ['digest', 'boundary', 'sequence'])
def test_invalid_prefix_preserves_source_and_any_failed_temporary_for_audit(tmp_path, damage):
    raw = b'{"step":1}\n{"step":2}\n'
    if damage == 'sequence':
        raw = raw.replace(b'"step":2', b'"step":3')
    source = tmp_path/'source'; source.write_bytes(raw); target = tmp_path/'target'
    with pytest.raises(ValueError):
        copy_prefix(source, target, length=len(raw)-(1 if damage == 'boundary' else 0),
                    expected_sha256='0'*64 if damage == 'digest' else hashlib.sha256(raw).hexdigest(),
                    completed=2, minimum_free_bytes=0)
    assert source.read_bytes() == raw and not target.exists()
    assert target.with_suffix('.pending').exists()
