from copy import deepcopy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from read_and_pack_billion_public_candidate_CPU_v1 import validate_source_receipt


def fixture():
    source = dict(family='code', path='data/000.parquet', size=123, sha256='a'*64)
    receipt = dict(source=source, source_number=1, actual_verified_original_file_bytes=123,
                   actual_verified_original_file_sha256='a'*64)
    catalogs = dict(code=[source], math=[dict(family='math', path='math/000.parquet', size=99, sha256='b'*64)], web=[])
    return source, receipt, catalogs


def test_every_source_catalog_prefix_position_and_pinned_byte_identity_are_required():
    source, receipt, catalogs = fixture()
    positions = dict(code=0, math=0, web=0)
    assert validate_source_receipt(receipt, 1, catalogs, positions, None, None) == 'code'
    assert positions['code'] == 1
    wrong = deepcopy(receipt);wrong['source']['path'] = 'data/001.parquet'
    with pytest.raises(ValueError, match='catalog prefix'):
        validate_source_receipt(wrong, 1, catalogs, dict(code=0, math=0, web=0), None, None)
    wrong = dict(receipt, actual_verified_original_file_bytes=122)
    with pytest.raises(ValueError, match='pinned whole original'):
        validate_source_receipt(wrong, 1, catalogs, dict(code=0, math=0, web=0), None, None)


def test_next_source_family_requires_every_previous_train_quota_and_never_skips_family():
    _, _, catalogs = fixture()
    source = catalogs['math'][0]
    receipt = dict(source=source, source_number=2, actual_verified_original_file_bytes=99,
                   actual_verified_original_file_sha256='b'*64)
    class Incomplete:
        def family_full(self, family):return False
    with pytest.raises(ValueError, match='unsatisfied training quota'):
        validate_source_receipt(receipt, 2, catalogs, dict(code=1, math=0, web=0), 'code', Incomplete())
    class Complete:
        def family_full(self, family):return True
    assert validate_source_receipt(receipt, 2, catalogs, dict(code=1, math=0, web=0), 'code', Complete()) == 'math'
