from pathlib import Path
import hashlib
import pytest

from slm_lab.photo_storage_policy_v1 import validate_reserve, migration_snapshot, required_bytes


@pytest.mark.parametrize('value', [26, 28, 35, 40.5])
def test_conservative_finite_reserves_are_accepted(value):
    assert validate_reserve(value) == value


@pytest.mark.parametrize('value', [25.99, 0, -1, float('nan'), float('inf'), True])
def test_unsafe_or_nonfinite_reserves_are_rejected(value):
    with pytest.raises(ValueError): validate_reserve(value)


def test_previous_source_requires_its_exact_committed_digest(tmp_path):
    p = tmp_path / 'old.py'; p.write_text('original collector source')
    digest = hashlib.sha256(p.read_bytes()).hexdigest()
    assert migration_snapshot(p, digest)['sha256'] == digest
    p.write_text('another collector')
    with pytest.raises(ValueError): migration_snapshot(p, digest)
    assert migration_snapshot(None, digest) is None


def test_disk_budget_covers_all_writers_plus_safety():
    assert required_bytes(8, 3, 3, 2, 8) == 24
    with pytest.raises(ValueError): required_bytes(8, 3, -3, 2)
