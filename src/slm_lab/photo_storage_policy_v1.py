"""Validate photo-collection reserve and explicit prior-source migration."""
import math
from pathlib import Path

from .corpus import file_hash

MINIMUM_RESERVE_GIB = 26.0


def validate_reserve(value):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value < MINIMUM_RESERVE_GIB):
        raise ValueError('Photo reserve must be finite and at least26GiB')
    return float(value)


def migration_snapshot(path, expected_digest):
    if path is None:
        return None
    path = Path(path).resolve()
    if not path.is_file() or file_hash(path) != expected_digest:
        raise ValueError('Prior collector snapshot differs from the committed configuration')
    return {'path': str(path), 'sha256': expected_digest}


def required_bytes(snapshot, export, archive, remaining_photos, safety=8 * 1024**3):
    values = [snapshot, export, archive, remaining_photos, safety]
    if any(type(x) is not int or x < 0 for x in values):
        raise ValueError('Disk budget requires nonnegative actual byte sizes')
    return sum(values)
