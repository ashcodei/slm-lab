"""Bounded atomic metadata publication above an unchanged disk reserve.

Transient checkpoint/archive writes can borrow filesystem headroom. Wait for
the full existing reserve and complete next metadata allocation to return;
never reduce the reserve, truncate a report or overwrite a pending artifact.
"""
import json
import os
from pathlib import Path

from .bounded_disk_reserve_wait_v1 import wait_for_reserve


def write_json(path, value, *, root, minimum_free_bytes, maximum_metadata_bytes, guard,
               maximum_wait_seconds=1800, free_bytes=None, now=None, pause=None):
    path, root = Path(path), Path(root)
    if (type(minimum_free_bytes) is not int or minimum_free_bytes < 1
            or type(maximum_metadata_bytes) is not int or maximum_metadata_bytes < 1 or not callable(guard)):
        raise ValueError('Explicit positive complete metadata bounds and an owning source/resource guard are required')
    raw = (json.dumps(value, indent=2, allow_nan=False) + '\n').encode()
    if len(raw) > maximum_metadata_bytes:
        raise ValueError('The complete metadata cannot be truncated to fit its declared bound')
    pending = path.with_suffix(path.suffix + '.pending')
    if pending.exists() or pending.is_symlink(): raise ValueError('Preserve the existing pending metadata artifact')
    # Fund the next complete allocation and small directory/inode overhead.
    needed = ((len(raw) + 4095) // 4096) * 4096 + 64 * 1024
    report = wait_for_reserve(root, minimum_bytes=minimum_free_bytes+needed, maximum_seconds=maximum_wait_seconds,
        interval_seconds=5, guard=guard, free_bytes=free_bytes, now=now, pause=pause)
    guard()
    with pending.open('xb') as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())
    os.replace(pending, path)
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try: os.fsync(directory)
    finally: os.close(directory)
    return dict(**report, complete_metadata_bytes=len(raw), unchanged_minimum_free_bytes=minimum_free_bytes)
