"""Capture externally committed bytes without binding a producer's live files.

Atomic replacement or journal growth may continue at the original paths. Only
exact independently committed bytes enter the immutable destination. Failed
captures retain their own partial files and never alter any producer input.
"""
import hashlib
import os
from pathlib import Path


def copy_committed(source,destination,*,expected_sha256,maximum_bytes,prefix_bytes=None,checked=lambda:None):
    source,destination=Path(source),Path(destination)
    if (not isinstance(expected_sha256,str) or len(expected_sha256)!=64
            or any(c not in '0123456789abcdef' for c in expected_sha256)
            or type(maximum_bytes) is not int or maximum_bytes<1
            or prefix_bytes is not None and (type(prefix_bytes) is not int or not 0<prefix_bytes<=maximum_bytes)
            or source.is_symlink() or not source.is_file() or destination.exists()):
        raise ValueError('A fresh destination and exact externally committed whole/prefix bytes are required')
    digest=hashlib.sha256();copied=0;checked()
    with source.open('rb') as src:
        stat=os.fstat(src.fileno());size=stat.st_size if prefix_bytes is None else prefix_bytes
        if not 0<size<=maximum_bytes or stat.st_size<size:raise ValueError('The whole committed source must fit its declared capture budget')
        with destination.open('xb') as dst:
            while copied<size:
                checked();block=src.read(min(1024**2,size-copied))
                if not block:raise ValueError('The original committed source ended before its complete exact boundary')
                dst.write(block);digest.update(block);copied+=len(block)
            dst.flush();os.fsync(dst.fileno())
    if copied!=size or digest.hexdigest()!=expected_sha256:
        raise ValueError('Every captured committed byte must match; preserve the unadmitted partial capture')
    checked()
    return dict(original_mutable_path=str(source.resolve()),immutable_capture=str(destination.resolve()),
        captured_bytes=copied,sha256=digest.hexdigest(),journal_prefix_only=prefix_bytes is not None,
        original_producer_path_not_bound_for_future_immutability=True)
