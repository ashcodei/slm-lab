"""Fork only an externally committed journal prefix; preserve its old tail."""
import hashlib
import json
import os
from pathlib import Path
import shutil


def copy_prefix(source, destination, *, length, expected_sha256, completed, minimum_free_bytes):
    source, destination = Path(source), Path(destination)
    pending = destination.with_suffix(destination.suffix+'.pending')
    if (type(length) is not int or length < 1 or type(completed) is not int or completed < 1
            or destination.exists() or pending.exists() or source.resolve() == destination.resolve()
            or shutil.disk_usage(destination.parent).free < length+minimum_free_bytes):
        raise ValueError('A fresh separately funded committed-prefix destination is required')
    digest = hashlib.sha256(); size = 0
    with source.open('rb') as original, pending.open('xb') as output:
        for step in range(1, completed+1):
            line = original.readline()
            if not line.endswith(b'\n') or json.loads(line).get('step') != step or size+len(line) > length:
                raise ValueError('Every original complete committed journal line must reconstruct')
            output.write(line); digest.update(line); size += len(line)
        if size != length or digest.hexdigest() != expected_sha256:
            raise ValueError('The complete original committed prefix length/SHA must match')
        output.flush(); os.fsync(output.fileno())
    # Never clobber an existing fork; preserve a failed temporary file for audit.
    os.link(pending, destination); pending.unlink()
    fd = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    return dict(path=str(destination.resolve()), bytes=size, sha256=digest.hexdigest(), completed=completed,
                original_uncommitted_tail_bytes_preserved=source.stat().st_size-length)
