"""Lossless opaque checkpoint archives with physical restore before retirement.

No tensor deserialization, model rewriting or precision conversion occurs.
The caller selects an inactive, unbound project artifact and supplies its exact
hash plus live-resource guards. Existing inference paths are never substituted.
"""
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil

BLOCK = 8 * 1024**2
FORMAT = 'own-opaque-checkpoint-exact-gzip-archive-v1'


def sha(path):
    with Path(path).open('rb') as stream: return hashlib.file_digest(stream, 'sha256').hexdigest()


def state(path):
    s = Path(path).stat()
    return dict(device=s.st_dev, inode=s.st_ino, bytes=s.st_size, mtime_ns=s.st_mtime_ns,
        mode=s.st_mode, uid=s.st_uid, gid=s.st_gid, links=s.st_nlink)


def atomic(path, value):
    path = Path(path); pending = path.with_suffix('.pending')
    with pending.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False); stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    pending.replace(path)
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try: os.fsync(fd)
    finally: os.close(fd)


def restore(archive, target):
    archive, target = Path(archive).resolve(), Path(target)
    manifest = json.loads((archive / 'manifest.json').read_text()); payload = archive / 'payload.gz'
    if (manifest.get('format') != FORMAT or manifest.get('complete_archive') is not True
            or type(manifest.get('original_bytes')) is not int or manifest['original_bytes'] < 1
            or sha(payload) != manifest.get('payload_sha256') or target.exists() or target.is_symlink()):
        raise ValueError('An exact complete archive and unused physical restore path are required')
    pending = target.with_name(target.name + '.restoring')
    if pending.exists() or pending.is_symlink(): raise ValueError('Preserve an earlier physical restore attempt')
    total, digest = 0, hashlib.sha256()
    try:
        with gzip.open(payload, 'rb') as source, pending.open('xb') as output:
            while block := source.read(BLOCK):
                total += len(block)
                if total > manifest['original_bytes']: raise ValueError('Archive exceeds the exact original byte count')
                output.write(block); digest.update(block)
            output.flush(); os.fsync(output.fileno())
        if total != manifest['original_bytes'] or digest.hexdigest() != manifest['original_sha256'] or sha(pending) != manifest['original_sha256']:
            raise ValueError('Every physically restored byte must match the entire original checkpoint SHA256')
        os.link(pending, target); pending.unlink()
        return dict(physically_restored_path=str(target.resolve()), actual_restored_bytes=total,
            actual_restored_sha256=digest.hexdigest(), independent_physical_file_rehash_passed=True)
    finally:
        pending.unlink(missing_ok=True)


def archive(source, destination, scratch, *, expected_sha256, maximum_payload_bytes, minimum_free_bytes,
            minimum_reclaimed_bytes, guard, progress=None, retire_original=False):
    source, destination, scratch = Path(source).absolute(), Path(destination).absolute(), Path(scratch).resolve()
    if (source.is_symlink() or not source.is_file() or destination.exists() or not callable(guard)
            or any(type(n) is not int or n < 1 for n in [maximum_payload_bytes, minimum_reclaimed_bytes])
            or type(minimum_free_bytes) is not int or minimum_free_bytes < 0 or not scratch.is_dir()
            or scratch.stat().st_dev == source.stat().st_dev):
        raise ValueError('An inactive ordinary artifact, fresh archive, independent scratch filesystem and explicit complete budgets are required')
    guard(); before = state(source)
    if before['links'] != 1 or sha(source) != expected_sha256:
        raise ValueError('Never retire a shared anchor or a checkpoint with different complete bytes')
    if shutil.disk_usage(scratch).free < before['bytes'] + 1024**3:
        raise ValueError('The entire original must fit on the separate scratch filesystem')
    destination.mkdir()
    payload = destination / 'payload.gz'
    total, raw_digest = 0, hashlib.sha256()
    class Bounded:
        def __init__(self, stream): self.stream, self.count = stream, 0
        def write(self, block):
            if (self.count + len(block) > maximum_payload_bytes
                    or shutil.disk_usage(destination).free - len(block) < minimum_free_bytes):
                raise ValueError('A complete exact compressed payload must fit above the original writers reserve')
            value = self.stream.write(block); self.count += value; return value
        def flush(self): self.stream.flush()
    with source.open('rb') as original, payload.open('xb') as stored:
        bounded = Bounded(stored)
        with gzip.GzipFile(filename='', mode='wb', compresslevel=1, mtime=0, fileobj=bounded) as compressed:
            while block := original.read(BLOCK):
                guard(); raw_digest.update(block); total += len(block); compressed.write(block)
                if progress is not None and (total % (256 * 1024**2) == 0 or total == before['bytes']):
                    progress(dict(original_bytes_processed=total, original_bytes=before['bytes'], compressed_bytes_written=bounded.count))
        stored.flush(); os.fsync(stored.fileno())
    if (state(source) != before or total != before['bytes'] or raw_digest.hexdigest() != expected_sha256
            or sha(source) != expected_sha256 or before['bytes'] - payload.stat().st_size < minimum_reclaimed_bytes):
        raise ValueError('The complete unchanged original and sufficient actual storage recovery are required; original retained')
    manifest = dict(format=FORMAT, complete_archive=True, original_path=str(source), original_state=before,
        original_bytes=total, original_sha256=expected_sha256, payload_bytes=payload.stat().st_size,
        payload_sha256=sha(payload), tensor_bytes_and_optimizer_RNG_metadata_unchanged=True,
        original_retired=False, physical_restore_checked=False, goal_achieved=False)
    atomic(destination / 'manifest.json', manifest)
    restore_path = scratch / 'physically-restored-original.pt'
    verification = restore(destination, restore_path)
    try:
        guard()
        if state(source) != before or sha(source) != expected_sha256 or sha(payload) != manifest['payload_sha256']:
            raise ValueError('Original artifact or complete archive changed before retirement')
        manifest.update(physical_restore_checked=True, physical_restore=verification,
            reclaimed_allocated_bytes_if_retired=source.stat().st_blocks * 512 - payload.stat().st_blocks * 512)
        atomic(destination / 'manifest.json', manifest)
        if retire_original:
            source.unlink()
            manifest['original_retired'] = True
            atomic(destination / 'manifest.json', manifest)
    finally:
        restore_path.unlink(missing_ok=True)
    return manifest
