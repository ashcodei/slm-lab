"""Lossless checkpoint XOR archives against an explicitly retained raw anchor.

Storage alignment is inherited from the independently tested V3 ZIP reader.
Every other byte remains raw. Both durable inputs and a complete physical
restore are hashed before an inactive original may be retired. No tensor
deserialization, precision conversion or model execution is involved.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil

import numpy as np
import zstandard as zstd

from .exact_checkpoint_xor_zstd_archive_v3 import segments
from .exact_checkpoint_zstd_archive_v2 import atomic, sha, state

FORMAT = 'own-opaque-checkpoint-retained-raw-anchor-XOR-zstd-exact-v4'
BLOCK = 8 * 1024**2


def anchor_spec(path):
    path = Path(path).absolute()
    if path.is_symlink() or not path.is_file() or path.resolve() != path:
        raise ValueError('An ordinary explicitly retained raw anchor is required')
    before = state(path); digest = sha(path)
    if state(path) != before:
        raise ValueError('The complete retained raw anchor changed while hashing')
    return dict(path=str(path), bytes=before['bytes'], sha256=digest, must_be_retained=True)


def restore(archive, target):
    archive, target = Path(archive).resolve(), Path(target).absolute()
    manifest = json.loads((archive / 'manifest.json').read_text()); payload = archive / 'payload.zst'
    spec = manifest.get('retained_raw_anchor', {})
    if (manifest.get('format') != FORMAT or manifest.get('complete_archive') is not True
            or type(manifest.get('original_bytes')) is not int or manifest['original_bytes'] < 1
            or not isinstance(manifest.get('segments'), list) or not manifest['segments']
            or not spec.get('path') or anchor_spec(spec['path']) != spec
            or payload.is_symlink() or sha(payload) != manifest.get('payload_sha256')
            or payload.stat().st_size != manifest.get('payload_bytes')
            or target.exists() or target.is_symlink() or target.resolve() != target):
        raise ValueError('Exact complete payload/raw anchor and a fresh physical output are required')
    pending = target.with_name(target.name + '.restoring')
    if pending.exists() or pending.is_symlink():
        raise ValueError('Preserve the earlier physical restoration attempt')
    total, offset, digest = 0, 0, hashlib.sha256()
    try:
        with Path(spec['path']).open('rb') as old, payload.open('rb') as stored, pending.open('xb') as output:
            for part in manifest['segments']:
                length, encoded = part.get('length'), part.get('payload_bytes')
                anchor_offset = part.get('anchor_offset')
                if (part.get('kind') not in ('raw', 'xor') or part.get('target_offset') != total
                        or part.get('payload_offset') != offset or type(length) is not int or length < 1
                        or type(encoded) is not int or encoded < 1 or total + length > manifest['original_bytes']
                        or offset + encoded > manifest['payload_bytes']
                        or part['kind'] == 'raw' and anchor_offset is not None
                        or part['kind'] == 'xor' and (type(anchor_offset) is not int or anchor_offset < 0
                            or anchor_offset + length > spec['bytes'])):
                    raise ValueError('Exact bounded contiguous target/payload/anchor segment ranges are required')
                stored.seek(offset)
                if part['kind'] == 'xor': old.seek(anchor_offset)
                class Limited:
                    def __init__(self): self.remaining = encoded
                    def read(self, size):
                        if not self.remaining: return b''
                        value = stored.read(min(size, self.remaining))
                        if not value: raise ValueError('A compressed segment is truncated')
                        self.remaining -= len(value); return value
                limited, count = Limited(), 0
                with zstd.ZstdDecompressor().stream_reader(limited, read_size=64*1024,
                        read_across_frames=True, closefd=False) as decoder:
                    while piece := decoder.read(BLOCK):
                        if count + len(piece) > length:
                            raise ValueError('A restored segment exceeds its complete original byte count')
                        if part['kind'] == 'xor':
                            previous = old.read(len(piece))
                            if len(previous) != len(piece): raise ValueError('The raw anchor storage is truncated')
                            piece = np.bitwise_xor(np.frombuffer(piece, dtype=np.uint8),
                                np.frombuffer(previous, dtype=np.uint8)).tobytes()
                        output.write(piece); digest.update(piece); count += len(piece); total += len(piece)
                if limited.remaining or count != length:
                    raise ValueError('Every encoded segment must restore its exact complete byte count')
                offset += encoded
            output.flush(); os.fsync(output.fileno())
        if (total != manifest['original_bytes'] or offset != manifest['payload_bytes']
                or digest.hexdigest() != manifest['original_sha256'] or sha(pending) != manifest['original_sha256']
                or anchor_spec(spec['path']) != spec or sha(payload) != manifest['payload_sha256']):
            raise ValueError('Every physical output byte and both durable archive inputs must remain exact')
        os.link(pending, target); pending.unlink()
        directory = os.open(target.parent, os.O_RDONLY | os.O_DIRECTORY)
        try: os.fsync(directory)
        finally: os.close(directory)
        return dict(physically_restored_path=str(target), actual_restored_bytes=total,
            actual_restored_sha256=digest.hexdigest(), independent_physical_file_rehash_passed=True,
            only_durable_payload_and_retained_raw_anchor_required=True)
    finally:
        pending.unlink(missing_ok=True)


def archive(source, anchor, destination, scratch, *, expected_sha256, expected_anchor_sha256,
            maximum_payload_bytes, minimum_free_bytes, minimum_reclaimed_bytes,
            guard, progress=None, retire_original=False):
    source, anchor, destination = [Path(p).absolute() for p in (source, anchor, destination)]
    scratch = Path(scratch).resolve()
    if (source.is_symlink() or not source.is_file() or source.resolve() != source
            or source == anchor or destination.exists() or destination.is_symlink()
            or destination.resolve() != destination or not scratch.is_dir()
            or scratch.stat().st_dev == source.stat().st_dev or not callable(guard)
            or any(type(v) is not int or v < 1 for v in (maximum_payload_bytes, minimum_reclaimed_bytes))
            or type(minimum_free_bytes) is not int or minimum_free_bytes < 0):
        raise ValueError('An inactive original, distinct retained anchor and independent full physical scratch are required')
    guard(); before = state(source); spec = anchor_spec(anchor)
    if before['links'] != 1 or sha(source) != expected_sha256 or spec['sha256'] != expected_anchor_sha256:
        raise ValueError('Never retire a shared/changed original or use a changed raw anchor')
    if shutil.disk_usage(scratch).free < before['bytes'] + 1024**3:
        raise ValueError('The complete original must fit on the independent scratch filesystem')
    destination.mkdir(); payload = destination / 'payload.zst'
    total, compressed, digest, actual_parts = 0, 0, hashlib.sha256(), []
    class Bounded:
        def __init__(self, stream): self.stream = stream
        def write(self, block):
            nonlocal compressed
            if (compressed + len(block) > maximum_payload_bytes
                    or shutil.disk_usage(destination).free - len(block) < minimum_free_bytes):
                raise ValueError('The complete exact XOR payload must fit above the unchanged writer reserve')
            n = self.stream.write(block); compressed += n; return n
        def flush(self): self.stream.flush()
    with source.open('rb') as original, anchor.open('rb') as old, payload.open('xb') as stored:
        bounded = Bounded(stored)
        for part in segments(source, anchor):
            guard(); original.seek(part['target_offset']); start, remaining = compressed, part['length']
            if part['kind'] == 'xor': old.seek(part['anchor_offset'])
            with zstd.ZstdCompressor(level=1, threads=0, write_checksum=True).stream_writer(
                    bounded, write_size=BLOCK, closefd=False) as writer:
                while remaining:
                    guard(); raw = original.read(min(BLOCK, remaining))
                    if not raw: raise ValueError('The complete original was truncated')
                    remaining -= len(raw); total += len(raw); digest.update(raw)
                    if part['kind'] == 'xor':
                        previous = old.read(len(raw))
                        if len(previous) != len(raw): raise ValueError('The retained raw anchor was truncated')
                        raw = np.bitwise_xor(np.frombuffer(raw, dtype=np.uint8),
                            np.frombuffer(previous, dtype=np.uint8)).tobytes()
                    writer.write(raw)
            actual_parts.append(dict(part, payload_offset=start, payload_bytes=compressed-start))
            if progress is not None:
                progress(dict(original_bytes_processed=total, original_bytes=before['bytes'], compressed_bytes_written=compressed))
        stored.flush(); os.fsync(stored.fileno())
    guard()
    if (state(source) != before or total != before['bytes'] or digest.hexdigest() != expected_sha256
            or sha(source) != expected_sha256 or anchor_spec(anchor) != spec
            or before['bytes'] - compressed < minimum_reclaimed_bytes):
        raise ValueError('Complete unchanged original/anchor and sufficient actual savings required; original retained')
    manifest = dict(format=FORMAT, complete_archive=True, original_path=str(source), original_state=before,
        original_bytes=total, original_sha256=expected_sha256, retained_raw_anchor=spec,
        anchor=str(anchor), anchor_sha256=expected_anchor_sha256, anchor_must_be_retained=True,
        source_files_sha256={str(anchor): expected_anchor_sha256}, segments=actual_parts,
        payload_bytes=compressed, payload_sha256=sha(payload), original_retired=False, physical_restore_checked=False,
        every_original_tensor_optimizer_RNG_and_metadata_byte_preserved=True, goal_achieved=False)
    atomic(destination / 'manifest.json', manifest)
    restored = scratch / ('physically-restored-raw-XOR-' + destination.name + '.pt')
    try:
        verification = restore(destination, restored); guard()
        if (state(source) != before or sha(source) != expected_sha256 or anchor_spec(anchor) != spec
                or sha(payload) != manifest['payload_sha256']):
            raise ValueError('Original, anchor and durable payload must remain exact before retirement')
        reclaimed = source.stat().st_blocks*512 - sum(p.stat().st_blocks*512 for p in destination.iterdir())
        if reclaimed < minimum_reclaimed_bytes:
            raise ValueError('The complete physical archive must reclaim the requested allocated space; original retained')
        manifest.update(physical_restore_checked=True, physical_restore=verification,
            reclaimed_allocated_bytes_if_retired=reclaimed)
        atomic(destination / 'manifest.json', manifest)
        if retire_original:
            source.unlink()
            fd = os.open(source.parent, os.O_RDONLY | os.O_DIRECTORY)
            try: os.fsync(fd)
            finally: os.close(fd)
            manifest['original_retired'] = True; atomic(destination / 'manifest.json', manifest)
    finally:
        restored.unlink(missing_ok=True)
    return manifest
