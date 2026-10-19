"""Exact opaque checkpoint bytes using tensor-aligned XOR and Zstandard.

The immutable anchor is an existing independently restorable V2 archive.
ZIP storage offsets are read without unpickling tensors. Numeric values,
precision, optimizer/RNG state and every non-tensor metadata byte are intact.
Physical restoration from both durable payloads is mandatory before retirement.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import tempfile
import zipfile

import numpy as np
import zstandard as zstd

from .exact_checkpoint_zstd_archive_v2 import (restore as restore_anchor, sha, state, atomic, FORMAT as ANCHOR)

FORMAT = 'own-opaque-checkpoint-tensor-aligned-XOR-zstd-exact-v3'
BLOCK = 8 * 1024**2


def storage_ranges(path):
    result = {}
    with zipfile.ZipFile(path) as archive, Path(path).open('rb') as raw:
        for item in archive.infolist():
            parts = item.filename.split('/')
            if len(parts) != 3 or parts[-2] != 'data' or not parts[-1].isdigit(): continue
            if item.compress_type != zipfile.ZIP_STORED or parts[-1] in result:
                raise ValueError('Distinct ordinary uncompressed checkpoint ZIP storages are required')
            raw.seek(item.header_offset); header = raw.read(30)
            if len(header) != 30 or header[:4] != b'PK\x03\x04': raise ValueError('Invalid original storage ZIP header')
            names, extra = struct.unpack_from('<HH', header, 26)
            start = item.header_offset + 30 + names + extra
            if start + item.file_size > Path(path).stat().st_size: raise ValueError('A storage exceeds the entire original file')
            if item.file_size: result[parts[-1]] = (start, item.file_size)
    if not result: raise ValueError('A complete own ZIP checkpoint with actual storages is required')
    return result


def segments(source, anchor):
    ranges, previous = storage_ranges(source), storage_ranges(anchor)
    parts, position = [], 0
    for key, (offset, length) in sorted(ranges.items(), key=lambda item: item[1][0]):
        if offset < position: raise ValueError('Original tensor byte ranges overlap')
        if offset > position: parts.append(dict(kind='raw', target_offset=position, length=offset-position, anchor_offset=None))
        old = previous.get(key)
        reuse = old is not None and old[1] == length
        parts.append(dict(kind='xor' if reuse else 'raw', target_offset=offset, length=length,
            anchor_offset=old[0] if reuse else None))
        position = offset + length
    size = Path(source).stat().st_size
    if position < size: parts.append(dict(kind='raw', target_offset=position, length=size-position, anchor_offset=None))
    if not any(p['kind']=='xor' for p in parts): raise ValueError('The exact anchor has no aligned storage candidate')
    return parts


def anchor_spec(path):
    path = Path(path).resolve(); manifest = path / 'manifest.json'; info = json.loads(manifest.read_text())
    if (info.get('format') != ANCHOR or info.get('complete_archive') is not True
            or info.get('physical_restore_checked') is not True or sha(path / 'payload.zst') != info.get('payload_sha256')):
        raise ValueError('An actual complete retained V2 anchor with prior independent physical restoration is required')
    return dict(archive=str(path), manifest_sha256=sha(manifest), payload_sha256=info['payload_sha256'],
        original_sha256=info['original_sha256'], original_bytes=info['original_bytes'], anchor_must_be_retained=True)


def restore(archive, target, scratch):
    archive, target, scratch = Path(archive).resolve(), Path(target).absolute(), Path(scratch).resolve()
    manifest = json.loads((archive / 'manifest.json').read_text()); payload = archive / 'payload.zst'
    if (manifest.get('format') != FORMAT or manifest.get('complete_archive') is not True
            or target.exists() or target.is_symlink() or not scratch.is_dir()
            or type(manifest.get('original_bytes')) is not int or manifest['original_bytes'] < 1
            or sha(payload) != manifest.get('payload_sha256')
            or anchor_spec(manifest['compressed_anchor']['archive']) != manifest['compressed_anchor']):
        raise ValueError('Exact complete target/anchor payloads, fresh output and existing scratch space are required')
    pending = target.with_name(target.name + '.restoring')
    if pending.exists() or pending.is_symlink(): raise ValueError('Preserve the earlier physical restore attempt')
    total, digest, payload_position = 0, hashlib.sha256(), 0
    try:
        with tempfile.TemporaryDirectory(prefix='slm-xor-anchor-', dir=scratch) as temporary:
            anchor = Path(temporary) / 'anchor.pt'
            restore_anchor(manifest['compressed_anchor']['archive'], anchor)
            if sha(anchor) != manifest['compressed_anchor']['original_sha256']: raise ValueError('Every restored anchor byte must agree')
            with anchor.open('rb') as old, payload.open('rb') as stored, pending.open('xb') as output:
                for part in manifest['segments']:
                    if (part.get('kind') not in ['raw','xor'] or part.get('target_offset') != total
                            or part.get('payload_offset') != payload_position or type(part.get('length')) is not int
                            or part['length'] < 1 or type(part.get('payload_bytes')) is not int or part['payload_bytes'] < 1
                            or part['kind']=='xor' and (type(part.get('anchor_offset')) is not int or part['anchor_offset'] < 0)):
                        raise ValueError('The exact complete target and compressed segment order must be contiguous')
                    stored.seek(payload_position); count = 0
                    if part['kind']=='xor': old.seek(part['anchor_offset'])
                    class Limited:
                        def __init__(self): self.remaining = part['payload_bytes']
                        def read(self, size):
                            if not self.remaining: return b''
                            block = stored.read(min(size, self.remaining))
                            if not block: raise ValueError('A compressed target segment is truncated')
                            self.remaining -= len(block)
                            return block
                    limited = Limited()
                    with zstd.ZstdDecompressor().stream_reader(limited,read_size=64*1024,read_across_frames=True,closefd=False) as decoder:
                        while piece := decoder.read(BLOCK):
                            if count + len(piece) > part['length']: raise ValueError('A restored segment exceeds its exact original byte count')
                            if part['kind']=='xor':
                                previous = old.read(len(piece))
                                if len(previous) != len(piece): raise ValueError('The complete anchor storage is truncated')
                                piece = np.bitwise_xor(np.frombuffer(piece,dtype=np.uint8),np.frombuffer(previous,dtype=np.uint8)).tobytes()
                            output.write(piece); digest.update(piece)
                            count += len(piece); total += len(piece)
                    if limited.remaining or count != part['length']:
                        raise ValueError('The original complete compressed segment and exact decoded length must agree')
                    payload_position += part['payload_bytes']
                output.flush(); os.fsync(output.fileno())
        if (total != manifest['original_bytes'] or payload_position != payload.stat().st_size
                or digest.hexdigest() != manifest['original_sha256'] or sha(pending) != manifest['original_sha256']):
            raise ValueError('Every independently physically restored original target byte must match its exact whole-file SHA256')
        os.link(pending, target); pending.unlink()
        return dict(physically_restored_path=str(target), actual_restored_bytes=total,
            actual_restored_sha256=digest.hexdigest(), independent_physical_file_rehash_passed=True,
            retained_compressed_anchor_also_physically_restored=True)
    finally: pending.unlink(missing_ok=True)


def archive(source, anchor_archive, destination, scratch, *, expected_sha256, maximum_payload_bytes,
            minimum_free_bytes, minimum_reclaimed_bytes, guard, progress=None, retire_original=False):
    source, anchor_archive, destination, scratch = [Path(p).absolute() for p in [source,anchor_archive,destination,scratch]]
    if (not source.is_file() or source.is_symlink() or destination.exists() or destination.is_symlink()
            or not scratch.is_dir() or scratch.stat().st_dev == source.stat().st_dev or not callable(guard)
            or destination.is_relative_to(anchor_archive) or anchor_archive.is_relative_to(destination)
            or any(type(n) is not int or n < 1 for n in [maximum_payload_bytes,minimum_reclaimed_bytes])
            or type(minimum_free_bytes) is not int or minimum_free_bytes < 0):
        raise ValueError('An inactive original, retained anchor, fresh archive and independent physical scratch filesystem are required')
    guard(); original_state = state(source); original_bytes = original_state['bytes']
    spec = anchor_spec(anchor_archive)
    if original_state['links'] != 1 or sha(source) != expected_sha256:
        raise ValueError('Never retire a shared or changed original checkpoint')
    if shutil.disk_usage(scratch).free < original_bytes + 2 * spec['original_bytes'] + 1024**3:
        raise ValueError('Both independent anchor restores and the complete target must fit on scratch')
    destination.mkdir(); payload = destination / 'payload.zst'
    total, compressed_bytes, digest, actual_parts = 0, 0, hashlib.sha256(), []
    class Bounded:
        def __init__(self, stream): self.stream = stream
        def write(self, block):
            nonlocal compressed_bytes
            if (compressed_bytes + len(block) > maximum_payload_bytes
                    or shutil.disk_usage(destination).free - len(block) < minimum_free_bytes):
                raise ValueError('The complete exact XOR payload must fit above the original writer reserve')
            n = self.stream.write(block); compressed_bytes += n; return n
        def flush(self): self.stream.flush()
    with tempfile.TemporaryDirectory(prefix='slm-xor-encode-anchor-',dir=scratch) as temporary:
        anchor = Path(temporary) / 'anchor.pt'; restore_anchor(anchor_archive, anchor)
        parts = segments(source, anchor)
        with source.open('rb') as original, anchor.open('rb') as old, payload.open('xb') as stored:
            bounded = Bounded(stored)
            for part in parts:
                guard(); original.seek(part['target_offset']); start = compressed_bytes; remaining = part['length']
                if part['kind']=='xor': old.seek(part['anchor_offset'])
                with zstd.ZstdCompressor(level=1,threads=0,write_checksum=True).stream_writer(bounded,write_size=BLOCK,closefd=False) as writer:
                    while remaining:
                        guard(); raw = original.read(min(BLOCK,remaining))
                        if not raw: raise ValueError('The original complete target was truncated')
                        remaining -= len(raw); total += len(raw); digest.update(raw)
                        if part['kind']=='xor':
                            previous = old.read(len(raw))
                            if len(previous)!=len(raw): raise ValueError('The original complete anchor was truncated')
                            raw=np.bitwise_xor(np.frombuffer(raw,dtype=np.uint8),np.frombuffer(previous,dtype=np.uint8)).tobytes()
                        writer.write(raw)
                actual_parts.append(dict(part,payload_offset=start,payload_bytes=compressed_bytes-start))
                if progress is not None:
                    progress(dict(original_bytes_processed=total,original_bytes=original_bytes,compressed_bytes_written=compressed_bytes))
            stored.flush(); os.fsync(stored.fileno())
    guard()
    if (state(source)!=original_state or total!=original_bytes or digest.hexdigest()!=expected_sha256
            or sha(source)!=expected_sha256 or original_bytes-compressed_bytes<minimum_reclaimed_bytes
            or anchor_spec(anchor_archive)!=spec):
        raise ValueError('Complete unchanged originals and sufficient actual exact storage recovery are required; original retained')
    manifest=dict(format=FORMAT,complete_archive=True,original_path=str(source),original_state=original_state,
        original_bytes=original_bytes,original_sha256=expected_sha256,compressed_anchor=spec,segments=actual_parts,
        anchor=str(anchor_archive/'payload.zst'),anchor_sha256=spec['payload_sha256'],anchor_must_be_retained=True,
        source_files_sha256={str(anchor_archive/'manifest.json'):spec['manifest_sha256'],str(anchor_archive/'payload.zst'):spec['payload_sha256']},
        payload_bytes=compressed_bytes,payload_sha256=sha(payload),original_retired=False,physical_restore_checked=False,
        every_original_tensor_optimizer_RNG_and_metadata_byte_preserved=True,goal_achieved=False)
    atomic(destination/'manifest.json',manifest)
    restore_path = scratch / ('physically-restored-XOR-original-' + destination.name + '.pt')
    verification=restore(destination,restore_path,scratch)
    try:
        guard()
        if (state(source)!=original_state or sha(source)!=expected_sha256 or sha(payload)!=manifest['payload_sha256']
                or anchor_spec(anchor_archive)!=spec):
            raise ValueError('Every original, target payload and retained anchor must remain exact before retirement')
        manifest.update(physical_restore_checked=True,physical_restore=verification,
            reclaimed_allocated_bytes_if_retired=source.stat().st_blocks*512-payload.stat().st_blocks*512)
        atomic(destination/'manifest.json',manifest)
        if retire_original:
            source.unlink();manifest['original_retired']=True;atomic(destination/'manifest.json',manifest)
    finally: restore_path.unlink(missing_ok=True)
    return manifest
