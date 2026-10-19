"""Lossless bounded JSON records with independently readable gzip members.

The fixed index stores offset, compressed/raw lengths and each raw SHA256.
Random reads decompress one record, avoiding repeated whole-corpus scans.
Partial writers remain inspectable and can never grant source admission.
"""
import gzip
import hashlib
import json
import os
from pathlib import Path
import struct
import zlib

from .bounded_disk_reserve_wait_v1 import wait_for_reserve
from .bounded_gzip_observation_stream_v1 import MAX_LINE, NEXT_RECORD

FORMAT = 'own-indexed-bounded-byte-LF-gzip-records-v1'
ENTRY = struct.Struct('<QQQ32s')


class Writer:
    def __init__(self, pack, index, *, root, maximum_raw_bytes, minimum_free_bytes, guard, reserve=wait_for_reserve):
        if (type(maximum_raw_bytes) is not int or maximum_raw_bytes < 1
                or type(minimum_free_bytes) is not int or minimum_free_bytes < 0
                or not callable(guard) or not callable(reserve)):
            raise ValueError('Explicit whole raw quota, other-writer reserve and source guard are required')
        self.pack, self.index, self.root = Path(pack), Path(index), Path(root)
        if self.pack.exists() or self.index.exists(): raise ValueError('Preserve every previous complete or interrupted indexed stream')
        self.maximum, self.minimum, self.guard, self.reserve = maximum_raw_bytes, minimum_free_bytes, guard, reserve
        self.packed = self.pack.open('xb')
        self.indexed = self.index.open('xb')
        self.count = self.raw_bytes = self.compressed_bytes = 0
        self.digest = hashlib.sha256()
        self.closed = False

    def append(self, value):
        if self.closed: raise ValueError('An indexed stream cannot reopen')
        raw = (json.dumps(value, ensure_ascii=True, allow_nan=False, separators=(',', ':')) + '\n').encode()
        if len(raw) > MAX_LINE or self.raw_bytes + len(raw) > self.maximum:
            raise ValueError('Every complete record must fit the raw source quota without truncation')
        self.reserve(self.root, minimum_bytes=self.minimum + NEXT_RECORD, guard=self.guard)
        encoded = gzip.compress(raw, compresslevel=3, mtime=0)
        offset = self.compressed_bytes
        self.packed.write(encoded)
        self.indexed.write(ENTRY.pack(offset, len(encoded), len(raw), hashlib.sha256(raw).digest()))
        self.count += 1
        self.raw_bytes += len(raw)
        self.compressed_bytes += len(encoded)
        self.digest.update(raw)
        return self.count - 1

    def fsync(self):
        for stream in [self.packed, self.indexed]: stream.flush(); os.fsync(stream.fileno())

    def close(self):
        if not self.closed:
            self.fsync()
            self.packed.close()
            self.indexed.close()
            self.closed = True

    def receipt(self):
        return dict(format=FORMAT, complete=self.closed, records=self.count, raw_bytes=self.raw_bytes,
            compressed_bytes=self.compressed_bytes, index_bytes=ENTRY.size * self.count,
            complete_ordered_raw_sha256=self.digest.hexdigest(), maximum_raw_bytes=self.maximum,
            minimum_other_writer_free_bytes=self.minimum, next_record_reserve_bytes=NEXT_RECORD)

    def __enter__(self): return self
    def __exit__(self, kind, value, trace): self.close()


class Records:
    def __init__(self, pack, index, *, count):
        self.pack, self.index = Path(pack), Path(index)
        if (type(count) is not int or count < 1 or any(not p.is_file() or p.is_symlink() for p in [self.pack, self.index])
                or self.index.stat().st_size != count * ENTRY.size):
            raise ValueError('Exact complete regular indexed records and declared population required')
        self.index_bytes = self.index.read_bytes()
        self.count = count
        offset = 0
        for i in range(count):
            start, packed, raw, digest = ENTRY.unpack_from(self.index_bytes, i * ENTRY.size)
            if start != offset or not 18 <= packed <= MAX_LINE + 4096 or not 1 <= raw <= MAX_LINE:
                raise ValueError('Every indexed compressed member must retain a contiguous bounded complete original record')
            offset += packed
        if offset != self.pack.stat().st_size:
            raise ValueError('The complete index must cover every packed byte exactly once')
        self.stream = self.pack.open('rb')

    def __len__(self): return self.count

    def read(self, position):
        if type(position) is not int or not 0 <= position < self.count:
            raise ValueError('Only an exact original indexed record position is allowed')
        offset, packed_size, raw_size, digest = ENTRY.unpack_from(self.index_bytes, position * ENTRY.size)
        encoded = os.pread(self.stream.fileno(), packed_size, offset)
        codec = zlib.decompressobj(wbits=31)
        try: raw = codec.decompress(encoded, raw_size + 1)
        except zlib.error as exc: raise ValueError('Corrupt compressed source member') from exc
        if (len(encoded) != packed_size or len(raw) != raw_size or not codec.eof or codec.unused_data or codec.unconsumed_tail
                or not raw.endswith(b'\n') or hashlib.sha256(raw).digest() != digest):
            raise ValueError('The actual complete compressed member and every original decoded byte must match its bounded index')
        return raw, json.loads(raw)

    def close(self): self.stream.close()
    def __enter__(self): return self
    def __exit__(self, kind, value, trace): self.close()
