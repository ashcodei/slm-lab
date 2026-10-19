"""Exact same joint frames with reads bounded by actual file length too.

Preserve every original byte, quota, checksum and failure reservation. Avoid
allocating the maximum32MiB header allowance for each tiny immutable header.
Growth/shrink during one read rejects; no source/record clipping is allowed.
"""
import copy
import gzip
import io
import json
import os

from .joint_indexed_history_v1 import Writer as OriginalWriter,JointRecords,FORMAT,PROTOCOL
from .indexed_native_history_v1 import (Reader as FramesReader,checked_root,limits,canonical,HEADER_LIMIT,RECEIPT_LIMIT,
    sha,RECEIPT)
from .parallel_function_repair_stage_v1 import value_digest


def read_regular(path,maximum_bytes):
    if type(maximum_bytes) is not int or maximum_bytes<1 or path.is_symlink() or not path.is_file():
        raise ValueError('One bounded complete regular immutable history file is required')
    with path.open('rb') as stream:
        initial=os.fstat(stream.fileno());size=initial.st_size
        if size>maximum_bytes:raise ValueError('The complete actual history file exceeds its original byte bound')
        raw=stream.read(size+1)
        final=os.fstat(stream.fileno())
        if (len(raw)!=size or stream.read(1) or (initial.st_ino,initial.st_size,initial.st_mtime_ns)!=(final.st_ino,final.st_size,final.st_mtime_ns)):
            raise ValueError('A complete original history file changed during its bounded actual-length read')
    return raw


def decompress(raw,maximum_bytes):
    if type(maximum_bytes) is not int or maximum_bytes<1:raise ValueError('The complete original expanded byte bound is required')
    chunks=[];total=0
    with gzip.GzipFile(fileobj=io.BytesIO(raw),mode='rb') as stream:
        while True:
            chunk=stream.read(min(64*1024,maximum_bytes-total+1))
            if not chunk:break
            total+=len(chunk)
            if total>maximum_bytes:raise ValueError('The entire expanded history exceeds its original byte bound')
            chunks.append(chunk)
    return b''.join(chunks)


class Writer(OriginalWriter):
    def check(self):
        self.guard();self.codec.check(self.codec.reference)
        if (value_digest(dict(root=str(self.root),starting_completed=self.starting_completed,budgets=self.budgets,header_sha256=self.header_sha256))!=self.initial_configuration
                or sha(read_regular(self.root/'reference.json.gz',2*HEADER_LIMIT))!=self.header_sha256):
            raise ValueError('The whole original history header,prefix and budgets must remain unchanged')


class Reader(FramesReader):
    def __init__(self,directory,*,expected_header_sha256,maximum_compact_record_bytes,maximum_expanded_record_bytes,guard):
        self.root=checked_root(directory)
        if (not callable(guard) or type(maximum_compact_record_bytes) is not int or maximum_compact_record_bytes<1
                or type(maximum_expanded_record_bytes) is not int or maximum_expanded_record_bytes<1):raise ValueError('Independent complete joint reader bounds and guard are required')
        self.guard=guard;self.guard();payload=read_regular(self.root/'reference.json.gz',2*HEADER_LIMIT)
        if sha(payload)!=expected_header_sha256:raise ValueError('The independently retained actual joint header differs')
        raw=decompress(payload,HEADER_LIMIT);self.header=json.loads(raw)
        if canonical(self.header)!=raw:raise ValueError('Every original header byte/type must remain')
        header=self.header;self.budgets=limits(header['budgets']);self.codec=JointRecords(header['reference_binding'])
        if (header['format']!=FORMAT or header['protocol']!=PROTOCOL or header['actual_directory']!=str(self.root)
                or header['reference_typed_sha256']!=self.codec.identity or type(header['starting_completed']) is not int
                or not 0<=header['starting_completed']<self.budgets['maximum_records']
                or self.budgets['maximum_compact_record_bytes']>maximum_compact_record_bytes
                or self.budgets['maximum_expanded_record_bytes']>maximum_expanded_record_bytes):
            raise ValueError('The original complete joint reference/prefix and independent bounds must remain')
        self.header_sha256=expected_header_sha256;self.header_physical_bytes=len(payload);self.guard()

    def compact(self, receipt):
        self.guard()
        self.codec.check(self.header['reference_binding'])
        if sha(read_regular(self.root / 'reference.json.gz', 2 * HEADER_LIMIT)) != self.header_sha256:
            raise ValueError('The original independently retained history header changed during readback')
        sequence = receipt.get('sequence')
        if type(sequence) is not int or not 1 <= sequence <= self.budgets['maximum_records'] - self.header['starting_completed']:
            raise ValueError('A complete original receipt sequence is required')
        name = 'record-' + str(sequence).zfill(12)
        if (receipt.get('format') != RECEIPT or receipt.get('header_sha256') != self.header_sha256
                or type(receipt.get('step')) is not int or receipt['step'] != self.header['starting_completed'] + sequence
                or receipt.get('record') != name + '.json.gz'):
            raise ValueError('The exact original record location and source prefix must remain')
        raw_receipt = read_regular(self.root / (name + '.receipt.json'), RECEIPT_LIMIT)
        if raw_receipt != canonical(receipt):
            raise ValueError('Every independently retained receipt byte must match its durable original')
        payload = read_regular(self.root / receipt['record'], 2 * self.budgets['maximum_compact_record_bytes'] + 64 * 1024)
        if len(payload) != receipt['compressed_bytes'] or sha(payload) != receipt['compressed_sha256']:
            raise ValueError('Every original compressed observation byte must match before expansion')
        raw = decompress(payload, self.budgets['maximum_compact_record_bytes'])
        row = json.loads(raw)
        if (len(raw) != receipt['compact_bytes'] or sha(raw) != receipt['compact_sha256']
                or canonical(row) != raw or value_digest(row) != receipt['compact_typed_sha256']
                or type(row.get('step')) is not int or row['step'] != receipt['step']):
            raise ValueError('Every original compact observation type, position and byte must remain')
        expanded = canonical(self.codec.expand(row, share_reference=True))
        if (len(expanded) != receipt['original_expanded_bytes'] or sha(expanded) != receipt['original_expanded_sha256']
                or len(expanded) > self.budgets['maximum_expanded_record_bytes']):
            raise ValueError('Every full original reference/action/probability/verdict/timing byte must reconstruct')
        self.guard()
        return row
