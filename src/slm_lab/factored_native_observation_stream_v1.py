"""Lossless framed native observations with one exact shared reference binding.

Only the known reference-binding subtree is factored. Every original expanded
JSON byte, action, probability, worker result and timing remains reconstructible
and hashed. Expanded logical and actual compressed physical quotas are separate.
This transport cannot admit data, GPU capacity, training or model quality.
"""
import copy
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import threading

from .bounded_disk_reserve_wait_v1 import wait_for_reserve

FORMAT = 'own-exact-factored-native-observation-framed-gzip-v1'
MAGIC = b'SLM-NATIVE-OBS-1\n'
FRAME = struct.Struct('>cQQ32s')
CHUNK = 64 * 1024
MAX_WIRE = 64 * 1024**2
MAX_EXPANDED = 128 * 1024**2


def canonical(value): return (json.dumps(value, ensure_ascii=True, allow_nan=False) + '\n').encode()
def digest(raw): return hashlib.sha256(raw).hexdigest()


def kernels(row):
    update = row.get('actual_complete_update')
    if not isinstance(update, dict) or not isinstance(update.get('actual_rank_observations'), list):
        raise ValueError('An original complete native update wrapper and all actual rank observations are required')
    observations = update['actual_rank_observations']
    if not 1 <= len(observations) <= 4 or any(not isinstance(r,dict) or 'kernel' not in r for r in observations):
        raise ValueError('Every original actual replica, including empty-rank tails, must remain')
    for row in observations:
        if row['kernel'] is not None:
            if not isinstance(row['kernel'],dict) or 'own_reference_binding' not in row['kernel']:
                raise ValueError('Every real native or CE-only kernel requires its exact own reference binding')
            yield row['kernel']


def factor(row, reference, reference_raw):
    wire = copy.deepcopy(row)
    marker = {'$own_native_reference_binding_sha256': digest(reference_raw)}
    for kernel in kernels(wire):
        if canonical(kernel['own_reference_binding']) != reference_raw:
            raise ValueError('Every factored reference must be byte-exact to the complete original binding')
        kernel['own_reference_binding'] = dict(marker)
    return wire


def expand(wire, reference, reference_sha256):
    row = copy.deepcopy(wire)
    marker = {'$own_native_reference_binding_sha256': reference_sha256}
    for kernel in kernels(row):
        if kernel['own_reference_binding'] != marker:
            raise ValueError('Every native reference marker must match the exact complete header binding')
        kernel['own_reference_binding'] = copy.deepcopy(reference)
    return row


class Writer:
    def __init__(self,path,*,reference_binding,root,maximum_logical_bytes,maximum_physical_bytes,
                 minimum_free_bytes,guard,maximum_wire_record_bytes=MAX_WIRE,maximum_expanded_record_bytes=MAX_EXPANDED):
        if (not isinstance(reference_binding,dict) or not reference_binding.get('sources')
                or not callable(guard) or type(minimum_free_bytes) is not int or minimum_free_bytes<0
                or any(type(n) is not int or n<1 for n in [maximum_logical_bytes,maximum_physical_bytes,
                    maximum_wire_record_bytes,maximum_expanded_record_bytes])):
            raise ValueError('An exact complete reference and separate positive logical/physical/record budgets are required')
        self.path,self.root,self.guard=Path(path),Path(root),guard
        self.thread=threading.get_ident();self.reference=copy.deepcopy(reference_binding)
        self.reference_raw=canonical(self.reference);self.reference_sha256=digest(self.reference_raw)
        self.logical_limit,self.physical_limit=maximum_logical_bytes,maximum_physical_bytes
        self.wire_limit,self.expanded_limit=maximum_wire_record_bytes,maximum_expanded_record_bytes
        self.reserve=minimum_free_bytes;self.records=self.update_bytes=self.logical_used=self.physical_used=0
        self.update_digest,self.physical_digest=hashlib.sha256(),hashlib.sha256();self.closed=False
        if len(self.reference_raw)>min(self.logical_limit,self.wire_limit,self.expanded_limit):
            raise ValueError('The entire unchanged reference header must fit before any stream is opened')
        guard()
        wait_for_reserve(self.root,minimum_bytes=self.reserve+len(self.reference_raw)+CHUNK,guard=guard)
        self.file=self.path.open('xb');owner=self
        class Physical:
            def write(self,raw):
                if (owner.physical_used+len(raw)>owner.physical_limit
                        or shutil.disk_usage(owner.root).free-len(raw)<owner.reserve):
                    raise ValueError('The complete actual compressed journal must remain inside its physical quota and reserve')
                n=owner.file.write(raw);owner.physical_used+=n;owner.physical_digest.update(raw[:n]);return n
            def flush(self):owner.file.flush()
        self.compressed=gzip.GzipFile(filename='',mode='wb',fileobj=Physical(),compresslevel=3,mtime=0)
        self.compressed.write(MAGIC)
        self._frame(b'R',self.reference_raw,self.reference_raw)
        self.logical_used=len(self.reference_raw)

    def _owned(self):
        if self.closed or threading.get_ident()!=self.thread:
            raise ValueError('Only the original live journal owner may append or close observations')

    def _frame(self,kind,wire,expanded):
        if len(wire)>self.wire_limit or len(expanded)>self.expanded_limit:
            raise ValueError('A complete unchanged native record exceeds its explicit framing capacity')
        self.compressed.write(FRAME.pack(kind,len(wire),len(expanded),hashlib.sha256(expanded).digest()))
        for start in range(0,len(wire),CHUNK):self.compressed.write(wire[start:start+CHUNK])

    def append(self,row):
        self._owned();self.guard()
        original=canonical(row);wire=canonical(factor(row,self.reference,self.reference_raw))
        restored=canonical(expand(json.loads(wire),self.reference,self.reference_sha256))
        if restored!=original:raise ValueError('Factoring cannot change any original expanded JSON byte')
        if (self.logical_used+len(original)>self.logical_limit or len(original)>self.expanded_limit or len(wire)>self.wire_limit):
            raise ValueError('The full expanded observation quota and complete per-record capacities must remain funded')
        wait_for_reserve(self.root,minimum_bytes=self.reserve+len(wire)+CHUNK,guard=self.guard)
        self._frame(b'U',wire,original)
        self.logical_used+=len(original);self.update_bytes+=len(original);self.update_digest.update(original);self.records+=1
        self.guard()

    def fsync(self):
        self._owned();self.compressed.flush();self.file.flush();os.fsync(self.file.fileno())

    def close(self):
        self._owned();self.guard()
        end=canonical(dict(format=FORMAT,records=self.records,expanded_update_bytes=self.update_bytes,
            expanded_update_stream_sha256=self.update_digest.hexdigest(),logical_budget_used=self.logical_used,
            reference_binding_sha256=self.reference_sha256,source_or_GPU_training_admission_not_granted=True))
        self._frame(b'E',end,end);self.compressed.close();self.file.flush();os.fsync(self.file.fileno());self.file.close();self.closed=True

    def receipt(self):
        if not self.closed:raise ValueError('Only an entirely closed framed stream has a complete transport receipt')
        return dict(format=FORMAT,complete=True,records=self.records,expanded_update_bytes=self.update_bytes,
            expanded_update_stream_sha256=self.update_digest.hexdigest(),logical_budget_used=self.logical_used,
            maximum_logical_bytes=self.logical_limit,actual_compressed_bytes=self.physical_used,
            compressed_sha256=self.physical_digest.hexdigest(),maximum_physical_bytes=self.physical_limit,
            reference_binding_sha256=self.reference_sha256,all_original_expanded_JSON_bytes_preserved=True,
            source_or_GPU_training_admission_not_granted=True,goal_achieved=False)


def exact(stream,n):
    result=bytearray()
    while len(result)<n:
        piece=stream.read(min(CHUNK,n-len(result)))
        if not piece:raise ValueError('A complete framed native observation is truncated')
        result.extend(piece)
    return bytes(result)


class Reader:
    def __init__(self,path,*,maximum_logical_bytes,maximum_wire_record_bytes=MAX_WIRE,
                 maximum_expanded_record_bytes=MAX_EXPANDED,expected_reference_sha256=None):
        if any(type(n) is not int or n<1 for n in [maximum_logical_bytes,maximum_wire_record_bytes,maximum_expanded_record_bytes]):
            raise ValueError('Explicit complete expanded and per-record read bounds are required')
        self.path=Path(path);self.logical_limit=maximum_logical_bytes;self.wire_limit=maximum_wire_record_bytes
        self.expanded_limit=maximum_expanded_record_bytes;self.expected_reference=expected_reference_sha256
        self.started=self.complete=False;self.records=self.update_bytes=self.logical_used=0
        self.update_digest=hashlib.sha256()

    def __iter__(self):
        if self.started:raise ValueError('Each complete independent framed readback uses a fresh reader')
        self.started=True
        reference=None;reference_sha=None
        with gzip.open(self.path,'rb') as stream:
            if exact(stream,len(MAGIC))!=MAGIC:raise ValueError('The exact native observation framing protocol is required')
            while True:
                kind,wire_size,expanded_size,expected=FRAME.unpack(exact(stream,FRAME.size))
                if not 0<wire_size<=self.wire_limit or not 0<expanded_size<=self.expanded_limit:
                    raise ValueError('The complete declared frame cannot exceed the actual bounded reader capacity')
                wire=exact(stream,wire_size)
                if kind==b'R':
                    if reference is not None or self.records:raise ValueError('Exactly one original reference header must precede every actual update')
                    reference=json.loads(wire);reference_sha=digest(wire)
                    if (not isinstance(reference,dict) or not reference.get('sources') or canonical(reference)!=wire
                            or self.expected_reference is not None and self.expected_reference!=reference_sha):
                        raise ValueError('The exact full original reference header does not match')
                    expanded=wire
                elif kind==b'U':
                    if reference is None:raise ValueError('No actual update may precede its complete reference header')
                    row=expand(json.loads(wire),reference,reference_sha);expanded=canonical(row)
                elif kind==b'E':
                    end=json.loads(wire);expanded=wire
                    wanted=dict(format=FORMAT,records=self.records,expanded_update_bytes=self.update_bytes,
                        expanded_update_stream_sha256=self.update_digest.hexdigest(),logical_budget_used=self.logical_used,
                        reference_binding_sha256=reference_sha,source_or_GPU_training_admission_not_granted=True)
                    if reference is None or end!=wanted:raise ValueError('Every actual original record and complete expanded-stream digest must reconcile')
                else:raise ValueError('An unknown observation frame cannot be interpreted or silently skipped')
                if len(expanded)!=expanded_size or hashlib.sha256(expanded).digest()!=expected:
                    raise ValueError('Every original complete expanded JSON byte must match its actual record hash and length')
                if kind==b'E':
                    if stream.read(1):raise ValueError('No hidden or trailing native observations may follow the complete footer')
                    self.complete=True;return
                self.logical_used+=len(expanded)
                if self.logical_used>self.logical_limit:raise ValueError('The original full expanded observation quota cannot be bypassed by factoring')
                if kind==b'U':
                    self.records+=1;self.update_bytes+=len(expanded);self.update_digest.update(expanded)
                    yield expanded,row
