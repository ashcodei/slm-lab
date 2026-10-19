"""Joint records using the proven allocated frame/failure storage components.

Only the immutable header codec changes; the original exact JSON/gzip frames,
receipt chains, physical reservations and independent typed failures remain.
The whole record is retained. Checkpoints may keep receipt envelopes only.
"""
import copy
import gzip
import json
from pathlib import Path
import shutil

from .indexed_native_history_v1 import (Writer as FramesWriter,Reader as FramesReader,
    checked_root,limits,canonical,HEADER_LIMIT,RECEIPT_LIMIT,exclusive,fsync_directory,
    sha,read_regular,decompress)
from .indexed_native_history_v2 import Writer as FailureWriter
from .parallel_function_repair_stage_v1 import value_digest

FORMAT='own-complete-joint-objective-indexed-history-v1'
PROTOCOL='own-complete-joint-records-one-exact-training-binding-v1'


class JointRecords:
    def __init__(self,reference):
        fields={'sources','model_metadata','own_text_reference','options','world_size','optimizer_settings_sha256'}
        if type(reference) is not dict or set(reference)!=fields or not reference['sources'] or type(reference['world_size']) is not int or not 1<=reference['world_size']<=4:
            raise ValueError('The entire actual joint source/model/reference/objective binding is required')
        self.reference=copy.deepcopy(reference);self.identity=value_digest(reference);self.original_bytes=canonical(reference)
        if value_digest(json.loads(self.original_bytes))!=self.identity:raise ValueError('The complete original training-binding types must remain')

    def check(self,reference):
        if value_digest(reference)!=self.identity or canonical(reference)!=self.original_bytes or canonical(self.reference)!=self.original_bytes:
            raise ValueError('Every immutable joint source/model/reference/objective value must remain exact')

    def pack(self,row):
        if type(row) is not dict or type(row.get('step')) is not int or row['step']<1:
            raise ValueError('One complete globally reduced original pre-Adam metric is required')
        if value_digest(json.loads(canonical(row)))!=value_digest(row):raise ValueError('Every original record type/value must survive JSON without conversion')
        return copy.deepcopy(row)

    def expand(self,row,*,share_reference=False):
        return self.pack(row)


class Writer(FailureWriter):
    def __init__(self,directory,*,reference_binding,starting_completed,budgets,guard):
        # Preserve the existing allocated success/failure methods verbatim.
        self.root=checked_root(directory);self.budgets=limits(budgets)
        if type(starting_completed) is not int or not 0<=starting_completed<self.budgets['maximum_records'] or not callable(guard):
            raise ValueError('An actual remaining joint source prefix and guard are required')
        self.codec=JointRecords(reference_binding);self.guard=guard;self.starting_completed=starting_completed
        self.records,self.ticket,self.failed=0,None,False
        header=dict(format=FORMAT,protocol=PROTOCOL,actual_directory=str(self.root),starting_completed=starting_completed,
            budgets=self.budgets,reference_binding=reference_binding,reference_typed_sha256=self.codec.identity)
        raw=canonical(header);payload=gzip.compress(raw,mtime=0)
        if (len(raw)>HEADER_LIMIT or len(payload)+RECEIPT_LIMIT>self.budgets['maximum_physical_bytes']
                or len(self.codec.original_bytes)>self.budgets['maximum_logical_bytes']):raise ValueError('The entire actual joint header must fit both original quotas')
        self.guard()
        if shutil.disk_usage(self.root.parent).free<self.budgets['minimum_free_bytes']+len(payload)+RECEIPT_LIMIT:
            raise ValueError('The complete actual joint header must fit above the other-writer reserve')
        self.root.mkdir(exist_ok=False);exclusive(self.root/'reference.json.gz',payload);fsync_directory(self.root.parent)
        self.header_sha256=sha(payload);self.last_receipt_sha256=self.header_sha256
        self.physical_bytes=len(payload);self.logical_bytes=len(self.codec.original_bytes)
        self.initial_configuration=value_digest(dict(root=str(self.root),starting_completed=self.starting_completed,
            budgets=self.budgets,header_sha256=self.header_sha256))
        self.failure_slot=self.failure_artifact=None;self.pending_model_commit=False;self.active_step=None
        self.guard()


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
