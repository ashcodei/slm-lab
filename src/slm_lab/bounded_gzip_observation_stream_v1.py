"""Lossless byte-LF JSON observations with an aggregate raw-byte quota.

Reserve the next bounded record above the original other-writer floor. The
full original uncompressed quota still applies; compression never drops a
record, changes a value or relabels a partial source population complete.
"""
import gzip
import hashlib
import json
import os
from pathlib import Path

from .bounded_disk_reserve_wait_v1 import wait_for_reserve

MAX_LINE = 8 * 1024**2
NEXT_RECORD = 16 * 1024**2


def lines(path):
    path = Path(path)
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rb') as stream:
        for raw in stream:
            if not raw.endswith(b'\n') or len(raw) > MAX_LINE:
                raise ValueError('Every bounded original JSON observation requires its complete byte-LF record')
            yield raw, json.loads(raw)


class RawBudget:
    def __init__(self, maximum_bytes, *, retained_source_bytes=0):
        if (type(maximum_bytes) is not int or type(retained_source_bytes) is not int
                or not 0 <= retained_source_bytes < maximum_bytes):
            raise ValueError('The complete original raw quota and explicitly retained source bytes are required')
        self.maximum, self.used = maximum_bytes, retained_source_bytes


class Writer:
    def __init__(self, path, *, root, budget, reserve_bytes, guard, reserve=wait_for_reserve):
        if (type(budget) is not RawBudget or type(reserve_bytes) is not int or reserve_bytes < 0
                or not callable(guard) or not callable(reserve)):
            raise ValueError('A shared complete raw quota and explicit guarded other-writer reserve are required')
        self.path, self.root, self.budget = Path(path), Path(root), budget
        self.reserve_bytes, self.guard, self.reserve = reserve_bytes, guard, reserve
        self.file = self.path.open('xb')
        self.gzip = gzip.GzipFile(filename='', mode='wb', fileobj=self.file, compresslevel=3, mtime=0)
        self.records = self.raw_bytes = 0
        self.digest = hashlib.sha256()
        self.closed = False

    def append_raw(self, raw):
        if (self.closed or not isinstance(raw, bytes) or not raw.endswith(b'\n') or len(raw) > MAX_LINE
                or self.budget.used + len(raw) > self.budget.maximum):
            raise ValueError('Every complete record must fit the unchanged raw observation quota')
        json.loads(raw)
        self.reserve(self.root, minimum_bytes=self.reserve_bytes + NEXT_RECORD, guard=self.guard)
        self.gzip.write(raw)
        self.gzip.flush()
        self.file.flush()
        self.records += 1
        self.raw_bytes += len(raw)
        self.budget.used += len(raw)
        self.digest.update(raw)

    def append(self, value):
        self.append_raw((json.dumps(value, ensure_ascii=True, allow_nan=False, separators=(',', ':')) + '\n').encode())

    def fsync(self):
        self.gzip.flush()
        self.file.flush()
        os.fsync(self.file.fileno())

    def receipt(self):
        return dict(records=self.records, actual_uncompressed_bytes=self.raw_bytes,
            actual_uncompressed_sha256=self.digest.hexdigest(), actual_compressed_bytes=self.path.stat().st_size,
            aggregate_original_raw_quota=self.budget.maximum, aggregate_actual_raw_bytes=self.budget.used,
            original_other_writer_reserve_bytes=self.reserve_bytes, next_complete_record_reserve_bytes=NEXT_RECORD,
            complete=self.closed)

    def close(self):
        if not self.closed:
            self.gzip.close()
            self.file.flush()
            os.fsync(self.file.fileno())
            self.file.close()
            self.closed = True

    def __enter__(self): return self

    def __exit__(self, kind, value, trace): self.close()
