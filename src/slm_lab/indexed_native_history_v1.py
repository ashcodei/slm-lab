"""Immutable per-update native observations with bounded resident history.

One exact reference header and individually compressed compact records stay
on disk. Checkpoints may retain only small independently hashed receipts.
Reservations allocate real blocks before sampling; a failed/interrupted
attempt is preserved and never reopened for writing. This storage component
does not admit source data, neural updates or production GPU training.
"""
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil

from .bounded_disk_reserve_wait_v1 import wait_for_reserve
from .native_reference_records_v1 import NativeReferenceRecords, PROTOCOL, canonical
from .parallel_function_repair_stage_v1 import value_digest

FORMAT = 'own-indexed-exact-native-history-v1'
RECEIPT = 'own-indexed-exact-native-history-record-v1'
HEADER_LIMIT = 16 * 1024**2
RECEIPT_LIMIT = 16 * 1024
LIMIT_KEYS = {'maximum_records', 'maximum_compact_record_bytes', 'maximum_expanded_record_bytes',
    'maximum_logical_bytes', 'maximum_physical_bytes', 'minimum_free_bytes'}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def fsync_directory(directory):
    fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    try: os.fsync(fd)
    finally: os.close(fd)


def exclusive(path, raw):
    with path.open('xb') as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    fsync_directory(path.parent)


def checked_root(directory):
    root = Path(directory).absolute()
    if root.resolve() != root or root.is_symlink():
        raise ValueError('The actual immutable history directory cannot be a symlink')
    return root


def read_regular(path, maximum_bytes):
    if path.is_symlink() or not path.is_file() or path.stat().st_size > maximum_bytes:
        raise ValueError('A complete bounded regular history artifact is required')
    with path.open('rb') as stream:
        raw = stream.read(maximum_bytes + 1)
    if len(raw) > maximum_bytes:
        raise ValueError('The history artifact grew past its complete bound')
    return raw


def decompress(raw, maximum_bytes):
    import io
    with gzip.GzipFile(fileobj=io.BytesIO(raw), mode='rb') as stream:
        output = stream.read(maximum_bytes + 1)
        if len(output) > maximum_bytes or stream.read(1):
            raise ValueError('The entire expanded history frame exceeds its original bound')
    return output


def limits(value):
    if (not isinstance(value, dict) or set(value) != LIMIT_KEYS
            or any(type(v) is not int or v < (0 if k == 'minimum_free_bytes' else 1) for k, v in value.items())
            or value['maximum_compact_record_bytes'] > value['maximum_expanded_record_bytes']):
        raise ValueError('All explicit positive complete history limits and a nonnegative reserve are required')
    return dict(value)


class Writer:
    def __init__(self, directory, *, reference_binding, starting_completed, budgets, guard):
        self.root = checked_root(directory)
        self.budgets = limits(budgets)
        if (type(starting_completed) is not int or not 0 <= starting_completed < budgets['maximum_records']
                or not callable(guard)):
            raise ValueError('A complete original remaining source prefix and owning guard are required')
        self.codec = NativeReferenceRecords(reference_binding)
        self.guard = guard
        self.starting_completed = starting_completed
        self.records, self.ticket, self.failed = 0, None, False
        header = dict(format=FORMAT, protocol=PROTOCOL, actual_directory=str(self.root),
            starting_completed=starting_completed, budgets=self.budgets,
            reference_binding=reference_binding, reference_typed_sha256=self.codec.identity)
        raw = canonical(header)
        if value_digest(json.loads(raw)) != value_digest(header):
            raise ValueError('The complete original header value types must survive JSON exactly')
        if len(raw) > HEADER_LIMIT:
            raise ValueError('The whole original reference header exceeds its explicit header capacity')
        payload = gzip.compress(raw, mtime=0)
        if len(payload) + RECEIPT_LIMIT > budgets['maximum_physical_bytes'] or len(self.codec.original_bytes) > budgets['maximum_logical_bytes']:
            raise ValueError('The entire original header must fit both complete budgets')
        self.guard()
        if shutil.disk_usage(self.root.parent).free < budgets['minimum_free_bytes'] + len(payload) + RECEIPT_LIMIT:
            raise ValueError('The complete header must fit above the unchanged other-writer reserve')
        self.root.mkdir(exist_ok=False)
        exclusive(self.root / 'reference.json.gz', payload)
        fsync_directory(self.root.parent)
        self.header_sha256 = sha(payload)
        self.last_receipt_sha256 = self.header_sha256
        self.physical_bytes = len(payload)
        self.logical_bytes = len(self.codec.original_bytes)
        self.initial_configuration = value_digest(dict(root=str(self.root), starting_completed=self.starting_completed,
            budgets=self.budgets, header_sha256=self.header_sha256))
        self.guard()

    def check(self):
        self.guard()
        self.codec.check(self.codec.reference)
        if (value_digest(dict(root=str(self.root), starting_completed=self.starting_completed,
                budgets=self.budgets, header_sha256=self.header_sha256)) != self.initial_configuration
                or sha(read_regular(self.root / 'reference.json.gz', 2 * HEADER_LIMIT)) != self.header_sha256):
            raise ValueError('The whole original history header, prefix and budgets must remain unchanged')

    def reserve_next(self):
        """Allocate worst-case next-frame blocks before any neural sampling."""
        if self.failed or self.ticket is not None or self.starting_completed + self.records >= self.budgets['maximum_records']:
            raise ValueError('Exactly one healthy remaining history reservation is required')
        self.check()
        compact = self.budgets['maximum_compact_record_bytes']
        # Deliberately loose: enough for gzip bytes, receipt and directory
        # bookkeeping without relying on a hoped-for compression ratio.
        allocation = 2 * compact + 2 * RECEIPT_LIMIT + 64 * 1024
        if (self.physical_bytes + allocation > self.budgets['maximum_physical_bytes']
                or self.logical_bytes + self.budgets['maximum_expanded_record_bytes'] > self.budgets['maximum_logical_bytes']):
            raise ValueError('The complete next possible observation must fit both quotas before sampling')
        # Directory/inode bookkeeping needs free blocks beyond the blocks
        # actually consumed by fallocate; keep this headroom outside the file.
        wait_for_reserve(self.root, minimum_bytes=self.budgets['minimum_free_bytes'] + allocation + 64 * 1024,
            maximum_seconds=1800, interval_seconds=5, guard=self.guard)
        sequence = self.records + 1
        path = self.root / ('record-' + str(sequence).zfill(12) + '.reserved')
        fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            os.posix_fallocate(fd, 0, allocation)
            os.fsync(fd)
            allocated = os.fstat(fd).st_blocks * 512
            if allocated < allocation:
                raise ValueError('Sparse length does not reserve actual physical blocks')
        except BaseException:
            self.failed = True
            raise
        finally:
            os.close(fd)
        fsync_directory(self.root)
        self.ticket = dict(sequence=sequence, step=self.starting_completed + sequence,
            path=path, allocation_bytes=allocation, actual_allocated_bytes=allocated)
        try:
            wait_for_reserve(self.root, minimum_bytes=self.budgets['minimum_free_bytes'] + 64 * 1024,
                maximum_seconds=1800, interval_seconds=5, guard=self.guard)
            self.check()
        except BaseException:
            self.failed = True
            raise
        return {k: str(v) if isinstance(v, Path) else v for k, v in self.ticket.items()}

    def append(self, row):
        """Publish into the funded inode; any failure preserves this attempt."""
        if self.failed or self.ticket is None:
            raise ValueError('A fresh physically funded pre-sampling reservation is required')
        ticket = self.ticket
        try:
            self.check()
            if type(row.get('step')) is not int or row['step'] != ticket['step']:
                raise ValueError('The exact original next complete update must match its funded step')
            original = canonical(row)
            packed = self.codec.pack(row)
            compact = canonical(packed)
            if value_digest(json.loads(compact)) != value_digest(packed):
                raise ValueError('Every original observation value type must survive JSON exactly')
            if (len(original) > self.budgets['maximum_expanded_record_bytes']
                    or len(compact) > self.budgets['maximum_compact_record_bytes']):
                raise ValueError('Every actual original byte must fit the predeclared complete record bounds')
            if canonical(self.codec.expand(packed)) != original:
                raise ValueError('The compact observation must reconstruct every original byte exactly')
            payload = gzip.compress(compact, mtime=0)
            if len(payload) + 2 * RECEIPT_LIMIT + 64 * 1024 > ticket['allocation_bytes']:
                raise ValueError('The complete actual gzip frame exceeds its funded physical allocation')
            sequence = ticket['sequence']
            name = 'record-' + str(sequence).zfill(12)
            record_path, receipt_path = self.root / (name + '.json.gz'), self.root / (name + '.receipt.json')
            if record_path.exists() or receipt_path.exists():
                raise ValueError('An original history artifact can never be replaced')
            receipt = dict(format=RECEIPT, header_sha256=self.header_sha256, sequence=sequence, step=ticket['step'],
                previous_receipt_sha256=self.last_receipt_sha256,
                record=name + '.json.gz', compressed_bytes=len(payload), compressed_sha256=sha(payload),
                compact_bytes=len(compact), compact_sha256=sha(compact), compact_typed_sha256=value_digest(packed),
                original_expanded_bytes=len(original), original_expanded_sha256=sha(original))
            receipt_raw = canonical(receipt)
            if len(receipt_raw) > RECEIPT_LIMIT:
                raise ValueError('The entire original receipt must fit without truncation')
            # Consume the already allocated inode instead of deleting its
            # blocks and hoping a second allocation succeeds after sampling.
            with ticket['path'].open('r+b') as stream:
                stream.write(payload)
                stream.truncate(len(payload))
                stream.flush()
                os.fsync(stream.fileno())
            os.link(ticket['path'], record_path)
            fsync_directory(self.root)
            exclusive(receipt_path, receipt_raw)
            self.guard()
            ticket['path'].unlink()
            fsync_directory(self.root)
            self.records += 1
            self.logical_bytes += len(original)
            self.physical_bytes += len(payload) + len(receipt_raw)
            self.last_receipt_sha256 = sha(receipt_raw)
            self.ticket = None
            return receipt
        except BaseException:
            self.failed = True
            raise


class Reader:
    def __init__(self, directory, *, expected_header_sha256, maximum_compact_record_bytes,
                 maximum_expanded_record_bytes, guard):
        self.root = checked_root(directory)
        if (not callable(guard) or type(maximum_compact_record_bytes) is not int or maximum_compact_record_bytes < 1
                or type(maximum_expanded_record_bytes) is not int or maximum_expanded_record_bytes < 1):
            raise ValueError('Independent full reader bounds and an owning guard are required')
        self.guard = guard
        self.guard()
        payload = read_regular(self.root / 'reference.json.gz', 2 * HEADER_LIMIT)
        if sha(payload) != expected_header_sha256:
            raise ValueError('The independently retained complete reference-header SHA differs')
        raw = decompress(payload, HEADER_LIMIT)
        self.header = json.loads(raw)
        if canonical(self.header) != raw:
            raise ValueError('The full original header types, order and JSON bytes must remain')
        header = self.header
        self.budgets = limits(header['budgets'])
        self.codec = NativeReferenceRecords(header['reference_binding'])
        if (header['format'] != FORMAT or header['protocol'] != PROTOCOL or header['actual_directory'] != str(self.root)
                or header['reference_typed_sha256'] != self.codec.identity
                or type(header['starting_completed']) is not int
                or not 0 <= header['starting_completed'] < self.budgets['maximum_records']
                or self.budgets['maximum_compact_record_bytes'] > maximum_compact_record_bytes
                or self.budgets['maximum_expanded_record_bytes'] > maximum_expanded_record_bytes):
            raise ValueError('The exact original complete reference, source prefix and independently funded bounds are required')
        self.header_sha256 = expected_header_sha256
        self.header_physical_bytes = len(payload)
        self.guard()

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

    def records(self, receipts, *, share_reference=False):
        """Stream a checkpoint prefix; never construct a full resident history."""
        previous = self.header_sha256
        logical, physical = len(self.codec.original_bytes), self.header_physical_bytes
        for index, receipt in enumerate(receipts, 1):
            if receipt.get('sequence') != index or receipt.get('previous_receipt_sha256') != previous:
                raise ValueError('No original prefix record may be omitted, reordered or replaced')
            row = self.compact(receipt)
            raw_receipt = canonical(receipt)
            previous = sha(raw_receipt)
            logical += receipt['original_expanded_bytes']
            physical += receipt['compressed_bytes'] + len(raw_receipt)
            if logical > self.budgets['maximum_logical_bytes'] or physical > self.budgets['maximum_physical_bytes']:
                raise ValueError('The entire original logical and physical prefix quotas must remain')
            yield self.codec.expand(row, share_reference=share_reference)
