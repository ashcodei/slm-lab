"""Funded success records and an independent typed failure-evidence slot.

The unchanged V1 format still stores each exact complete observation. A
separate physically allocated slot survives source-guard or post-Adam errors,
including nonfinite scalar/tensor observations. Failure evidence certifies no
successful optimizer commit. This component cannot admit GPU production.
"""
import hashlib
import json
import os
from pathlib import Path
import struct

import torch

from .bounded_disk_reserve_wait_v1 import wait_for_reserve
from .indexed_native_history_v1 import (Writer as RecordsWriter, checked_root, read_regular,
    exclusive, fsync_directory, canonical, sha, RECEIPT_LIMIT)

PROTOCOL = 'own-indexed-native-history-independent-funded-failure-slot-v2'
FAILURE = 'own-native-complete-tensor-primitive-failure-v1'
RECEIPT = 'own-native-independent-failure-receipt-v1'


def observation_digest(value):
    """Hash original order, collection types, float bits and exact tensor data."""
    result = hashlib.sha256()
    def framed(tag, raw):
        result.update(tag + str(len(raw)).encode() + b':' + raw)
    def visit(item):
        if isinstance(item, torch.Tensor):
            data = item.detach().cpu().contiguous()
            framed(b'T', json.dumps([str(data.dtype), list(data.shape)]).encode())
            framed(b'B', data.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif type(item) is float:
            framed(b'F', struct.pack('>d', item))
        elif type(item) is bytes:
            framed(b'bytes', item)
        elif item is None or type(item) in [str, int, bool]:
            framed(type(item).__name__.encode(), json.dumps(item, ensure_ascii=True).encode())
        elif type(item) in [list, tuple]:
            framed(type(item).__name__.encode(), str(len(item)).encode())
            for child in item: visit(child)
        elif type(item) is dict:
            framed(b'D', str(len(item)).encode())
            for key, child in item.items(): visit(key); visit(child)
        else:
            raise ValueError('Only the complete original tensor/primitive failure domain may be serialized')
    visit(value)
    return result.hexdigest()


def freeze_observation(value):
    """Capture CPU values and separate original tensor-device/gradient metadata."""
    metadata = []
    def visit(item, path):
        if isinstance(item, torch.Tensor):
            metadata.append(dict(path=path, device=str(item.device), dtype=str(item.dtype),
                shape=list(item.shape), requires_grad=item.requires_grad))
            return item.detach().to(device='cpu', copy=True)
        if item is None or type(item) in [float, str, int, bool, bytes]: return item
        if type(item) in [list, tuple]:
            values = [visit(child, path + [dict(sequence_index=i)]) for i, child in enumerate(item)]
            return tuple(values) if type(item) is tuple else values
        if type(item) is dict:
            return {visit(k, path + [dict(key_position=i)]):
                visit(v, path + [dict(value_position=i)]) for i, (k, v) in enumerate(item.items())}
        raise ValueError('The complete original failure report cannot silently drop unsupported values')
    copied = visit(value, [])
    if observation_digest(copied) != observation_digest(value):
        raise ValueError('Every original failure bit and value type must survive the CPU snapshot')
    return copied, metadata


def file_sha(path):
    with path.open('rb') as stream: return hashlib.file_digest(stream, 'sha256').hexdigest()


def slot_state(path):
    if path.is_symlink() or not path.is_file(): raise ValueError('The original real allocated failure inode is required')
    stat = path.stat()
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_blocks, stat.st_mtime_ns)


class BoundedFailureFile:
    def __init__(self, stream, maximum): self.stream, self.maximum = stream, maximum
    def write(self, value):
        if self.stream.tell() + len(value) > self.maximum:
            raise ValueError('The complete actual failure exceeds its funded original bound')
        return self.stream.write(value)
    def flush(self): return self.stream.flush()


class Writer(RecordsWriter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.failure_slot = None
        self.failure_artifact = None
        self.pending_model_commit = False
        self.active_step = None

    def reserve_next(self):
        if (self.failed or self.ticket is not None or self.pending_model_commit
                or self.starting_completed + self.records >= self.budgets['maximum_records']):
            raise ValueError('Exactly one healthy remaining source step and completed prior model boundary are required')
        self.check()
        primary = 2 * self.budgets['maximum_compact_record_bytes'] + 2 * RECEIPT_LIMIT + 64 * 1024
        failure = 2 * self.budgets['maximum_expanded_record_bytes'] + 2 * RECEIPT_LIMIT + 64 * 1024
        if (self.physical_bytes + primary + failure > self.budgets['maximum_physical_bytes']
                or self.logical_bytes + 2 * self.budgets['maximum_expanded_record_bytes'] > self.budgets['maximum_logical_bytes']):
            raise ValueError('Both the complete next observation and independent failure evidence must fit before sampling')
        if self.failure_slot is not None and slot_state(self.failure_slot['path']) != self.failure_slot['state']:
            raise ValueError('The independent failure slot changed before sampling')
        needed = primary + (failure if self.failure_slot is None else 0)
        wait_for_reserve(self.root, minimum_bytes=self.budgets['minimum_free_bytes'] + needed + 128 * 1024,
            maximum_seconds=1800, interval_seconds=5, guard=self.guard)
        self.active_step = self.starting_completed + self.records + 1
        try:
            def allocate(path, size):
                fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600)
                try:
                    os.posix_fallocate(fd, 0, size)
                    os.fsync(fd)
                    if os.fstat(fd).st_blocks * 512 < size:
                        raise ValueError('The entire success/failure allocation must use actual physical blocks')
                finally: os.close(fd)
                fsync_directory(self.root)
            if self.failure_slot is None:
                path = self.root / 'independent-failure.reserved'
                allocate(path, failure)
                self.failure_slot = dict(path=path, allocation_bytes=failure, state=slot_state(path))
            path = self.root / ('record-' + str(self.records + 1).zfill(12) + '.reserved')
            allocate(path, primary)
            self.ticket = dict(sequence=self.records + 1, step=self.active_step, path=path,
                allocation_bytes=primary, actual_allocated_bytes=path.stat().st_blocks * 512)
            wait_for_reserve(self.root, minimum_bytes=self.budgets['minimum_free_bytes'] + 64 * 1024,
                maximum_seconds=1800, interval_seconds=5, guard=self.guard)
            self.check()
            return dict(step=self.active_step, success_allocated_bytes=self.ticket['actual_allocated_bytes'],
                failure_allocated_bytes=self.failure_slot['path'].stat().st_blocks * 512,
                retained_one_failure_slot_per_attempt=True)
        except BaseException:
            self.failed = True
            raise

    def append(self, row):
        if self.failure_slot is None or self.pending_model_commit:
            raise ValueError('The independent pre-sampling failure slot is required')
        receipt = super().append(row)
        self.pending_model_commit = True
        return receipt

    def acknowledge_success(self, step):
        """Call only after every actual peer confirms its complete Adam step."""
        if (self.failed or not self.pending_model_commit or type(step) is not int
                or step != self.active_step or self.ticket is not None
                or self.starting_completed + self.records != step):
            raise ValueError('The exact published observation and actual successful model boundary are required')
        self.check()
        if slot_state(self.failure_slot['path']) != self.failure_slot['state']:
            raise ValueError('The funded emergency evidence slot changed after the model update')
        self.pending_model_commit = False

    def preserve_failure(self, observation):
        """Use funded blocks even when the owning source guard now rejects.

        This deliberately does not call the failed data/source guard. It
        certifies only exact captured evidence, never a usable neural state.
        The independent inode cannot overwrite a published success record.
        """
        if self.failure_slot is None or self.failure_artifact is not None or self.active_step is None:
            raise ValueError('Exactly one original funded failure slot and attempted step are required')
        slot = self.failure_slot
        if slot_state(slot['path']) != slot['state']:
            raise ValueError('Preserve the foreign or already consumed failure artifact without replacing it')
        self.failed = True
        copied, metadata = freeze_observation(observation)
        payload = dict(format=FAILURE, protocol=PROTOCOL, attempted_step=self.active_step,
            expected_reference_header_sha256=self.header_sha256,
            original_observation=copied, original_tensor_metadata=metadata,
            successful_optimizer_commit_certified=False, production_training_admitted=False)
        expected = observation_digest(payload)
        limit = self.budgets['maximum_expanded_record_bytes']
        with slot['path'].open('r+b') as stream:
            torch.save(payload, BoundedFailureFile(stream, limit))
            size = stream.tell()
            stream.truncate(size)
            stream.flush()
            os.fsync(stream.fileno())
        actual = torch.load(slot['path'], map_location='cpu', weights_only=True, mmap=True)
        if observation_digest(actual) != expected:
            raise ValueError('Every original failure tensor, float bit, collection type and metadata must reload exactly')
        name = 'failure-step-' + str(self.active_step).zfill(12)
        target, receipt_path = self.root / (name + '.pt'), self.root / (name + '.receipt.json')
        receipt = dict(format=RECEIPT, protocol=PROTOCOL, attempted_step=self.active_step,
            expected_reference_header_sha256=self.header_sha256, file=target.name, actual_bytes=size,
            file_sha256=file_sha(slot['path']), original_typed_observation_sha256=expected,
            successful_optimizer_commit_certified=False, production_training_admitted=False)
        if target.exists() or receipt_path.exists(): raise ValueError('An original complete failure artifact must never be replaced')
        os.link(slot['path'], target)
        fsync_directory(self.root)
        exclusive(receipt_path, canonical(receipt))
        # Keep the original failure-slot link as well: both immutable names
        # point to the same blocks and preserve an interrupted attempt.
        self.failure_artifact = receipt
        self.logical_bytes += size
        self.physical_bytes += size + len(canonical(receipt))
        return receipt


def read_failure(directory, receipt, *, maximum_bytes):
    root = checked_root(directory)
    step = receipt.get('attempted_step')
    if type(maximum_bytes) is not int or maximum_bytes < 1 or type(step) is not int or step < 1:
        raise ValueError('Independent complete failure capacity and attempted source step are required')
    name = 'failure-step-' + str(step).zfill(12)
    if (receipt.get('format') != RECEIPT or receipt.get('protocol') != PROTOCOL
            or receipt.get('file') != name + '.pt' or type(receipt.get('actual_bytes')) is not int
            or not 0 < receipt['actual_bytes'] <= maximum_bytes
            or receipt.get('successful_optimizer_commit_certified') is not False
            or receipt.get('production_training_admitted') is not False):
        raise ValueError('An exact failure-only receipt cannot certify a completed model update')
    if read_regular(root / (name + '.receipt.json'), RECEIPT_LIMIT) != canonical(receipt):
        raise ValueError('The exact independently retained failure receipt must match')
    path = root / receipt['file']
    if path.is_symlink() or not path.is_file() or path.stat().st_size != receipt['actual_bytes'] or file_sha(path) != receipt['file_sha256']:
        raise ValueError('Every original bounded failure-payload byte must match before loading')
    payload = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    if (observation_digest(payload) != receipt['original_typed_observation_sha256']
            or payload['format'] != FAILURE or payload['protocol'] != PROTOCOL or payload['attempted_step'] != step
            or payload['expected_reference_header_sha256'] != receipt['expected_reference_header_sha256']
            or payload['successful_optimizer_commit_certified'] is not False or payload['production_training_admitted'] is not False):
        raise ValueError('Every original typed failure value and original boundary must reload exactly')
    return payload
