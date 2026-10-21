"""Bound immutable CPU snapshots while preserving repeated tensor aliases.

Only serialization, fsync and checksum computation run in the writer thread.
All tensor copies happen before submit returns. The tensor-byte bound excludes
Python metadata overhead; production host/RSS capacity still needs measurement.
"""
from pathlib import Path

import torch

from .async_durable_checkpoint_v1 import AsyncCheckpoint
from .codenet_checkpoint_v1 import atomic_checkpoint
from .corpus import file_hash


def freeze_cpu_bounded(value, maximum_tensor_bytes):
    if type(maximum_tensor_bytes) is not int or maximum_tensor_bytes < 1:
        raise ValueError('A positive explicit private snapshot tensor budget is required')
    seen = set(); active = set(); size = 0

    def inspect(item):
        nonlocal size
        if item is None or isinstance(item, (bool, int, float, str, bytes)):
            return
        key = id(item)
        if key in active:
            raise ValueError('Recursive checkpoint containers are unsupported')
        if key in seen:
            return
        seen.add(key)
        if isinstance(item, torch.Tensor):
            if item.device.type != 'cpu' or item.layout != torch.strided or item.is_quantized:
                raise ValueError('Only already staged ordinary CPU tensors may reach this writer')
            size += item.numel()*item.element_size()
            if size > maximum_tensor_bytes:
                raise ValueError('Private snapshot tensor-byte budget exceeded before copying')
            return
        if not isinstance(item, (dict, list, tuple)):
            raise TypeError('Unsupported mutable checkpoint value: '+str(type(item)))
        active.add(key)
        if isinstance(item, dict):
            for name, child in item.items():
                if not isinstance(name, (str, int)):
                    raise TypeError('Checkpoint mapping keys must be immutable strings or integers')
                inspect(child)
        else:
            for child in item:
                inspect(child)
        active.remove(key)

    inspect(value)
    memo = {}

    def clone(item):
        if item is None or isinstance(item, (bool, int, float, str, bytes)):
            return item
        key = id(item)
        if key in memo:
            return memo[key]
        if isinstance(item, torch.Tensor):
            result = item.detach().clone(memory_format=torch.preserve_format)
        elif isinstance(item, dict):
            result = {name: clone(child) for name, child in item.items()}
        elif isinstance(item, list):
            result = [clone(child) for child in item]
        else:
            result = tuple(clone(child) for child in item)
        memo[key] = result
        return result

    return clone(value), size


class BoundedAsyncCheckpoint(AsyncCheckpoint):
    def __init__(self, *, maximum_snapshot_tensor_bytes, writer=atomic_checkpoint):
        if type(maximum_snapshot_tensor_bytes) is not int or maximum_snapshot_tensor_bytes < 1:
            raise ValueError('An explicit positive snapshot tensor budget is required')
        super().__init__(writer=writer)
        self.maximum_snapshot_tensor_bytes = maximum_snapshot_tensor_bytes
        self.last_commit = None
        self.last_snapshot_tensor_bytes = 0

    def _commit(self, path, snapshot, maximum_bytes, minimum_free_bytes):
        self.writer(path, snapshot, maximum_bytes, minimum_free_bytes)
        # No subsequent replacement is submitted until this checksum closes.
        path = Path(path)
        return dict(path=str(path.resolve()), sha256=file_hash(path), bytes=path.stat().st_size)

    def submit(self, path, bundle, maximum_bytes, minimum_free_bytes):
        if self.closed:
            raise ValueError('Checkpoint writer is closed')
        self.flush()
        snapshot, size = freeze_cpu_bounded(bundle, self.maximum_snapshot_tensor_bytes)
        self.last_snapshot_tensor_bytes = size
        self.pending = self.pool.submit(self._commit, path, snapshot, maximum_bytes, minimum_free_bytes)

    def poll(self, *, wait=False):
        if self.pending is None or not wait and not self.pending.done():
            return None
        result = self.pending.result()
        self.pending = None
        self.last_commit = result
        return result

    def check(self):
        return self.poll()

    def flush(self):
        return self.poll(wait=True)
