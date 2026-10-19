"""One immutable CPU snapshot in flight, with atomic durable replacement.

Snapshot construction and copying happen on the caller thread at an optimizer
boundary. Only CPU serialization and filesystem I/O run in the writer thread.
The caller must collect every worker's RNG before submission and propagate
writer errors collectively before another distributed update.
"""
from concurrent.futures import ThreadPoolExecutor

import torch

from .codenet_checkpoint_v1 import atomic_checkpoint


def freeze_cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().to(device='cpu', copy=True)
    if isinstance(value, dict):
        return {key: freeze_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [freeze_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(freeze_cpu(item) for item in value)
    if value is None or isinstance(value, (bool, int, float, str, bytes)):
        return value
    raise TypeError('Unsupported mutable checkpoint value: ' + str(type(value)))


class AsyncCheckpoint:
    def __init__(self, *, writer=atomic_checkpoint):
        self.writer = writer
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='durable-checkpoint')
        self.pending = None
        self.closed = False

    def submit(self, path, bundle, maximum_bytes, minimum_free_bytes):
        if self.closed:
            raise ValueError('Checkpoint writer is closed')
        self.flush()
        # No alias to model, Adam, metrics, or private RNG may reach the thread.
        snapshot = freeze_cpu(bundle)
        self.pending = self.pool.submit(self.writer, path, snapshot, maximum_bytes, minimum_free_bytes)

    def check(self):
        if self.pending is not None and self.pending.done():
            self.pending.result()

    def flush(self):
        if self.pending is not None:
            self.pending.result()
            self.pending = None

    def close(self):
        try:
            self.flush()
        finally:
            self.closed = True
            self.pool.shutdown(wait=True)
