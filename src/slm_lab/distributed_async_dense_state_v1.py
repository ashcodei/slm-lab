"""Collective background durability for the existing complete dense state.

Call check collectively before each optimizer update, save collectively at an
optimizer boundary, and drain/close collectively before exporting or resuming.
One private CPU snapshot is in flight. Full-size GPU/host capacity is external.
"""
from pathlib import Path

from torch import distributed as dist

from .async_durable_checkpoint_v2 import BoundedAsyncCheckpoint
from .codenet_checkpoint_v1 import atomic_checkpoint
from .distributed_dense_code_state_v1 import make
from .distributed_dense_code_step_v1 import agree, gather


class DistributedAsyncDenseState:
    def __init__(self, *, maximum_snapshot_tensor_bytes, object_group=None, writer=atomic_checkpoint):
        if not dist.is_initialized() or dist.get_world_size() != 4:
            raise ValueError('All four initialized original replicas are required')
        self.group = object_group
        budgets = gather(maximum_snapshot_tensor_bytes, self.group)
        if (any(type(value) is not int or value < 1 for value in budgets)
                or any(value != budgets[0] for value in budgets)):
            raise ValueError('All four replicas need the same positive private snapshot tensor budget')
        self.writer = BoundedAsyncCheckpoint(maximum_snapshot_tensor_bytes=maximum_snapshot_tensor_bytes,
                                            writer=writer) if dist.get_rank() == 0 else None
        self.inflight = None
        self.last_commit = None
        self.closed = False

    def _status(self, *, wait=False):
        error = None; receipt = None
        if self.writer is not None:
            try:
                result = self.writer.poll(wait=wait)
                if result is not None:
                    if self.inflight is None:
                        raise ValueError('A completed write lacks its original submitted source cursor')
                    receipt = dict(**result, **self.inflight)
            except Exception as exc:
                error = repr(exc)
        agree(error, self.group)
        receipt = gather(receipt, self.group)[0]
        if receipt is not None:
            self.last_commit = receipt
            self.inflight = None
        return receipt

    def check(self):
        if self.closed:
            raise ValueError('Collective checkpoint writer is closed')
        return self._status()

    def drain(self):
        if self.closed:
            raise ValueError('Collective checkpoint writer is closed')
        return self._status(wait=True)

    def save(self, model, optimizer, path, *, sources, completed, planned, metrics, device,
             maximum_file_bytes, minimum_free_bytes):
        self.drain()
        request = None; error = None
        try:
            request = dict(path=str(Path(path).resolve()), sources=sources, completed=completed, planned=planned,
                           maximum_file_bytes=maximum_file_bytes, minimum_free_bytes=minimum_free_bytes)
        except Exception as exc:
            error = repr(exc)
        agree(error, self.group)
        requests = gather(request, self.group)
        if any(other != requests[0] for other in requests):
            raise ValueError('Every rank must request the same complete source cursor and durable destination')
        state = make(model, optimizer, sources=sources, completed=completed, planned=planned,
                     metrics=metrics, device=device, object_group=self.group)
        self.inflight = dict(completed=completed, planned=planned, sources=dict(sources))
        error = None
        if self.writer is not None:
            try:
                self.writer.submit(path, state, maximum_file_bytes, minimum_free_bytes)
            except Exception as exc:
                error = repr(exc)
        agree(error, self.group)
        # Every GPU-to-CPU copy and private immutable clone has completed.
        # Training can now advance independently of serialization/fsync/hash.
        dist.barrier(group=self.group)

    def close(self):
        try:
            return self.drain()
        finally:
            error = None
            if self.writer is not None:
                try:
                    self.writer.close()
                except Exception as exc:
                    error = repr(exc)
            self.closed = True
            agree(error, self.group)
