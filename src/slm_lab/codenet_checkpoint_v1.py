"""Exact per-worker RNG and atomic, capacity-bounded complete-code snapshots."""
import os
from pathlib import Path
import random
import shutil

import numpy as np
import torch
from torch import distributed as dist


def capture_rng(device):
    numpy_state = np.random.get_state()
    return {'python': random.getstate(), 'torch': torch.get_rng_state(),
            'numpy': {'kind': numpy_state[0], 'keys': numpy_state[1].tolist(), 'position': numpy_state[2],
                      'has_gauss': numpy_state[3], 'cached_gaussian': numpy_state[4]},
            'cuda': torch.cuda.get_rng_state(device) if torch.device(device).type == 'cuda' else None}


def restore_rng(state, device):
    if (state['cuda'] is None) != (torch.device(device).type != 'cuda'):
        raise ValueError('Resume must retain the original CPU/CUDA RNG contract')
    random.setstate(state['python'])
    torch.set_rng_state(state['torch'].cpu())
    n = state['numpy']
    np.random.set_state((n['kind'], np.asarray(n['keys'], dtype=np.uint32), n['position'],
                         n['has_gauss'], n['cached_gaussian']))
    if state['cuda'] is not None:
        torch.cuda.set_rng_state(state['cuda'].cpu(), device)


def collect_worker_rng(device, world):
    state = capture_rng(device)
    if world == 1:
        return [state]
    if not dist.is_initialized() or dist.get_world_size() != world:
        raise ValueError('Exact initialized worker population required for RNG collection')
    states = [None] * world
    dist.all_gather_object(states, state)
    return states


class _BoundedFile:
    def __init__(self, stream, maximum_bytes, minimum_free_bytes):
        self.stream, self.maximum_bytes, self.minimum_free_bytes = stream, maximum_bytes, minimum_free_bytes

    def write(self, data):
        if self.stream.tell() + len(data) > self.maximum_bytes:
            raise ValueError('Checkpoint would exceed its declared maximum bytes')
        if shutil.disk_usage(Path(self.stream.name).parent).free - len(data) < self.minimum_free_bytes:
            raise ValueError('Checkpoint would consume the persistent writers reserve')
        return self.stream.write(data)

    def flush(self):
        self.stream.flush()


def atomic_checkpoint(path, bundle, maximum_bytes, minimum_free_bytes):
    """Keep the previous resumable state intact until the complete new file is durable."""
    if (type(maximum_bytes) is not int or maximum_bytes < 1
        or type(minimum_free_bytes) is not int or minimum_free_bytes < 0):
        raise ValueError('Explicit positive snapshot budget and nonnegative reserve required')
    path = Path(path)
    pending = path.with_suffix(path.suffix + '.pending')
    if pending.exists():
        raise ValueError('Earlier incomplete snapshot requires inspection')
    if shutil.disk_usage(path.parent).free < maximum_bytes + minimum_free_bytes:
        raise ValueError('Insufficient free space for a complete atomic checkpoint above reserve')
    try:
        with pending.open('xb') as stream:
            torch.save(bundle, _BoundedFile(stream, maximum_bytes, minimum_free_bytes))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(pending, path)
        fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except Exception:
        # Only this call's unsaved temporary file is removed; the old checkpoint survives.
        pending.unlink(missing_ok=True)
        raise


def validate_resume(bundle, plan_sha256, world, planned_steps, allow_completed=False):
    if (bundle.get('training_kind') != 'complete_code_rehearsal_v1'
        or bundle.get('run_plan_sha256') != plan_sha256 or bundle.get('world_size') != world
        or bundle.get('planned_steps') != planned_steps or len(bundle.get('worker_rng_states', [])) != world
        or 'optimizer' not in bundle or type(bundle.get('steps')) is not int
        or not 0 < bundle['steps'] <= planned_steps
        or (bundle['steps'] == planned_steps and not allow_completed)):
        raise ValueError('Exact resumable complete-code plan, optimizer, budget and worker RNG states required')
    return bundle['steps']


def save_collective(path, make_bundle, device, world, rank, maximum_bytes, minimum_free_bytes):
    """All workers participate; rank-zero I/O failure reaches every worker."""
    states = collect_worker_rng(device, world)
    outcome = [None]
    if rank == 0:
        try:
            atomic_checkpoint(path, make_bundle(states), maximum_bytes, minimum_free_bytes)
        except Exception as error:
            outcome[0] = str(error)
    if world > 1:
        dist.broadcast_object_list(outcome, src=0)
    if outcome[0] is not None:
        raise ValueError('Collective snapshot failed: ' + outcome[0])


class FileBindings:
    def __init__(self, files_sha256):
        from .corpus import file_hash
        if not files_sha256:
            raise ValueError('Nonempty exact runtime/data binding required')
        self.files = {Path(name): digest for name, digest in files_sha256.items()}
        self.states = {}
        for path, digest in self.files.items():
            if file_hash(path) != digest:
                raise ValueError('Bound runtime/data file differs: ' + str(path))
            self.states[path] = self._state(path)

    @staticmethod
    def _state(path):
        s = path.stat()
        return s.st_ino, s.st_size, s.st_mtime_ns

    def check(self, full=False):
        from .corpus import file_hash
        for path, digest in self.files.items():
            # This filesystem can retain the same timestamp for rapid same-size writes.
            # Hash runtime/plan/report files every time; large immutable payloads also
            # receive a complete digest check at initialization and each snapshot.
            check_bytes = full or self.states[path][1] <= 8 * 1024**2
            if self._state(path) != self.states[path] or (check_bytes and file_hash(path) != digest):
                raise ValueError('Bound runtime/data changed during training: ' + str(path))
