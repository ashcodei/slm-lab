"""Use recovered native metadata with the unchanged four-stage handoff.

Only the native source-preparation location and parent selector change.
Original outputs, full scopes, funding checks, numerical workers and owned
GPU lifecycles remain identical. Bindings are local to this observer process.
"""
from contextlib import contextmanager

from . import full_experiment_launch_queue_v1 as original
from .metadata_observer_recovery_v1 import INDEXED_OUT

STAGES = (*original.STAGES[:3], dict(original.STAGES[3], preparation=str(INDEXED_OUT), driver='train_native_code_suffix_aligned_GPU_v5.py'))


@contextmanager
def binding():
    previous = original.STAGES
    if ([{k: v for k, v in row.items() if k not in ['preparation', 'driver']} for row in previous]
            != [{k: v for k, v in row.items() if k not in ['preparation', 'driver']} for row in STAGES]):
        raise ValueError('Every original experiment output,count and full assessment scope must remain')
    original.STAGES = STAGES
    try:
        yield
    finally:
        original.STAGES = previous


def run(*args, **kwargs):
    with binding():
        return original.run(*args, **kwargs)
