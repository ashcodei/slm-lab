"""Verify a terminal metadata-only failure before creating a new observer.

Never relabel an old owner successful, adopt a launched GPU experiment or
restart its process. Preserve its source bytes, plans, progress and traceback.
"""
import hashlib
import json
from pathlib import Path

from .exact_waiting_process_retirement_v1 import terminal

ORIGINAL_OUT = Path('/tmp/slm-native-full-recipe-recovery-CPU-v1')
INDEXED_OUT = Path('/tmp/slm-native-indexed-recipe-recovery-CPU-v1')
QUEUE_OUT = Path('/tmp/slm-full-owned-experiment-launch-queue-v2')
DISK_ERROR = 'ValueError: Original disk reserve remains unavailable after the bounded wait'


def read(path): return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def capture(folder, log, *, kind):
    folder, log = Path(folder).resolve(), Path(log).resolve()
    if kind not in ['serial', 'source']:
        raise ValueError('Only a metadata-only serial observer or source preparer can be recovered')
    plan = read(folder / 'plan.json')
    if not terminal(plan['owner']):
        raise ValueError('The exact original observer is still live;do not create a replacement')
    if ((folder / 'complete.json').exists() or (folder / 'ready.json').exists()
            or any(folder.glob('*-launched.json')) or any(folder.glob('*-launch-intent.json'))
            or not log.read_text().strip().endswith(DISK_ERROR)):
        raise ValueError('Only a preserved terminal metadata-reserve failure with no GPU launch may be recovered')
    progress = read(folder / 'progress.json')
    if kind == 'serial':
        if progress.get('launched_experiments') != 0 or progress.get('current_driver_launched') is not False:
            raise ValueError('A serial observer with actual launched work needs a different explicit recovery')
    elif any(plan.get(k) != 0 for k in ['models_constructed', 'GPU_contexts', 'production_updates']):
        raise ValueError('Source recovery cannot adopt a neural or GPUproducer')
    files = dict(plan['source_files_sha256'])
    for path, digest in files.items():
        if sha(path) != digest:
            raise ValueError('Every original captured source must remain unchanged before observer recovery')
    for path in [folder / 'plan.json', folder / 'progress.json', log]:
        files[str(path)] = sha(path)
    return dict(kind=kind, exact_retired_owner=plan['owner'], terminal_verified=True,
        original_plan_sha256=sha(folder / 'plan.json'), original_log=str(log), original_log_sha256=sha(log),
        original_metadata_failure_preserved=True, original_owner_not_relabelled_successful=True,
        original_models_or_GPU_work_launched=False, source_files_sha256=files,
        new_versioned_metadata_owner_required=True, neural_retry_or_training_admission=False)
