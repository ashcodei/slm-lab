"""Bounded two-generation checkpoints and the actual collective training loop."""
from datetime import datetime, timezone
import os
from pathlib import Path
import time

import torch
from torch import distributed as dist

from .code_residual_collective_checkpoint_v2 import _digest, fingerprint, _same, _validated
from .corpus import file_hash
from .diverse_images import atomic_json


FORMAT = 'owned-code-residual-checkpoint-rotation-v4'
CHECKPOINT_FILES = {'resume.pt', 'previous.pt', 'pending.pt', 'inference.pt'}


def _report_budget(path, value, maximum_bytes):
    import json
    path = Path(path)
    if type(maximum_bytes) is not int or maximum_bytes < 1:
        raise ValueError('Positive aggregate report budget required')
    payload_size = len((json.dumps(value, indent=2) + '\n').encode())
    occupied = sum(item.stat().st_size for item in path.parent.iterdir()
        if item.is_file() and item.name not in CHECKPOINT_FILES and item != path)
    if occupied + payload_size > maximum_bytes:
        raise ValueError('Owned stage metadata would exceed its declared aggregate report budget')


def report_json(path, value, maximum_bytes):
    """Bound aggregate metadata separately from the declared model-state slots."""
    _report_budget(path, value, maximum_bytes)
    atomic_json(path, value)


def rank_zero(fn):
    outcome = [None]
    if dist.get_rank() == 0:
        try:
            outcome[0] = dict(ok=True, value=fn())
        except Exception as error:
            outcome[0] = dict(ok=False, error=type(error).__name__ + ': ' + str(error))
    dist.broadcast_object_list(outcome, src=0)
    if not outcome[0]['ok']:
        raise ValueError('Collective lifecycle I/O failed: ' + outcome[0]['error'])
    return outcome[0]['value']


def checkpoint_layout(output, plan_sha256, *, pending_allowed=False):
    """Rank-zero readback; never guess which unjournaled file should resume."""
    import json
    output = Path(output)
    if not _digest(plan_sha256):
        raise ValueError('Exact run plan identity required for checkpoint ownership')
    for name in ['pending.pt.pending', 'resume.pt.pending', 'previous.pt.pending', 'checkpoint-ledger.tmp']:
        if (output / name).exists():
            raise ValueError('Earlier incomplete checkpoint/journal needs inspection: ' + name)
    if not pending_allowed and (output / 'pending.pt').exists():
        raise ValueError('Earlier unpublished complete checkpoint needs explicit inspection')
    path = output / 'checkpoint-ledger.json'
    if not path.exists():
        if any((output / name).exists() for name in ('resume.pt', 'previous.pt')):
            raise ValueError('Existing unjournaled training state needs explicit inspection')
        return dict(format=FORMAT, run_plan_sha256=plan_sha256, latest=None, previous=None)
    ledger = json.loads(path.read_text())
    if (set(ledger) != {'format', 'run_plan_sha256', 'latest', 'previous'}
            or ledger['format'] != FORMAT or ledger['run_plan_sha256'] != plan_sha256):
        raise ValueError('Exact owned checkpoint journal/run plan required')
    for key, name in [('latest', 'resume.pt'), ('previous', 'previous.pt')]:
        record, target = ledger[key], output / name
        if record is None:
            if target.exists():
                raise ValueError('Unjournaled previous checkpoint requires inspection')
            continue
        if (set(record) != {'checkpoint', 'checkpoint_sha256', 'checkpoint_bytes', 'steps'}
                or Path(record['checkpoint']).resolve() != target.resolve() or type(record['steps']) is not int
                or record['steps'] < 1 or type(record['checkpoint_bytes']) is not int
                or record['checkpoint_bytes'] < 1 or not _digest(record['checkpoint_sha256'])
                or target.stat().st_size != record['checkpoint_bytes'] or file_hash(target) != record['checkpoint_sha256']):
            raise ValueError('Owned latest/previous complete state bytes or identity changed')
    if ledger['latest'] is None or (ledger['previous'] and ledger['previous']['steps'] >= ledger['latest']['steps']):
        raise ValueError('Strictly ordered complete latest/previous generations required')
    return ledger


def publish_pending(output, plan_sha256, receipt, maximum_report_bytes=512 * 1024**2):
    """Two atomic renames; crashes between operations require journal inspection."""
    output = Path(output)
    ledger = checkpoint_layout(output, plan_sha256, pending_allowed=True)
    pending, latest, previous = (output / name for name in ('pending.pt', 'resume.pt', 'previous.pt'))
    if (Path(receipt['checkpoint']).resolve() != pending.resolve() or type(receipt['steps']) is not int
            or receipt['steps'] < 1 or pending.stat().st_size != receipt['checkpoint_bytes']
            or file_hash(pending) != receipt['checkpoint_sha256']
            or (ledger['latest'] and receipt['steps'] <= ledger['latest']['steps'])):
        raise ValueError('Exact newer completed pending state required before publishing')
    old = ledger['latest']
    def record(path, value):
        return dict(checkpoint=str(path), checkpoint_sha256=value['checkpoint_sha256'],
            checkpoint_bytes=value['checkpoint_bytes'], steps=value['steps'])
    updated = dict(format=FORMAT, run_plan_sha256=plan_sha256, latest=record(latest, receipt),
        previous=record(previous, old) if old else None)
    journal = output / 'checkpoint-ledger.json'
    _report_budget(journal, updated, maximum_report_bytes)
    if old is not None:
        os.replace(latest, previous)
    os.replace(pending, latest)
    report_json(journal, updated, maximum_report_bytes)
    with journal.open('rb') as stream:
        os.fsync(stream.fileno())
    fd = os.open(output, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    return updated['latest']


def run_updates(kernel, output, *, maximum_checkpoint_bytes, minimum_free_bytes,
                checkpoint_every, log_every, until_step=None, maximum_report_bytes=512 * 1024**2):
    """The production loop; a bounded stop retains an explicitly incomplete stage."""
    output = Path(output).resolve()
    stop = kernel.steps if until_step is None else until_step
    def arguments():
        if (not dist.is_initialized() or dist.get_world_size() != 4
                or any(type(value) is not int or value < 1 for value in [maximum_checkpoint_bytes, checkpoint_every, log_every, maximum_report_bytes])
                or type(minimum_free_bytes) is not int or minimum_free_bytes < 0
                or type(stop) is not int or not 0 <= kernel.completed <= stop <= kernel.steps or stop < 1):
            raise ValueError('Exact four-worker complete-state storage/update/logging budget required')
        return fingerprint(dict(output=str(output), plan=kernel.sources['run_plan_sha256'],
            start=kernel.completed, stop=stop, planned=kernel.steps, maximum=maximum_checkpoint_bytes,
            reserve=minimum_free_bytes, maximum_report_bytes=maximum_report_bytes,
            checkpoint_every=checkpoint_every, log_every=log_every))
    _same(_validated(arguments))
    rank_zero(lambda: output.mkdir(parents=True, exist_ok=True))
    ledger = rank_zero(lambda: checkpoint_layout(output, kernel.sources['run_plan_sha256']))
    latest = ledger['latest']
    if (kernel.completed == 0 and latest is not None) or (kernel.completed > 0 and (latest is None or latest['steps'] != kernel.completed)):
        raise ValueError('Explicitly restored kernel and journal must identify the same complete state')
    for _ in range(kernel.completed, stop):
        if torch.device(kernel.device).type == 'cuda':
            torch.cuda.synchronize(kernel.device)
        tick = time.monotonic()
        metric = kernel.update()
        if torch.device(kernel.device).type == 'cuda':
            torch.cuda.synchronize(kernel.device)
        duration = torch.tensor(time.monotonic() - tick, dtype=torch.float64, device=kernel.device)
        dist.all_reduce(duration, op=dist.ReduceOp.MAX)
        seconds = float(duration)
        if kernel.completed == 1 or kernel.completed % log_every == 0 or kernel.completed == stop:
            def progress():
                report_json(output / 'progress.json', dict(status='training; capability unverified',
                    step=kernel.completed, planned_steps=kernel.steps, latest_metric=metric,
                    step_seconds=seconds, global_target_tokens_per_second=metric['total_target_tokens'] / seconds,
                    run_plan_sha256=kernel.sources['run_plan_sha256'], updated_at=datetime.now(timezone.utc).isoformat(),
                    candidate_promoted=False, goal_achieved=False), maximum_report_bytes)
            rank_zero(progress)
        if kernel.completed % checkpoint_every == 0 or kernel.completed == stop:
            rank_zero(lambda: checkpoint_layout(output, kernel.sources['run_plan_sha256']))
            receipt = kernel.snapshot(output / 'pending.pt', maximum_bytes=maximum_checkpoint_bytes,
                minimum_free_bytes=minimum_free_bytes)
            latest = rank_zero(lambda: publish_pending(output, kernel.sources['run_plan_sha256'], receipt, maximum_report_bytes))
    return dict(completed_steps=kernel.completed, planned_steps=kernel.steps,
        training_complete=kernel.completed == kernel.steps, latest_checkpoint=latest,
        full_update_history_retained=len(kernel.metrics) == kernel.completed,
        candidate_promoted=False, goal_achieved=False)
