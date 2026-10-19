"""Serial exact archival of explicitly inspected inactive phase snapshots.

Wait for the exact earlier archival owner and its terminal success/failure
evidence. Retain the original trained anchor and every checkpoint byte. All
raw retirements use the tested V4 full physical restore. No shared caches,
original workers, training inputs, scopes or numerical weights are changed.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
import traceback

from archive_inactive_code_capacity_CPU_v1 import metadata_bindings, no_readers_or_anchors
from slm_lab.exact_waiting_process_retirement_v1 import terminal

ROOT = Path(__file__).resolve().parents[1]
PREVIOUS = ROOT / 'reports/inactive-code-capacity-after1-exact-raw-XOR-v2'
PREVIOUS_OWNER = dict(pid=298358, ticks='61323700')
PHASE_OWNER = dict(pid=3096450, ticks='38665081')
STEPS = (34816, 32768, 30720, 26624, 24576, 22528, 20480, 18432, 16384, 12853, 12288, 10240, 8192)
RESERVE = 28 * 1024**3
MAXIMUM = 13 * 1024**3 // 4
MINIMUM_SAVING = 256 * 1024**2


def read(path): return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as stream: return hashlib.file_digest(stream, 'sha256').hexdigest()


def predecessor(folder, expected_owner):
    """A live birth stays waiting; terminal requires exact physical evidence."""
    folder = Path(folder); plan = read(folder / 'plan.json')
    if plan['owner'] != expected_owner:
        raise ValueError('The exact original archival owner must remain bound')
    if not terminal(expected_owner): return None
    if (folder / 'failure.json').exists():
        failure = read(folder / 'failure.json')
        if (failure.get('owner') != expected_owner or failure.get('original_retained') is not True
                or failure.get('discarded_unadmitted_duplicate', {}).get('complete_original_and_anchor_still_exact') is not True
                or failure['discarded_unadmitted_duplicate'].get('only_this_new_unadmitted_duplicate_discarded') is not True):
            raise ValueError('Failed predecessor must retain its exact original/anchor and close its unadmitted duplicate')
        return dict(terminal_failed_attempt_preserved=True, receipt=str(folder / 'failure.json'))
    if not (folder / 'complete.json').exists():
        raise ValueError('A missing process without a complete physical receipt is not archival success')
    complete = read(folder / 'complete.json'); actual = complete.get('actual_archive_manifest', {})
    if (complete.get('complete') is not True or complete.get('owner') != expected_owner
            or actual.get('original_retired') is not True or actual.get('physical_restore_checked') is not True
            or actual.get('physical_restore', {}).get('independent_physical_file_rehash_passed') is not True):
        raise ValueError('Completed predecessor must prove its actual complete physical restore and raw retirement')
    return dict(terminal_complete_physical_archive=True, receipt=str(folder / 'complete.json'))


def main(output):
    from slm_lab.local_source_closure_v2 import python_sources
    output = Path(output).absolute()
    if (output.exists() or output.resolve() != output or os.environ.get('CUDA_VISIBLE_DEVICES') != ''
            or os.nice(0) != 19 or any(os.environ.get(k) != '1' for k in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'])
            or 'torch' in sys.modules):
        raise ValueError('A fresh CUDA-hidden one-thread nice19 non-neural archival queue is required')
    os.sched_setscheduler(0, os.SCHED_IDLE, os.sched_param(0)); output.mkdir()
    package = Path('/opt/conda/lib/python3.11/site-packages/zstandard')
    codec_files = [package / '__init__.py', package / 'backend_c.cpython-311-x86_64-linux-gnu.so']
    old_bindings = read(ROOT / 'reports/inactive-code-capacity-after1-exact-zstd-v1/plan.json')['source_files_sha256']
    if any(old_bindings.get(str(p)) != sha(p) for p in codec_files):
        raise ValueError('The installed codec used in the complete physical checks must remain unchanged')
    if importlib.util.find_spec('zstandard') is None:
        spec = importlib.util.spec_from_file_location('zstandard', package / '__init__.py', submodule_search_locations=[str(package)])
        module = importlib.util.module_from_spec(spec); sys.modules['zstandard'] = module; spec.loader.exec_module(module)
    from slm_lab.exact_checkpoint_raw_anchor_v4 import archive, atomic, state
    owner = dict(pid=os.getpid(), ticks=Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19])
    tests = ROOT / 'reports/exact-checkpoint-raw-anchor-v4-tests.log'
    proof = ROOT / 'reports/phase-joint-v3'
    original = read(proof / 'plan.json'); done = read(proof / 'complete.json')
    if (not terminal(PHASE_OWNER) or {k: original['owner'][k] for k in PHASE_OWNER} != PHASE_OWNER
            or original['planned_steps'] != 35284 or read(proof / 'training-complete.json')['actual_updates'] != 35284
            or done.get('full_selected_expanded_family_source_epoch_complete') is not True or '4 passed' not in tests.read_text()):
        raise ValueError('The exact complete original phase and tested full physical codec must precede any archival work')
    paths = [PREVIOUS / 'plan.json', proof / 'plan.json', proof / 'training-complete.json', proof / 'complete.json', tests,
        *codec_files, *python_sources(ROOT, [Path(__file__), ROOT / 'scripts/restore_exact_checkpoint_raw_anchor_v4.py'])]
    files = {str(p): sha(p) for p in paths}
    signatures = {p: state(p) for p in files}
    def base_guard():
        if (os.sched_getscheduler(0) != os.SCHED_IDLE or os.nice(0) != 19 or not terminal(PHASE_OWNER)
                or 'torch' in sys.modules or any(state(p) != before for p, before in signatures.items())):
            raise ValueError('Every immutable archival source, completed phase and idle CPU-only context must remain')
    atomic(output / 'plan.json', dict(owner=owner, at=datetime.now(timezone.utc).isoformat(), source_files_sha256=files,
        predecessor=PREVIOUS_OWNER, phase_owner=PHASE_OWNER, exact_snapshot_steps=list(STEPS),
        retained_anchor=str(ROOT / 'checkpoints/phase-joint-v3/trained/model.pt'),
        maximum_each_payload_bytes=MAXIMUM, minimum_each_actual_reclaimed_bytes=MINIMUM_SAVING,
        unchanged_original_writer_reserve_bytes=RESERVE, actual_prior_immutable_source_bindings=0,
        actual_prior_live_readers_or_anchor_dependencies=0, full_physical_restore_required_for_each_raw_retirement=True,
        no_retry_of_failed_original_code_archive=True, GPU_contexts=0, neural_updates=0, goal_achieved=False))
    completed = []; started = time.monotonic()
    try:
        while True:
            base_guard(); ready = predecessor(PREVIOUS, PREVIOUS_OWNER)
            if ready is not None: break
            atomic(output / 'progress.json', dict(phase='waiting for exact prior archival owner', completed_snapshots=0))
            if time.monotonic()-started > 24*3600: raise ValueError('Exact predecessor observation lifetime expired; no restart')
            time.sleep(30)
        atomic(output / 'predecessor-terminal.json', dict(**ready, receipt_sha256=sha(ready['receipt'])))
        anchor = ROOT / 'checkpoints/phase-joint-v3/trained/model.pt'; anchor_state = state(anchor); anchor_sha = sha(anchor)
        atomic(output / 'retained-anchor.json', dict(path=str(anchor), sha256=anchor_sha, original_state=anchor_state,
            source_files_sha256={str(anchor): anchor_sha}, must_be_retained=True))
        for step in STEPS:
            base_guard(); source = ROOT / f'checkpoints/phase-joint-v3/step-{step}/model.pt'
            destination = source.with_name(source.name + '.exact-raw-XOR-v1')
            report = ROOT / f'reports/inactive-phase-step-{step}-exact-raw-XOR-v1'
            if (source.is_symlink() or source.stat().st_nlink != 1 or source.stat().st_size != 3937741157
                    or destination.exists() or report.exists()):
                raise ValueError('Every explicitly selected inactive snapshot needs its fresh physical archive and receipt')
            no_readers_or_anchors(source); mentions = metadata_bindings(source)
            before, digest = state(source), sha(source); report.mkdir(); last = [time.monotonic()]; encoded = [False]
            atomic(report / 'plan.json', dict(owner=owner, source_files_sha256=files, original_snapshot_step=step,
                source=str(source), original_sha256=digest, original_state=before, retained_raw_anchor=str(anchor),
                retained_raw_anchor_sha256=anchor_sha, archive=str(destination), inspected_original_metadata=mentions,
                actual_immutable_source_bindings=0, actual_live_readers=0, original_writer_reserve_bytes=RESERVE,
                complete_physical_restore_required=True, goal_achieved=False))
            def guard():
                base_guard()
                if state(source) != before or state(anchor) != anchor_state or time.monotonic()-started > 24*3600:
                    raise ValueError('The complete selected source and retained anchor must remain exact')
                if encoded[0] or time.monotonic()-last[0] >= 30:
                    no_readers_or_anchors(source); metadata_bindings(source); last[0] = time.monotonic()
            def progress(value):
                encoded[0] = value['original_bytes_processed'] == before['bytes']
                atomic(output / 'progress.json', dict(phase='encoding complete inactive native snapshot', step=step,
                    completed_snapshots=len(completed), actual_reclaimed_bytes=sum(r['reclaimed_bytes'] for r in completed), **value))
            guard()
            if shutil.disk_usage(ROOT).free < RESERVE + MAXIMUM or shutil.disk_usage('/tmp').free < RESERVE + before['bytes']:
                raise ValueError('Complete persistent payload and independent full physical restore must fit before encoding')
            try:
                with tempfile.TemporaryDirectory(prefix='slm-phase-exact-raw-XOR-', dir='/tmp') as scratch:
                    result = archive(source, anchor, destination, scratch, expected_sha256=digest, expected_anchor_sha256=anchor_sha,
                        maximum_payload_bytes=MAXIMUM, minimum_free_bytes=RESERVE, minimum_reclaimed_bytes=MINIMUM_SAVING,
                        guard=guard, progress=progress, retire_original=True)
            except BaseException as exc:
                payload = destination / 'payload.zst'; discarded = None
                if source.is_file() and state(source) == before and sha(source) == digest and sha(anchor) == anchor_sha and payload.is_file():
                    discarded = dict(path=str(payload), bytes=payload.stat().st_size, sha256=sha(payload), complete_original_and_anchor_still_exact=True)
                    payload.unlink()
                atomic(report / 'failure.json', dict(error=repr(exc), traceback=traceback.format_exc(), original_retained=source.exists(),
                    only_this_unadmitted_duplicate_discarded=discarded, automatic_retry=False, goal_achieved=False)); raise
            if not result['physical_restore_checked'] or not result['original_retired'] or source.exists():
                raise ValueError('Only complete physical verification and actual raw retirement count as recovery')
            receipt = dict(complete=True, actual_archive_manifest=result, actual_archive_manifest_sha256=sha(destination / 'manifest.json'),
                explicit_restore_command='.venv/bin/python scripts/restore_exact_checkpoint_raw_anchor_v4.py --archive '
                    +str(destination)+' --output '+str(source), shared_caches_and_active_inputs_unchanged=True, goal_achieved=False)
            atomic(report / 'complete.json', receipt)
            completed.append(dict(step=step, archive=str(destination), receipt=str(report / 'complete.json'),
                receipt_sha256=sha(report / 'complete.json'), reclaimed_bytes=result['reclaimed_allocated_bytes_if_retired']))
            atomic(output / 'completed-snapshots.json', dict(snapshots=completed, actual_reclaimed_bytes=sum(r['reclaimed_bytes'] for r in completed)))
        base_guard()
        atomic(output / 'complete.json', dict(complete=True, owner=owner, source_files_sha256=files, actual_snapshots=completed,
            actual_reclaimed_bytes=sum(r['reclaimed_bytes'] for r in completed), each_original_fully_physically_restored=True,
            retained_trained_anchor_unchanged=sha(anchor) == anchor_sha, model_quality_gain=False, goal_achieved=False))
    except BaseException as exc:
        atomic(output / 'failure.json', dict(owner=owner, error=repr(exc), traceback=traceback.format_exc(), completed_snapshots=completed,
            prior_failure_and_all_original_evidence_preserved=True, automatic_retry=False, goal_achieved=False)); raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('/tmp/slm-inactive-phase-snapshots-exact-raw-XOR-v1'))
    main(parser.parse_args().output)
