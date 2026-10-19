"""Archive one inactive initial native against an unchanged retained initial.

All sampled tensor differences were zero, but only a complete opaque encode
and physical restore can establish exact recoverability or actual savings.
Fund both this256MiBmaximum and the live serial queue's3.25GiBmaximum above
the original28GiBfloor. Preserve every original byte and all failed evidence.
"""
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
ORIGINAL_OWNER = dict(pid=3075160, ticks='36900588')
SERIAL_OWNER = dict(pid=298748, ticks='61375675')
MAXIMUM = 256 * 1024**2
OTHER_MAXIMUM = 13 * 1024**3 // 4
FLOOR = 28 * 1024**3


def read(p): return json.loads(Path(p).read_text())


def sha(p):
    with Path(p).open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()


def main():
    if (os.environ.get('CUDA_VISIBLE_DEVICES') != '' or os.nice(0) != 19
            or any(os.environ.get(k) != '1' for k in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'])):
        raise ValueError('An actual CUDA-hidden nice19 single-thread archival context is required')
    os.sched_setscheduler(0, os.SCHED_IDLE, os.sched_param(0))
    package = Path('/opt/conda/lib/python3.11/site-packages/zstandard')
    binary = package / 'backend_c.cpython-311-x86_64-linux-gnu.so'
    original_codec = read(ROOT / 'reports/inactive-code-capacity-after1-exact-zstd-v1/plan.json')['source_files_sha256']
    if any(original_codec.get(str(p)) != sha(p) for p in [package / '__init__.py', binary]):
        raise ValueError('The exact previously physically checked installed codec must remain unchanged')
    if importlib.util.find_spec('zstandard') is None:
        spec = importlib.util.spec_from_file_location('zstandard', package / '__init__.py', submodule_search_locations=[str(package)])
        module = importlib.util.module_from_spec(spec); sys.modules['zstandard'] = module; spec.loader.exec_module(module)
    from slm_lab.exact_checkpoint_raw_anchor_v4 import archive, atomic, state
    from slm_lab.local_source_closure_v2 import python_sources
    source = ROOT / 'checkpoints/fast-joint-v2/initial/model.pt'
    anchor = ROOT / 'checkpoints/protected-joint-v1/initial/model.pt'
    destination = source.with_name(source.name + '.exact-matching-initial-v1')
    output = ROOT / 'reports/inactive-fast-joint-initial-exact-v1'
    serial_plan = Path('/tmp/slm-inactive-phase-snapshots-exact-raw-XOR-v1/plan.json')
    original_plan = ROOT / 'reports/fast-joint-v2/plan.json'
    readback = ROOT / 'reports/inactive-phase-first-native-independent-restore-v1/complete.json'
    inspected = Path('/tmp/slm-additional-inactive-checkpoint-samples-v1.json')
    rows = [r for r in read(inspected) if r['source'] == str(source) and r['anchor'] == str(anchor)]
    complete = read(readback); serial = read(serial_plan)
    if (output.exists() or destination.exists() or not terminal(ORIGINAL_OWNER)
            or {k: read(original_plan)['owner'][k] for k in ORIGINAL_OWNER} != ORIGINAL_OWNER
            or serial['owner'] != SERIAL_OWNER or serial['maximum_each_payload_bytes'] != OTHER_MAXIMUM
            or len(rows) != 1 or rows[0]['sampled_bytes'] != 4 * 1024**2
            or rows[0]['sampled_zero_difference_bytes'] != rows[0]['sampled_bytes']
            or source.is_symlink() or anchor.is_symlink() or source.stat().st_nlink != 1
            or source.stat().st_size != 3937741157 or anchor.stat().st_size != 3937741157
            or complete.get('complete') is not True or not terminal(complete['owner'])
            or complete.get('all_original_bytes_recoverable_from_durable_archive_and_retained_anchor') is not True
            or shutil.disk_usage(ROOT).free < FLOOR + OTHER_MAXIMUM + MAXIMUM + 1024**3
            or shutil.disk_usage('/tmp').free < 40 * 1024**3 + source.stat().st_size):
        raise ValueError('Exact inactive originals, actual complete codec proof and both full concurrent physical budgets must fit')
    for p, h in complete['source_files_sha256'].items():
        if p.endswith('.py') and sha(p) != h:
            raise ValueError('Every actual independent full physical restore runtime byte must remain unchanged')
    no_readers_or_anchors(source); mentions = metadata_bindings(source)
    paths = [original_plan, serial_plan, readback, inspected, package / '__init__.py', binary,
        *python_sources(ROOT, [Path(__file__), ROOT / 'scripts/restore_exact_checkpoint_raw_anchor_v4.py'])]
    files = {str(p): sha(p) for p in paths}; signatures = {p: state(p) for p in files}
    output.mkdir(); owner = dict(pid=os.getpid(), ticks=Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19])
    atomic(output / 'owner-plan.json', dict(owner=owner, source_files_sha256=files, at=datetime.now(timezone.utc).isoformat(),
        source=str(source), retained_anchor=str(anchor), actual_immutable_source_bindings=0, actual_live_readers=0,
        original_writer_floor_bytes=FLOOR, other_live_serial_maximum_payload_bytes=OTHER_MAXIMUM,
        maximum_own_payload_bytes=MAXIMUM, both_complete_payloads_physically_funded=True,
        sampled_matches_are_not_complete_archive_proof=True, GPU_contexts=0, neural_updates=0, goal_achieved=False))
    atomic(output / 'progress.json', dict(phase='hashing complete original and retained anchor', original_retired=False))
    before, anchor_before = state(source), state(anchor); original_sha, anchor_sha = sha(source), sha(anchor)
    atomic(output / 'plan.json', dict(owner=owner, source_files_sha256=files, source=str(source), original_state=before,
        original_sha256=original_sha, retained_anchor=str(anchor), retained_anchor_sha256=anchor_sha, archive=str(destination),
        inspected_original_metadata=mentions, maximum_payload_bytes=MAXIMUM, minimum_archive_free_bytes=FLOOR + OTHER_MAXIMUM,
        minimum_reclaimed_bytes=3 * 1024**3, full_physical_restore_required_before_retirement=True, goal_achieved=False))
    started = time.monotonic(); last = [started]; encoded = [False]
    def guard():
        if (os.sched_getscheduler(0) != os.SCHED_IDLE or os.nice(0) != 19 or not terminal(ORIGINAL_OWNER)
                or state(source) != before or state(anchor) != anchor_before or 'torch' in sys.modules
                or time.monotonic()-started > 3*3600 or any(state(p) != s for p, s in signatures.items())):
            raise ValueError('Exact inactive original, retained anchor and complete idle archival/runtime context must remain')
        if encoded[0] or time.monotonic()-last[0] >= 30:
            no_readers_or_anchors(source); metadata_bindings(source); last[0] = time.monotonic()
    def progress(value):
        encoded[0] = value['original_bytes_processed'] == before['bytes']
        atomic(output / 'progress.json', dict(phase='complete exact archive of matching inactive initial', **value, original_retired=False))
    try:
        with tempfile.TemporaryDirectory(prefix='slm-matching-initial-exact-v1-', dir='/tmp') as scratch:
            result = archive(source, anchor, destination, scratch, expected_sha256=original_sha, expected_anchor_sha256=anchor_sha,
                maximum_payload_bytes=MAXIMUM, minimum_free_bytes=FLOOR + OTHER_MAXIMUM,
                minimum_reclaimed_bytes=3 * 1024**3, guard=guard, progress=progress, retire_original=True)
        if not result['physical_restore_checked'] or not result['original_retired'] or source.exists():
            raise ValueError('Only a fully physically verified archive and actual raw retirement reclaim storage')
        atomic(output / 'complete.json', dict(complete=True, owner=owner, actual_archive_manifest=result,
            actual_archive_manifest_sha256=sha(destination / 'manifest.json'), actual_reclaimed_allocated_bytes=result['reclaimed_allocated_bytes_if_retired'],
            explicit_restore_command='.venv/bin/python scripts/restore_exact_checkpoint_raw_anchor_v4.py --archive '
                +str(destination)+' --output '+str(source), retained_anchor_must_remain=str(anchor),
            all_original_bytes_retained=True, shared_caches_and_original_workers_unchanged=True, model_quality_gain=False, goal_achieved=False))
        print(json.dumps(dict(complete=True, reclaimed_allocated_bytes=result['reclaimed_allocated_bytes_if_retired'])), flush=True)
    except BaseException as exc:
        payload = destination / 'payload.zst'; discarded = None
        if source.exists() and state(source) == before and sha(source) == original_sha and sha(anchor) == anchor_sha and payload.is_file():
            discarded = dict(path=str(payload), bytes=payload.stat().st_size, sha256=sha(payload), complete_original_and_anchor_still_exact=True)
            payload.unlink()
        atomic(output / 'failure.json', dict(owner=owner, error=repr(exc), traceback=traceback.format_exc(), original_retained=source.exists(),
            only_this_new_unadmitted_duplicate_discarded=discarded, automatic_retry=False, goal_achieved=False)); raise


if __name__ == '__main__': main()
