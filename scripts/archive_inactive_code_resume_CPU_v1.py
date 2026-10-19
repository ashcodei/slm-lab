"""Physically verify an exact inactive code-resume archive before retirement.

Only the closed original7558update resume snapshot is eligible. The trained
raw model remains its durable anchor. Fund this full8GiBmaximum alongside
the older serial3.25GiBmaximum above its original28GiBwriter reserve.
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
from slm_lab.local_source_closure_v2 import python_sources

ROOT = Path(__file__).resolve().parents[1]
OWNER = dict(pid=3798674, ticks='42848849')
SERIAL_OWNER = dict(pid=298748, ticks='61375675')
MAXIMUM = 8*1024**3
OTHER_MAXIMUM = 13*1024**3//4
RESERVE = 28*1024**3+OTHER_MAXIMUM


def read(path): return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as stream: return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    if (os.environ.get('CUDA_VISIBLE_DEVICES') != '' or os.nice(0) != 19
            or any(os.environ.get(k) != '1' for k in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'])):
        raise ValueError('Only a CUDA-hidden nice19 single-thread source-only archival process is eligible')
    os.sched_setscheduler(0, os.SCHED_IDLE, os.sched_param(0))
    package = Path('/opt/conda/lib/python3.11/site-packages/zstandard')
    codec_files = [package/'__init__.py', package/'backend_c.cpython-311-x86_64-linux-gnu.so']
    tested = read(ROOT/'reports/inactive-code-capacity-after1-exact-zstd-v1/plan.json')['source_files_sha256']
    if any(tested.get(str(p)) != sha(p) for p in codec_files):
        raise ValueError('Every exact previously physically tested installed codec byte must remain')
    if importlib.util.find_spec('zstandard') is None:
        spec = importlib.util.spec_from_file_location('zstandard', package/'__init__.py', submodule_search_locations=[str(package)])
        module = importlib.util.module_from_spec(spec); sys.modules['zstandard'] = module; spec.loader.exec_module(module)
    from slm_lab.exact_checkpoint_raw_anchor_v4 import archive, atomic, state
    source = ROOT/'checkpoints/code-first-continuation-v3/resume.pt'
    anchor = source.parent/'trained/model.pt'
    destination = source.with_name(source.name+'.exact-raw-XOR-v1')
    output = ROOT/'reports/inactive-code-final-resume-exact-raw-XOR-v1'
    original_plan = ROOT/'reports/code-first-continuation-v3/plan.json'
    original_done = original_plan.with_name('training-complete.json')
    serial_plan = Path('/tmp/slm-inactive-phase-snapshots-exact-raw-XOR-v1/plan.json')
    physical_proof = ROOT/'reports/inactive-phase-first-native-independent-restore-v1/complete.json'
    tests = ROOT/'reports/exact-checkpoint-raw-anchor-v4-tests.log'
    proof = read(physical_proof); serial = read(serial_plan)
    if (output.exists() or destination.exists() or not terminal(OWNER)
            or {k: read(original_plan)['owner'][k] for k in OWNER} != OWNER
            or read(original_done)['actual_updates'] != 7558
            or serial['owner'] != SERIAL_OWNER or serial['maximum_each_payload_bytes'] != OTHER_MAXIMUM
            or source.is_symlink() or anchor.is_symlink() or source.stat().st_nlink != 1
            or source.stat().st_size != 11807057403 or anchor.stat().st_size != 3937741157
            or proof.get('complete') is not True or not terminal(proof['owner'])
            or proof.get('all_original_bytes_recoverable_from_durable_archive_and_retained_anchor') is not True
            or '4 passed' not in tests.read_text()
            or shutil.disk_usage(ROOT).free < RESERVE+MAXIMUM+1024**3
            or shutil.disk_usage('/tmp').free < 40*1024**3+source.stat().st_size):
        raise ValueError('Exact closed originals, tested physical restoration and both complete concurrent payload budgets must fit')
    for p, digest in proof['source_files_sha256'].items():
        if p.endswith('.py') and sha(p) != digest:
            raise ValueError('Every actually independently tested full restore source must remain unchanged')
    no_readers_or_anchors(source); mentions = metadata_bindings(source)
    paths = [original_plan, original_done, serial_plan, physical_proof, tests, *codec_files,
        *python_sources(ROOT, [Path(__file__), ROOT/'scripts/restore_exact_checkpoint_raw_anchor_v4.py'])]
    files = {str(p): sha(p) for p in paths}; signatures = {p: state(p) for p in files}
    source_state, anchor_state = state(source), state(anchor)
    owner = dict(pid=os.getpid(), ticks=Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19])
    output.mkdir()
    atomic(output/'owner-plan.json', dict(owner=owner, at=datetime.now(timezone.utc).isoformat(), source_files_sha256=files,
        original_experiment_owner=OWNER, original_complete_source_updates=7558,
        source=str(source), retained_raw_anchor=str(anchor), retained_raw_anchor_must_not_be_retired=True,
        actual_inspected_original_metadata=mentions, actual_immutable_source_bindings=0, actual_live_readers=0,
        maximum_own_payload_bytes=MAXIMUM, other_serial_maximum_payload_bytes=OTHER_MAXIMUM,
        minimum_original_writer_free_bytes=28*1024**3, minimum_own_writer_free_bytes=RESERVE,
        both_complete_payloads_physically_funded=True,
        full_physical_restore_and_independent_rehash_required_before_retirement=True,
        shared_cache_and_active_inputs_unchanged=True, GPU_contexts=0, neural_updates=0, goal_achieved=False))
    atomic(output/'progress.json', dict(phase='hashing whole original resume and retained trained anchor', original_retired=False))
    original_sha, anchor_sha = sha(source), sha(anchor)
    atomic(output/'plan.json', dict(owner=owner, source_files_sha256=files, source=str(source), original_sha256=original_sha,
        original_state=source_state, retained_raw_anchor=str(anchor), retained_raw_anchor_sha256=anchor_sha,
        retained_raw_anchor_state=anchor_state, archive=str(destination), maximum_payload_bytes=MAXIMUM,
        minimum_own_writer_free_bytes=RESERVE, minimum_reclaimed_bytes=2*1024**3,
        only_this_attempt_unadmitted_duplicate_may_be_discarded_after_exact_original_and_anchor_rehash=True,
        full_physical_restore_and_independent_rehash_required_before_retirement=True, goal_achieved=False))
    started = time.monotonic(); last = [started]
    def guard():
        if (os.sched_getscheduler(0) != os.SCHED_IDLE or os.nice(0) != 19 or not terminal(OWNER)
                or state(source) != source_state or state(anchor) != anchor_state or time.monotonic()-started > 3*3600):
            raise ValueError('The exact inactive source, durable anchor and bounded idle archival context must remain')
        for p, expected in signatures.items():
            if state(p) != expected: raise ValueError('An exact source or original closed completion receipt changed')
        if time.monotonic()-last[0] >= 30:
            no_readers_or_anchors(source); metadata_bindings(source); last[0] = time.monotonic()
    def progress(row):
        atomic(output/'progress.json', dict(phase='encoding whole final inactive resume', **row,
            original_retired=False, all_original_bytes_retained=True))
    try:
        guard()
        with tempfile.TemporaryDirectory(prefix='slm-inactive-final-code-resume-', dir='/tmp') as scratch:
            result = archive(source, anchor, destination, Path(scratch), expected_sha256=original_sha,
                expected_anchor_sha256=anchor_sha, maximum_payload_bytes=MAXIMUM, minimum_free_bytes=RESERVE,
                minimum_reclaimed_bytes=2*1024**3, guard=guard, progress=progress, retire_original=True)
        if source.exists() or not result['original_retired'] or not result['physical_restore_checked']:
            raise ValueError('Only a complete independent full physical restoration may retire the original raw duplicate')
        atomic(output/'complete.json', dict(complete=True, owner=owner, at=datetime.now(timezone.utc).isoformat(),
            source_files_sha256=files, actual_archive_manifest=result, actual_archive_manifest_sha256=sha(destination/'manifest.json'),
            actual_reclaimed_allocated_bytes=result['reclaimed_allocated_bytes_if_retired'],
            explicit_restore_command='.venv/bin/python scripts/restore_exact_checkpoint_raw_anchor_v4.py --archive '
                +str(destination)+' --output '+str(source), retained_raw_anchor=str(anchor),
            all_original_tensor_optimizer_RNG_and_metadata_bytes_retained=True,
            shared_cache_and_active_inputs_unchanged=True, full_future_GPU_experiment_funding_still_requires_actual_free_check=True,
            model_quality_gain=False, goal_achieved=False))
        print(json.dumps(dict(complete=True, actual_reclaimed_bytes=result['reclaimed_allocated_bytes_if_retired'])), flush=True)
    except BaseException as error:
        payload = destination/'payload.zst'; discarded = None
        if (source.is_file() and state(source) == source_state and sha(source) == original_sha
                and state(anchor) == anchor_state and sha(anchor) == anchor_sha and payload.is_file()):
            discarded = dict(path=str(payload), bytes=payload.stat().st_size, sha256=sha(payload),
                only_this_new_unadmitted_duplicate_discarded=True, complete_original_and_anchor_still_exact=True)
            payload.unlink()
        atomic(output/'failure.json', dict(owner=owner, error=repr(error), traceback=traceback.format_exc(),
            original_retained=source.exists(), discarded_unadmitted_duplicate=discarded,
            all_original_and_partial_metadata_preserved=True, automatic_retry=False, goal_achieved=False))
        raise


if __name__ == '__main__': main()
