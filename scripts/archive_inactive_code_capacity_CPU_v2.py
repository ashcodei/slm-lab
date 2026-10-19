"""Retain an exact inactive code snapshot as a raw-anchor XOR archive.

The unsuccessful plain-compression attempt and all original evidence remain.
Only the one inactive raw duplicate may be retired, after a complete physical
restore from the durable delta and the unchanged retained initial checkpoint.
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

from archive_inactive_code_capacity_CPU_v1 import metadata_bindings, no_readers_or_anchors, OWNER, SIZES

ROOT = Path(__file__).resolve().parents[1]
RESERVE = 28 * 1024**3
MAXIMUM = 8 * 1024**3


def main():
    if (os.environ.get('CUDA_VISIBLE_DEVICES') != '' or os.nice(0) != 19
            or any(os.environ.get(k) != '1' for k in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'])):
        raise ValueError('Only a CUDA-hidden nice19 one-thread archival process is eligible')
    os.sched_setscheduler(0, os.SCHED_IDLE, os.sched_param(0))
    prior = ROOT / 'reports/inactive-code-capacity-after1-exact-zstd-v1'
    old = json.loads((prior / 'plan.json').read_text()); failed = json.loads((prior / 'failure.json').read_text())
    package = Path('/opt/conda/lib/python3.11/site-packages/zstandard')
    codec_files = [package/'__init__.py', package/'backend_c.cpython-311-x86_64-linux-gnu.so']
    for p in codec_files:
        with p.open('rb') as stream: digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        if old['source_files_sha256'].get(str(p)) != digest:
            raise ValueError('The exact previously physically tested installed codec must remain')
    if importlib.util.find_spec('zstandard') is None:
        spec = importlib.util.spec_from_file_location('zstandard', package/'__init__.py', submodule_search_locations=[str(package)])
        module = importlib.util.module_from_spec(spec); sys.modules['zstandard'] = module; spec.loader.exec_module(module)
    from slm_lab.exact_checkpoint_raw_anchor_v4 import archive, atomic, sha, state
    from slm_lab.exact_waiting_process_retirement_v1 import terminal
    from slm_lab.local_source_closure_v2 import python_sources
    source = ROOT / 'checkpoints/code-first-continuation-v3/capacity-after1.pt'
    anchor = source.parent / 'initial/model.pt'
    destination = source.with_name(source.name + '.exact-raw-XOR-v2')
    output = ROOT / 'reports/inactive-code-capacity-after1-exact-raw-XOR-v2'
    tests = ROOT / 'reports/exact-checkpoint-raw-anchor-v4-tests.log'
    original_plan = ROOT / 'reports/code-first-continuation-v3/plan.json'
    original_done = original_plan.with_name('training-complete.json')
    if (not terminal(old['owner']) or not terminal(OWNER) or old['source'] != str(source)
            or old['original_experiment_owner'] != OWNER or not failed.get('original_retained')
            or not failed.get('discarded_unadmitted_duplicate', {}).get('complete_original_still_exact')
            or state(source) != old['original_state'] or source.stat().st_size != SIZES['after1']
            or output.exists() or destination.exists() or '4 passed' not in tests.read_text()
            or {k: json.loads(original_plan.read_text())['owner'][k] for k in OWNER} != OWNER
            or json.loads(original_done.read_text())['actual_updates'] != 7558
            or shutil.disk_usage(ROOT).free < RESERVE + MAXIMUM
            or shutil.disk_usage('/tmp').free < RESERVE + source.stat().st_size):
        raise ValueError('Exact preserved originals, tested codec and full durable/independent physical budgets are required')
    no_readers_or_anchors(source); inspected = metadata_bindings(source)
    original_sha, original_state = old['original_sha256'], state(source)
    anchor_state = state(anchor); anchor_sha = sha(anchor)
    paths = [prior/'plan.json', prior/'failure.json', tests, original_plan, original_done,
        *codec_files, *python_sources(ROOT, [Path(__file__), ROOT/'scripts/restore_exact_checkpoint_raw_anchor_v4.py'])]
    files = {str(p): sha(p) for p in paths}
    signatures = {p: (Path(p).stat().st_dev, Path(p).stat().st_ino, Path(p).stat().st_size,
        Path(p).stat().st_mtime_ns, Path(p).stat().st_ctime_ns) for p in files}
    output.mkdir(); owner = dict(pid=os.getpid(), ticks=Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19])
    atomic(output/'plan.json', dict(owner=owner, at=datetime.now(timezone.utc).isoformat(), source_files_sha256=files,
        original_experiment_owner=OWNER, original_complete_source_updates=7558, source=str(source), original_sha256=original_sha,
        original_state=original_state, retained_raw_anchor=str(anchor), retained_raw_anchor_sha256=anchor_sha,
        retained_raw_anchor_state=anchor_state, retained_raw_anchor_must_not_be_retired=True, archive=str(destination),
        prior_failed_plain_archive=str(prior), actual_inspected_original_metadata=inspected,
        actual_immutable_source_bindings=0, actual_live_readers=0,
        maximum_payload_bytes=MAXIMUM, minimum_original_writer_free_bytes=RESERVE, minimum_reclaimed_bytes=1024**3,
        full_physical_restore_and_independent_rehash_required_before_retirement=True,
        only_this_attempt_unadmitted_duplicate_may_be_discarded_if_original_and_anchor_remain_exact=True,
        shared_cache_and_active_inputs_unchanged=True, GPU_contexts=0, new_neural_updates=0, goal_achieved=False))
    started = time.monotonic(); last_inspection = [started]
    def guard():
        if (os.sched_getscheduler(0) != os.SCHED_IDLE or os.nice(0) != 19 or not terminal(OWNER)
                or state(source) != original_state or state(anchor) != anchor_state
                or time.monotonic()-started > 3*3600):
            raise ValueError('Exact inactive original/retained anchor and idle bounded archival context must remain')
        for p, before in signatures.items():
            st = Path(p).stat()
            if before != (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns):
                raise ValueError('An exact tested archival source or original completed receipt changed')
        if time.monotonic()-last_inspection[0] >= 30:
            no_readers_or_anchors(source); metadata_bindings(source); last_inspection[0] = time.monotonic()
    def progress(value):
        atomic(output/'progress.json', dict(phase='complete exact raw-anchor XOR encoding', **value,
            source_retired=False, all_original_bytes_retained=True, goal_achieved=False))
    before_free = shutil.disk_usage(ROOT).free
    try:
        with tempfile.TemporaryDirectory(prefix='slm-inactive-raw-XOR-v2-', dir='/tmp') as temporary:
            result = archive(source, anchor, destination, Path(temporary), expected_sha256=original_sha,
                expected_anchor_sha256=anchor_sha, maximum_payload_bytes=MAXIMUM, minimum_free_bytes=RESERVE,
                minimum_reclaimed_bytes=1024**3, guard=guard, progress=progress, retire_original=True)
        if not result['original_retired'] or not result['physical_restore_checked'] or source.exists():
            raise ValueError('Only complete physical verification and actual raw retirement reclaim space')
        atomic(output/'complete.json', dict(complete=True, owner=owner, at=datetime.now(timezone.utc).isoformat(),
            source_files_sha256=files, actual_archive_manifest=result, actual_archive_manifest_sha256=sha(destination/'manifest.json'),
            actual_reclaimed_allocated_bytes=result['reclaimed_allocated_bytes_if_retired'],
            observed_free_bytes_before=before_free, observed_free_bytes_after=shutil.disk_usage(ROOT).free,
            explicit_restore_command='.venv/bin/python scripts/restore_exact_checkpoint_raw_anchor_v4.py --archive '
                +str(destination)+' --output '+str(source), retained_raw_anchor=str(anchor),
            all_original_tensor_optimizer_RNG_and_metadata_bytes_retained=True,
            prior_failure_preserved=True, shared_cache_and_active_inputs_unchanged=True,
            full_future_GPU_experiments_not_funded_by_reclamation_alone=True, model_quality_gain=False, goal_achieved=False))
        print(json.dumps(dict(complete=True, reclaimed_allocated_bytes=result['reclaimed_allocated_bytes_if_retired'])), flush=True)
    except BaseException as exc:
        payload = destination/'payload.zst'; discarded = None
        if (source.is_file() and state(source) == original_state and sha(source) == original_sha
                and state(anchor) == anchor_state and sha(anchor) == anchor_sha and payload.is_file()):
            discarded = dict(path=str(payload), bytes=payload.stat().st_size, sha256=sha(payload),
                only_this_new_unadmitted_duplicate_discarded=True, complete_original_and_anchor_still_exact=True)
            payload.unlink()
        atomic(output/'failure.json', dict(owner=owner, error=repr(exc), traceback=traceback.format_exc(),
            original_retained=source.exists(), discarded_unadmitted_duplicate=discarded,
            original_and_partial_metadata_preserved=True, automatic_retry=False, goal_achieved=False))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.parse_args(); main()
