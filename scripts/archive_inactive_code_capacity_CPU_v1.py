"""Losslessly archive an unbound capacity snapshot from the completed code run.

Preserve every original tensor, optimizer, RNG and metadata byte. The existing
opaque codec physically restores and independently rehashes the complete file
on separate scratch before retiring the raw duplicate. No model construction
or GPU work, no active input changes, and no shared-cache cleanup.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
RESERVE = 28 * 1024**3
OWNER = dict(pid=3798674, ticks='42848849')
SIZES = {'after1': 11388025211, 'after7': 11801119723}


def metadata_bindings(source):
    result = subprocess.run(['rg', '--threads', '1', '-l', '-F', '-e', str(source), '-e', str(source.relative_to(ROOT)),
        str(ROOT / 'reports'), '-g', '*.json'], capture_output=True, text=True)
    if result.returncode not in [0, 1]: raise ValueError('Complete original metadata binding inspection failed')
    def inspect(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if isinstance(key, str) and '/' in key and (ROOT / key).resolve() == source:
                    raise ValueError('An existing immutable original source binding names this artifact')
                inspect(item)
        elif isinstance(value, list):
            for item in value: inspect(item)
    for name in result.stdout.splitlines(): inspect(json.loads(Path(name).read_text()))
    return result.stdout.splitlines()


def no_readers_or_anchors(source):
    from slm_lab.source_segment_archive_v1 import no_readers
    no_readers(source)
    for path in (ROOT / 'checkpoints').rglob('manifest.json'):
        data = json.loads(path.read_text())
        for name in ['anchor', 'source_checkpoint']:
            if data.get(name) and Path(data[name]).resolve() == source:
                raise ValueError('A retained full archive anchor cannot be retired')


def main(target):
    if (target not in SIZES or os.environ.get('CUDA_VISIBLE_DEVICES') != '' or os.nice(0) != 19
            or any(os.environ.get(k) != '1' for k in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'])):
        raise ValueError('An exact CUDA-hidden nice19 single-thread inactive-source archive is required')
    os.sched_setscheduler(0, os.SCHED_IDLE, os.sched_param(0)); allowed = os.sched_getaffinity(0)
    os.sched_setaffinity(0, {22 if 22 in allowed else max(allowed)})
    if importlib.util.find_spec('zstandard') is None:
        # Load only the exact package already used by the closed large-file
        # restore proof. Do not add a whole foreign site-packages directory.
        tested_bindings = json.loads((ROOT / 'reports/inactive-phase-capacity-exact-zstd-v2/plan.json').read_text())['source_files_sha256']
        package = Path('/opt/conda/lib/python3.11/site-packages/zstandard')
        for path in [package / '__init__.py', package / 'backend_c.cpython-311-x86_64-linux-gnu.so']:
            with path.open('rb') as stream: digest = hashlib.file_digest(stream, 'sha256').hexdigest()
            if tested_bindings.get(str(path)) != digest:
                raise ValueError('The exact previously tested installed codec must remain unchanged')
        spec = importlib.util.spec_from_file_location('zstandard', package / '__init__.py', submodule_search_locations=[str(package)])
        module = importlib.util.module_from_spec(spec); sys.modules['zstandard'] = module; spec.loader.exec_module(module)
    from slm_lab.exact_checkpoint_zstd_archive_v2 import archive, atomic, sha, state
    from slm_lab.exact_waiting_process_retirement_v1 import terminal
    from slm_lab.local_source_closure_v2 import python_sources
    import zstandard
    source = ROOT / 'checkpoints/code-first-continuation-v3' / ('capacity-' + target + '.pt')
    destination = source.with_name(source.name + '.exact-zstd-v1')
    output = ROOT / 'reports' / ('inactive-code-capacity-' + target + '-exact-zstd-v1')
    maximum = (9 if target == 'after1' else 9.5) * 1024**3; maximum = int(maximum)
    if (output.exists() or destination.exists() or source.is_symlink() or source.stat().st_nlink != 1
            or source.stat().st_size != SIZES[target] or shutil.disk_usage(ROOT).free < RESERVE + maximum
            or shutil.disk_usage('/tmp').free < RESERVE + source.stat().st_size):
        raise ValueError('Fresh complete persistent payload and independent full physical restore budgets must fit')
    plan_path = ROOT / 'reports/code-first-continuation-v3/plan.json'
    completed_path = plan_path.with_name('training-complete.json'); original = json.loads(plan_path.read_text())
    if ({k: original['owner'][k] for k in OWNER} != OWNER or not terminal(OWNER)
            or original['planned_steps'] != 7558 or json.loads(completed_path.read_text())['actual_updates'] != 7558):
        raise ValueError('The exact original complete7558update source experiment must remain terminal')
    tested = ROOT / 'reports/inactive-phase-capacity-exact-zstd-v2'
    tested_plan = json.loads((tested / 'plan.json').read_text())
    tested_done = json.loads((tested / 'complete.json').read_text())
    codec = ROOT / 'src/slm_lab/exact_checkpoint_zstd_archive_v2.py'
    tests = ROOT / 'reports/exact-checkpoint-zstd-archive-v2-venv-tests.log'
    if (not terminal(tested_plan['owner']) or tested_done.get('complete') is not True
            or tested_done.get('actual_archive_manifest', {}).get('physical_restore_checked') is not True
            or tested_plan['source_files_sha256'].get(str(codec)) != sha(codec) or '3 passed' not in tests.read_text()):
        raise ValueError('The unchanged actually tested codec and its previous complete large physical restore must remain bound')
    no_readers_or_anchors(source); examined = metadata_bindings(source)
    original_state = state(source); original_sha = sha(source)
    paths = [plan_path, completed_path, tested / 'plan.json', tested / 'complete.json', tests,
        Path(zstandard.__file__).resolve(), Path(importlib.util.find_spec('zstandard.backend_c').origin).resolve(),
        *python_sources(ROOT, [Path(__file__), codec])]
    files = {str(p): sha(p) for p in paths}
    signatures = {p: (Path(p).stat().st_dev, Path(p).stat().st_ino, Path(p).stat().st_size, Path(p).stat().st_mtime_ns, Path(p).stat().st_ctime_ns) for p in files}
    output.mkdir(); owner = dict(pid=os.getpid(), ticks=Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19])
    atomic(output / 'plan.json', dict(owner=owner, at=datetime.now(timezone.utc).isoformat(), source_files_sha256=files,
        original_experiment_owner=OWNER, original_complete_source_updates=7558, original_capacity_target=target,
        source=str(source), original_sha256=original_sha, original_state=original_state, archive=str(destination),
        actual_inspected_original_metadata=examined, actual_immutable_source_bindings=0, actual_live_readers=0,
        maximum_payload_bytes=maximum, minimum_original_writer_free_bytes=RESERVE, minimum_reclaimed_bytes=1024**3,
        complete_physical_restore_and_independent_rehash_required_before_retirement=True,
        all_original_bytes_retained_and_historical_original_path_requires_explicit_restore=True,
        only_this_attempt_unadmitted_duplicate_may_be_discarded_after_failure_if_original_remains_exact=True,
        no_shared_cache_or_active_training_input_changes=True, GPU_contexts=0, new_neural_updates=0, goal_achieved=False))
    started = time.monotonic(); last_inspection = [started]
    def guard():
        if (os.sched_getscheduler(0) != os.SCHED_IDLE or os.nice(0) != 19 or not terminal(OWNER)
                or state(source) != original_state or time.monotonic() - started > 3600):
            raise ValueError('The complete original inactive source, idle scheduling and bounded archive lifetime must remain')
        for p, before in signatures.items():
            st = Path(p).stat()
            if before != (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns):
                raise ValueError('An exact completed source or immutable tested archive runtime changed')
        if time.monotonic() - last_inspection[0] >= 10:
            no_readers_or_anchors(source); metadata_bindings(source); last_inspection[0] = time.monotonic()
    def progress(value):
        atomic(output / 'progress.json', dict(phase='lossless encode of exact inactive original code capacity', **value,
            source_retired=False, all_original_model_optimizer_RNG_metadata_retained=True, goal_achieved=False))
    before_free = shutil.disk_usage(ROOT).free
    try:
        with tempfile.TemporaryDirectory(prefix='slm-inactive-code-capacity-' + target + '-', dir='/tmp') as temporary:
            result = archive(source, destination, Path(temporary), expected_sha256=original_sha, maximum_payload_bytes=maximum,
                minimum_free_bytes=RESERVE, minimum_reclaimed_bytes=1024**3, guard=guard, progress=progress, retire_original=True)
        if not result['original_retired'] or not result['physical_restore_checked'] or source.exists():
            raise ValueError('Only a complete actual physical restore and verified original retirement count as disk reclamation')
        atomic(output / 'complete.json', dict(complete=True, owner=owner, at=datetime.now(timezone.utc).isoformat(),
            source_files_sha256=files, actual_archive_manifest=result, actual_archive_manifest_sha256=sha(destination / 'manifest.json'),
            actual_reclaimed_allocated_bytes=result['reclaimed_allocated_bytes_if_retired'],
            observed_free_bytes_before=before_free, observed_free_bytes_after=shutil.disk_usage(ROOT).free,
            explicit_restore_command='.venv/bin/python scripts/restore_exact_checkpoint_archive_v2.py --archive ' + str(destination) + ' --output ' + str(source),
            all_original_bytes_retained=True, active_inputs_and_workers_unchanged=True, shared_caches_untouched=True,
            full_future_GPU_experiments_not_funded_by_reclamation_alone=True, model_quality_gain=False, goal_achieved=False))
        print(json.dumps(dict(complete=True, source=str(source), reclaimed_allocated_bytes=result['reclaimed_allocated_bytes_if_retired'])), flush=True)
    except BaseException as exc:
        discarded = None
        payload = destination / 'payload.zst'
        if source.is_file() and state(source) == original_state and sha(source) == original_sha and payload.is_file():
            discarded = dict(path=str(payload), bytes=payload.stat().st_size, sha256=sha(payload),
                only_this_new_unadmitted_duplicate_discarded=True, complete_original_still_exact=True)
            payload.unlink()
        atomic(output / 'failure.json', dict(owner=owner, error=repr(exc), traceback=traceback.format_exc(),
            original_retained=source.exists(), discarded_unadmitted_duplicate=discarded,
            all_original_and_partial_metadata_preserved=True, automatic_retry=False, goal_achieved=False))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--target', choices=SIZES, required=True)
    main(parser.parse_args().target)
