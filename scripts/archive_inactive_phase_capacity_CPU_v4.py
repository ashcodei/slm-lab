"""Retain the inactive post-update capacity snapshot with an exact compressed-anchor XOR archive."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/inactive-phase-capacity-exact-xor-zstd-v4'
SOURCE = ROOT / 'checkpoints/phase-joint-v3/capacity-post-step4.pt'
DESTINATION = SOURCE.with_name(SOURCE.name + '.exact-xor-zstd-v4')
ANCHOR = ROOT / 'checkpoints/phase-joint-v3/capacity-raw-step1.pt.exact-zstd-v2'
OWNER = dict(pid=3096450, ticks='38665081')
RESERVE = 28 * 1024**3


def main():
    if (OUT.exists() or DESTINATION.exists() or os.environ.get('CUDA_VISIBLE_DEVICES') != '' or os.nice(0) < 19
            or any(os.environ.get(k) != '1' for k in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'])):
        raise ValueError('A fresh CUDA-hidden nice19 one-thread archive with all prior attempts preserved is required')
    os.sched_setscheduler(0, os.SCHED_OTHER, os.sched_param(0))
    from slm_lab.exact_checkpoint_xor_zstd_archive_v3 import archive, atomic, sha, state, anchor_spec
    from slm_lab.exact_waiting_process_retirement_v1 import terminal
    from slm_lab.local_source_closure_v2 import python_sources
    import zstandard
    import importlib.util
    prior = ROOT / 'reports/inactive-phase-capacity-exact-zstd-v2'
    old_plan = json.loads((prior / 'plan.json').read_text())
    prior_archive = json.loads((prior / 'complete.json').read_text())
    if (not terminal(old_plan['owner']) or prior_archive.get('complete') is not True
            or prior_archive['actual_archive_manifest'].get('physical_restore_checked') is not True):
        raise ValueError('The previous exact lossless archive must be complete and terminal')
    failed = ROOT / 'reports/inactive-phase-capacity-exact-zstd-v3'
    failed_plan = json.loads((failed / 'plan.json').read_text())
    failed_attempt = json.loads((failed / 'failure.json').read_text())
    if not terminal(failed_plan['owner']) or failed_attempt.get('original_retained') is not True:
        raise ValueError('The exact failed plain-Zstandard attempt must remain terminal and preserved')
    retained_anchor = anchor_spec(ANCHOR)
    plan_path = ROOT / 'reports/phase-joint-v3/plan.json'
    complete_path = ROOT / 'reports/phase-joint-v3/training-complete.json'
    owner = json.loads(plan_path.read_text())['owner']
    if ({k: owner[k] for k in OWNER} != OWNER or not terminal(OWNER)
            or json.loads(complete_path.read_text())['actual_updates'] != 35284):
        raise ValueError('The exact original whole-source phase must be terminal and completed')
    tests = ROOT / 'reports/exact-checkpoint-xor-zstd-archive-v3-tests.log'
    if '4 passed' not in tests.read_text(): raise ValueError('Actual physical restore, corruption and overflow controls must pass')
    if SOURCE.is_symlink() or SOURCE.stat().st_nlink != 1 or SOURCE.stat().st_size != 11388026875:
        raise ValueError('Only the exact ordinary unshared old capacity artifact is eligible')

    def no_readers_or_anchors():
        if not terminal(OWNER): raise ValueError('The original completed owner cannot become active')
        for proc in Path('/proc').iterdir():
            if not proc.name.isdigit() or int(proc.name) == os.getpid(): continue
            try:
                command = (proc / 'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace')
                if str(SOURCE) in command or str(SOURCE) in (proc / 'maps').read_text():
                    raise ValueError('An old capacity artifact has an actual live user')
                for fd in (proc / 'fd').iterdir():
                    try:
                        if fd.resolve() == SOURCE: raise ValueError('An old capacity artifact is open in another live process')
                    except (FileNotFoundError, PermissionError, ProcessLookupError): pass
            except (FileNotFoundError, PermissionError, ProcessLookupError): pass
        for path in (ROOT / 'checkpoints').rglob('manifest.json'):
            try: metadata = json.loads(path.read_text())
            except (ValueError, OSError): continue
            for key in ['anchor', 'source_checkpoint']:
                if metadata.get(key) and Path(metadata[key]).resolve() == SOURCE:
                    raise ValueError('A retained exact archive anchor can never be retired')

    def bound_metadata():
        result = subprocess.run(['rg', '-l', '-F', '-e', str(SOURCE), '-e', str(SOURCE.relative_to(ROOT)),
            str(ROOT / 'reports'), '-g', '*.json'], capture_output=True, text=True)
        if result.returncode not in [0, 1]: raise ValueError('Complete report binding inspection failed: ' + result.stderr)
        examined = result.stdout.splitlines()
        def inspect(value):
            if isinstance(value, dict):
                for key, item in value.items():
                    if isinstance(key, str) and ('/' in key) and (ROOT / key).resolve() == SOURCE:
                        raise ValueError('An existing recorded immutable source binding names the artifact: ' + key)
                    inspect(item)
            elif isinstance(value, list):
                for item in value: inspect(item)
        for name in examined: inspect(json.loads(Path(name).read_text()))
        return examined

    no_readers_or_anchors(); examined = bound_metadata()
    original_sha = sha(SOURCE); original_state = state(SOURCE)
    if original_sha != failed_plan['original_sha256']: raise ValueError('Every original failed-attempt target byte must remain intact')
    seeds = [Path(__file__), ROOT / 'src/slm_lab/exact_checkpoint_xor_zstd_archive_v3.py',
        ROOT / 'tests/test_exact_checkpoint_xor_zstd_archive_v3.py']
    codec = [Path(zstandard.__file__).resolve(), Path(importlib.util.find_spec('zstandard.backend_c').origin).resolve()]
    files = {str(p): sha(p) for p in [plan_path, complete_path, tests, prior / 'complete.json', prior / 'plan.json', failed / 'plan.json', failed / 'failure.json', ANCHOR / 'manifest.json', ANCHOR / 'payload.zst', *codec, *python_sources(ROOT, seeds)]}
    states = {p: (Path(p).stat().st_ino, Path(p).stat().st_size, Path(p).stat().st_mtime_ns) for p in files}
    OUT.mkdir()
    atomic(OUT / 'plan.json', dict(at=datetime.now(timezone.utc).isoformat(),
        owner=dict(pid=os.getpid(), ticks=Path('/proc/self/stat').read_text().split(') ', 1)[1].split()[19]),
        original_experiment_owner=OWNER, previous_complete_lossless_archive_preserved=prior_archive,
        previous_plain_Zstandard_saving_threshold_failure_preserved=failed_attempt, retained_compressed_anchor=retained_anchor,
        archive_method='tensor-aligned exact byte XOR against independently restorable retained compressed anchor',
        maximum_archive_wall_seconds=900,
        codec=zstandard.__version__, actual_scheduler='SCHED_OTHER', actual_nice=19, actual_threads=1, original_path=str(SOURCE), original_sha256=original_sha, original_state=original_state,
        archive_path=str(DESTINATION), source_files_sha256=files, inspected_matching_report_metadata=examined,
        actual_live_readers=0, recorded_checkpoint_source_bindings=0, original_model_weights_optimizer_RNG_and_all_metadata_retained=True,
        required_independent_physical_restore_before_original_retirement=True, minimum_original_writer_free_bytes=RESERVE,
        original_artifact_only_requires_explicit_restore_before_historical_reuse=True, no_active_inference_source_or_worker_modified=True,
        shared_outside_project_caches_untouched=True, neural_forwards=0, new_GPUcontexts=0, goal_achieved=False))
    free_before = shutil.disk_usage(ROOT).free
    started = time.monotonic()
    def guard():
        if (os.sched_getscheduler(0) != os.SCHED_OTHER or not terminal(OWNER) or state(SOURCE) != original_state
                or time.monotonic() - started > 900):
            raise ValueError('The exact inactive artifact and single-thread normal nice19 source-only archive scope must remain')
        if any((Path(p).stat().st_ino, Path(p).stat().st_size, Path(p).stat().st_mtime_ns) != v for p, v in states.items()):
            raise ValueError('An immutable archive runtime or completed-source receipt changed')
    def progress(value):
        atomic(OUT / 'progress.json', dict(phase='encode exact inactive old capacity artifact against retained compressed anchor',
            **value, original_retired=False, original_writer_reserve_bytes=RESERVE, goal_achieved=False))
    try:
        with tempfile.TemporaryDirectory(prefix='slm-exact-capacity-restore-', dir='/tmp') as directory:
            # The expensive complete reader/anchor/binding inspection runs once
            # more at the final physical-restore boundary, before retirement.
            def guarded(): guard()
            def final_guard():
                guard(); no_readers_or_anchors(); bound_metadata()
                if any(sha(p) != h for p, h in files.items()): raise ValueError('All exact runtime/phase inputs must remain')
            result = archive(SOURCE, ANCHOR, DESTINATION, Path(directory), expected_sha256=original_sha,
                maximum_payload_bytes=original_state['bytes'] + 16*1024**2, minimum_free_bytes=RESERVE,
                minimum_reclaimed_bytes=2 * 1024**3, guard=guarded, progress=progress, retire_original=False)
            final_guard()
            if (not result['physical_restore_checked'] or result['original_sha256'] != original_sha
                    or sha(DESTINATION / 'payload.zst') != result['payload_sha256'] or sha(SOURCE) != original_sha):
                raise ValueError('Every original byte must physically restore and rehash before retirement')
            SOURCE.unlink(); result['original_retired'] = True
            atomic(DESTINATION / 'manifest.json', result)
        atomic(OUT / 'complete.json', dict(complete=True, at=datetime.now(timezone.utc).isoformat(),
            original_path=str(SOURCE), original_sha256=original_sha, actual_archive_manifest=result,
            original_bytes=original_state['bytes'], compressed_bytes=result['payload_bytes'],
            actual_reclaimed_allocated_bytes=result['reclaimed_allocated_bytes_if_retired'],
            free_bytes_before=free_before, free_bytes_after=shutil.disk_usage(ROOT).free,
            exact_old_capacity_snapshot_retained_with_physical_restore_proof=True,
            explicit_archive_restore_required_for_original_historical_path=True,
            active_training_inputs_and_sources_and_workers_unchanged=True, goal_achieved=False))
        print(json.dumps(dict(complete=True, reclaimed_allocated_bytes=result['reclaimed_allocated_bytes_if_retired'])), flush=True)
    except BaseException as exc:
        # An unadmitted new duplicate cannot consume the original writers'
        # reserve. Keep the failure receipt; never remove the original here.
        payload = DESTINATION / 'payload.zst'; removed = None
        if SOURCE.exists() and sha(SOURCE) == original_sha and payload.is_file() and not payload.is_symlink():
            removed = dict(path=str(payload), bytes=payload.stat().st_size, sha256=sha(payload)); payload.unlink()
        atomic(OUT / 'failure.json', dict(error=repr(exc), original_retained=SOURCE.exists(),
            discarded_only_unadmitted_new_duplicate=removed, automatic_retry=False, goal_achieved=False))
        raise


if __name__ == '__main__': main()
