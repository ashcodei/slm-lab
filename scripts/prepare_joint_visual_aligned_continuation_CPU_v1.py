"""Idle source-only preparation; a later normal-scheduled launch is separate.

No model, CUDAcontext, GPUlease or existing pipeline mutation. Wait for the
exact full canonical native and source owners, then freeze a reviewable launch
recipe. This process never spawns a GPUworker, even when preparation finishes.
"""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/joint-visual-aligned-continuation-preparation-v1'
RESERVE = 40 * 1024**3
MAX_METADATA = 64 * 1024**2
MAX_ASSESSMENTS = 4 * 1024**3


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def state(path):
    s = Path(path).stat()
    return s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns


def atomic(path, value):
    path = Path(path)
    encoded = (json.dumps(value, indent=2, allow_nan=False) + '\n').encode()
    if len(encoded) > MAX_METADATA:
        raise ValueError('The complete source-only preparation recipe must fit its declared metadata bound')
    pending = path.with_suffix(path.suffix + '.pending')
    with pending.open('xb') as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    pending.replace(path)


def immutable_report(path):
    name = Path(path).name
    return not (name.endswith('queue.json') or 'progress' in name or name in ['children.json', 'LATEST.json']
        or 'training-metrics' in name or name.endswith('.pending'))


def main():
    if (OUT.exists() or os.environ.get('CUDA_VISIBLE_DEVICES') != '' or os.nice(0) < 19
            or any(os.environ.get(k) != '1' for k in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'])):
        raise ValueError('Fresh one-thread CUDA-hidden nice19 source-only preparation required')
    os.sched_setscheduler(0, os.SCHED_IDLE, os.sched_param(0))
    from slm_lab.bounded_disk_reserve_wait_v1 import wait_for_reserve
    from slm_lab.exact_waiting_process_retirement_v1 import terminal
    from slm_lab.joint_visual_runtime_contract_v1 import (ready, PIPELINE, PIPELINE_QUEUE, FINALIZER, FINALIZER_QUEUE, SOURCE, ADMISSION)
    from slm_lab.local_source_closure_v2 import python_sources
    import train_joint_visual_aligned_continuation_GPU_v1 as driver
    if 'torch' in sys.modules:
        raise ValueError('Idle preparation must not import neural runtime modules')
    checks = [('joint-visual-cold-overlay-and-runtime-contract-v1-tests.log', '20 passed'),
        ('joint-visual-controller-v1-tests.log', '14 passed'), ('visual-native-assessment-binding-v1-tests.log', '6 passed'),
        ('visual-aligned-GPU-driver-v1-tests.log', '10 passed')]
    for name, expected in checks:
        if expected not in (ROOT / 'reports' / name).read_text():
            raise ValueError('All actual visual runtime/controller and unchanged full-scope dispatch controls must pass')
    predecessors = dict(pipeline_owner=read(ROOT / PIPELINE_QUEUE)['owner'],
        finalizer_owner=read(ROOT / FINALIZER_QUEUE)['owner'], source_owner=read(ROOT / SOURCE / 'plan.json')['owner'])
    if not all(not terminal(owner) for owner in predecessors.values()):
        raise ValueError('Capture the exact actually live original canonical/source owners before queuing')
    case = ROOT / 'reports/joint-visual-actual-capacity-cases-CPU-v1'
    entrypoints = [Path(__file__), ROOT / 'scripts/train_joint_visual_aligned_continuation_GPU_v1.py',
        ROOT / 'scripts/compare_visual_native_endpoints_v1.py', ROOT / 'tests/test_visual_aligned_GPU_driver_v1.py',
        ROOT / 'tests/test_visual_native_assessment_binding_v1.py',
        ROOT / 'scripts/assess_machine_standalone_full_v2_assessment_recovery_gate_v1.py']
    paths = [ROOT / PIPELINE_QUEUE, ROOT / FINALIZER_QUEUE, ROOT / SOURCE / 'plan.json',
        case / 'complete.json', case / 'actual-complete-source-capacity-plan.json',
        *[ROOT / 'reports' / name for name, _ in checks], *python_sources(ROOT, entrypoints)]
    files = {str(p): sha(p) for p in paths}
    states = {p: state(p) for p in files}
    owner = dict(pid=os.getpid(), ticks=Path('/proc/self/stat').read_text().split(') ', 1)[1].split()[19])
    OUT.mkdir()
    atomic(OUT / 'plan.json', dict(owner=owner, at=datetime.now(timezone.utc).isoformat(), source_files_sha256=files,
        exact_predecessors=predecessors, initializer='fixed fully completed aligned machine-DPO standalone native student',
        original_source_views=302627, original_global_updates=4733, original_positive_response_EOS_targets=4669243,
        all_original_human_RL_and_nine_endpoint_scopes_preserved=True,
        full_new_candidate_suites=list(driver.SUITES), full_new_QA=16961, full_new_AI2D=3088, AI2D_conditions=3,
        full_new_MBPP=600, full_new_human_pairs=4642, full_new_word_crops=3429, word_conditions=['proper', 'blank', 'wrong'],
        exact_completed_canonical_native_baseline_reused_without_selection=True,
        idle_preparation_cannot_escape_to_normal_CPU_scheduling_on_this_host=True,
        later_GPU_launch_requires_a_new_normal_scheduled_nice0_process=True,
        GPU_workers_never_spawned_by_this_preparation=True, no_existing_worker_modified=True,
        minimum_other_writer_reserve_bytes=RESERVE, maximum_private_state_bytes=driver.MAX_STATE,
        maximum_candidate_assessment_bytes=MAX_ASSESSMENTS, maximum_metadata_bytes=MAX_METADATA,
        scheduler='SCHED_IDLE', CPU_threads=1, CUDA_hidden=True, models_constructed=0, neural_forwards=0,
        GPU_contexts=0, GPU_leases=0, production_updates=0, automatic_retry=False, goal_achieved=False))

    def guard():
        if os.sched_getscheduler(0) != os.SCHED_IDLE or any(state(p) != v for p, v in states.items()):
            raise ValueError('Every exact frozen source, queue scope and idle preparation policy must remain unchanged')

    try:
        deadline = time.monotonic() + 35 * 86400
        while True:
            guard()
            if ready(ROOT, **predecessors):
                break
            if time.monotonic() > deadline:
                raise ValueError('Preparation observation lifetime ended; no original owner was interrupted')
            atomic(OUT / 'progress.json', dict(owner=owner, phase='waiting for exact full aligned native and complete visual-source admission',
                live_predecessors={k: not terminal(v) for k, v in predecessors.items()},
                models_constructed=0, GPU_leases=0, production_updates=0, goal_achieved=False))
            time.sleep(30)
        wait_for_reserve(ROOT, minimum_bytes=RESERVE, guard=guard)
        final = read(ROOT / FINALIZER / 'complete.json')
        canonical = read(ROOT / PIPELINE / 'plan.json')
        admission = read(ROOT / ADMISSION)
        initializer = dict(checkpoint=final['checkpoint'], checkpoint_sha256=final['checkpoint_sha256'],
            sources=final['standalone_sources'], tokenizer_sha256=final['tokenizer_sha256'])
        baseline = ROOT / PIPELINE / 'standalone'
        original_cold = read(ROOT / 'reports/full-function-repair-comparison-GPU-v2/plan.json')['cold_inputs']
        cold_path = (ROOT / original_cold['path']).resolve()
        if sha(cold_path) != original_cold['sha256']:
            raise ValueError('All original complete4096text448photo64x448word cold tensors must retain their bytes')
        # Every original baseline result and actual action/delivery journal is
        # now closed. Mutable queue/progress/child registries are not imported.
        for mapping in [canonical['source_files_sha256'], admission['source_files_sha256']]:
            for path, expected in mapping.items():
                if '/reports/' in str(path) and not immutable_report(path):
                    continue
                if path in files and files[path] != expected:
                    raise ValueError('The canonical final native and original source evidence disagree')
                files[path] = expected
        for path in [ROOT / ADMISSION, ROOT / FINALIZER / 'complete.json', ROOT / FINALIZER / 'plan.json',
                ROOT / PIPELINE / 'plan.json', ROOT / PIPELINE / 'complete.json', ROOT / PIPELINE / 'restored.json',
                Path(final['checkpoint']), Path(final['checkpoint']).parent / 'tokenizer.json', cold_path,
                *[p for p in baseline.rglob('*') if p.is_file() and immutable_report(p)]]:
            files[str(path)] = sha(path)
        if any(sha(p) != expected for p, expected in files.items()):
            raise ValueError('Every complete original source, baseline action and actual native initializer must match')
        maximum_native = max(4 * 1024**3, Path(final['checkpoint']).stat().st_size + 64 * 1024**2)
        # Peak training uses five state-sized slots including the atomic
        # pending file. Final evaluation retains four slots, one complete
        # native, all full new reports, and the bounded complete journal.
        budget = RESERVE + max(5 * driver.MAX_STATE,
            4 * driver.MAX_STATE + maximum_native + MAX_ASSESSMENTS + 512 * 1024**2)
        if 'torch' in sys.modules:
            raise ValueError('Source-only preparation must not import neural runtime modules')
        atomic(OUT / 'ready.json', dict(complete=True, at=datetime.now(timezone.utc).isoformat(), owner=owner,
            exact_predecessors=predecessors, source_files_sha256=files, initializer=initializer,
            cold_inputs=dict(path=str(cold_path), sha256=original_cold['sha256']),
            cached_baseline=dict(folder=str(baseline), same_full_protocol=True, exact_fixed_native_reused_not_score_selected=True),
            cached_own_reference=canonical['cached_own_reference'], pulse_sources=canonical['pulse_sources'],
            maximum_native_bytes=maximum_native, maximum_candidate_assessment_bytes=MAX_ASSESSMENTS,
            required_free_bytes_before_launch=budget, observed_free_bytes=shutil.disk_usage(ROOT).free,
            explicit_later_normal_scheduled_launch_required=True, GPU_work_started=False,
            original_human_RL_and_all_nine_canonical_endpoints_preserved=True,
            full_size_visual_GPUcapacity_and_complete_candidate_training_export_assessment_still_required=True,
            production_updates=0, candidate_promoted=False, goal_achieved=False))
        atomic(OUT / 'complete.json', dict(complete=True, ready_sha256=sha(OUT / 'ready.json'),
            no_models_GPUcontexts_leases_or_workers_started=True, later_owned_launch_still_required=True, goal_achieved=False))
        print(json.dumps(dict(complete=True, ready=str(OUT / 'ready.json'), GPU_work_started=False)))
    except BaseException:
        atomic(OUT / 'failure.json', dict(owner=owner, traceback=traceback.format_exc(), automatic_retry=False,
            original_owners_untouched=True, GPU_work_started=False, goal_achieved=False))
        raise


if __name__ == '__main__':
    main()
