"""Run actual native training/recovery integration after the exact capacity trial."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/native-suffix-training-controller-v1-queue'
CAPACITY = ROOT / 'reports/native-suffix-capacity-controller-v2-pytest/test_actual_four_replicas_both0'


def read(p): return json.loads(Path(p).read_text())


def sha(p):
    with Path(p).open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()


def atomic(path, value):
    pending = path.with_suffix('.pending')
    with pending.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False); stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    pending.replace(path)


def identity(pid):
    fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
    return dict(pid=pid, ticks=fields[19], state=fields[0], nice=int(fields[16]), scheduler=os.sched_getscheduler(pid))


def main(parent):
    if (OUT.exists() or os.environ.get('CUDA_VISIBLE_DEVICES') != '' or os.nice(0) < 19
            or any(os.environ.get(k) != '1' for k in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'])):
        raise ValueError('Fresh CUDA-hidden nice19 one-thread native mechanics queue required')
    os.sched_setscheduler(0, os.SCHED_IDLE, os.sched_param(0))
    from slm_lab.exact_waiting_process_retirement_v1 import terminal
    from slm_lab.local_source_closure_v2 import python_sources
    prior = ROOT / 'reports/native-suffix-capacity-controller-v2-readiness.json'
    expected = read(prior)['corrected_actual_full_trial_owner']
    if parent != {k: expected[k] for k in ['pid', 'ticks']}: raise ValueError('Exact originally declared capacity trial birth required')
    gates = ROOT / 'reports/native-suffix-training-controller-v1-gates-tests.log'
    if '2 passed' not in gates.read_text(): raise ValueError('Actual small/partial native source rejection controls must pass first')
    runner = ROOT / 'scripts/run_native_suffix_training_controller_v1_tests.py'
    seeds = [Path(__file__), runner, ROOT / 'tests/test_native_suffix_training_controller_v1.py']
    files = {str(p): sha(p) for p in [prior, gates, *python_sources(ROOT, seeds)]}
    if 'torch' in sys.modules: raise ValueError('A waiting queue must not import the neural runtime')
    OUT.mkdir()
    atomic(OUT / 'plan.json', dict(owner=identity(os.getpid()), at=datetime.now(timezone.utc).isoformat(),
        exact_capacity_trial_parent=parent, source_files_sha256=files, automatic_retry=False,
        own_small_fixture_parameters=267718, repeated_original_human_programs=1, fixture_source_presentations=9,
        explicit_fixture_horizon=5, full_original_native_horizon_preserved=512,
        full_source_scope_not_replaced_by_fixture=True, new_GPUcontexts=0, new_production_updates=0, goal_achieved=False))
    child = None
    try:
        deadline = time.monotonic() + 7 * 86400
        while not terminal(parent):
            if time.monotonic() > deadline: raise ValueError('Declared source-only mechanics waiting deadline expired')
            if 'torch' in sys.modules or any(sha(p) != h for p, h in files.items()): raise ValueError('Waiting queue/source bindings changed')
            atomic(OUT / 'progress.json', dict(phase='waiting for exact actual four-replica native capacity trial',
                exact_parent_still_live=True, new_GPUcontexts=0, production_native_RL_admitted=False))
            time.sleep(15)
        original_log = ROOT / 'reports/native-suffix-capacity-controller-v2-tests.log'
        if '3 passed' not in original_log.read_text(): raise ValueError('The exact whole native capacity trial failed or stopped without success')
        closed = [original_log]
        for profile in ['anchor_CE', 'anchor_execution']:
            closed.extend(CAPACITY / (profile + '-rank-' + str(rank) + '.json') for rank in range(4))
            latest = CAPACITY / profile / 'durable/LATEST.json'
            pointer = read(latest); receipt = read(pointer['receipt'])
            if pointer['completed'] != 3 or sha(pointer['receipt']) != pointer['receipt_sha256'] or sha(receipt['checkpoint']) != receipt['checkpoint_sha256']:
                raise ValueError('The actual complete small capacity state/receipt/pointer bytes differ')
            closed.extend([latest, Path(pointer['receipt']), Path(receipt['checkpoint'])])
        closed_files = {str(p): sha(p) for p in closed}
        atomic(OUT / 'actual-closed-capacity-inputs.json', dict(source_files_sha256=closed_files,
            original_parent_terminal=True, actual_capacity_test_success=True, full_size_GPUcapacity=False))
        log = ROOT / 'reports/native-suffix-training-controller-v1-full-tests.log'
        with log.open('x') as stream:
            child = subprocess.Popen([str(ROOT / '.venv/bin/python'), '-u', str(runner)], cwd=ROOT,
                env=dict(os.environ, CUDA_VISIBLE_DEVICES=''), stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            atomic(OUT / 'child.json', dict(actual_child=identity(child.pid), actual_small_neural_integration=True,
                no_full_size_model_or_GPUjob_launched=True))
            code = child.wait()
        if code != 0 or '3 passed' not in log.read_text(): raise ValueError('Actual native training/readback/recovery integration failed; preserve outputs')
        if any(sha(p) != h for p, h in {**files, **closed_files}.items()): raise ValueError('Actual complete capacity inputs or integration sources changed')
        atomic(OUT / 'complete.json', dict(complete=True, test_exit_code=code, actual_test_log_sha256=sha(log),
            actual_four_replica_small_native_controller_integration_passed=True,
            fresh_capacity_readback_fixed_group_horizon_complete_training_and_preserved_attempt_recovery_passed=True,
            full_size_GPUcapacity_passed=False, full_size_GPUtraining_updates=0,
            production_native_RL_admitted=False, new_model_quality_gain=False, goal_achieved=False))
    except BaseException as exc:
        atomic(OUT / 'failure.json', dict(error=repr(exc), traceback=traceback.format_exc(), automatic_retry=False,
            actual_child_pid=None if child is None else child.pid, all_actual_partial_outputs_preserved=True, goal_achieved=False))
        raise


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--parent-pid', type=int, required=True)
    parser.add_argument('--parent-ticks', required=True)
    args = parser.parse_args()
    main(dict(pid=args.parent_pid, ticks=args.parent_ticks))
