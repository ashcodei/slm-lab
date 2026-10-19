"""Run full actual native-data integration once its exact publisher is terminal."""
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
OUT = ROOT / 'reports/recovered-native-suffix-data-complete-v2-queue'
ADMISSION = ROOT / 'reports/native-human-code-suffix-data-admission-CPU-v2'


def read(path): return json.loads(path.read_text())


def sha(path):
    with Path(path).open('rb') as stream: return hashlib.file_digest(stream, 'sha256').hexdigest()


def atomic(path, value):
    temporary = path.with_suffix('.pending')
    with temporary.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False); stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    temporary.replace(path)


def identity(pid):
    fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
    return dict(pid=pid, ticks=fields[19], state=fields[0], nice=int(fields[16]), scheduler=os.sched_getscheduler(pid))


def main(parent):
    if (OUT.exists() or os.environ.get('CUDA_VISIBLE_DEVICES') != '' or os.nice(0) < 19
            or any(os.environ.get(k) != '1' for k in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'])):
        raise ValueError('Fresh CUDA-hidden nice19 one-thread source-only integration queue required')
    os.sched_setscheduler(0, os.SCHED_IDLE, os.sched_param(0))
    from slm_lab.exact_waiting_process_retirement_v1 import terminal
    from slm_lab.local_source_closure_v2 import python_sources
    if read(ADMISSION / 'plan.json')['owner'] != parent:
        raise ValueError('The exact new source-data publisher birth is required')
    seeds = [Path(__file__), ROOT / 'scripts/run_recovered_native_suffix_data_complete_v2_tests.py',
        ROOT / 'tests/test_recovered_native_suffix_data_complete_v2.py']
    sources = {str(p): sha(p) for p in [ADMISSION / 'plan.json', *python_sources(ROOT, seeds)]}
    if 'torch' in sys.modules: raise ValueError('The waiting parent cannot import the neural runtime')
    OUT.mkdir()
    owner = identity(os.getpid())
    atomic(OUT / 'plan.json', dict(owner=owner, at=datetime.now(timezone.utc).isoformat(), exact_data_publisher=parent,
        source_files_sha256=sources, full_actual_original_source_scope_required=True,
        original_sources=3721, original_native_views=14884, CPU_fixture_cannot_substitute=True,
        automatic_retry=False, neural_forwards=0, optimizer_updates=0, GPUcontexts=0, goal_achieved=False))
    child = None
    try:
        deadline = time.monotonic() + 7 * 86400
        while not terminal(parent):
            if time.monotonic() > deadline: raise ValueError('The declared source-only integration waiting deadline expired')
            if any(sha(p) != h for p, h in sources.items()) or 'torch' in sys.modules:
                raise ValueError('Every source-only queue and complete-scope test binding must remain exact')
            atomic(OUT / 'progress.json', dict(owner=identity(os.getpid()),
                phase='waiting for exact complete recovered native-data publisher', exact_publisher_still_live=True,
                full_source_integration_passed=False, GPUcontexts=0, neural_forwards=0))
            time.sleep(15)
        if (ADMISSION / 'failure.json').exists() or not read(ADMISSION / 'complete.json')['complete']:
            raise ValueError('The complete publisher failed or stopped without success; preserve all actual outputs')
        log = ROOT / 'reports/recovered-native-suffix-data-complete-v2-tests.log'
        with log.open('x') as stream:
            child = subprocess.Popen([str(ROOT / '.venv/bin/python'), '-u', str(seeds[1])], cwd=ROOT,
                env=dict(os.environ, CUDA_VISIBLE_DEVICES=''), stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            atomic(OUT / 'child.json', dict(actual_source_only_test=identity(child.pid),
                no_model_or_GPUwork_launched=True, all_14884_actual_admitted_records_required=True))
            code = child.wait()
        if code != 0 or '3 passed' not in log.read_text():
            raise ValueError('Full actual recovered native-data integration failed; preserve evidence without automatic retry')
        if any(sha(p) != h for p, h in sources.items()): raise ValueError('Full-scope integration sources changed')
        atomic(OUT / 'complete.json', dict(complete=True, actual_test_exit_code=code, actual_test_log_sha256=sha(log),
            full_actual_source_data_integration_passed=True, actual_original_sources=3721, actual_native_views=14884,
            both_optimization_and_capacity_loaders_tested=True, actual_admitted_capacity_selection_tested=True,
            neural_forwards=0, optimizer_updates=0, full_size_GPUcapacity_passed=False,
            model_quality_gain=False, production_RL_admitted=False, goal_achieved=False))
    except BaseException as exc:
        atomic(OUT / 'failure.json', dict(error=repr(exc), traceback=traceback.format_exc(), automatic_retry=False,
            actual_child_pid=None if child is None else child.pid, all_partial_outputs_preserved=True, goal_achieved=False))
        raise


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--parent-pid', type=int, required=True)
    parser.add_argument('--parent-ticks', required=True)
    args = parser.parse_args()
    main(dict(pid=args.parent_pid, ticks=args.parent_ticks))
