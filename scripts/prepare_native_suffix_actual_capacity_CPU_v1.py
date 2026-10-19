"""Wait for actual complete source integration, then freeze all native cases.

Only source data/geometry are read. This process never constructs a neural
model, launches GPU work, obtains a lease or authorizes production training.
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
OUT = ROOT / 'reports/native-human-code-suffix-actual-capacity-CPU-v1'
PARENT = ROOT / 'reports/recovered-native-suffix-data-complete-v4-queue'
EXPECTED_PARENT = dict(pid=200154, ticks='57669936')


def read(path): return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as stream: return hashlib.file_digest(stream, 'sha256').hexdigest()


def atomic(path, value):
    raw = (json.dumps(value, indent=2, allow_nan=False) + '\n').encode()
    if len(raw) > 32 * 1024**2 or shutil.disk_usage(ROOT).free - len(raw) < 40 * 1024**3:
        raise ValueError('Complete source metadata must fit above the existing40GiB other-writer reserve')
    temporary = path.with_suffix(path.suffix + '.pending')
    with temporary.open('xb') as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())
    temporary.replace(path)


def require_complete_source(parent):
    if ((PARENT / 'failure.json').exists() or not (PARENT / 'complete.json').is_file()
            or read(PARENT / 'plan.json')['owner']['pid'] != parent['pid']
            or read(PARENT / 'plan.json')['owner']['ticks'] != parent['ticks']):
        raise ValueError('The exact complete full-source integration must succeed; preserve every failure')
    completed = read(PARENT / 'complete.json')
    log = ROOT / 'reports/recovered-native-suffix-data-complete-v3-tests.log'
    if (completed.get('complete') is not True or completed.get('actual_test_exit_code') != 0
            or completed.get('full_actual_source_data_integration_passed') is not True
            or completed.get('actual_original_sources') != 3721 or completed.get('actual_native_views') != 14884
            or completed.get('both_optimization_and_capacity_loaders_tested') is not True
            or completed.get('actual_admitted_capacity_selection_tested') is not True
            or completed.get('actual_test_log_sha256') != sha(log) or '3 passed' not in log.read_text()):
        raise ValueError('All actual original sources and both native data roles must pass full integration first')
    return [PARENT / 'complete.json', log]


def main():
    if (OUT.exists() or os.environ.get('CUDA_VISIBLE_DEVICES') != '' or os.nice(0) < 19
            or any(os.environ.get(k) != '1' for k in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'])):
        raise ValueError('A fresh CUDA-hidden nice19 one-thread source-only preparer is required')
    os.sched_setscheduler(0, os.SCHED_IDLE, os.sched_param(0))
    from slm_lab.exact_waiting_process_retirement_v1 import terminal
    from slm_lab.local_source_closure_v2 import python_sources
    actual_parent = read(PARENT / 'plan.json')['owner']
    if any(actual_parent[k] != v for k, v in EXPECTED_PARENT.items()):
        raise ValueError('The exact originally captured complete-source integration owner must remain')
    import train_native_code_suffix_aligned_GPU_v1 as driver
    if 'torch' in sys.modules: raise ValueError('The waiting source-only preparer cannot import a neural runtime')
    log = ROOT / 'reports/native-code-suffix-owned-driver-v1-tests.log'
    prefill_log = ROOT / 'reports/native-text-chunked-prefill-v1-tests.log'
    if '10 passed' not in log.read_text() or '3 passed' not in prefill_log.read_text():
        raise ValueError('Actual connected native-driver and whole-context prefill checks are required')
    entrypoints = [Path(__file__), Path(driver.__file__), ROOT / 'src/slm_lab/native_suffix_capacity_cases_v1.py',
        ROOT / 'src/slm_lab/native_suffix_capacity_evidence_v1.py', ROOT / 'src/slm_lab/recovered_native_suffix_train_data_v3.py',
        ROOT / 'tests/test_native_code_suffix_owned_driver_v1.py', ROOT / 'tests/test_native_text_chunked_prefill_v1.py']
    files = {str(p): sha(p) for p in [PARENT / 'plan.json', log, prefill_log, *python_sources(ROOT, entrypoints)]}
    metadata = {p: (Path(p).stat().st_ino, Path(p).stat().st_size, Path(p).stat().st_mtime_ns) for p in files}
    owner = dict(pid=os.getpid(), ticks=Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19])
    OUT.mkdir()
    atomic(OUT / 'plan.json', dict(owner=owner, at=datetime.now(timezone.utc).isoformat(), exact_full_data_integration_owner=EXPECTED_PARENT,
        source_files_sha256=files, original_sources=3721, original_native_views=14884, epochs=3, group_size=8,
        all_failed_and_CE_only_sources_retained=True, models_constructed=0, neural_forwards=0,
        GPU_contexts=0, GPU_leases=0, optimizer_updates=0, goal_achieved=False))
    def guard():
        if any((Path(p).stat().st_ino, Path(p).stat().st_size, Path(p).stat().st_mtime_ns) != state for p, state in metadata.items()):
            raise ValueError('Every complete tested source byte must remain unchanged while waiting')
    try:
        deadline = time.monotonic() + 14 * 86400
        while not terminal(EXPECTED_PARENT):
            guard()
            if 'torch' in sys.modules or time.monotonic() > deadline:
                raise ValueError('The exact source-only waiting context or declared lifetime no longer holds')
            atomic(OUT / 'progress.json', dict(owner=owner, phase='waiting for exact full native-data integration',
                exact_integration_owner_live=True, full_native_source_capacity_selected=False,
                models_constructed=0, neural_forwards=0, GPU_contexts=0))
            time.sleep(30)
        guard()
        extra = require_complete_source(EXPECTED_PARENT)
        import torch
        from slm_lab.recovered_native_suffix_train_data_v3 import load_recovered_native_suffix_data
        from slm_lab.native_suffix_capacity_cases_v1 import choose_admitted
        from slm_lab.native_suffix_capacity_evidence_v1 import stress_positions
        from slm_lab.native_suffix_owned_runtime_contract_v1 import FULL
        data = load_recovered_native_suffix_data(ROOT, purpose='capacity')
        capacity = choose_admitted(data, seed=driver.STAGE_OPTIONS['seed'], epochs=driver.STAGE_OPTIONS['epochs'])
        positions = stress_positions(data)
        if (any(capacity['complete_training_budget'].get(k) != v for k, v in FULL.items())
                or not set(positions) <= {p for row in capacity['capacity_entries'] for p in row}):
            raise ValueError('All original whole-source budgets and every full-horizon maximum must remain selected')
        data.guard.check(full=True)
        if torch.cuda.is_initialized(): raise ValueError('Source preparation cannot initialize CUDA')
        guard()
        if any(sha(p) != h for p, h in files.items()): raise ValueError('Full source bytes changed during preparation')
        files.update(data.source_files_sha256)
        files.update({str(p): sha(p) for p in extra})
        case = OUT / 'actual-complete-source-capacity-plan.json'
        atomic(case, capacity)
        files[str(case)] = sha(case)
        maxima = [data.information(p) for p in positions]
        KV = max(2 * 8 * 35 * 8 * 64 * (i['conditional_prefix_tokens'] + i['full_native_action_budget'] - 1) * 4 for i in maxima)
        atomic(OUT / 'complete.json', dict(complete=True, at=datetime.now(timezone.utc).isoformat(), source_files_sha256=files,
            complete_training_budget=capacity['complete_training_budget'], actual_capacity_budget=capacity['capacity_budget'],
            actual_complete_capacity_entries=len(capacity['capacity_entries']), actual_fixed_horizon_source_positions=positions,
            predicted_full_size_selected_logical_KV_bytes=KV, declared_cache_bound_bytes=driver.STAGE_OPTIONS['maximum_cache_bytes'],
            every_original_CE_anchor_and_full_native_horizon_preserved=True, source_selection_only=True,
            models_constructed=0, neural_forwards=0, GPU_contexts=0, GPU_leases=0, optimizer_updates=0,
            full_size_GPUcapacity_passed=False, production_training_admitted=False, new_model_quality_gain=False, goal_achieved=False))
    except BaseException as exc:
        atomic(OUT / 'failure.json', dict(owner=owner, error=repr(exc), traceback=traceback.format_exc(),
            all_partial_outputs_preserved=True, automatic_retry=False, GPU_work_started=False, goal_achieved=False))
        raise


if __name__ == '__main__': main()
