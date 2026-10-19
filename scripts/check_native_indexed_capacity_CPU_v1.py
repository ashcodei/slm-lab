"""Owned CUDA-hidden capacity, fixed fixture horizon and fresh cold readback."""
import hashlib
import json
import os
from pathlib import Path
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/native-indexed-capacity-v2-proof-v1'


def main():
    os.chdir(ROOT)
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    for key in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']:
        os.environ[key] = '1'
    os.environ['PYTEST_DISABLE_PLUGIN_AUTOLOAD'] = '1'
    os.setpriority(os.PRIO_PROCESS, 0, 19)
    os.sched_setscheduler(0, os.SCHED_IDLE, os.sched_param(0))
    os.sched_setaffinity(0, {22})
    predecessor = ROOT / 'reports/native-indexed-neural-v4-proof-v1'
    old_plan = json.loads((predecessor / 'plan.json').read_text())
    old_done = json.loads((predecessor / 'complete.json').read_text())
    old_proof_path = predecessor / 'pytest/test_actual_four_replicas_both0/complete.json'
    old_proof = json.loads(old_proof_path.read_text())
    if (old_done['pytest_exit_code'] != 0 or not old_done['every_actual_original_baseline_and_new_source_byte_unchanged']
            or old_proof['actual_physical_small_CPU_AdamW_calls'] != 32
            or not old_proof['every_pre_Adam_observation_durable_before_actual_optimizer_call']
            or not old_proof['all_complete_peer_failure_samples_and_verdicts_retained']):
        raise ValueError('The complete actual four-replica indexed neural/failure predecessor must pass')
    required = [('native-history-summary-v2-tests.log', '9 passed'),
        ('native-source-batch-and-allocated-guard-v1-tests.log', '8 passed'),
        ('native-indexed-capacity-v2-rejection-tests.log', '2 passed')]
    if any(text not in (ROOT / 'reports' / name).read_text() for name, text in required):
        raise ValueError('All streamed accounting, actual allocated-block, source-boundary and admission checks must pass first')
    paths = [Path(__file__).resolve(), old_proof_path, predecessor / 'complete.json',
        ROOT / 'src/slm_lab/native_suffix_batched_history_stage_v5.py',
        ROOT / 'src/slm_lab/native_history_summary_v1.py', ROOT / 'src/slm_lab/native_allocated_history_guard_v1.py',
        ROOT / 'src/slm_lab/native_suffix_capacity_controller_v2.py',
        ROOT / 'src/slm_lab/native_suffix_capacity_evidence_v2.py',
        ROOT / 'src/slm_lab/native_suffix_capacity_readback_v2.py',
        ROOT / 'src/slm_lab/native_suffix_full_horizon_capacity_v2.py',
        ROOT / 'tests/test_native_suffix_batched_history_stage_v5.py',
        ROOT / 'tests/test_native_suffix_capacity_indexed_v2.py',
        *[ROOT / 'reports' / name for name, _ in required]]
    bindings = dict(old_plan['source_files_sha256'])
    bindings.update({str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    def intact():
        return all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == h for p, h in bindings.items())
    if not intact(): raise ValueError('Every actual original and new code/proof byte must remain unchanged')
    OUT.mkdir(parents=True, exist_ok=False)
    owner = dict(pid=os.getpid(), ticks=Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19])
    (OUT / 'plan.json').write_text(json.dumps(dict(at=datetime.now(timezone.utc).isoformat(), owner=owner,
        source_files_sha256=bindings, actual_CPU_replicas=4, profiles=['anchor_CE', 'anchor_execution'],
        planned_physical_small_AdamW_calls=32, declared_complete_CPU_source_presentations=9,
        distinct_original_human_programs=1, CPU_fixture_horizon=5, original_native_horizon=512,
        complete_fixed_fixture_horizon_positions=[0, 8], independent_fresh_readback_optimizer_calls=0,
        actual_block_budget_and_source_boundary_batching_required=True,
        full_size_GPUcapacity_or_quality_not_claimed=True, production_updates=0, goal_achieved=False), indent=2) + '\n')
    print('Starting owned indexed-capacity/fixed-horizon/fresh-cold proof', owner, flush=True)
    import pytest
    code = pytest.main(['-q', '-s',
        'tests/test_native_suffix_capacity_indexed_v2.py::test_actual_four_replicas_both_profiles_indexed_capacity_horizon_fresh_readback_and_cold',
        '--basetemp=' + str(OUT / 'pytest')])
    unchanged = intact()
    (OUT / ('complete.json' if code == 0 and unchanged else 'failure.json')).write_text(json.dumps(dict(owner=owner,
        at=datetime.now(timezone.utc).isoformat(), pytest_exit_code=int(code),
        all_actual_source_and_predecessor_bytes_unchanged=unchanged,
        full_size_GPUcapacity_or_quality=False, production_updates=0, goal_achieved=False), indent=2) + '\n')
    if not unchanged: raise ValueError('A closed original/new source changed during the proof')
    return int(code)


if __name__ == '__main__':
    sys.exit(main())
