"""Owned idle CPU proof behind the exact complete indexed capacity owner."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/native-indexed-training-v3-proof-v1'
CAPACITY = ROOT / 'reports/native-indexed-capacity-v2-proof-v1'


def main():
    os.chdir(ROOT)
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['PYTEST_DISABLE_PLUGIN_AUTOLOAD'] = '1'
    for name in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']: os.environ[name] = '1'
    os.setpriority(os.PRIO_PROCESS, 0, 19)
    os.sched_setscheduler(0, os.SCHED_IDLE, os.sched_param(0))
    os.sched_setaffinity(0, {22})
    from slm_lab.corpus import file_hash
    from slm_lab.exact_waiting_process_retirement_v1 import terminal
    from slm_lab.bounded_disk_reserve_wait_v1 import wait_for_reserve
    from slm_lab.reserved_atomic_metadata_v1 import write_json

    cap_plan = json.loads((CAPACITY / 'plan.json').read_text())
    owner = dict(pid=os.getpid(), ticks=Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19])
    files = ['src/slm_lab/native_training_runtime_bindings_v1.py',
        'src/slm_lab/native_indexed_history_inventory_v3.py', 'tests/test_native_indexed_history_inventory_v3.py',
        'reports/native-indexed-history-inventory-v3-tests.log',
        'src/slm_lab/native_suffix_training_controller_v3.py', 'src/slm_lab/native_suffix_standalone_export_v3.py',
        'tests/test_native_indexed_training_contract_v2.py', 'tests/test_native_indexed_export_records_v2.py',
        'tests/test_native_indexed_training_recovery_v3.py', 'scripts/check_native_indexed_training_CPU_v2.py',
        'reports/native-indexed-training-contract-v2-tests.log', 'reports/native-indexed-export-records-v2-tests.log']
    previous = ROOT / 'reports/native-indexed-training-v2-proof-v1'
    failed = json.loads((previous / 'failure.json').read_text())
    previous_plan = json.loads((previous / 'plan.json').read_text())
    if (not terminal(previous_plan['owner']) or failed['pytest_exit_code'] != 1
            or not failed['all_actual_capacity_and_runtime_source_bytes_unchanged']
            or 'Each retained attempt must keep its physically funded independent failure slot'
                not in (ROOT / 'reports/native-indexed-training-v2-proof-v1.log').read_text()
            or '8 passed' not in (ROOT / 'reports/native-indexed-history-inventory-v3-tests.log').read_text()):
        raise ValueError('Preserve the actual terminal consumed-slot regression and require the corrected complete readback')
    files.extend(['reports/native-indexed-training-v2-proof-v1/plan.json',
        'reports/native-indexed-training-v2-proof-v1/failure.json', 'reports/native-indexed-training-v2-proof-v1.log'])
    # All previous successful neural boundaries and both real captured
    # failures stay immutable inputs; this is a distinct corrected experiment.
    files.extend(str(p.relative_to(ROOT)) for p in (previous / 'pytest').rglob('*') if p.is_file())
    bindings = dict(cap_plan['source_files_sha256'])
    bindings[str(CAPACITY / 'plan.json')] = file_hash(CAPACITY / 'plan.json')
    bindings.update({str(ROOT / path): file_hash(ROOT / path) for path in files})
    def intact():
        if any(file_hash(path) != digest for path, digest in bindings.items()):
            raise ValueError('Every original capacity/source and additional runtime byte must remain exact')
    intact()
    if ('4 passed' not in (ROOT / 'reports/native-indexed-training-contract-v2-tests.log').read_text()
            or '8 passed' not in (ROOT / 'reports/native-indexed-export-records-v2-tests.log').read_text()):
        raise ValueError('Actual closed-chain streaming and runtime contract checks must first pass')
    OUT.mkdir(parents=True, exist_ok=False)
    minimum = 40 * 1024**3
    def publish(name, value):
        write_json(OUT / name, value, root=OUT, minimum_free_bytes=minimum,
            maximum_metadata_bytes=4 * 1024**2, guard=intact)
    publish('plan.json', dict(at=datetime.now(timezone.utc).isoformat(), owner=owner,
        exact_capacity_predecessor=cap_plan['owner'], source_files_sha256=bindings,
        original_policy_data_sources_not_replaced_by_additional_runtime_bindings=True,
        profiles=['anchor_CE', 'anchor_execution'], actual_CPU_replicas=4,
        planned_small_CPU_physical_AdamW_calls=56,
        declared_source_presentations=9, distinct_original_human_programs=1,
        explicit_CPU_fixture_horizon=5, original_native_horizon=512,
        post_Adam_root_failure_and_original_uncommitted_record_retention_required=True,
        whole_declared_fixture_training_and_exact_recovery_and_native_cold_required=True,
        production_updates=0, full_size_GPUcapacity_or_quality_gain=False, goal_achieved=False))
    print('Waiting for exact indexed capacity owner', cap_plan['owner'], 'new owner', owner, flush=True)
    while not terminal(cap_plan['owner']):
        intact(); time.sleep(5)
    done_path = CAPACITY / 'complete.json'
    if not done_path.exists(): raise ValueError('The exact capacity owner is terminal without successful complete evidence')
    done = json.loads(done_path.read_text())
    fixture = CAPACITY / 'pytest/test_actual_four_replicas_both0'
    proof = json.loads((fixture / 'complete.json').read_text())
    if (done['pytest_exit_code'] != 0 or not done['all_actual_source_and_predecessor_bytes_unchanged']
            or proof['actual_physical_small_CPU_AdamW_calls'] != 32
            or not proof['complete_dynamic_history_not_materialized']):
        raise ValueError('Every actual indexed capacity/replay/horizon/fresh-cold check must pass unchanged')
    artifacts = {str(path): file_hash(path) for path in fixture.rglob('*') if path.is_file()}
    artifacts[str(done_path)] = file_hash(done_path)
    def all_intact():
        intact()
        if any(file_hash(path) != digest for path, digest in artifacts.items()):
            raise ValueError('Every original successful capacity artifact must remain unchanged')
    all_intact()
    # Three history attempts per profile, 128 MiB independently funded failure
    # slots apiece, original complete records/states/events and four exports.
    # Preserve the original producer's 40 GiB floor throughout this CPU proof.
    wait_for_reserve(OUT, minimum_bytes=minimum + 1024**3, guard=all_intact)
    publish('ready.json', dict(at=datetime.now(timezone.utc).isoformat(), owner=owner,
        capacity_owner_terminal_and_all_actual_checks_passed=True, capacity_artifact_files_sha256=artifacts,
        actual_other_writer_floor_bytes=minimum, initial_additional_CPU_proof_space_bound=1024**3,
        goal_achieved=False))
    print('Starting complete indexed training/recovery/export CPU proof', owner, flush=True)
    import pytest
    code = pytest.main(['-q', '-s',
        'tests/test_native_indexed_training_recovery_v3.py::test_actual_four_replica_complete_training_post_Adam_failure_recovery_and_standalone_cold',
        '--basetemp=' + str(OUT / 'pytest')])
    all_intact()
    publish('complete.json' if code == 0 else 'failure.json', dict(at=datetime.now(timezone.utc).isoformat(),
        owner=owner, pytest_exit_code=int(code), all_actual_capacity_and_runtime_source_bytes_unchanged=True,
        full_size_GPUcapacity_or_quality_gain=False, production_updates=0, goal_achieved=False))
    return int(code)


if __name__ == '__main__':
    sys.exit(main())
