"""Source-bound four-replica indexed-history neural proof, never GPU training."""
import hashlib
import json
import os
from pathlib import Path
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/native-indexed-neural-v4-proof-v1'


def main():
    os.chdir(ROOT)
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    for key in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']:
        os.environ[key] = '1'
    os.environ['PYTEST_DISABLE_PLUGIN_AUTOLOAD'] = '1'
    os.setpriority(os.PRIO_PROCESS, 0, 19)
    os.sched_setscheduler(0, os.SCHED_IDLE, os.sched_param(0))
    os.sched_setaffinity(0, {22})
    baseline = ROOT / 'reports/native-compact-neural-v3-proof-v1'
    old_plan = json.loads((baseline / 'plan.json').read_text())
    old_done = json.loads((baseline / 'complete.json').read_text())
    old_proof = baseline / 'pytest/test_actual_four_replicas_both0/complete.json'
    old = json.loads(old_proof.read_text())
    if (old_done['pytest_exit_code'] != 0 or not old_done['all_original_source_bytes_unchanged']
            or old['actual_physical_small_CPU_AdamW_calls'] != 56
            or not old['all_matched_neural_updates_and_durable_replay_exact']):
        raise ValueError('The complete exact original 56-call baseline must remain successful')
    if '1 passed' not in (ROOT / 'reports/native-indexed-stage-v4-smoke-tests.log').read_text():
        raise ValueError('The actual integrated single-replica neural/state/recovery check must pass first')
    if ('6 passed' not in (ROOT / 'reports/indexed-native-history-v2-tests.log').read_text()
            or '1 passed' not in (ROOT / 'reports/indexed-native-history-v2-prefix-tests.log').read_text()):
        raise ValueError('All six unchanged and one corrected-prefix failure-storage cases must pass first')
    paths = [Path(__file__).resolve(), ROOT / 'src/slm_lab/native_suffix_indexed_stage_v4.py',
        ROOT / 'src/slm_lab/indexed_native_history_v1.py', ROOT / 'src/slm_lab/indexed_native_history_v2.py',
        ROOT / 'tests/test_native_suffix_indexed_stage_v4.py', old_proof, baseline / 'complete.json',
        ROOT / 'reports/native-indexed-stage-v4-smoke-tests.log']
    for profile in ['anchor_CE', 'anchor_execution']:
        folder = baseline / 'pytest/test_actual_four_replicas_both0' / profile
        paths.append(folder / 'baseline.nativeobs.gz')
        paths.extend((folder / 'durable').glob('*'))
        paths.extend((folder.parent).glob(profile + '-rank-*.json'))
    bindings = dict(old_plan['source_files_sha256'])
    bindings.update({str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if p.is_file()})
    def intact():
        return all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == h for p, h in bindings.items())
    if not intact(): raise ValueError('Every original baseline and actual indexed source byte must remain exact')
    OUT.mkdir(parents=True, exist_ok=False)
    owner = dict(pid=os.getpid(), ticks=Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19])
    plan = dict(at=datetime.now(timezone.utc).isoformat(), owner=owner, source_files_sha256=bindings,
        actual_CPU_replicas=4, profiles=['anchor_CE', 'anchor_execution'],
        planned_physical_small_AdamW_calls=32, baseline_56_calls_reused_without_retraining=True,
        explicit_real_sampling_failure_groups_without_Adam=4,
        one_original_human_program_repeated=9, CPU_fixture_horizon=5, original_native_horizon=512,
        only_verified_additional_source_metadata_normalized_for_baseline_comparison=True,
        pre_Adam_all_peer_exact_durable_readback_required=True,
        no_full_size_GPU_capacity_or_new_quality_claim=True, production_updates=0, goal_achieved=False)
    (OUT / 'plan.json').write_text(json.dumps(plan, indent=2) + '\n')
    print('Starting owned indexed-history four-replica neural proof', owner, flush=True)
    import pytest
    code = pytest.main(['-q', '-s',
        'tests/test_native_suffix_indexed_stage_v4.py::test_actual_four_replicas_both_profiles_pre_Adam_durable_receipts_fresh_replay_and_complete_peer_failure',
        '--basetemp=' + str(OUT / 'pytest')])
    unchanged = intact()
    report = dict(owner=owner, at=datetime.now(timezone.utc).isoformat(), pytest_exit_code=int(code),
        every_actual_original_baseline_and_new_source_byte_unchanged=unchanged,
        production_updates=0, new_quality_gain=False, goal_achieved=False)
    (OUT / ('complete.json' if code == 0 and unchanged else 'failure.json')).write_text(json.dumps(report, indent=2) + '\n')
    if not unchanged: raise ValueError('The owned proof source or original baseline changed')
    return int(code)


if __name__ == '__main__':
    sys.exit(main())
