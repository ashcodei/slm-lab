"""Restore all ten unstarted scopes after their exact owners ended on eval failure.

This sends no signals. A new optional repair study and all canonical human,
code, RL and full final benchmark controllers wait behind the independently
validated complete epoch and separate full native assessment recovery.
"""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import time
import traceback

from slm_lab.codenet_checkpoint_v1 import FileBindings
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.exact_waiting_process_retirement_v1 import terminal
from slm_lab.function_repair_GPU_contract_v2 import QUEUE as GPU_QUEUE, REPORT as GPU_REPORT, ADMISSION
from slm_lab.local_source_closure_v2 import python_sources
from slm_lab.public_pretraining_assessment_recovery_v1 import completed_training_after_observer_failure
import own_training_gpu_lease_v2 as owner

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/assessment-recovery-lineage-v1'
MAP = ROOT / 'reports/assessment-recovery-lineage-v1-file-map.json'
OLD = ROOT / 'reports/function-repair-GPU-lineage-v1/admission.json'


def read(path):
    return json.loads(Path(path).read_text())


def scope(queue):
    return {k: v for k, v in queue.items() if k not in {'at', 'status', 'source_files_sha256', 'expires_at_unix'}
        and 'owner' not in k and 'predecessor' not in k}


def no_learned_outputs(row):
    text = (ROOT / 'scripts' / row['previous_script']).read_text()
    paths = re.findall(r"(?:REPORTS|OUT|OUTPUT)\s*=\s*ROOT\s*/\s*['\"]([^'\"]+)['\"]", text)
    if not paths:
        raise ValueError('Explicit original scientific output paths required')
    for name in paths:
        p = ROOT / name
        if not p.exists():
            continue
        if (row['stage'] == 'AI2D-fixed-exposure-strata-v5'
                and name == 'reports/AI2D-fixed-exposure-strata-repair-gate-v1'
                and p.is_dir() and set(x.name for x in p.iterdir()) == {'plan.json', 'failure.json'}
                and read(p / 'plan.json') == read(ROOT / row['previous_queue'])):
            continue
        raise ValueError('Existing scientific output cannot be substituted or overwritten: ' + name)


def launch(script, queue, log):
    path = ROOT / queue
    if path.exists():
        raise ValueError('Every actual new controller queue must be fresh')
    with (OUT / log).open('xb') as stream:
        child = subprocess.Popen([str(ROOT / '.venv/bin/python'), '-u', str(ROOT / 'scripts' / script)], cwd=ROOT,
            stdout=stream, stderr=subprocess.STDOUT, start_new_session=True, env=dict(os.environ))
    birth = owner.resources.identity(child.pid)
    if not birth:
        raise ValueError('New controller ended before registration')
    while not path.exists():
        if child.poll() is not None:
            raise ValueError('New controller ended before publishing its actual queue: ' + script)
        time.sleep(2)
    queue_value = read(path)
    if (any(queue_value['owner'][k] != birth[k] for k in ['pid', 'ticks']) or terminal(queue_value['owner'])):
        raise ValueError('Actual new queue and live controller birth must match')
    FileBindings(queue_value['source_files_sha256']).check()
    return queue_value


def main():
    if (os.environ.get('CUDA_VISIBLE_DEVICES') != '' or OUT.exists()
            or (ROOT / GPU_QUEUE).exists() or (ROOT / GPU_REPORT).exists()):
        raise ValueError('Fresh CUDA-hidden recovery handoff required')
    os.nice(max(0, 19 - os.nice(0)))
    os.sched_setscheduler(0, os.SCHED_IDLE, os.sched_param(0))
    if not completed_training_after_observer_failure(ROOT):
        raise ValueError('Full verified epoch and actual terminal original GPU runtime required')
    tests = ROOT / 'reports/assessment-recovery-lineage-v1-tests.log'
    if '5 passed' not in tests.read_text():
        raise ValueError('Exact ten scopes, numerical functions and fail-closed recovery routing tests required')
    mapping = read(MAP)
    previous = read(OLD)['new_controllers']
    if len(mapping['stages']) != 10 or len(previous) != 10:
        raise ValueError('All ten original canonical scopes required')
    files = dict(mapping['source_files_sha256'])
    for p in [MAP, OLD, tests, *python_sources(ROOT, [Path(__file__), ROOT / 'tests/test_assessment_recovery_lineage_v1.py'])]:
        files[str(p)] = file_hash(p)
    old_gpu = read(ROOT / 'reports/full-function-repair-comparison-GPU-v1-queue.json')['owner']
    if not terminal(old_gpu):
        raise ValueError('Exact previous optional study controller must already be terminal')
    snapshots = []
    for row, actual in zip(mapping['stages'], previous):
        if row['stage'] != actual['stage'] or row['previous_script'] != actual['script']:
            raise ValueError('Every exact immediately preceding canonical controller required')
        q = read(ROOT / row['previous_queue'])
        if any(q['owner'][k] != actual['owner'][k] for k in ['pid', 'ticks']) or not terminal(q['owner']):
            raise ValueError('Only already terminal exact old births may be replaced')
        no_learned_outputs(row)
        snapshots.append(q)
    guard = FileBindings(files)
    guard.check(full=True)
    OUT.mkdir()
    atomic_json(OUT / 'plan.json', dict(at=datetime.now(timezone.utc).isoformat(), owner=owner.resources.identity(os.getpid()),
        source_files_sha256=files, exact_old_owners=[q['owner'] for q in snapshots], old_optional_GPU_owner=old_gpu,
        all_old_owners_already_terminal=True, no_signals_sent=True, no_existing_neural_job_changed=True,
        old_failed_runtime_preserved=True, canonical_initializer_and_all_nine_final_endpoints_unchanged=True,
        goal_achieved=False))
    launched = []
    gpu = None
    try:
        gpu = launch('train_full_function_repair_comparison_GPU_v2.py', GPU_QUEUE, 'GPU-controller-launcher.log')
        for row, old in zip(mapping['stages'], snapshots):
            new = launch(row['recovery_script'], row['queue'], row['stage'] + '-launcher.log')
            if scope(old) != scope(new):
                raise ValueError('Full original numerical source or benchmark scope changed: ' + row['stage'])
            launched.append(dict(stage=row['stage'], queue=row['queue'], owner=new['owner'], script=row['recovery_script'],
                original_queue_scope=scope(old), actual_new_queue_scope=scope(new)))
            atomic_json(OUT / 'progress.json', dict(actual_new_controllers=len(launched), new_controllers=launched,
                admission_open=False, goal_achieved=False))
        if terminal(gpu['owner']) or any(terminal(q['owner']) for q in launched):
            raise ValueError('All gated replacement owners must be actually live before admission')
        guard.check(full=True)
        admission = dict(complete=True, at=datetime.now(timezone.utc).isoformat(), GPU_study_owner=gpu['owner'],
            old_ten_waiting_owners_retired=True, old_ten_owners_already_terminal_before_recovery=True,
            no_signals_sent=True, all_original_neural_functions_and_benchmark_populations_preserved=True,
            new_controllers=launched, source_files_sha256=files, model_updates=0,
            human_pairs_each_arm=85786, coding_sources_each_arm=3721, machine_pairs_each_arm=1620,
            all_nine_full_final_endpoints_preserved=True, canonical_human_RL_initializer_unchanged=True,
            failed_runtime_not_relabelled_successful=True, candidate_promoted=False, goal_achieved=False)
        atomic_json(ROOT / ADMISSION, admission)
        atomic_json(OUT / 'complete.json', dict(**admission, admission_sha256=file_hash(ROOT / ADMISSION)))
        print(json.dumps(dict(admission=str(ROOT / ADMISSION), actual_new_controllers=10, neural_workers_signalled=0)), flush=True)
    except BaseException as error:
        atomic_json(OUT / 'failure.json', dict(error=repr(error), traceback=traceback.format_exc(), new_controllers=launched,
            GPU_study_owner=None if gpu is None else gpu['owner'], admission_open=(ROOT / ADMISSION).exists(),
            automatic_retry=False, no_signals_sent=True, goal_achieved=False))
        raise


if __name__ == '__main__':
    main()
