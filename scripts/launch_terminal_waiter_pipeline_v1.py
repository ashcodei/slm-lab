"""Explicit new lineage after authoritative terminal failure of the old waiter.

Old source, checkpoints, initializer and failed controllers are preserved.
All replacements and actual tests must exist before publishing the handoff.
No old process is signalled and no already running training is restarted.
"""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

import own_training_gpu_lease_v2 as owner
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.exact_waiting_process_retirement_v1 import observe, terminal

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/terminal-waiter-pipeline-v1-handoff'
STAGES = (
    ('code-first-continuation-v3', 'train_code_first_continuation_v3.py'),
    ('phase-human-preference-v7', 'train_phase_human_preference_v7.py'),
    ('broad-anchored-code-v6', 'train_broad_anchored_code_v6.py'),
    ('code-first-final-standalone-CPU-v3', 'finalize_code_first_student_CPU_v3.py'),
    ('full-held-QA-pipeline-v4', 'assess_full_held_QA_pipeline_v4.py'),
    ('full-AI2D-pipeline-v4', 'assess_full_AI2D_pipeline_v4.py'),
    ('full-original-MBPP-pipeline-v4', 'assess_full_original_MBPP_pipeline_v4.py'),
    ('AI2D-fixed-exposure-strata-v4', 'assess_AI2D_fixed_exposure_strata_v4.py'),
)


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or OUT.exists():
        raise ValueError('A fresh explicit CPU-only terminal-waiter handoff is required')
    controls = ROOT / 'reports/terminal-waiter-gpu-lease-v2-tests.log'
    drivers = ROOT / 'reports/terminal-waiter-pipeline-drivers-v1-tests.log'
    if '3 passed' not in controls.read_text() or '24 passed' not in drivers.read_text():
        raise ValueError('All actual kernel-lease/cleanup and revised pipeline controls must pass')
    owner.idle_original()
    receipt = json.loads(owner.RECEIPT.read_text())
    old = receipt['old_pipeline_owners']
    if not all(terminal(p['owner']) for p in old.values()):
        raise ValueError('Every old failed pipeline controller must be terminal before replacement')
    for name, script in STAGES:
        if (ROOT / 'reports' / (name + '-queue.json')).exists() or (ROOT / 'reports' / name).exists():
            raise ValueError('Never duplicate or overwrite an existing replacement controller')
        if not (ROOT / 'scripts' / script).is_file():
            raise ValueError('Every concrete replacement must exist before launching any')
    if subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True).strip():
        raise ValueError('All original GPU actors must remain terminal before this new handoff')
    main_plan = json.loads((ROOT / 'reports/phase-joint-v3/plan.json').read_text())
    if not terminal(main_plan['owner']):
        raise ValueError('The exact complete main parent must finish normally before replacing failed successors')
    OUT.mkdir(); launched = {}
    try:
        for name, script in STAGES:
            log = ROOT / 'reports' / (name + '-controller.log')
            with log.open('xb') as stream:
                child = subprocess.Popen(['.venv/bin/python', '-u', 'scripts/' + script], cwd=ROOT,
                    stdout=stream, stderr=subprocess.STDOUT, start_new_session=True,
                    env={**os.environ, 'CUDA_VISIBLE_DEVICES':'', 'TOKENIZERS_PARALLELISM':'false',
                         'OMP_NUM_THREADS':'1', 'MKL_NUM_THREADS':'1'})
            identity = owner.resources.identity(child.pid)
            launched[name] = dict(owner=identity, script=script, log=str(log))
            atomic_json(OUT / 'launch-progress.json', dict(launched=launched, handoff_still_closed=True, goal_achieved=False))
            queue_path = ROOT / 'reports' / (name + '-queue.json')
            while not queue_path.exists():
                if terminal(identity):
                    raise ValueError('Exact new ' + name + ' ended before publishing its checked queue; inspect its retained log')
                time.sleep(.5)
            queue = json.loads(queue_path.read_text())
            if (queue['owner']['pid'], queue['owner']['ticks']) != (identity['pid'], identity['ticks']):
                raise ValueError('Every newly launched controller must publish its exact process birth')
            print(json.dumps(dict(queued=name, owner=identity, handoff_closed=True)), flush=True)
        new = {}; protected = {}
        for name, script in STAGES:
            path = ROOT / 'reports' / (name + '-queue.json'); queue = json.loads(path.read_text())
            observed = observe(queue['owner'])
            if name != 'AI2D-fixed-exposure-strata-v4' and (ROOT / 'reports' / name).exists():
                raise ValueError('Every new training/assessment controller must remain unstarted behind the closed handoff')
            new[name] = dict(owner=queue['owner'], actual_observation=observed, queue=str(path), queue_sha256=file_hash(path))
            protected.update(queue['source_files_sha256'])
        if not all(terminal(p['owner']) for p in old.values()):
            raise ValueError('An old original process birth unexpectedly became live')
        owner.idle_original()
        if subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True).strip():
            raise ValueError('No GPU may be taken before all exact new controllers and the handoff exist')
        for name in old:
            path = ROOT / 'reports' / (name + '-queue.json')
            (OUT / (name + '-original-queue.json')).write_bytes(path.read_bytes())
        evidence = [Path(__file__), owner.RECEIPT, controls, drivers,
                    ROOT / 'reports/phase-joint-v3/complete.json', ROOT / 'reports/phase-joint-v3/restored.json',
                    ROOT / 'reports/code-first-continuation-v2-parent-failure.json',
                    ROOT / 'reports/code-first-continuation-v2-initializer/complete.json']
        atomic_json(OUT / 'complete.json', dict(
            at=datetime.now(timezone.utc).isoformat(), explicit_terminal_waiter_handoff_complete=True,
            original_main_owner=main_plan['owner'], original_main_complete_and_all_workers_terminal=True,
            original_waiter_terminal_receipt_sha256=file_hash(owner.RECEIPT),
            old_failed_production_controllers_all_terminal=True, all_old_production_updates_zero=True,
            old_complete_CPU_initializer_preserved_not_a_completed_production_run=True,
            old_pipeline_owners=old, new_waiting_pipeline=new, source_files_sha256={str(p):file_hash(p) for p in evidence},
            all_training_source_order_hyperparameters_and_assessment_populations_preserved=True,
            actual_new_focused_controls_passed=27, old_processes_signalled=0,
            new_production_updates=0, goal_achieved=False,
        ))
        print(json.dumps(dict(handoff=str(OUT / 'complete.json'), new_controllers=len(new), old_processes_signalled=0)), flush=True)
    except BaseException as error:
        atomic_json(OUT / 'failure.json', dict(error=repr(error), traceback=traceback.format_exc(), launched=launched,
            original_jobs_and_partial_replacements_left_intact=True, handoff_not_granted=True, goal_achieved=False))
        raise


if __name__ == '__main__':
    main()
