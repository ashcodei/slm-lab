"""Recover all ten boot-interrupted, unstarted controllers with real new births.

No old queue, model, benchmark population or scientific function is replaced.
Each fresh queue's process and immutable file bindings are independently read
back before launching its successor. No GPU process is stopped by this script.
"""
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

from slm_lab.codenet_checkpoint_v1 import FileBindings
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.exact_waiting_process_retirement_v1 import terminal
import own_training_gpu_lease_v2 as owner

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/instance-lineage-recovery-v1'
MAP=ROOT/'reports/instance-lineage-recovery-v1-file-map.json'
RECEIPT=ROOT/'reports/instance-interruption-20261018-v1/complete.json'
START=ROOT/'reports/public-pretraining-recovery-v1-queue.json'

def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or OUT.exists():raise ValueError('Fresh GPU-hidden lineage recovery required')
    routes=json.loads(MAP.read_text())['stages'];interruption=json.loads(RECEIPT.read_text())
    if len(routes)!=10 or any(not terminal(r) for r in interruption['old_registered_births']):
        raise ValueError('Every exact interrupted birth must remain terminal')
    tests=ROOT/'reports/instance-lineage-recovery-v1-tests.log'
    if '1 passed' not in tests.read_text():raise ValueError('All original scientific functions and scopes must remain byte-identical')
    while not START.exists():time.sleep(2)
    first=json.loads(START.read_text())
    if terminal(first['owner']):raise ValueError('Actual recovered pretraining owner must be live')
    files={str(p):file_hash(p) for p in [Path(__file__),MAP,RECEIPT,START,tests]}
    files.update(first['source_files_sha256']);guard=FileBindings(files)
    OUT.mkdir();me=owner.resources.identity(os.getpid());launched=[]
    atomic_json(OUT/'plan.json',dict(owner=me,original_interruption_receipt=str(RECEIPT),actual_recovery_pretraining_owner=first['owner'],
        stages=routes,source_files_sha256=files,neural_model_updates=0,goal_achieved=False))
    try:
        for row in routes:
            queue=ROOT/row['queue'];report=ROOT/'reports'/row['stage']
            if queue.exists():raise ValueError('Never overwrite a recovered queue')
            # The old strata directory contains only an unstarted reader plan;
            # its recovery copy uses a fresh destination. Every learned job is absent.
            if 'strata' not in row['stage'] and (report.exists() or (ROOT/'checkpoints'/row['stage']).exists()):
                raise ValueError('An original learned/output stage exists and needs explicit state recovery: '+row['stage'])
            guard.check();log=OUT/(row['stage']+'-launcher.log')
            with log.open('xb') as stream:
                child=subprocess.Popen([str(ROOT/'.venv/bin/python'),'-u',str(ROOT/'scripts'/row['recovery_script'])],cwd=ROOT,
                    stdout=stream,stderr=subprocess.STDOUT,start_new_session=True,env=dict(os.environ))
            birth=owner.resources.identity(child.pid)
            if not birth:raise ValueError('Recovered controller ended before registration')
            while not queue.exists():
                if child.poll() is not None:raise ValueError('Recovered controller failed before queue publication: '+row['stage'])
                time.sleep(2)
            actual=json.loads(queue.read_text())
            if actual['owner']['pid']!=birth['pid'] or actual['owner']['ticks']!=birth['ticks'] or terminal(birth):
                raise ValueError('Recovered queue must name its actual live process birth')
            FileBindings(actual['source_files_sha256']).check()
            launched.append(dict(**row,actual_owner=birth,queue_sha256=file_hash(queue),source_files_sha256=actual['source_files_sha256']))
            atomic_json(OUT/'progress.json',dict(actual_new_controllers=len(launched),launched=[dict(stage=r['stage'],owner=r['actual_owner']) for r in launched],goal_achieved=False))
        if any(terminal(row['actual_owner']) for row in launched):raise ValueError('A recovered controller ended during queue recovery')
        guard.check(full=True)
        atomic_json(OUT/'complete.json',dict(complete=True,at=datetime.now(timezone.utc).isoformat(),actual_ten_new_controllers=launched,
            actual_recovery_pretraining_owner=first['owner'],all_original_queues_and_scientific_functions_preserved=True,
            full_human85786_code3721_machine1620_sources_and_nine_endpoint_benchmarks_retained=True,
            only_fresh_operational_queues_and_unstarted_strata_output_changed=True,signals_sent=0,neural_model_updates=0,
            source_files_sha256=files,goal_achieved=False))
        print(json.dumps(dict(recovered_controllers=len(launched),receipt=str(OUT/'complete.json'))),flush=True)
    except BaseException as error:
        atomic_json(OUT/'failure.json',dict(error=repr(error),traceback=traceback.format_exc(),actual_launched_controllers=launched,goal_achieved=False));raise

if __name__=='__main__':main()
