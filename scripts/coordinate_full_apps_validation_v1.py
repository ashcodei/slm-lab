"""Reconcile complete held-source scores; use only idle owned GPU windows."""
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import subprocess
import time

import train_apps_full_language_v1 as owner
from slm_lab.apps_human_solution_view_v1 import AppsHumanSolutions
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json

FOLDER=owner.REPORTS/'APPS-validation'


def combine(phase):
    target=FOLDER/(phase+'-complete.json')
    if target.exists():return json.loads(target.read_text())
    source=AppsHumanSolutions(owner.SOURCE,role='validation',purpose='audit')
    reports=[json.loads((FOLDER/(phase+'-'+str(i)+'.json')).read_text()) for i in range(3)]
    records=sorted([r for x in reports for r in x['records']],key=lambda r:r['position'])
    if len(records)!=12068 or [r['position'] for r in records]!=list(range(12068)):
        raise ValueError('Every original validation program exactly once required')
    if len({r['checkpoint_sha256'] for r in reports})!=1:raise ValueError('Same fixed own policy across every shard required')
    targets=correct=0;nll=0.
    for actual,original in zip(records,source.records):
        if (actual['id']!=original['id'] or actual['response_targets']!=original['target_tokens']
                or not math.isfinite(actual['mean_NLL']) or actual['mean_NLL']<0
                or not 0<=actual['correct_teacher_forced_targets']<=actual['response_targets']):
            raise ValueError('Every original source identity/full target span and finite score must reconcile')
        n=actual['response_targets'];targets+=n;correct+=actual['correct_teacher_forced_targets'];nll+=n*actual['mean_NLL']
    if targets!=3851168:raise ValueError('All original held response/EOS targets required')
    result=dict(observed_at=datetime.now(timezone.utc).isoformat(),phase=phase,checkpoint_sha256=reports[0]['checkpoint_sha256'],
        original_validation_programs=12068,response_targets=targets,token_mean_NLL=nll/targets,
        correct_teacher_forced_targets=correct,teacher_forced_token_accuracy=correct/targets,
        source_admission_sha256=file_hash(owner.SOURCE/'admission.json'),
        files_sha256={str(FOLDER/(phase+'-'+str(i)+'.json')):file_hash(FOLDER/(phase+'-'+str(i)+'.json')) for i in range(3)},
        all_source_IDs_and_response_targets_reconciled=True,model_updates=0,
        used_for_gradients_early_stopping_or_policy_selection=False,functional_correctness_measured=False,
        public_internal_validation_not_certified_independent_final_benchmark=True,goal_achieved=False)
    atomic_json(target,result);print(json.dumps({k:v for k,v in result.items() if k!='files_sha256'}),flush=True)
    return result


def main():
    plan=owner.register('read-only-full-APPS-validation-coordinator')
    base=combine('baseline')
    while True:
        owner.ownership(plan)
        state=json.loads((owner.REPORTS/'status.json').read_text())
        if state.get('phase')=='candidate' and all((owner.REPORTS/'candidate'/('suite-'+str(i)+'-complete.json')).exists() for i in [1,2,3]):break
        time.sleep(5)
    if (FOLDER/'candidate-launches.json').exists():raise ValueError('No implicit duplicate validation launch')
    actors=[];children=[]
    for i in range(3):
        with (FOLDER/('candidate-'+str(i)+'.log')).open('xb') as stream:
            p=subprocess.Popen(['.venv/bin/python','-u','scripts/assess_full_apps_validation_v1.py','--phase','candidate','--shard',str(i)],
                cwd=owner.ROOT,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True,
                env={**os.environ,'CUDA_VISIBLE_DEVICES':str(i+1),'OMP_NUM_THREADS':'1'})
        children.append(p);actors.append(dict(**owner.resources.identity(p.pid),shard=i))
    atomic_json(FOLDER/'candidate-launches.json',dict(actors=actors,model_updates=0,goal_achieved=False))
    while any(p.poll() is None for p in children):
        owner.ownership(plan)
        if any(p.poll() not in [None,0] for p in children):raise ValueError('A checked full-validation worker failed; preserve partial observations')
        time.sleep(2)
    if any(p.returncode!=0 for p in children):raise ValueError('Full validation did not complete')
    candidate=combine('candidate')
    atomic_json(FOLDER/'comparison.json',dict(baseline=base,candidate=candidate,
        token_NLL_change=candidate['token_mean_NLL']-base['token_mean_NLL'],
        teacher_forced_token_accuracy_change=candidate['teacher_forced_token_accuracy']-base['teacher_forced_token_accuracy'],
        measured_functional_code_gain=False,candidate_promoted=False,goal_achieved=False))


if __name__=='__main__':main()
