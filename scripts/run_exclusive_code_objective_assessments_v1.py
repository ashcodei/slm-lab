"""Full fixed code/QA for three frozen models, without checkpoint-writing work."""
import argparse
import csv
from datetime import datetime,timezone
import fcntl
import gc
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import torch

import assess_codenet_model_exclusive_v1 as code
import assess_knowledge_exclusive_v1 as knowledge
from assess_code_residual_lr_pilot_v1 import compare_fixed_results
from slm_lab.code_repetition_inference_v1 import load_objective_policy
from slm_lab.code_residual_repeated_inference_v1 import load_repeated_source_policy
from slm_lab.codenet_checkpoint_v1 import FileBindings
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.exclusive_inference_resources_v1 import (ENV,MINIMUM_FREE,REPORT_BUDGET,
    checkpoint_writers,identity,inference_reservation,retire_idle_waiter,validate_active_lease)

ROOT=Path(__file__).resolve().parents[1]
REPORTS=ROOT/'reports/code-objective-exclusive-assessment-v1'


def read(path):return json.loads(path.read_text())


def status(message,**fields):
    atomic_json(REPORTS/'status.json',dict(status=message,updated_at=datetime.now(timezone.utc).isoformat(),
        candidate_promoted=False,goal_achieved=False,**fields))


def subject_spec(name,result,plan,run_plan_sha):
    return dict(name=name,checkpoint=result['checkpoint'],checkpoint_sha256=result['checkpoint_sha256'],
        base_checkpoint=plan['original_base_checkpoint'],base_sha256=plan['original_base_checkpoint_sha256'],
        run_plan_sha256=run_plan_sha,starting_policy_sha256=plan['starting_policy_sha256'],
        selection_sha256=plan['original_selection_sha256'],profile=result.get('profile'),
        stage='code_residual_repeated_source_pilot_v1' if name=='repeated_source' else 'code_repetition_objective_pilot_v1',
        steps=512 if name=='repeated_source' else 128)


def run_subject(name,device):
    plan=read(REPORTS/'plan.json');subject=next(s for s in plan['subjects'] if s['name']==name)
    guard=FileBindings(plan['source_files_sha256']);inference_reservation()
    sub=REPORTS/name
    if sub.exists():raise ValueError('Preserve earlier partial/full subject assessment')
    sub.mkdir()
    def state(message,**fields):
        atomic_json(sub/'status.json',dict(status=message,updated_at=datetime.now(timezone.utc).isoformat(),
            goal_achieved=False,**fields))
    checkpoint=Path(subject['checkpoint']);base=Path(subject['base_checkpoint']);token=checkpoint.parent/'tokenizer.json'
    def loader(path,device='cpu',training_state=True):
        if Path(path).resolve()!=checkpoint.resolve() or training_state is not False:raise ValueError('Only exact frozen inference checkpoint may load')
        kwargs=dict(expected_checkpoint_sha256=subject['checkpoint_sha256'],expected_base_sha256=subject['base_sha256'],
            expected_plan_sha256=subject['run_plan_sha256'],expected_starting_policy_sha256=subject['starting_policy_sha256'],
            expected_selection_sha256=subject['selection_sha256'],device=device)
        if name=='repeated_source':return load_repeated_source_policy(checkpoint,base,token,**kwargs)
        return load_objective_policy(checkpoint,base,token,expected_profile=subject['profile'],**kwargs)
    try:
        code.load_checkpoint=loader;knowledge.load_checkpoint=loader
        state('generating all28 original raw code questions',device=device)
        code.evaluate(checkpoint,sub/'code',device,threads=1,memory_fraction=.14,
            expected_metadata=dict(stage=subject['stage'],steps=subject['steps'],planned_steps=subject['steps'],
                data_sha256=file_hash(ROOT/'data/codenet-durable-pack-v2/manifest.json')),
            label='own857M_exclusive_'+name,expected_checkpoint_sha256=subject['checkpoint_sha256'])
        guard.check(full=True);validate_active_lease();gc.collect();torch.cuda.empty_cache()
        state('evaluating all559 original raw knowledge questions',device=device)
        knowledge.evaluate(checkpoint,ROOT/'data/knowledge-eval-v1/questions.jsonl',sub/'knowledge.json',
            device=device,max_examples=559,max_new_tokens=32)
        guard.check(full=True);validate_active_lease()
        temporary=Path('/tmp')/f'slm-exclusive-comparison-{os.getpid()}.json'
        comparison=compare_fixed_results(sub/'code',sub/'knowledge.json',temporary);temporary.unlink()
        comparison.update(subject=name,training_checkpoint_sha256=subject['checkpoint_sha256'],
            objective_profile=subject['profile'],resource_policy='Exclusive report-only inference; actual GPU leases block idle-GPU curriculum restart, no active checkpoint writer,16GiB remaining reserve and400MiB aggregate report budget.',
            scope='Frozen own857M '+name+' candidate versus exact own706M completed-language base. Complete unchanged28 raw code questions/2048 tokens and559 raw QA/32 tokens; generation sealed before case execution. Original development/earlier-corpus exposure qualifications remain, not final independent or frontier benchmarks. No model update, decoder repair, feedback selection or automatic promotion.')
        atomic_json(sub/'matched-comparison.json',comparison)
        state('complete unchanged code and raw knowledge assessment',code=comparison['candidate_code'],
            knowledge=comparison['candidate_code_route_raw_knowledge'])
    except Exception as error:
        state('assessment stopped; preserve partial evidence and inspect',error=str(error));raise


def parent():
    if REPORTS.exists():raise ValueError('Preserve earlier exclusive assessment; no implicit restart')
    if shutil.disk_usage(ROOT).free<MINIMUM_FREE+REPORT_BUDGET:raise ValueError('Fund400MiB all reports above16GiB remaining space')
    if checkpoint_writers():raise ValueError('Use existing shared assessments while an original checkpoint writer is live')
    reports_pair=ROOT/'reports/code-repetition-objective-pilot-v1'
    receipt=read(ROOT/'reports/code-repetition-objective-pilot-v1-launch.json')
    actor=identity(receipt['pid'])
    if actor and actor['ticks']==str(receipt['start_ticks']) and actor['state'] not in {'Z','X'}:
        raise ValueError('Wait for exact private checkpoint writer to terminate')
    paired=read(reports_pair/'results.json');assert read(reports_pair/'independent-complete-readback.json')['actual_Adam_updates_rederived']==256
    old=ROOT/'reports/code-residual-repeated-source-pilot-v1';repeated=read(old/'results.json')
    assert read(old/'independent-complete-readback.json')['actual_Adam_updates']==512
    subjects=[subject_spec('repeated_source',repeated,read(old/'plan.json'),repeated['plan_sha256'])]
    for result in paired['profiles']:
        subjects.append(subject_spec(result['profile']['name'],result,read(reports_pair/'plan.json'),paired['plan_sha256']))
    old_generation_paths=[ROOT/'reports/code-residual-repeated-source-held-code-v1',
        ROOT/'reports/code-repetition-objective-assessment-v1/CE_control',
        ROOT/'reports/code-repetition-objective-assessment-v1/CE_plus_repetition_UL']
    if any(p.exists() for p in old_generation_paths):raise ValueError('Earlier full assessment already started; let it finish')
    names=['scripts/assess_codenet_model_exclusive_v1.py','scripts/assess_knowledge_exclusive_v1.py',
        'scripts/assess_codenet_model_v2.py','scripts/assess_code_residual_lr_pilot_v1.py',
        'src/slm_lab/knowledge_evaluation.py','src/slm_lab/exclusive_inference_resources_v1.py',
        'src/slm_lab/code_repetition_inference_v1.py','src/slm_lab/code_residual_repeated_inference_v1.py',
        'src/slm_lab/code_residual_branch_v1.py','src/slm_lab/code_residual_checkpoint_v1.py',
        'src/slm_lab/model.py','src/slm_lab/training.py','reports/exclusive-inference-driver-derivation-v1.json',
        'reports/exclusive-inference-resources-v1-focused-tests.log',
        'reports/exclusive-inference-driver-and-comparison-v1-tests.log',
        'scripts/resume_curriculum_after_disk_cleanup_v1.py',
        'reports/curriculum-disk-cleanup-resumption-v1/plan.json',
        'data/codenet-algorithm-assessment-v1/manifest.json','data/knowledge-eval-v1/questions.jsonl',
        'data/knowledge-eval-v1/questions.audit.json']
    paths=[Path(__file__),reports_pair/'results.json',reports_pair/'plan.json',reports_pair/'independent-complete-readback.json',
        old/'results.json',old/'plan.json',old/'independent-complete-readback.json']+[ROOT/n for n in names]
    bound={str(p):file_hash(p) for p in paths}
    for subject in subjects:
        checkpoint=Path(subject['checkpoint']);bound[str(checkpoint)]=subject['checkpoint_sha256']
        bound[str(Path(subject['base_checkpoint']))]=subject['base_sha256']
        bound[str(checkpoint.parent/'tokenizer.json')]=file_hash(checkpoint.parent/'tokenizer.json')
    guard=FileBindings(bound);REPORTS.mkdir()
    atomic_json(REPORTS/'plan.json',dict(created_at=datetime.now(timezone.utc).isoformat(),source_files_sha256=bound,
        subjects=subjects,devices=['cuda:0','cuda:2','cuda:3'],raw_code_questions_per_subject=28,
        code_generation_budget=2048,raw_knowledge_questions_per_subject=559,raw_knowledge_generation_budget=32,
        maximum_report_bytes=REPORT_BUDGET,minimum_remaining_home_bytes=MINIMUM_FREE,
        original_checkpoint25GiB_and_shared_inference26GiB_guards_unchanged=True,
        no_checkpoint_writers_may_overlap=True,actual_GPU_allocations_block_checked_idle_GPU_resumer=True,
        zero_occupied_file_credit=True,no_model_or_generation_or_scoring_changes=True,
        subject_model_selection_or_promotion=False,goal_achieved=False))
    lock=(REPORTS/'owner.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    buffers=[];children=[]
    try:
        torch.set_num_threads(1)
        for index in [0,2,3]:
            torch.cuda.set_device(index)
            if torch.cuda.mem_get_info(index)[0]<12*1024**3:raise ValueError('Initially12GiB free on every evaluation GPU required')
            torch.cuda.set_per_process_memory_fraction(.10,index)
            buffers.append(torch.empty(256*1024**2,dtype=torch.uint8,device=f'cuda:{index}'))
        if checkpoint_writers():raise ValueError('Original writer started during claim; leave existing queue intact')
        uuids={int(row[0]):row[1].strip() for row in csv.reader(subprocess.check_output(
            ['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True).splitlines())}
        stopped={str(ROOT/n):file_hash(ROOT/n) for n in ['reports/extended-curriculum-v1/status.json','reports/dialogue-retention-v1/status.json']}
        owner=identity(os.getpid())
        atomic_json(REPORTS/'lease.json',dict(format='slm-exclusive-inference-lease-v1',parent_pid=os.getpid(),
            parent_start_ticks=owner['ticks'],parent_source_sha256=file_hash(Path(__file__)),
            reserved_GPU_UUIDs=[uuids[i] for i in [0,2,3]],stopped_predecessor_status_sha256=stopped))
        os.environ[ENV]=str(REPORTS/'lease.json');inference_reservation()
        retired=[]
        for launch_name,status_name in [('code-residual-repeated-source-assessment-v1','code-residual-repeated-source-assessment-v1'),
                ('code-repetition-objective-assessment-v1','code-repetition-objective-assessment-v1')]:
            prior=read(ROOT/'reports'/f'{launch_name}-launch.json')
            retired.append(dict(launch=prior,outcome=retire_idle_waiter(prior,
                ROOT/'reports'/status_name/'status.json',old_generation_paths)))
        atomic_json(REPORTS/'superseded-idle-waiters.json',dict(retired=retired,previous_sources_logs_and_outputs_preserved=True))
        assert not any(p.exists() for p in old_generation_paths)
        guard.check(full=True);inference_reservation()
        launches=[]
        for subject,device in zip(subjects,['cuda:0','cuda:2','cuda:3']):
            log=(REPORTS/f'{subject["name"]}.log').open('xb')
            args=[str(ROOT/'.venv/bin/python'),'-u',str(Path(__file__)), '--subject',subject['name'],'--device',device]
            process=subprocess.Popen(args,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            child=identity(process.pid);children.append(process)
            launches.append(dict(subject=subject['name'],device=device,pid=process.pid,start_ticks=child['ticks'],argv=args))
        atomic_json(REPORTS/'subject-launches.json',dict(children=launches))
        while any(p.poll() is None for p in children):
            status('full fixed code/QA inference is running; original training waits for storage',
                children=[dict(**launch,exit_code=p.poll()) for launch,p in zip(launches,children)],
                free_home_bytes=shutil.disk_usage(ROOT).free)
            time.sleep(10)
        if any(p.returncode!=0 for p in children):raise ValueError('One or more full assessments stopped; inspect preserved child logs/results')
        comparisons=[read(REPORTS/s['name']/'matched-comparison.json') for s in subjects]
        if sum(p.stat().st_size for p in REPORTS.rglob('*') if p.is_file())>REPORT_BUDGET:raise ValueError('Total400MiB report budget exceeded')
        guard.check(full=True)
        atomic_json(REPORTS/'results.json',dict(completed_at=datetime.now(timezone.utc).isoformat(),subjects=comparisons,
            all_original_questions_and_primary_scores_recomputed=True,candidate_promoted=False,goal_achieved=False))
        status('all three complete fixed code/QA assessments finished; broader capability unverified')
    except Exception as error:
        status('exclusive assessment stopped; preserve evidence and inspect',error=str(error))
        # Keep the owner and allocations alive until already-started readers
        # observe their next guard or finish. Never silently orphan a live reader.
        for process in children:
            if process.poll() is None:process.wait()
        raise
    finally:
        buffers.clear();gc.collect();torch.cuda.empty_cache();lock.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--subject');parser.add_argument('--device')
    args=parser.parse_args()
    if args.subject:
        if not args.device:raise ValueError('Explicit child device required')
        run_subject(args.subject,args.device)
    else:parent()
