"""Freshly rerun all96decoded final code-first programs without neural work."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import time
import traceback

from slm_lab.codenet_checkpoint_v1 import FileBindings
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.exact_waiting_process_retirement_v1 import terminal
from slm_lab.local_source_closure_v2 import python_sources
from slm_lab.code_tasks import TASKS
from slm_lab.decoded_native_code_reexecution_v1 import replay

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'reports/code-first-continuation-v3'
OUT=ROOT/'reports/code-first-complete-native-program-reexecution-CPU-v1'
QUEUE=ROOT/'reports/code-first-complete-native-program-reexecution-CPU-v1-queue.json'


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or OUT.exists() or QUEUE.exists():raise ValueError('Fresh GPU-hidden source-preserving reexecution queue required')
    tests=ROOT/'reports/decoded-native-code-reexecution-v1-tests.log'
    if '2 passed' not in tests.read_text():raise ValueError('Actual Python/JS/ordinary reference and wrong-algorithm controls required')
    source_plan=SOURCE/'plan.json';plan=json.loads(source_plan.read_text());identity=plan['owner']
    files={str(p):file_hash(p) for p in [source_plan,tests,*python_sources(ROOT,[Path(__file__),ROOT/'tests/test_decoded_native_code_reexecution_v1.py'])]}
    guard=FileBindings(files);me=dict(pid=os.getpid(),ticks=Path('/proc/self/stat').read_text().split(') ',1)[1].split()[19])
    atomic_json(QUEUE,dict(owner=me,exact_original_GPU_owner=identity,source_files_sha256=files,
        original_decoded_native_deliveries=96,fresh_native_program_executions=96,additional_ordinary_Python_executions=48,
        original_generated_token_actions_not_retained_by_GPU_producer=True,new_neural_forwards=0,model_updates=0,goal_achieved=False))
    while not terminal(identity):guard.check();time.sleep(5)
    if ((SOURCE/'failure.json').exists() or any(not (SOURCE/n).exists() for n in ['complete.json','training-complete.json','restored.json','native-assessment-inputs.json'])):
        raise ValueError('The exact original GPU source epoch and all native controls must finish successfully')
    if any(not terminal(r) for r in json.loads((SOURCE/'children.json').read_text())):raise ValueError('Every exact original registered GPU child must be terminal')
    trained=json.loads((SOURCE/'training-complete.json').read_text());done=json.loads((SOURCE/'complete.json').read_text())
    if (trained.get('actual_updates')!=plan['planned_steps'] or not done.get('original_local_code_and_broad_assessments_complete')):
        raise ValueError('The complete original full epoch and unchanged native assessments are required')
    inputs=json.loads((SOURCE/'native-assessment-inputs.json').read_text());jobs=[];profiles=[]
    for arm in ['initial','trained']:
        checkpoint=ROOT/'checkpoints/code-first-continuation-v3'/arm/'model.pt'
        if file_hash(checkpoint)!=inputs[arm]['checkpoint_sha256']:raise ValueError('Exact original complete native source weights required')
        files[str(checkpoint)]=inputs[arm]['checkpoint_sha256']
        for frame in ['completion','instruction']:
            p=SOURCE/arm/('code-'+frame+'.json');report=json.loads(p.read_text())
            if (report['checkpoint_sha256']!=inputs[arm]['checkpoint_sha256'] or report['generation_budget']!=512
                    or report['prompt_mode']!=frame or report['task_count']!=24
                    or [(r['language'],r['task']) for r in report['results']]!=[(n,t[0]) for n in ['Python','JavaScript'] for t in TASKS]):
                raise ValueError('Every original task/frame/language and unchanged512action budget must remain')
            profiles.append({k:report[k] for k in ['generation_budget','decoding','suite_sha256','tokenizer_sha256','inference_device_type','inference_precision']})
            if report['suite_sha256']!=file_hash(ROOT/'src/slm_lab/code_tasks.py') or report['tokenizer_sha256']!=plan['tokenizer_sha256']:
                raise ValueError('The exact original suite and own BPE must remain')
            files[str(p)]=file_hash(p);files.update(report['source_files_sha256'])
            jobs.extend((arm,frame,row) for row in report['results'])
    if len(jobs)!=96 or any(p!=profiles[0] for p in profiles):raise ValueError('All96matched original native deliveries require identical runtime and budget profiles')
    for p in [SOURCE/'complete.json',SOURCE/'restored.json',SOURCE/'training-complete.json',SOURCE/'native-assessment-inputs.json']:
        files[str(p)]=file_hash(p)
    guard=FileBindings(files);OUT.mkdir();atomic_json(OUT/'plan.json',dict(owner=me,source_files_sha256=files,
        actual_matched_native_profile=profiles[0],original_GPU_updates=plan['planned_steps'],neural_forwards=0,
        no_claim_about_original_generated_token_IDs_or_EOS_stopping=True,goal_achieved=False))
    def execute(job):
        arm,frame,case=job
        return dict(endpoint=arm,**replay(case,frame))
    rows=[]
    try:
        with (OUT/'all96-fresh-program-executions.jsonl').open('x') as journal,ThreadPoolExecutor(max_workers=4) as pool:
            for row in pool.map(execute,jobs):
                rows.append(row);journal.write(json.dumps(row,ensure_ascii=False)+'\n');journal.flush()
                atomic_json(OUT/'progress.json',dict(actual_complete_native_programs=len(rows),planned_native_programs=96,
                    actual_complete_secondary_Python_programs=sum(r['language']=='Python' for r in rows),neural_updates=0,goal_achieved=False))
        outcomes={}
        for arm in ['initial','trained']:
            group=[r for r in rows if r['endpoint']==arm]
            outcomes[arm]=dict(tasks=len(group),syntax=sum(bool(r['fresh_native_execution'].get('syntax_valid')) for r in group),
                functional=sum(bool(r['fresh_native_execution'].get('functional_passed')) for r in group),
                ordinary_Python_tasks=sum(r['language']=='Python' for r in group),
                ordinary_Python_functional=sum(bool(r['fresh_ordinary_Python_execution']['functional_passed']) for r in group if r['language']=='Python'),
                producer_empirical_efficiency_passes_not_fresh_timing_comparison=sum(r['producer_empirical_efficiency_pass'] for r in group))
        guard.check(full=True)
        atomic_json(OUT/'complete.json',dict(at=datetime.now(timezone.utc).isoformat(),complete=True,actual_matched_endpoint_totals=outcomes,
            all96_original_decoded_program_bytes_reconstructed_and_fresh_native_verdicts_match=True,
            all48_original_Python_programs_additionally_run_in_ordinary_stdlib_protocol=True,
            all_original_finite_cases_frames_languages_and_budgets_retained=True,
            original_generated_token_IDs_and_EOS_stopping_not_certified=True,no_new_neural_forward_or_generated_program_repair=True,
            source_files_sha256={**files,str(OUT/'all96-fresh-program-executions.jsonl'):file_hash(OUT/'all96-fresh-program-executions.jsonl')},
            finite_cases_not_arbitrary_code_correctness_or_asymptotic_efficiency=True,model_updates=0,candidate_promoted=False,goal_achieved=False))
        print(json.dumps(dict(complete=str(OUT/'complete.json'),actual_matched_endpoint_totals=outcomes)),flush=True)
    except BaseException as error:
        atomic_json(OUT/'failure.json',dict(error=repr(error),traceback=traceback.format_exc(),goal_achieved=False));raise


if __name__=='__main__':main()
