"""Check one complete original Python3 reference per screened training problem.

Uses every original public/private/generated test when the whole population
fits the declared bounded runtime. Other cases receive explicit qualifications,
remain retained in the original source, and are not relabelled wrong code.
This builds execution evidence, not an execution reward admission or RL run.
"""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime,timezone
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

from slm_lab.code_contests_CE_view_v1 import CodeContestsCEView
from slm_lab.code_contests_Python_cases_v1 import MAX_CASES,case_contract,execute_reference
from slm_lab.codenet_checkpoint_v1 import FileBindings
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.local_source_closure_v2 import python_sources
from slm_lab.source_jsonl_records_v1 import records

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/CodeContests-full-Python-references-v1'
SCHEMA=ROOT/'data/code-contests-protobuf-schema-v1'


def initialize_worker():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('Reference workers must remain CPU hidden')
    schema=json.loads((SCHEMA/'manifest.json').read_text())
    sys.path.insert(0,schema['isolated_reader_site_packages']);sys.path.insert(0,str(SCHEMA))


def check_one(task):
    import contest_problem_pb2 as proto
    row=task['native_row'];source=task['original_record']
    with open(source['extracted_record_file'],'rb') as stream:
        stream.seek(source['extracted_record_offset']);raw=stream.read(source['extracted_record_bytes'])
    if hashlib.sha256(raw).hexdigest()!=source['actual_original_record_sha256']:raise ValueError('Original complete problem record changed')
    problem=proto.ContestProblem()
    if problem.ParseFromString(raw)!=len(raw):raise ValueError('Complete original official-schema record required')
    solution=problem.solutions[row['original_solution_position']]
    if (int(solution.language)!=3 or solution.solution!=row['response']
            or hashlib.sha256(solution.solution.encode()).hexdigest()!=row['original_program_sha256']):
        raise ValueError('Exact original selected Python3 code required')
    counts={kind:len(getattr(problem,kind)) for kind in ['public_tests','private_tests','generated_tests']}
    result=dict(original_problem_id=source['id'],original_problem_name=source['original_name'],
        original_platform=source['original_source'],original_record_sha256=source['actual_original_record_sha256'],
        original_program_sha256=row['original_program_sha256'],original_solution_position=row['original_solution_position'],
        original_view_position=task['view_position'],original_pack_position=row['token_pack_index'],
        original_publisher_split='train',admitted_source_role='train',original_solution_language='Python3',
        complete_original_test_counts=counts,complete_original_test_population=sum(counts.values()),
        no_original_program_or_case_clipped=True,publisher_correctness_label=True,
        execution_reward_training_admitted=False,model_updates=0)
    if any(problem.HasField(field) and getattr(problem,field) for field in ['input_file','output_file']):
        return dict(result,status='original_file_interface_not_supported_by_this_stdin_protocol',all_original_finite_cases_pass=False)
    if not 1<=sum(counts.values())<=MAX_CASES:
        return dict(result,status='whole_original_case_count_requires_review',all_original_finite_cases_pass=False)
    cases=[dict(kind=kind,position=i,input=case.input,expected=case.output)
           for kind in ['public_tests','private_tests','generated_tests'] for i,case in enumerate(getattr(problem,kind))]
    try:contract=case_contract(cases)
    except ValueError as error:
        return dict(result,status='whole_original_case_payload_or_encoding_requires_review',diagnostic=str(error),all_original_finite_cases_pass=False)
    try:execution=execute_reference(solution.solution,cases,seconds_per_case=3.,seconds_total=120.)
    except ValueError as error:
        return dict(result,status='original_program_local_runtime_qualification',diagnostic=str(error),all_original_finite_cases_pass=False)
    if execution['complete_original_tests_sha256']!=contract['complete_original_tests_sha256']:
        raise ValueError('Every original complete test input/output must retain the same execution identity')
    return dict(result,status=execution['status'],actual_execution=execution,
        all_original_finite_cases_pass=execution['all_original_finite_cases_pass'])


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or OUT.exists():raise ValueError('Fresh CPU-only original-reference assessment required')
    if shutil.disk_usage(ROOT).free<32*1024**3:raise ValueError('Preserve original persistent writer reserve')
    test_log=ROOT/'reports/code-contests-Python-cases-v1-final-tests.log'
    if 'Ran 3 tests' not in test_log.read_text() or not test_log.read_text().rstrip().endswith('OK'):
        raise ValueError('Actual full-program, expected-value separation, coverage and resource tests required')
    view=CodeContestsCEView(ROOT)
    original_path=ROOT/'data/code-contests-native-train-records-v1/manifest.json'
    original=json.loads(original_path.read_text())
    sources={r['id']:r for r in records(ROOT/'data/code-contests-native-train-records-v1/all-original-problem-records.jsonl')}
    choices={}
    for position,row in enumerate(view.rows):
        if row['language']=='Python3':choices.setdefault(row['source_problem_id'],position)
    tasks=[dict(view_position=position,native_row=view.row(position),original_record=sources[source]) for source,position in choices.items()]
    if not tasks:raise ValueError('Every available screened training problem with selected Python3 code must be considered')
    files={str(p):file_hash(p) for p in [original_path,SCHEMA/'manifest.json',view.folder/'manifest.json',view.folder/'admission.json',
        test_log,ROOT/'tests/test_code_contests_Python_cases_v1.py',
        ROOT/'reports/CodeContests-held-text-view-v1-readback/complete.json']}
    files.update(original['files_sha256'])
    files.update({str(p):file_hash(p) for p in python_sources(ROOT,[Path(__file__),ROOT/'tests/test_code_contests_Python_cases_v1.py'])})
    guard=FileBindings(files);OUT.mkdir();(OUT/'original-problem-executions').mkdir();started=time.monotonic()
    owner=dict(pid=os.getpid(),ticks=Path('/proc/self/stat').read_text().split(') ',1)[1].split()[19])
    atomic_json(OUT/'plan.json',dict(at=datetime.now(timezone.utc).isoformat(),owner=owner,source_files_sha256=files,
        actual_screened_train_problems_with_selected_Python3_code=len(tasks),
        source_selection='First selected original publisher-order Python3 reference per entire screened training problem',
        no_problem_selected_by_model_score_or_reference_success=True,CPU_worker_limit=4,CUDA_VISIBLE_DEVICES='',
        seconds_per_original_case=3.,seconds_per_complete_original_program=120.,maximum_original_case_count=1000,
        maximum_complete_original_input_and_expected_bytes=8*1024**2,
        every_original_public_private_generated_case_required_for_full_case_success=True,
        publisher_source_labels_not_automatically_local_execution_passes=True,
        incompatible_file_interfaces_versions_checkers_payloads_and_resources_are_qualifications_not_wrong_code_labels=True,
        no_original_held_question_or_source_role_changed=True,models_constructed=0,model_updates=0))
    atomic_json(OUT/'all-original-selected-reference-identities.json',[dict(view_position=t['view_position'],
        original_problem_id=t['original_record']['id'],original_record_sha256=t['original_record']['actual_original_record_sha256'],
        original_program_sha256=t['native_row']['original_program_sha256'],original_solution_position=t['native_row']['original_solution_position']) for t in tasks])
    counts=Counter();files_out={}; completed=0;submitted=0
    try:
        context=multiprocessing.get_context('spawn')
        with ProcessPoolExecutor(max_workers=4,mp_context=context,initializer=initialize_worker) as pool:
            active={}
            while submitted<len(tasks) and len(active)<4:
                task=tasks[submitted];active[pool.submit(check_one,task)]=submitted;submitted+=1
            while active:
                done,_=wait(active,return_when=FIRST_COMPLETED)
                for future in done:
                    slot=active.pop(future);result=future.result()
                    destination=OUT/'original-problem-executions'/f'{slot:05d}.json'
                    atomic_json(destination,result);files_out[str(destination)]=file_hash(destination)
                    counts[result['status']]+=1;counts['all_original_finite_cases_pass']+=bool(result['all_original_finite_cases_pass'])
                    counts['actual_original_case_runs']+=result.get('actual_execution',{}).get('actual_observed_case_count',0)
                    completed+=1
                    if submitted<len(tasks):
                        task=tasks[submitted];active[pool.submit(check_one,task)]=submitted;submitted+=1
                if shutil.disk_usage(ROOT).free<28*1024**3:raise ValueError('Preserve original persistent writer reserve during reference checks')
                atomic_json(OUT/'progress.json',dict(owner=owner,original_training_problems_completed=completed,
                    original_training_problems_total=len(tasks),actual_counts=dict(counts),
                    elapsed_seconds=time.monotonic()-started,models_constructed=0,model_updates=0,goal_achieved=False))
                if completed%32<len(done):print(json.dumps(dict(completed=completed,total=len(tasks),counts=dict(counts))),flush=True)
        if completed!=len(tasks):raise ValueError('Every original declared training problem must have an explicit execution disposition')
        guard.check(full=True)
        atomic_json(OUT/'all-original-problem-execution-files.json',files_out)
        atomic_json(OUT/'complete.json',dict(at=datetime.now(timezone.utc).isoformat(),complete=True,
            all_declared_original_training_problems_have_execution_dispositions=True,
            actual_original_training_problems=len(tasks),actual_counts=dict(counts),
            original_source_labels_and_local_runtime_qualifications_kept_distinct=True,
            every_complete_case_success_includes_all_original_public_private_generated_cases=True,
            no_general_correctness_efficiency_or_original_judge_claim=True,
            independent_actual_stdout_case_verdict_readback_pending=True,execution_reward_admission_pending=True,
            source_files_sha256={str(p):file_hash(p) for p in [OUT/'plan.json',OUT/'all-original-selected-reference-identities.json',OUT/'all-original-problem-execution-files.json',
                view.folder/'admission.json']},models_constructed=0,model_updates=0,candidate_promoted=False,goal_achieved=False))
        print(json.dumps(dict(complete=str(OUT/'complete.json'),original_problems=len(tasks),counts=dict(counts))),flush=True)
    except BaseException as error:
        atomic_json(OUT/'failure.json',dict(error=repr(error),traceback=traceback.format_exc(),
            actual_completed_original_problems=completed,model_updates=0,goal_achieved=False));raise


if __name__=='__main__':main()
