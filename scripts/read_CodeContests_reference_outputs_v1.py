"""Stream an independent full-source readback of retained reference outputs.

Waits for immutable per-problem files while the original worker continues.
Every original declared problem gets a readback. No program is rerun here and
the original reference job is never restarted or altered on a reader deadline.
"""
from collections import Counter
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

from slm_lab.code_contests_CE_view_v1 import CodeContestsCEView
from slm_lab.code_contests_execution_readback_v1 import verify
from slm_lab.codenet_checkpoint_v1 import FileBindings
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.local_source_closure_v2 import python_sources
from slm_lab.source_jsonl_records_v1 import records

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'reports/CodeContests-full-Python-references-v1'
SCHEMA=ROOT/'data/code-contests-protobuf-schema-v1'
OUT=ROOT/'reports/CodeContests-reference-outputs-v1-readback'


def alive(owner):
    path=Path('/proc')/str(owner['pid'])/'stat'
    if not path.exists():return False
    values=path.read_text().split(') ',1)[1].split()
    return values[19]==owner['ticks'] and values[0] not in {'Z','X'}


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or OUT.exists():raise ValueError('Fresh CPU-only actual output readback required')
    tests=ROOT/'reports/code-contests-execution-readback-v1-tests.log'
    if '3 passed' not in tests.read_text():raise ValueError('Actual case-output corruption, partial coverage and syntax controls required')
    source_plan_path=SOURCE/'plan.json';source_plan=json.loads(source_plan_path.read_text())
    identities_path=SOURCE/'all-original-selected-reference-identities.json';identities=json.loads(identities_path.read_text())
    view=CodeContestsCEView(ROOT)
    choices={}
    for position,row in enumerate(view.rows):
        if row['language']=='Python3':choices.setdefault(row['source_problem_id'],position)
    if (len(identities)!=source_plan['actual_screened_train_problems_with_selected_Python3_code']
            or [(r['original_problem_id'],r['view_position']) for r in identities]!=list(choices.items())):
        raise ValueError('All first original selected Python3 references in the complete screened training view required')
    index_path=ROOT/'data/code-contests-native-train-records-v1/all-original-problem-records.jsonl'
    sources={r['id']:r for r in records(index_path)}
    schema_path=SCHEMA/'manifest.json';schema=json.loads(schema_path.read_text())
    sys.path.insert(0,schema['isolated_reader_site_packages']);sys.path.insert(0,str(SCHEMA))
    import contest_problem_pb2 as proto
    files={str(p):file_hash(p) for p in [source_plan_path,identities_path,index_path,schema_path,tests,
        view.folder/'manifest.json',view.folder/'admission.json',ROOT/'tests/test_code_contests_execution_readback_v1.py',
        *python_sources(ROOT,[Path(__file__),ROOT/'tests/test_code_contests_execution_readback_v1.py'])]}
    files.update({str(SCHEMA/n):v for n,v in schema['files_sha256'].items()})
    guard=FileBindings(files);OUT.mkdir();started=time.monotonic();deadline=started+7*86400
    owner=dict(pid=os.getpid(),ticks=Path('/proc/self/stat').read_text().split(') ',1)[1].split()[19])
    atomic_json(OUT/'plan.json',dict(at=datetime.now(timezone.utc).isoformat(),owner=owner,
        exact_original_reference_owner=source_plan['owner'],source_files_sha256=files,
        original_reference_problems=len(identities),all_original_source_code_and_cases_reconstructed=True,
        actual_stdout_tokens_compared_directly_without_producer_comparator=True,
        no_neural_models_or_new_program_executions=True,
        original_sandbox_execution_not_independently_repeated=True,
        observation_deadline_seconds=7*86400,automatic_retry=False,model_updates=0,goal_achieved=False))
    counts=Counter();bindings={};position=0;qualified=[]
    ledger=OUT/'all-original-reference-output-readbacks.jsonl'
    try:
        with ledger.open('x') as stream:
            while position<len(identities):
                guard.check(full=False)
                if (SOURCE/'failure.json').exists():raise ValueError('Original reference producer failed; preserve partial output readback')
                path=SOURCE/'original-problem-executions'/f'{position:05d}.json'
                if not path.exists():
                    if not alive(source_plan['owner']):raise ValueError('Exact original owner ended before every declared result')
                    if time.monotonic()>deadline:raise ValueError('Reader observation deadline; original job left intact')
                    time.sleep(15);continue
                declared=identities[position];row=view.row(declared['view_position']);source=sources[declared['original_problem_id']]
                with open(source['extracted_record_file'],'rb') as original:
                    original.seek(source['extracted_record_offset']);raw=original.read(source['extracted_record_bytes'])
                if hashlib.sha256(raw).hexdigest()!=declared['original_record_sha256']:
                    raise ValueError('Actual complete original protobuf record differs')
                problem=proto.ContestProblem()
                if problem.ParseFromString(raw)!=len(raw):raise ValueError('Every original problem field must parse completely')
                program=problem.solutions[declared['original_solution_position']]
                if int(program.language)!=3 or program.solution!=row['response']:
                    raise ValueError('Actual original selected Python3 source must equal the entire reference program')
                result=json.loads(path.read_text());binding=file_hash(path)
                if (result['original_record_sha256']!=declared['original_record_sha256']
                        or result['original_view_position']!=declared['view_position']
                        or result['original_problem_name']!=problem.name
                        or result['original_platform']!=source['original_source']):
                    raise ValueError('Original result source fields differ')
                checked=verify(row,problem,result)
                bindings[str(path)]=binding
                if checked['fully_qualified']:qualified.append(dict(original_reference_position=position,**declared))
                counts['original_problem_outputs_read_back']+=1
                counts['fully_qualified_original_references']+=checked['fully_qualified']
                counts['actual_retained_case_outputs_checked']+=checked['actual_retained_case_outputs_checked']
                counts[checked['status']]+=1
                stream.write(json.dumps(dict(original_reference_position=position,original_problem_id=declared['original_problem_id'],
                    source_result_sha256=binding,**checked),ensure_ascii=False)+'\n');stream.flush()
                position+=1
                if position%32==0 or position==len(identities):
                    atomic_json(OUT/'progress.json',dict(owner=owner,actual_counts=dict(counts),
                        original_problems=len(identities),elapsed_seconds=time.monotonic()-started,
                        reference_producer_not_modified=True,model_updates=0,goal_achieved=False))
        while alive(source_plan['owner']) or not (SOURCE/'complete.json').exists():
            if (SOURCE/'failure.json').exists():raise ValueError('Original reference producer failed before finalization')
            if not alive(source_plan['owner']):raise ValueError('Missing original reference completion')
            if time.monotonic()>deadline:raise ValueError('Reader observation deadline; original job left intact')
            time.sleep(15)
        final=json.loads((SOURCE/'complete.json').read_text())
        if (final['actual_original_training_problems']!=position
                or counts['fully_qualified_original_references']!=final['actual_counts']['all_original_finite_cases_pass']
                or counts['actual_retained_case_outputs_checked']!=final['actual_counts']['actual_original_case_runs']):
            raise ValueError('Every full result and actual retained-case denominator must reconcile')
        guard.check(full=True);FileBindings(bindings).check(full=True)
        atomic_json(OUT/'all-qualified-original-reference-identities.json',qualified)
        atomic_json(OUT/'all-source-result-bindings.json',bindings)
        atomic_json(OUT/'complete.json',dict(at=datetime.now(timezone.utc).isoformat(),complete=True,
            every_declared_original_reference_output_and_complete_source_rebuilt=True,
            actual_counts=dict(counts),all_success_and_qualification_denominators_reconciled=True,
            every_retained_stdout_compared_directly_against_original_expected_tokens=True,
            no_program_rerun_or_independent_OS_execution_claim=True,
            no_publisher_judge_reproduction_or_general_correctness_claim=True,
            source_files_sha256={str(p):file_hash(p) for p in [OUT/'plan.json',ledger,OUT/'all-source-result-bindings.json',
                OUT/'all-qualified-original-reference-identities.json',SOURCE/'complete.json']},
            public_execution_supervision_and_reward_admission_not_granted=True,
            models_constructed=0,model_updates=0,goal_achieved=False))
        print(json.dumps(dict(complete=str(OUT/'complete.json'),actual_counts=dict(counts))),flush=True)
    except BaseException as error:
        atomic_json(OUT/'failure.json',dict(error=repr(error),traceback=traceback.format_exc(),
            original_results_checked=position,model_updates=0,goal_achieved=False));raise


if __name__=='__main__':main()
