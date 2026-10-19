"""Stream public execution-prediction candidates from complete reference checks.

Each reference result is immutable before use. Finalization waits for every
original declared problem and the exact source owner to terminate. This is
CPU data preparation only; no partial candidate pack is admitted to training.
"""
from collections import Counter
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

from tokenizers import Tokenizer

from slm_lab.code_contests_CE_view_v1 import CodeContestsCEView
from slm_lab.code_contests_public_execution_trace_v1 import public_trace
from slm_lab.codenet_checkpoint_v1 import FileBindings
from slm_lab.codenet_token_pack_v1 import CodePackWriter
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.local_source_closure_v2 import python_sources
from slm_lab.source_jsonl_records_v1 import records

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'reports/CodeContests-full-Python-references-v1'
OUT=ROOT/'data/code-contests-public-execution-traces-candidate-v1'
REPORTS=ROOT/'reports/CodeContests-public-execution-traces-candidate-v1'
SCHEMA=ROOT/'data/code-contests-protobuf-schema-v1'


def alive(owner):
    path=Path('/proc')/str(owner['pid'])/'stat'
    if not path.exists():return False
    values=path.read_text().split(') ',1)[1].split()
    return values[19]==owner['ticks'] and values[0] not in {'Z','X'}


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or OUT.exists() or REPORTS.exists():raise ValueError('Fresh CPU-only public execution-trace preparation required')
    tests=ROOT/'reports/code-contests-public-execution-trace-v1-final-tests.log'
    if 'Ran 3 tests' not in tests.read_text() or not tests.read_text().rstrip().endswith('OK'):
        raise ValueError('Actual source-role/stdout/private-value/context formatting tests required')
    source_plan_path=SOURCE/'plan.json';source_plan=json.loads(source_plan_path.read_text())
    identities_path=SOURCE/'all-original-selected-reference-identities.json';identities=json.loads(identities_path.read_text())
    if len(identities)!=source_plan['actual_screened_train_problems_with_selected_Python3_code']:
        raise ValueError('Every original declared training reference must be dispositioned')
    view=CodeContestsCEView(ROOT);tokenizer=Tokenizer.from_file(str(view.tokenizer_path))
    original_index=ROOT/'data/code-contests-native-train-records-v1/all-original-problem-records.jsonl'
    sources={r['id']:r for r in records(original_index)}
    schema_path=SCHEMA/'manifest.json';schema=json.loads(schema_path.read_text())
    sys.path.insert(0,schema['isolated_reader_site_packages']);sys.path.insert(0,str(SCHEMA))
    import contest_problem_pb2 as proto
    files={str(p):file_hash(p) for p in [source_plan_path,identities_path,original_index,schema_path,
        view.folder/'manifest.json',view.folder/'admission.json',view.tokenizer_path,tests,
        ROOT/'tests/test_code_contests_public_execution_trace_v1.py']}
    files.update({str(SCHEMA/n):v for n,v in schema['files_sha256'].items()})
    files.update({str(p):file_hash(p) for p in python_sources(ROOT,[Path(__file__),ROOT/'tests/test_code_contests_public_execution_trace_v1.py'])})
    guard=FileBindings(files)
    OUT.mkdir();REPORTS.mkdir();started=time.monotonic()
    owner=dict(pid=os.getpid(),ticks=Path('/proc/self/stat').read_text().split(') ',1)[1].split()[19])
    atomic_json(REPORTS/'plan.json',dict(at=datetime.now(timezone.utc).isoformat(),owner=owner,
        exact_original_reference_owner=source_plan['owner'],source_files_sha256=files,
        original_declared_reference_problems=len(identities),
        selection='One first complete fitting nonempty original public-input execution prediction per fully qualified reference; stable original problem order',
        every_original_reference_problem_gets_a_disposition=True,
        all_actual_public_private_generated_reference_cases_required_for_eligibility=True,
        original_private_and_generated_input_output_values_never_used_in_model_prompts_or_targets=True,
        targets_are_actual_retained_reference_stdout_not_publisher_gold_or_teacher_completions=True,
        all_original_referenced_record_payload_SHA256_checked_before_trace=True,
        no_additional_neural_parameters_or_teacher=True,maximum_candidate_bytes=512*1024**2,
        partial_pack_not_usable_for_optimization=True,automatic_retry=False,
        models_constructed=0,model_updates=0,dataset_admitted_to_training=False,goal_achieved=False))
    shutil.copy2(view.tokenizer_path,OUT/'tokenizer.json')
    shutil.copy2(ROOT/'configs/model.json',OUT/'model-config.json')
    cache=ROOT/'data/code-contests-original-train-cache-v2'
    shutil.copy2(cache/'publisher-LICENSE',OUT/'DATASET-LICENSE.txt');shutil.copy2(cache/'publisher-README.md',OUT/'publisher-README.md')
    writer=CodePackWriter(OUT,maximum_bytes=512*1024**2,minimum_free_gib=26)
    dispositions=REPORTS/'all-original-reference-trace-dispositions.jsonl';input_bindings={};counts=Counter();position=0
    deadline=time.monotonic()+7*86400
    try:
        with dispositions.open('x') as journal:
            while position<len(identities):
                guard.check(full=False)
                if (SOURCE/'failure.json').exists():raise ValueError('Original reference assessment failed; preserve partial candidate without admission')
                path=SOURCE/'original-problem-executions'/f'{position:05d}.json'
                if not path.exists():
                    if not alive(source_plan['owner']):raise ValueError('Exact original reference owner terminated before every declared disposition')
                    if time.monotonic()>deadline:raise ValueError('Candidate observation deadline; original reference process not restarted')
                    time.sleep(15);continue
                expected=identities[position];reference=json.loads(path.read_text());row=view.row(expected['view_position'])
                input_bindings[str(path)]=file_hash(path)
                if (row['original_program_sha256']!=expected['original_program_sha256']
                        or reference['original_problem_id']!=expected['original_problem_id']
                        or reference['original_program_sha256']!=expected['original_program_sha256']
                        or reference['original_record_sha256']!=expected['original_record_sha256']):
                    raise ValueError('Every original reference result, source problem and unchanged program must match')
                trace=None
                if reference['all_original_finite_cases_pass']:
                    source=sources[expected['original_problem_id']]
                    with open(source['extracted_record_file'],'rb') as stream:
                        stream.seek(source['extracted_record_offset']);raw=stream.read(source['extracted_record_bytes'])
                    if hashlib.sha256(raw).hexdigest()!=expected['original_record_sha256']:raise ValueError('Complete original publisher record changed')
                    problem=proto.ContestProblem()
                    if problem.ParseFromString(raw)!=len(raw):raise ValueError('Complete original official-schema record required')
                    original=problem.solutions[row['original_solution_position']]
                    if int(original.language)!=3 or original.solution!=row['response']:
                        raise ValueError('Original human program must match the qualified reference and public trace prompt')
                    trace,disposition=public_trace(row,problem,reference,tokenizer)
                    counts['fully_qualified_original_references']+=1
                else:disposition=dict(status='reference_not_qualified_for_all_original_finite_cases',original_reference_status=reference['status'])
                if trace is not None:
                    ids,labels=trace.pop('ids'),trace.pop('labels');writer.add(trace,ids,labels)
                    counts['complete_public_execution_traces']+=1
                    counts['complete_native_context_positions']+=len(ids);counts['actual_stdout_EOS_targets']+=sum(v!=-100 for v in labels)
                counts['original_reference_problems_dispositioned']+=1;counts[disposition['status']]+=1
                journal.write(json.dumps(dict(original_reference_position=position,original_problem_id=expected['original_problem_id'],
                    reference_result_sha256=input_bindings[str(path)],**disposition),ensure_ascii=False)+'\n');journal.flush()
                position+=1
                atomic_json(REPORTS/'progress.json',dict(owner=owner,actual_counts=dict(counts),
                    original_declared_reference_problems=len(identities),elapsed_seconds=time.monotonic()-started,
                    partial_pack_not_admitted_to_training=True,models_constructed=0,model_updates=0,goal_achieved=False))
                if position%32==0:print(json.dumps(dict(original_problems=position,public_traces=counts['complete_public_execution_traces'])),flush=True)
        while alive(source_plan['owner']) or not (SOURCE/'complete.json').exists():
            if (SOURCE/'failure.json').exists():raise ValueError('Original reference owner failed before finalization')
            if not alive(source_plan['owner']) and not (SOURCE/'complete.json').exists():raise ValueError('Missing complete original reference assessment')
            if time.monotonic()>deadline:raise ValueError('Candidate observation deadline; no reference restart attempted')
            time.sleep(15)
        final=json.loads((SOURCE/'complete.json').read_text())
        if final['actual_original_training_problems']!=len(identities):raise ValueError('Every declared original reference problem required')
        guard.check(full=True);FileBindings(input_bindings).check(full=True)
        manifest=writer.finish(dict(original_reference_plan_sha256=file_hash(source_plan_path),original_reference_completion_sha256=file_hash(SOURCE/'complete.json'),
            original_screened_CE_view_manifest_sha256=file_hash(view.folder/'manifest.json'),actual_counts=dict(counts),
            original_source_terms=str(cache/'publisher-README.md'),
            target_origin='Actual full-qualified reference stdout for original public input only',
            auxiliary_shared_language_head_task_no_parameter_or_teacher_addition=True,
            no_private_generated_or_held_inputs_or_outputs_used_as_model_text=True,
            independent_source_stdout_target_and_context_readback_pending=True))
        atomic_json(REPORTS/'all-original-reference-result-bindings.json',input_bindings)
        atomic_json(REPORTS/'complete.json',dict(at=datetime.now(timezone.utc).isoformat(),complete=True,
            every_original_reference_problem_dispositioned=True,actual_counts=dict(counts),actual_native_counts=manifest['counts'],
            actual_reference_stdout_source_and_target_readback_pending=True,
            source_files_sha256={str(p):file_hash(p) for p in [REPORTS/'plan.json',dispositions,REPORTS/'all-original-reference-result-bindings.json',OUT/'manifest.json']},
            dataset_admitted_to_training=False,models_constructed=0,model_updates=0,candidate_promoted=False,goal_achieved=False))
        print(json.dumps(dict(complete=str(REPORTS/'complete.json'),public_traces=counts['complete_public_execution_traces'],training_admitted=False)),flush=True)
    except BaseException as error:
        writer.close()
        atomic_json(REPORTS/'failure.json',dict(error=repr(error),traceback=traceback.format_exc(),
            original_reference_problems_dispositioned=position,dataset_admitted_to_training=False,model_updates=0,goal_achieved=False));raise


if __name__=='__main__':main()
