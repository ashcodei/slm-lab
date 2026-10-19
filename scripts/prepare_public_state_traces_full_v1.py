"""Prepare dense public-state candidates for all3959 declared source references.

CPU-only data experiment. The candidate stays unadmitted and cannot alter any
existing source-bound training queue. No neural model or teacher is loaded.
"""
from collections import Counter
from datetime import datetime,timezone
import base64
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
from slm_lab.public_python_state_trace_v1 import capture,implementation_sources
from slm_lab.public_state_trace_target_v1 import target
from slm_lab.source_jsonl_records_v1 import records

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'reports/CodeContests-full-Python-references-v1'
OUT=ROOT/'reports/public-state-traces-full-v1'
DATA=ROOT/'data/public-state-traces-full-candidate-v1'
SCHEMA=ROOT/'data/code-contests-protobuf-schema-v1'


def alive(identity):
    path=Path('/proc')/str(identity['pid'])/'stat'
    if not path.exists():return False
    fields=path.read_text().split(') ',1)[1].split()
    return fields[19]==identity['ticks'] and fields[0] not in {'Z','X'}


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or OUT.exists() or DATA.exists():raise ValueError('Fresh CPU-hidden all-reference state preparation required')
    test_names={'public-python-state-trace-v1-final-tests.log':4,'public-state-trace-target-v1-tests.log':2}
    for name,count in test_names.items():
        if f'{count} passed' not in (ROOT/'reports'/name).read_text():raise ValueError('Actual source-state/runtime/bound/mask controls required')
    source_plan=SOURCE/'plan.json';identities_path=SOURCE/'all-original-selected-reference-identities.json'
    identities=json.loads(identities_path.read_text());view=CodeContestsCEView(ROOT)
    view.guard.check(full=True);tokenizer=Tokenizer.from_file(str(view.tokenizer_path))
    source_identity=json.loads(source_plan.read_text())['owner']
    if len(identities)!=3959:raise ValueError('Every original declared screened Python3 reference required')
    selected=[dict(original_reference_position=i,**identity,reference_result=str(SOURCE/'original-problem-executions'/f'{i:05d}.json')) for i,identity in enumerate(identities)]
    pilot_path=ROOT/'reports/public-state-trace-pilot-v1-readback/complete.json'
    pilot=json.loads(pilot_path.read_text())
    if not pilot.get('all29_whole_state_traces_reproduced') or pilot['actual_counts']['all29_complete_native_targets_rebuilt']!=29:
        raise ValueError('Actual whole-source/native/state-repeat pilot required before full preparation')
    index_path=ROOT/'data/code-contests-native-train-records-v1/all-original-problem-records.jsonl'
    index={r['id']:r for r in records(index_path)}
    schema_path=SCHEMA/'manifest.json';schema=json.loads(schema_path.read_text())
    sys.path.insert(0,schema['isolated_reader_site_packages']);sys.path.insert(0,str(SCHEMA))
    import contest_problem_pb2 as proto
    sources=implementation_sources()
    extra=[source_plan,identities_path,index_path,schema_path,view.folder/'manifest.json',view.folder/'admission.json',view.tokenizer_path,
        ROOT/'tests/test_public_python_state_trace_v1.py',ROOT/'tests/test_public_state_trace_target_v1.py',
        pilot_path,*[ROOT/'reports'/n for n in test_names],*python_sources(ROOT,[Path(__file__)])]
    sources.update({str(p):file_hash(p) for p in extra})
    sources.update({str(SCHEMA/n):h for n,h in schema['files_sha256'].items()})
    OUT.mkdir();DATA.mkdir();(OUT/'actual-observations').mkdir();(OUT/'repeated-observations').mkdir();owner=dict(pid=os.getpid(),ticks=Path('/proc/self/stat').read_text().split(') ',1)[1].split()[19])
    atomic_json(OUT/'plan.json',dict(at=datetime.now(timezone.utc).isoformat(),owner=owner,source_files_sha256=sources,
        fixed_original_reference_cohort=selected,selection='All3959 declared original screened source references in original publisher order; all receive dispositions, and only all-original-case-qualified references can produce state targets',
        public_input_selection='Original public case0 only; no fallback to a later/easier input',
        maximum_trace_events=256,maximum_complete_trace_bytes=256*1024,maximum_native_context_positions=4096,
        whole_trace_or_value_overflow_qualifies_out_without_clipping=True,
        expected_or_private_or_generated_case_values_not_sent_to_instrumented_worker=True,
        actual_instrumented_stdout_must_byte_match_original_retained_reference_stdout=True,
        actual_all_public_private_generated_cases_passed_before_each_instrumented_source_execution=True,
        each_native_candidate_state_trace_must_reproduce_exactly_in_a_second_fresh_interpreter=True,
        source_frame_call_ids_are_CPython_call_segments_including_generator_resumes=True,
        traces_observe_non_dunder_locals_and_opaque_non_data_types_not_every_Python_or_external_state=True,
        candidate_not_admitted_to_any_training_and_existing_queues_unchanged=True,
        complete_original_reference_epoch_required_for_candidate_finalization=True,
        independent_state_target_and_held_overlap_admission_still_required=True,
        related_primary_sources=['https://arxiv.org/abs/2305.05383','https://arxiv.org/abs/2404.14662','https://aclanthology.org/2024.acl-srw.44/'],
        pretrained_weights_or_LLM_teacher_loaded=False,new_neural_parameters=0,models_constructed=0,model_updates=0,goal_achieved=False))
    guard=FileBindings(sources);guard.check(full=True)
    shutil.copyfile(view.tokenizer_path,DATA/'tokenizer.json');shutil.copyfile(ROOT/'configs/model.json',DATA/'model-config.json')
    cache=ROOT/'data/code-contests-original-train-cache-v2'
    shutil.copyfile(cache/'publisher-LICENSE',DATA/'DATASET-LICENSE.txt');shutil.copyfile(cache/'publisher-README.md',DATA/'publisher-README.md')
    counts=Counter();writer=None;started=time.monotonic();ledger=OUT/'all3959-original-source-dispositions.jsonl';reference_bindings={}
    try:
        with ledger.open('x') as stream:
            for ordinal,chosen in enumerate(selected):
                guard.check();reference_path=Path(chosen['reference_result'])
                while not reference_path.is_file():
                    if (SOURCE/'failure.json').exists():raise ValueError('Original source epoch failed; preserve partial state candidates')
                    if not alive(source_identity):raise ValueError('Exact original source owner ended before every declared disposition')
                    if time.monotonic()-started>7*86400:raise ValueError('Observation deadline; original owner not restarted')
                    time.sleep(10)
                chosen=dict(chosen,reference_result_sha256=file_hash(reference_path));reference_bindings[str(reference_path)]=chosen['reference_result_sha256']
                reference=json.loads(reference_path.read_text());row=view.row(chosen['view_position'])
                if reference['original_problem_id']!=chosen['original_problem_id'] or reference['original_program_sha256']!=chosen['original_program_sha256']:
                    raise ValueError('Every original immutable source result and whole program identity must match')
                if not reference['all_original_finite_cases_pass']:
                    counts['original_source_dispositions']+=1;counts['status:original_reference_not_fully_qualified']+=1
                    stream.write(json.dumps(dict(original_reference_position=chosen['original_reference_position'],original_problem_id=chosen['original_problem_id'],
                        reference_result_sha256=chosen['reference_result_sha256'],status='original_reference_not_fully_qualified',
                        original_reference_status=reference['status']))+'\n');stream.flush()
                    continue
                counts['all_original_case_qualified_sources']+=1
                if row['original_program_sha256']!=chosen['original_program_sha256'] or row['source_problem_id']!=chosen['original_problem_id']:
                    raise ValueError('Exact original admitted source/whole program required')
                source=index[chosen['original_problem_id']]
                with open(source['extracted_record_file'],'rb') as original:
                    original.seek(source['extracted_record_offset']);payload=original.read(source['extracted_record_bytes'])
                if hashlib.sha256(payload).hexdigest()!=chosen['original_record_sha256']:raise ValueError('Whole publisher payload must match')
                problem=proto.ContestProblem()
                if problem.ParseFromString(payload)!=len(payload) or problem.solutions[row['original_solution_position']].solution!=row['response']:
                    raise ValueError('Original complete protobuf program must match')
                # This existing helper checks all original-case qualification,
                # direct public identities and source roles. It supplies no
                # private/generated values to the observer or model text.
                _,qualification=public_trace(row,problem,reference,tokenizer)
                disposition=dict(original_reference_position=chosen['original_reference_position'],original_problem_id=chosen['original_problem_id'],
                    reference_result_sha256=chosen['reference_result_sha256'],original_program_sha256=row['original_program_sha256'],
                    original_public_case_position=0,source_reference_qualification=qualification['status'])
                native=None
                if not problem.public_tests:status='no_original_public_input'
                else:
                    public=problem.public_tests[0];observed=capture(row['response'],public.input)
                    path=OUT/'actual-observations'/f'{ordinal:04d}.json';atomic_json(path,observed)
                    disposition.update(observation=str(path),observation_sha256=file_hash(path),original_public_input_sha256=hashlib.sha256(public.input.encode()).hexdigest())
                    counts['actual_fresh_instrumented_public_runs']+=1
                    if not observed['eligible_complete_trace']:status=observed['status']
                    else:
                        raw=observed['actual_observations'];counts['complete_whole_source_frame_traces']+=1;counts['actual_complete_source_events']+=len(raw['events'])
                        actual=base64.b64decode(raw['actual_stdout_b64'],validate=True)
                        prior=reference['actual_execution']['original_case_verdicts'][0]
                        if prior['kind']!='public_tests' or prior['original_kind_position']!=0 or prior['original_input_sha256']!=disposition['original_public_input_sha256']:
                            raise ValueError('First exact original public case identity required')
                        retained=base64.b64decode(prior['actual_stdout_b64'],validate=True)
                        if actual!=retained:status='instrumented_stdout_differs_from_original_retained_execution'
                        else:
                            counts['whole_trace_and_byte_exact_original_stdout']+=1
                            native,reason=target(row['response'],public.input,raw,tokenizer)
                            status='whole_trace_native_candidate' if native is not None else reason
                if native is not None:
                    repeated=capture(row['response'],public.input)
                    repeat_path=OUT/'repeated-observations'/f'{ordinal:04d}.json';atomic_json(repeat_path,repeated)
                    disposition.update(repeated_observation=str(repeat_path),repeated_observation_sha256=file_hash(repeat_path))
                    counts['actual_fresh_native_candidate_trace_repetitions']+=1
                    if not repeated['eligible_complete_trace'] or repeated['actual_observations']!=raw:
                        counts['nonreproducible_whole_trace_candidates']+=1;native=None;status='whole_state_or_stdout_not_reproduced_in_second_fresh_interpreter'
                    else:counts['whole_trace_and_stdout_identical_in_two_fresh_interpreters']+=1
                if native is not None:
                    if writer is None:writer=CodePackWriter(DATA,maximum_bytes=64*1024**2,minimum_free_gib=26)
                    ids,labels=native.pop('ids'),native.pop('labels')
                    candidate=dict(id=chosen['original_problem_id']+':public-state-trace:0',group_id=row['group_id'],split='train',publisher_split='train',
                        language='Python3_public_source_state_prediction',**native,original_program=row['response'],original_public_input=public.input,
                        source_problem_id=chosen['original_problem_id'],original_solution_position=row['original_solution_position'],
                        original_program_sha256=row['original_program_sha256'],original_public_case_position=0,
                        actual_observation_sha256=disposition['observation_sha256'],source_original_reference_position=chosen['original_reference_position'],
                        candidate_not_admitted_to_training=True,no_private_generated_or_held_fields_used_for_model_text=True)
                    writer.add(candidate,ids,labels);counts['complete_native_candidates']+=1
                    counts['complete_native_context_positions']+=len(ids);counts['actual_state_trace_response_EOS_targets']+=sum(t!=-100 for t in labels)
                    counts['stdout_only_response_EOS_targets_same_selected_cases']+=len(tokenizer.encode(actual.decode()).ids)+1
                    counts['native_candidate_source_events']+=native['source_events']
                counts['original_source_dispositions']+=1;counts['status:'+status]+=1
                stream.write(json.dumps(dict(**disposition,status=status),ensure_ascii=False)+'\n');stream.flush()
                atomic_json(OUT/'progress.json',dict(owner=owner,actual_counts=dict(counts),elapsed_seconds=time.monotonic()-started,model_updates=0,training_admitted=False))
        while alive(source_identity):
            if (SOURCE/'failure.json').exists():raise ValueError('Original source epoch failed before state candidate finalization')
            if time.monotonic()-started>7*86400:raise ValueError('Observation deadline; source left unchanged')
            guard.check();time.sleep(10)
        original_complete=SOURCE/'complete.json'
        if (SOURCE/'failure.json').exists() or not original_complete.is_file():raise ValueError('Complete exact original reference epoch required')
        if json.loads(original_complete.read_text())['actual_original_training_problems']!=3959 or counts['original_source_dispositions']!=3959:
            raise ValueError('Every original declared source requires a retained final disposition')
        FileBindings(reference_bindings).check(full=True)
        atomic_json(OUT/'all3959-reference-result-bindings.json',reference_bindings)
        manifest=writer.finish(dict(candidate_preparation_only=True,actual_counts=dict(counts),source_plan_sha256=file_hash(OUT/'plan.json'),
            independently_reexecuted_state_readback_and_held_overlap_admission_still_required=True)) if writer is not None else None
        guard.check(full=True);view.guard.check(full=True)
        observations=sorted((OUT/'actual-observations').glob('*.json'))+sorted((OUT/'repeated-observations').glob('*.json'))
        atomic_json(OUT/'complete.json',dict(at=datetime.now(timezone.utc).isoformat(),complete=True,original_predeclared_source_population=3959,
            actual_counts=dict(counts),actual_native_counts=manifest['counts'] if manifest else {},
            response_target_density_ratio_same_cases=(counts['actual_state_trace_response_EOS_targets']/counts['stdout_only_response_EOS_targets_same_selected_cases']) if counts['stdout_only_response_EOS_targets_same_selected_cases'] else None,
            source_files_sha256={str(p):file_hash(p) for p in [OUT/'plan.json',ledger,original_complete,OUT/'all3959-reference-result-bindings.json',*observations,*([DATA/'manifest.json'] if manifest else [])]},
            actual_program_state_labels_not_neural_teacher_rationales=True,
            every_native_candidate_state_trace_repeated_in_fresh_interpreter=True,
            repeated_instrumentation_uses_same_worker_not_independent_trace_engine=True,
            benchmark_accuracy_or_generalization_gain_not_measured=True,
            no_candidate_admission_or_existing_training_queue_change=True,dataset_admitted_to_training=False,
            models_constructed=0,model_updates=0,candidate_promoted=False,goal_achieved=False))
        print(json.dumps(dict(complete=str(OUT/'complete.json'),actual_counts=dict(counts),training_admitted=False)),flush=True)
    except BaseException as error:
        if writer is not None:
            try:writer.close()
            except Exception:pass
        atomic_json(OUT/'failure.json',dict(error=repr(error),traceback=traceback.format_exc(),actual_counts=dict(counts),training_admitted=False,model_updates=0,goal_achieved=False));raise


if __name__=='__main__':main()
