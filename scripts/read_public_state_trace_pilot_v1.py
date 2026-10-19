"""Read every pilot source/target and repeat all29 candidate state executions.

No neural model, GPU ownership, original-source change or data admission.
Repeated instrumentation tests stability; it is not a second implementation
of the CPython tracing mechanism or a proof of general program semantics.
"""
from collections import Counter
from datetime import datetime,timezone
import base64
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback

from tokenizers import Tokenizer

from slm_lab.code_contests_CE_view_v1 import CodeContestsCEView
from slm_lab.code_contests_execution_readback_v1 import verify as verify_original_outputs
from slm_lab.codenet_checkpoint_v1 import FileBindings
from slm_lab.codenet_token_pack_v1 import CompleteCodePack
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.exact_waiting_process_retirement_v1 import terminal
from slm_lab.local_source_closure_v2 import python_sources
from slm_lab.public_python_state_trace_v1 import capture
from slm_lab.public_state_trace_readback_v1 import rebuild
from slm_lab.source_jsonl_records_v1 import records

ROOT=Path(__file__).resolve().parents[1]
PILOT=ROOT/'reports/public-state-trace-pilot-v1'
DATA=ROOT/'data/public-state-trace-pilot-candidate-v1'
OUT=ROOT/'reports/public-state-trace-pilot-v1-readback'
SCHEMA=ROOT/'data/code-contests-protobuf-schema-v1'


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or OUT.exists():raise ValueError('Fresh CPU-hidden source/target readback required')
    plan_path=PILOT/'plan.json';plan=json.loads(plan_path.read_text());complete_path=PILOT/'complete.json';complete=json.loads(complete_path.read_text())
    if not terminal(plan['owner']) or not complete['complete'] or (PILOT/'failure.json').exists():raise ValueError('Exact complete terminal64-source pilot required')
    FileBindings(plan['source_files_sha256']).check(full=True);FileBindings(complete['source_files_sha256']).check(full=True)
    tests=ROOT/'reports/public-state-trace-readback-v1-tests.log'
    if '1 passed' not in tests.read_text():raise ValueError('Actual native target/source/input/raw-value corruption controls required')
    cohort=plan['fixed_original_reference_cohort'];dispositions=list(records(PILOT/'all64-original-source-dispositions.jsonl'))
    if len(cohort)!=64 or len(dispositions)!=64:raise ValueError('Every predeclared original source required')
    pack=CompleteCodePack(DATA,'train','audit');tokenizer=Tokenizer.from_file(str(DATA/'tokenizer.json'))
    with gzip.open(DATA/'original-rows.jsonl.gz','rt') as stream:candidates=[json.loads(line) for line in stream]
    native={r['source_original_reference_position']:(i,r) for i,r in enumerate(candidates)}
    if len(native)!=29 or len(pack)!=29:raise ValueError('All29 retained original native targets required')
    view=CodeContestsCEView(ROOT);index_path=ROOT/'data/code-contests-native-train-records-v1/all-original-problem-records.jsonl'
    index={r['id']:r for r in records(index_path)}
    schema_path=SCHEMA/'manifest.json';schema=json.loads(schema_path.read_text());sys.path.insert(0,schema['isolated_reader_site_packages']);sys.path.insert(0,str(SCHEMA))
    import contest_problem_pb2 as proto
    files=dict(plan['source_files_sha256']);files.update(complete['source_files_sha256'])
    files.update({str(p):file_hash(p) for p in [complete_path,tests,ROOT/'tests/test_public_state_trace_readback_v1.py',
        *python_sources(ROOT,[Path(__file__),ROOT/'tests/test_public_state_trace_readback_v1.py'])]})
    guard=FileBindings(files);OUT.mkdir();(OUT/'repeated-observations').mkdir()
    owner=dict(pid=os.getpid(),ticks=Path('/proc/self/stat').read_text().split(') ',1)[1].split()[19])
    atomic_json(OUT/'plan.json',dict(at=datetime.now(timezone.utc).isoformat(),owner=owner,source_files_sha256=files,
        all64_original_sources_and_all29_native_targets_required=True,
        each_original_source_public_private_generated_stdout_verdict_rebuilt_without_producer_comparator=True,
        compact_state_target_and_native_masks_rebuilt_without_producer_formatter=True,
        original_tracing_worker_repeated_for_every_native_candidate=True,
        independent_second_trace_implementation_or_OS_confinement_certificate_not_claimed=True,
        original_instrumentation_sources_and_all_raw_artifacts_preserved=True,
        all_prior_training_and_source_owners_untouched=True,dataset_admission=False,models_constructed=0,model_updates=0,goal_achieved=False))
    counts=Counter();unstable=[];diagnoses=[]
    try:
        for ordinal,(chosen,disposition) in enumerate(zip(cohort,dispositions)):
            guard.check();row=view.row(chosen['view_position']);source=index[chosen['original_problem_id']]
            if disposition['original_reference_position']!=chosen['original_reference_position'] or disposition['original_problem_id']!=chosen['original_problem_id']:
                raise ValueError('Every original source order and disposition must match')
            path=Path(chosen['reference_result'])
            if file_hash(path)!=chosen['reference_result_sha256']:raise ValueError('Original completed reference changed')
            reference=json.loads(path.read_text())
            with open(source['extracted_record_file'],'rb') as stream:
                stream.seek(source['extracted_record_offset']);payload=stream.read(source['extracted_record_bytes'])
            if hashlib.sha256(payload).hexdigest()!=chosen['original_record_sha256']:raise ValueError('Every whole original protobuf payload required')
            problem=proto.ContestProblem()
            if problem.ParseFromString(payload)!=len(payload):raise ValueError('Whole official original record required')
            verified=verify_original_outputs(row,problem,reference)
            if not verified['fully_qualified']:raise ValueError('Every pilot source requires full original finite-case success')
            counts['all64_full_original_reference_dispositions_rebuilt']+=1
            counts['actual_original_reference_case_outputs_checked']+=verified['actual_retained_case_outputs_checked']
            if not problem.public_tests:
                if disposition['status']!='no_original_public_input':raise ValueError('No-public-input source must retain its qualification')
                counts['no_original_public_input']+=1;continue
            public=problem.public_tests[0];program=row['response'];observation_path=Path(disposition['observation'])
            if file_hash(observation_path)!=disposition['observation_sha256']:raise ValueError('Every raw observation must remain exact')
            observed=json.loads(observation_path.read_text());counts['original_observation_receipts_checked']+=1
            if observed['program_sha256']!=hashlib.sha256(program.encode()).hexdigest() or observed['public_input_sha256']!=hashlib.sha256(public.input.encode()).hexdigest():
                raise ValueError('Exact original whole code and public case0 only required')
            if not observed['eligible_complete_trace']:
                if chosen['original_reference_position'] in native:raise ValueError('A qualified-out partial observation cannot become a target')
                if disposition['status']=='invalid_observer_transport_qualification':
                    text=base64.b64decode(observed['raw_observation_stdout_b64']).decode()
                    try:
                        first,end=json.JSONDecoder().raw_decode(text)
                        diagnoses.append(dict(ordinal=ordinal,original_problem_id=chosen['original_problem_id'],status='extra_output_after_observer_JSON',
                            first_JSON_format=first.get('format'),extra_output_after_complete_JSON=text[end:],
                            excluded_without_salvaging_or_rewriting_original_source=True))
                    except ValueError:diagnoses.append(dict(ordinal=ordinal,status='invalid_JSON_transport_retained_without_salvage'))
                counts['whole_observer_trace_qualifications_retained']+=1;continue
            candidate=rebuild(program,public.input,observed,tokenizer);counts['complete_raw_state_traces_checked']+=1
            raw=observed['actual_observations'];prior=reference['actual_execution']['original_case_verdicts'][0]
            stdout=base64.b64decode(raw['actual_stdout_b64'],validate=True)
            if (prior['kind']!='public_tests' or prior['original_kind_position']!=0
                    or prior['original_input_sha256']!=hashlib.sha256(public.input.encode()).hexdigest()
                    or stdout!=base64.b64decode(prior['actual_stdout_b64'],validate=True)
                    or stdout.split()!=public.output.encode().split()):
                raise ValueError('Actual instrumented output must retain the exact original public execution and token verdict')
            if candidate is None:
                if chosen['original_reference_position'] in native:raise ValueError('A nonfitting whole target must stay excluded')
                counts['whole_native_context_qualifications_retained']+=1;continue
            position,packed=native[chosen['original_reference_position']];ids,labels=pack.sequence(position)
            if (packed['original_program']!=program or packed['original_public_input']!=public.input
                    or packed['prompt']!=candidate['prompt'] or packed['response']!=candidate['response']
                    or ids!=candidate['ids'] or labels!=candidate['labels'] or packed['source_events']!=candidate['events']
                    or packed['actual_observation_sha256']!=disposition['observation_sha256']):
                raise ValueError('Every complete source state/event/native token/EOS/mask must independently rebuild')
            counts['all29_complete_native_targets_rebuilt']+=1;counts['complete_native_context_positions']+=len(ids)
            counts['actual_state_trace_response_EOS_targets']+=sum(v!=-100 for v in labels)
            counts['stdout_only_response_EOS_targets_same_cases']+=len(tokenizer.encode(candidate['actual_stdout']).ids)+1
            repeated=capture(program,public.input,maximum_events=256,maximum_trace_bytes=256*1024)
            destination=OUT/'repeated-observations'/f'{ordinal:03d}.json';atomic_json(destination,repeated)
            same=(repeated['eligible_complete_trace'] and repeated['actual_observations']==raw)
            counts['actual_fresh_candidate_trace_repetitions']+=1;counts['whole_source_states_and_stdout_identical_across_repetition']+=same
            if not same:unstable.append(dict(ordinal=ordinal,original_problem_id=chosen['original_problem_id'],actual_repeat=str(destination)))
            atomic_json(OUT/'progress.json',dict(owner=owner,actual_counts=dict(counts),unstable_candidates=len(unstable),model_updates=0,training_admitted=False))
        expected=complete['actual_counts']
        if (counts['all64_full_original_reference_dispositions_rebuilt']!=64 or counts['all29_complete_native_targets_rebuilt']!=29
                or counts['actual_state_trace_response_EOS_targets']!=expected['actual_state_trace_response_EOS_targets']
                or counts['complete_native_context_positions']!=expected['complete_native_context_positions']):
            raise ValueError('Every original source and every native target denominator required')
        guard.check(full=True)
        atomic_json(OUT/'transport-qualification-diagnosis.json',diagnoses)
        repeated_paths=sorted((OUT/'repeated-observations').glob('*.json'))
        atomic_json(OUT/'complete.json',dict(at=datetime.now(timezone.utc).isoformat(),complete=True,actual_counts=dict(counts),
            unstable_candidates=unstable,all29_whole_state_traces_reproduced=(not unstable),
            response_target_density_ratio_same_cases=counts['actual_state_trace_response_EOS_targets']/counts['stdout_only_response_EOS_targets_same_cases'],
            source_files_sha256={str(p):file_hash(p) for p in [OUT/'plan.json',OUT/'transport-qualification-diagnosis.json',*repeated_paths]},
            independent_source_stdout_target_tokens_and_masks_readback_complete=True,
            repetition_uses_original_trace_worker_not_independent_trace_engine=True,
            full_reference_epoch_and_held_overlap_checks_still_required_before_any_separate_admission=True,
            no_original_data_or_training_queue_modified=True,dataset_admitted_to_training=False,
            models_constructed=0,model_updates=0,candidate_promoted=False,goal_achieved=False))
        print(json.dumps(dict(complete=str(OUT/'complete.json'),actual_counts=dict(counts),unstable_candidates=len(unstable),training_admitted=False)),flush=True)
    except BaseException as error:
        atomic_json(OUT/'failure.json',dict(error=repr(error),traceback=traceback.format_exc(),actual_counts=dict(counts),model_updates=0,goal_achieved=False));raise


if __name__=='__main__':main()
