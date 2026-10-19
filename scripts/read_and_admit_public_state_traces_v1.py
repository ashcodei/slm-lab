"""Rebuild every declared state candidate and admit a separate screened CE pack.

Waits for the original reference, original independent output reader and full
state producer to finish. It neither starts training nor changes any existing
candidate, model, source epoch or source-bound training queue.
"""
from collections import Counter
from datetime import datetime,timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

import numpy as np
from tokenizers import Tokenizer

from slm_lab.code_contests_CE_view_v1 import CodeContestsCEView
from slm_lab.code_contests_execution_readback_v1 import verify as verify_original_outputs
from slm_lab.codenet_checkpoint_v1 import FileBindings
from slm_lab.codenet_token_pack_v1 import CompleteCodePack,CodePackWriter
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.exact_waiting_process_retirement_v1 import terminal
from slm_lab.held_text_overlap_v1 import HeldTextOverlap
from slm_lab.local_source_closure_v2 import python_sources
from slm_lab.source_jsonl_records_v1 import records
from slm_lab.state_trace_source_decision_v1 import assess

ROOT=Path(__file__).resolve().parents[1]
REFERENCE=ROOT/'reports/CodeContests-full-Python-references-v1'
REFERENCE_READER=ROOT/'reports/CodeContests-reference-outputs-v1-readback'
STATES=ROOT/'reports/public-state-traces-full-v1'
CANDIDATE=ROOT/'data/public-state-traces-full-candidate-v1'
OUT=ROOT/'reports/public-state-traces-full-v1-readback-admission'
ADMITTED=ROOT/'data/public-state-traces-admitted-v1'
SCHEMA=ROOT/'data/code-contests-protobuf-schema-v1'


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or OUT.exists() or ADMITTED.exists():raise ValueError('Fresh CPU-hidden separate state-data admission required')
    tests=ROOT/'reports/state-trace-source-decision-v1-tests.log'
    if '3 passed' not in tests.read_text():raise ValueError('Actual all-case/whole-state/repeat/context/source-role controls required')
    folders={'reference':REFERENCE,'reference_reader':REFERENCE_READER,'states':STATES}
    plans={name:json.loads((folder/'plan.json').read_text()) for name,folder in folders.items()}
    registry_path=ROOT/'reports/CodeContests-native-held-text-v2/unchanged-held-text-registry.json'
    schema_path=SCHEMA/'manifest.json';schema=json.loads(schema_path.read_text())
    files={str(p):file_hash(p) for p in [tests,registry_path,schema_path,ROOT/'tests/test_state_trace_source_decision_v1.py',
        *(folder/'plan.json' for folder in folders.values()),*python_sources(ROOT,[Path(__file__),ROOT/'tests/test_state_trace_source_decision_v1.py'])]}
    files.update({str(SCHEMA/n):h for n,h in schema['files_sha256'].items()})
    guard=FileBindings(files);OUT.mkdir();started=time.monotonic();deadline=started+7*86400
    owner=dict(pid=os.getpid(),ticks=Path('/proc/self/stat').read_text().split(') ',1)[1].split()[19])
    atomic_json(OUT/'plan.json',dict(at=datetime.now(timezone.utc).isoformat(),owner=owner,source_files_sha256=files,
        exact_dependency_owners={k:p['owner'] for k,p in plans.items()},original_declared_sources=3959,
        all_original_source_dispositions_and_all_complete_native_candidates_required=True,
        every_retained_original_case_stdout_verdict_rebuilt_without_producer_comparator=True,
        state_targets_and_own_native_tokens_masks_EOS_rebuilt_without_producer_formatter=True,
        both_actual_whole_state_observations_must_match_before_native_admission=True,
        no_new_program_execution_or_neural_model_in_this_reader=True,
        all32953_unchanged_held_stems_scanned_on_actual_full_native_inputs=True,
        positive_exposure_or_unscannable_NUL_excludes_entire_source_component=True,
        no_negative_search_semantic_or_all_historical_independence_certificate=True,
        candidate_and_existing_training_queues_left_unchanged=True,
        admitted_scope='Public-input observed-source-state prediction CE only; no execution-reward or human-preference admission',
        models_constructed=0,model_updates=0,automatic_retry=False,goal_achieved=False))
    writer=None
    try:
        while True:
            ready=True;guard.check()
            for name,folder in folders.items():
                if (folder/'failure.json').exists():raise ValueError('Original '+name+' failed; no partial candidate admission')
                if not terminal(plans[name]['owner']):ready=False
                elif not (folder/'complete.json').is_file():raise ValueError('Exact original '+name+' ended without completion')
            if ready:break
            if time.monotonic()>deadline:raise ValueError('Admission observation deadline; original jobs left untouched')
            atomic_json(OUT/'waiting.json',dict(owner=owner,dependency_live={k:not terminal(p['owner']) for k,p in plans.items()},
                dataset_admitted_to_training=False,models_constructed=0,model_updates=0,goal_achieved=False))
            time.sleep(10)
        finals={name:json.loads((folder/'complete.json').read_text()) for name,folder in folders.items()}
        if (finals['reference']['actual_original_training_problems']!=3959
                or not finals['reference_reader']['every_declared_original_reference_output_and_complete_source_rebuilt']
                or not finals['reference_reader']['every_retained_stdout_compared_directly_against_original_expected_tokens']
                or finals['states']['original_predeclared_source_population']!=3959
                or finals['states']['actual_counts']['original_source_dispositions']!=3959
                or finals['states']['dataset_admitted_to_training'] is not False):
            raise ValueError('Exact complete original epochs, reference-output reader and full unadmitted state candidate required')
        sealed={}
        for name,folder in folders.items():
            sealed.update(finals[name]['source_files_sha256']);sealed[str(folder/'complete.json')]=file_hash(folder/'complete.json')
        sealed_guard=FileBindings(sealed)
        candidate=CompleteCodePack(CANDIDATE,'train','audit');tokenizer=Tokenizer.from_file(str(CANDIDATE/'tokenizer.json'))
        view=CodeContestsCEView(ROOT)
        if file_hash(CANDIDATE/'tokenizer.json')!=view.provenance['tokenizer_sha256']:raise ValueError('Exactly one original own tokenizer required')
        with gzip.open(CANDIDATE/'original-rows.jsonl.gz','rt') as stream:rows=[json.loads(line) for line in stream]
        if len(rows)!=len(candidate) or len({r['source_problem_id'] for r in rows})!=len(rows):raise ValueError('Every unique state candidate required')
        identities=json.loads((REFERENCE/'all-original-selected-reference-identities.json').read_text())
        dispositions=list(records(STATES/'all3959-original-source-dispositions.jsonl'));declared=plans['states']['fixed_original_reference_cohort']
        if any(len(v)!=3959 for v in [identities,dispositions,declared]):raise ValueError('All3959 original source roles and dispositions required')
        original_index={r['id']:r for r in records(ROOT/'data/code-contests-native-train-records-v1/all-original-problem-records.jsonl')}
        registry=json.loads(registry_path.read_text())
        if len(registry)!=32953:raise ValueError('Every named held stem must remain unchanged')
        exposure=HeldTextOverlap(registry)
        sys.path.insert(0,schema['isolated_reader_site_packages']);sys.path.insert(0,str(SCHEMA))
        import contest_problem_pb2 as proto
        counts=Counter();position=0;rebuilt=[];blocked=set();actual_bindings={}
        with (OUT/'all3959-source-state-native-readbacks.jsonl').open('x') as ledger:
            for original_position,(identity,disposition,claim) in enumerate(zip(identities,dispositions,declared)):
                guard.check()
                if (disposition['original_reference_position']!=original_position or claim['original_reference_position']!=original_position
                        or any(claim[k]!=identity[k] for k in ['view_position','original_problem_id','original_record_sha256','original_program_sha256','original_solution_position'])
                        or disposition['original_problem_id']!=identity['original_problem_id']):raise ValueError('Every original source position must match the declared complete cohort')
                result_path=REFERENCE/'original-problem-executions'/f'{original_position:05d}.json'
                sha=file_hash(result_path);actual_bindings[str(result_path)]=sha
                if sha!=disposition['reference_result_sha256']:raise ValueError('Original reference result changed')
                result=json.loads(result_path.read_text());row=view.row(identity['view_position']);source=original_index[identity['original_problem_id']]
                with open(source['extracted_record_file'],'rb') as stream:
                    stream.seek(source['extracted_record_offset']);payload=stream.read(source['extracted_record_bytes'])
                if hashlib.sha256(payload).hexdigest()!=identity['original_record_sha256']:raise ValueError('Every whole original publisher record must remain exact')
                problem=proto.ContestProblem()
                if problem.ParseFromString(payload)!=len(payload):raise ValueError('Whole official publisher record required')
                program=problem.solutions[identity['original_solution_position']]
                if int(program.language)!=3 or program.solution!=row['response']:raise ValueError('Exact whole original Python3 source required')
                checked=verify_original_outputs(row,problem,result);counts['retained_original_case_stdout_verdicts_checked']+=checked['actual_retained_case_outputs_checked']
                loaded={}
                for field,subfolder in [('observation','actual-observations'),('repeated_observation','repeated-observations')]:
                    loaded[field]=None
                    if field in disposition:
                        path=STATES/subfolder/f'{original_position:04d}.json'
                        if str(path)!=disposition[field] or file_hash(path)!=disposition[field+'_sha256']:raise ValueError('Every original observation path and actual bytes required')
                        actual_bindings[str(path)]=disposition[field+'_sha256'];loaded[field]=json.loads(path.read_text())
                decision=assess(row,problem,result,disposition,loaded['observation'],loaded['repeated_observation'],tokenizer)
                native=decision['target'];literal={};lexical={};scan_qualification=None
                if native is not None:
                    if position>=len(rows):raise ValueError('Missing complete native candidate')
                    actual=rows[position];ids,labels=candidate.sequence(position)
                    if (actual['source_original_reference_position']!=original_position or actual['source_problem_id']!=identity['original_problem_id']
                            or actual['group_id']!=row['group_id'] or actual['original_program']!=row['response']
                            or actual['original_public_input']!=problem.public_tests[0].input or actual['original_public_case_position']!=0
                            or actual['original_solution_position']!=identity['original_solution_position']
                            or actual['original_program_sha256']!=identity['original_program_sha256']
                            or actual['split']!='train' or actual['publisher_split']!='train'
                            or actual['language']!='Python3_public_source_state_prediction' or actual['token_pack_index']!=position
                            or actual['candidate_not_admitted_to_training'] is not True
                            or actual['no_private_generated_or_held_fields_used_for_model_text'] is not True
                            or actual['actual_observation_sha256']!=disposition['observation_sha256']
                            or actual['source_events']!=native['events'] or actual['prompt']!=native['prompt'] or actual['response']!=native['response']
                            or ids!=native['ids'] or labels!=native['labels'] or actual['target_tokens']!=sum(v!=-100 for v in labels)
                            or actual['complete_input_ids_sha256']!=hashlib.sha256(np.asarray(ids,dtype='<u2').tobytes()).hexdigest()):
                        raise ValueError('Every whole source state/event/native input/target token and EOS mask must reconstruct')
                    decoded=tokenizer.decode(ids[1:-1],skip_special_tokens=False)
                    if '\0' in decoded:scan_qualification='native_NUL_boundary_not_supported_by_held_scan'
                    else:literal=exposure.literal(decoded);lexical=exposure.lexical_copies(decoded)
                    if literal or lexical or scan_qualification:blocked.add(actual['group_id'])
                    rebuilt.append((actual,ids,labels));position+=1
                    counts['complete_native_candidates_rebuilt']+=1;counts['complete_native_context_positions']+=len(ids)
                    counts['actual_state_trace_response_EOS_targets']+=sum(v!=-100 for v in labels)
                    counts['candidate_inputs_with_literal_held_stems']+=bool(literal)
                    counts['candidate_inputs_with_lexical_copy_signal']+=bool(lexical)
                    counts['candidate_inputs_with_scan_qualification']+=bool(scan_qualification)
                counts['all_original_source_dispositions_rebuilt']+=1;counts['status:'+decision['status']]+=1
                ledger.write(json.dumps(dict(original_reference_position=original_position,original_problem_id=identity['original_problem_id'],
                    status=decision['status'],has_complete_native_candidate=native is not None,literal_held_stem_signals=literal,
                    lexical_held_copy_signals=lexical,scan_qualification=scan_qualification),ensure_ascii=False)+'\n')
                if original_position%64==0:
                    ledger.flush();atomic_json(OUT/'progress.json',dict(owner=owner,actual_counts=dict(counts),models_constructed=0,model_updates=0))
        actual_counts=dict(examples=position,context_tokens=counts['complete_native_context_positions'],target_tokens=counts['actual_state_trace_response_EOS_targets'])
        if (position!=len(rows) or actual_counts!=candidate.manifest['counts']['train']
                or counts['retained_original_case_stdout_verdicts_checked']!=finals['reference_reader']['actual_counts']['actual_retained_case_outputs_checked']):
            raise ValueError('Every original case observation and complete native token denominator must reconcile')
        atomic_json(OUT/'candidate-source-component-exclusions.json',dict(blocked_source_components=sorted(blocked),unchanged_held_stems=32953,
            exposure_configuration=exposure.configuration,whole_native_input_scan_includes_actual_state_targets=True,
            source_exposure_signals_not_semantic_or_full_ancestry_certificates=True))
        ADMITTED.mkdir()
        for name in ['tokenizer.json','model-config.json','DATASET-LICENSE.txt','publisher-README.md']:shutil.copyfile(CANDIDATE/name,ADMITTED/name)
        writer=CodePackWriter(ADMITTED,maximum_bytes=128*1024**2,minimum_free_gib=26);admitted_sequences=[]
        for row,ids,labels in rebuilt:
            if row['group_id'] in blocked:counts['native_candidates_excluded_by_source_component']+=1;continue
            exported={k:v for k,v in row.items() if k not in {'token_pack_index','complete_input_ids_sha256','target_tokens','candidate_not_admitted_to_training'}}
            exported['original_state_candidate_position']=row['token_pack_index'];exported['admitted_scope']='Public source-state prediction CE only'
            writer.add(exported,ids,labels);admitted_sequences.append((ids,labels))
        manifest=writer.finish(dict(original_candidate_manifest_sha256=file_hash(CANDIDATE/'manifest.json'),
            complete_original_reference_epoch_sha256=file_hash(REFERENCE/'complete.json'),independent_original_output_reader_sha256=file_hash(REFERENCE_READER/'complete.json'),
            every_source_disposition_and_every_actual_state_target_and_native_mask_rebuilt=True,
            every_admitted_candidate_has_two_matching_actual_whole_source_state_observations=True,
            model_text_from_original_train_program_public_input_and_actual_source_observations_only=True,
            no_neural_teacher_or_additional_model_parameters=True,all32953_held_stems_scanned=True,
            actual_candidate_counts=dict(counts),admitted_scope='Public source-state prediction CE only; not execution rewards or human preferences',
            independent_negative_search_semantic_all_history_or_independent_trace_engine_certificate_not_claimed=True))
        writer=None;check=CompleteCodePack(ADMITTED,'train','audit')
        if not len(check) or len(check)!=len(admitted_sequences):raise ValueError('Nonempty exact separately written admitted pack required')
        for i,expected in enumerate(admitted_sequences):
            if check.sequence(i)!=expected:raise ValueError('Every admitted native token and EOS mask must cold-read exactly')
        guard.check(full=True);sealed_guard.check(full=True);FileBindings(actual_bindings).check(full=True)
        atomic_json(OUT/'original-source-and-observation-bindings.json',actual_bindings)
        evidence={**files,**sealed,**actual_bindings,**{str(p):file_hash(p) for p in [OUT/'plan.json',CANDIDATE/'manifest.json',ADMITTED/'manifest.json',
            OUT/'all3959-source-state-native-readbacks.jsonl',OUT/'candidate-source-component-exclusions.json',OUT/'original-source-and-observation-bindings.json']}}
        atomic_json(ADMITTED/'admission.json',dict(at=datetime.now(timezone.utc).isoformat(),dataset_manifest_sha256=file_hash(ADMITTED/'manifest.json'),
            admitted_to_training=True,admitted_scope='Public source-state prediction CE only',source_files_sha256=evidence,
            actual_native_counts=manifest['counts']['train'],every_original_source_and_native_target_independently_rebuilt=True,
            actual_all32953_held_stems_scanned_and_flagged_components_excluded=True,
            every_admitted_sequence_cold_read_exact=True,original_candidate_unmodified_and_unadmitted=True,
            execution_reward_training_admitted=False,human_preference_training_admitted=False,
            models_constructed=0,model_updates=0,goal_achieved=False))
        atomic_json(OUT/'complete.json',dict(at=datetime.now(timezone.utc).isoformat(),complete=True,actual_counts=dict(counts),
            original_source_dispositions=3959,admitted_native_train_counts=manifest['counts']['train'],blocked_source_components=len(blocked),
            every_original_source_and_every_state_target_and_mask_rebuilt=True,
            every_separately_written_native_sequence_cold_read_exact=True,dataset_admitted_to_auxiliary_CE_training=True,
            execution_reward_or_human_preference_admission=False,existing_training_queues_and_original_candidate_unchanged=True,
            source_files_sha256={str(p):file_hash(p) for p in [OUT/'plan.json',ADMITTED/'manifest.json',ADMITTED/'admission.json']},
            models_constructed=0,model_updates=0,candidate_promoted=False,goal_achieved=False))
        print(json.dumps(dict(complete=str(OUT/'complete.json'),admitted_native_train_counts=manifest['counts']['train'],model_training_started=False)),flush=True)
    except BaseException as error:
        if writer is not None:writer.close()
        atomic_json(OUT/'failure.json',dict(error=repr(error),traceback=traceback.format_exc(),original_sources_and_training_queues_unchanged=True,
            models_constructed=0,model_updates=0,goal_achieved=False));raise


if __name__=='__main__':main()
