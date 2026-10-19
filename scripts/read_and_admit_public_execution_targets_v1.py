"""Reconstruct every public execution target, screen held text, admit a new pack.

Waits for both original producers and the independent full-output reader to end
successfully. The original candidate remains unchanged and unadmitted. Only
public-input program-output supervision is admitted, never execution rewards.
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
from slm_lab.codenet_checkpoint_v1 import FileBindings
from slm_lab.codenet_token_pack_v1 import CompleteCodePack,CodePackWriter
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.held_text_overlap_v1 import HeldTextOverlap
from slm_lab.local_source_closure_v2 import python_sources
from slm_lab.public_execution_target_readback_v1 import rebuild
from slm_lab.source_jsonl_records_v1 import records

ROOT=Path(__file__).resolve().parents[1]
REFERENCE=ROOT/'reports/CodeContests-full-Python-references-v1'
CANDIDATE=ROOT/'data/code-contests-public-execution-traces-candidate-v1'
PREPARATION=ROOT/'reports/CodeContests-public-execution-traces-candidate-v1'
REFERENCE_READER=ROOT/'reports/CodeContests-reference-outputs-v1-readback'
OUT=ROOT/'reports/public-execution-targets-v1-readback-admission'
ADMITTED=ROOT/'data/code-contests-public-execution-traces-admitted-v1'
SCHEMA=ROOT/'data/code-contests-protobuf-schema-v1'


def alive(owner):
    path=Path('/proc')/str(owner['pid'])/'stat'
    if not path.exists():return False
    values=path.read_text().split(') ',1)[1].split()
    return values[19]==owner['ticks'] and values[0] not in {'Z','X'}


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or os.environ.get('TOKENIZERS_PARALLELISM')!='false' or OUT.exists() or ADMITTED.exists():
        raise ValueError('Fresh CPU-only separately admitted execution-supervision pack required')
    tests=ROOT/'reports/public-execution-target-readback-v1-tests.log'
    if '3 passed' not in tests.read_text():raise ValueError('Actual stdout-not-gold, public-only, source-role, first-fit and no-clipping reconstruction controls required')
    dependencies={name:json.loads((folder/'plan.json').read_text()) for name,folder in
        [('reference',REFERENCE),('preparation',PREPARATION),('reference_reader',REFERENCE_READER)]}
    registry_path=ROOT/'reports/CodeContests-native-held-text-v2/unchanged-held-text-registry.json'
    schema_path=SCHEMA/'manifest.json';schema=json.loads(schema_path.read_text())
    files={str(p):file_hash(p) for p in [tests,registry_path,schema_path,
        ROOT/'tests/test_public_execution_target_readback_v1.py',
        *[folder/'plan.json' for folder in [REFERENCE,PREPARATION,REFERENCE_READER]],
        *python_sources(ROOT,[Path(__file__),ROOT/'tests/test_public_execution_target_readback_v1.py'])]}
    files.update({str(SCHEMA/n):v for n,v in schema['files_sha256'].items()})
    guard=FileBindings(files);OUT.mkdir();started=time.monotonic();deadline=started+7*86400
    owner=dict(pid=os.getpid(),ticks=Path('/proc/self/stat').read_text().split(') ',1)[1].split()[19])
    atomic_json(OUT/'plan.json',dict(at=datetime.now(timezone.utc).isoformat(),owner=owner,source_files_sha256=files,
        exact_dependency_owners={k:v['owner'] for k,v in dependencies.items()},
        every_original_declared_reference_disposition_and_every_candidate_native_target_reconstructed=True,
        complete_original_program_and_public_input_only_as_model_text=True,
        target_is_actual_retained_stdout_not_publisher_gold_or_model_teacher=True,
        all32953original_held_stems_retained_and_actual_native_input_screened=True,
        positive_literal_or_lexical_exposure_excludes_entire_candidate_source_component=True,
        independent_negative_exposure_search_or_semantic_independence_not_certified=True,
        original_candidate_remains_unchanged_and_unadmitted=True,
        auxiliary_CE_admission_only_execution_reward_and_human_preference_admission_forbidden=True,
        observation_deadline_seconds=7*86400,automatic_retry=False,models_constructed=0,model_updates=0,goal_achieved=False))
    writer=None
    try:
        while True:
            guard.check(full=False);ready=True
            for name,folder in [('reference',REFERENCE),('preparation',PREPARATION),('reference_reader',REFERENCE_READER)]:
                if (folder/'failure.json').exists():raise ValueError('Original '+name+' failed; preserve all partial sources')
                if alive(dependencies[name]['owner']):ready=False
                elif not (folder/'complete.json').exists():raise ValueError('Original '+name+' owner ended without complete evidence')
            if ready:break
            if time.monotonic()>deadline:raise ValueError('Reader observation deadline; all original jobs left intact')
            atomic_json(OUT/'waiting.json',dict(owner=owner,status='waiting for all original producers and full-output readback to finish',
                dependency_alive={k:alive(v['owner']) for k,v in dependencies.items()},
                model_updates=0,dataset_admitted_to_training=False,goal_achieved=False))
            time.sleep(15)
        finals={name:json.loads((folder/'complete.json').read_text()) for name,folder in
            [('reference',REFERENCE),('preparation',PREPARATION),('reference_reader',REFERENCE_READER)]}
        if (not finals['reference_reader']['every_declared_original_reference_output_and_complete_source_rebuilt']
                or not finals['reference_reader']['every_retained_stdout_compared_directly_against_original_expected_tokens']
                or not finals['preparation']['every_original_reference_problem_dispositioned']
                or finals['preparation']['dataset_admitted_to_training'] is not False):
            raise ValueError('Every original case-output and candidate disposition must actually be completed')
        complete_bindings={}
        for name,folder in [('reference',REFERENCE),('preparation',PREPARATION),('reference_reader',REFERENCE_READER)]:
            complete_bindings.update(finals[name]['source_files_sha256'])
            complete_bindings[str(folder/'complete.json')]=file_hash(folder/'complete.json')
        complete_guard=FileBindings(complete_bindings)
        candidate=CompleteCodePack(CANDIDATE,'train','audit');tokenizer=Tokenizer.from_file(str(CANDIDATE/'tokenizer.json'))
        view=CodeContestsCEView(ROOT)
        if file_hash(CANDIDATE/'tokenizer.json')!=view.provenance['tokenizer_sha256']:
            raise ValueError('One exact own BPE required')
        registry=json.loads(registry_path.read_text())
        if len(registry)!=32953:raise ValueError('Every original named held stem required')
        exposure=HeldTextOverlap(registry)
        identities=json.loads((REFERENCE/'all-original-selected-reference-identities.json').read_text())
        sources={r['id']:r for r in records(ROOT/'data/code-contests-native-train-records-v1/all-original-problem-records.jsonl')}
        dispositions=list(records(PREPARATION/'all-original-reference-trace-dispositions.jsonl'))
        if len(dispositions)!=len(identities):raise ValueError('Every original declared reference disposition required')
        with gzip.open(CANDIDATE/'original-rows.jsonl.gz','rt') as stream:rows=[json.loads(line) for line in stream]
        if len(rows)!=len(candidate) or len({r['source_problem_id'] for r in rows})!=len(rows):
            raise ValueError('Every unique candidate and original native position required')
        sys.path.insert(0,schema['isolated_reader_site_packages']);sys.path.insert(0,str(SCHEMA))
        import contest_problem_pb2 as proto
        counts=Counter();position=0;rebuilt=[];blocked=set();source_result_bindings={}
        with (OUT/'every-original-reference-and-native-target-readback.jsonl').open('x') as ledger:
            for original_position,(identity,disposition) in enumerate(zip(identities,dispositions)):
                result_path=REFERENCE/'original-problem-executions'/f'{original_position:05d}.json'
                result=json.loads(result_path.read_text());result_sha=file_hash(result_path)
                source_result_bindings[str(result_path)]=result_sha
                if (disposition['original_reference_position']!=original_position
                        or disposition['original_problem_id']!=identity['original_problem_id']
                        or disposition['reference_result_sha256']!=result_sha):
                    raise ValueError('Original reference disposition identity differs')
                expected=None;literal={};lexical={};row=view.row(identity['view_position'])
                if result['all_original_finite_cases_pass']:
                    source=sources[identity['original_problem_id']]
                    with open(source['extracted_record_file'],'rb') as stream:
                        stream.seek(source['extracted_record_offset']);raw=stream.read(source['extracted_record_bytes'])
                    if hashlib.sha256(raw).hexdigest()!=identity['original_record_sha256']:
                        raise ValueError('Complete original publisher record differs')
                    problem=proto.ContestProblem()
                    if problem.ParseFromString(raw)!=len(raw):raise ValueError('Every complete original source field required')
                    program=problem.solutions[identity['original_solution_position']]
                    if int(program.language)!=3 or program.solution!=row['response']:raise ValueError('Entire original accepted program differs')
                    expected=rebuild(row,problem,result,tokenizer)
                if expected is not None:
                    if position>=len(rows):raise ValueError('Missing original complete candidate target')
                    actual=rows[position];ids,labels=candidate.sequence(position)
                    for key,value in expected.items():
                        if key not in {'ids','labels'} and actual.get(key)!=value:raise ValueError('Independent public-only source/stdout target reconstruction differs: '+key)
                    expected_id=row['source_problem_id']+':public-execution:'+str(row['original_solution_position'])+':'+str(expected['original_public_case_position'])
                    if (actual['id']!=expected_id or actual['source_problem_id']!=row['source_problem_id']
                            or actual['group_id']!=row['group_id'] or actual['parent_original_presentation_id']!=row['id']
                            or actual['original_program_sha256']!=row['original_program_sha256']
                            or actual['original_solution_position']!=row['original_solution_position']
                            or actual['split']!='train' or actual['publisher_split']!='train'
                            or actual['language']!='Python3_execution_prediction' or actual['token_pack_index']!=position
                            or actual['no_private_or_generated_test_values_in_model_prompt_or_target'] is not True
                            or actual['target_is_actual_observed_stdout_not_generated_teacher_text'] is not True
                            or ids!=expected['ids'] or labels!=expected['labels']
                            or actual['target_tokens']!=sum(v!=-100 for v in labels)
                            or actual['complete_input_ids_sha256']!=hashlib.sha256(np.asarray(ids,dtype='<u2').tobytes()).hexdigest()
                            or disposition['status']!='first_complete_fitting_public_execution_trace_selected'
                            or disposition['selected_public_case_position']!=expected['original_public_case_position']):
                        raise ValueError('Whole unchanged candidate source, actual tokens, EOS mask or stable first-fitting public selection differs')
                    decoded=tokenizer.decode(ids[1:-1],skip_special_tokens=False)
                    literal=exposure.literal(decoded);lexical=exposure.lexical_copies(decoded)
                    if literal or lexical:blocked.add(actual['group_id'])
                    rebuilt.append((actual,ids,labels))
                    counts['actual_complete_native_targets_rebuilt']+=1;counts['actual_complete_context_positions']+=len(ids)
                    counts['actual_stdout_EOS_targets']+=sum(v!=-100 for v in labels)
                    counts['candidate_inputs_with_literal_held_stems']+=bool(literal)
                    counts['candidate_inputs_with_lexical_copy_signal']+=bool(lexical)
                    position+=1
                else:
                    expected_status='no_complete_fitting_nonempty_public_execution_trace' if result['all_original_finite_cases_pass'] else 'reference_not_qualified_for_all_original_finite_cases'
                    if disposition['status']!=expected_status:raise ValueError('Original source qualification/nonfit disposition differs')
                counts['all_original_reference_dispositions_checked']+=1
                ledger.write(json.dumps(dict(original_reference_position=original_position,original_problem_id=identity['original_problem_id'],
                    source_result_sha256=result_sha,has_complete_public_execution_target=expected is not None,
                    literal_held_stem_signals=literal,lexical_held_copy_signals=lexical),ensure_ascii=False)+'\n')
        expected_counts=dict(examples=position,context_tokens=counts['actual_complete_context_positions'],target_tokens=counts['actual_stdout_EOS_targets'])
        if position!=len(rows) or expected_counts!=candidate.manifest['counts']['train']:
            raise ValueError('Every original candidate native token and target denominator must reconcile')
        atomic_json(OUT/'candidate-exposure-component-exclusions.json',dict(blocked_source_components=sorted(blocked),
            unchanged_held_stems=32953,exposure_configuration=exposure.configuration,
            exposure_signals_conservative_not_semantic_equivalence_labels=True))
        ADMITTED.mkdir()
        for name in ('tokenizer.json','model-config.json','DATASET-LICENSE.txt','publisher-README.md'):
            shutil.copy2(CANDIDATE/name,ADMITTED/name)
        writer=CodePackWriter(ADMITTED,maximum_bytes=512*1024**2,minimum_free_gib=26)
        admitted_rows=[]
        for row,ids,labels in rebuilt:
            if row['group_id'] in blocked:
                counts['candidate_targets_excluded_by_source_component']+=1;continue
            original_position=row['token_pack_index']
            export={k:v for k,v in row.items() if k not in {'token_pack_index','complete_input_ids_sha256','target_tokens'}}
            export['original_execution_candidate_position']=original_position
            writer.add(export,ids,labels);admitted_rows.append((ids,labels))
        manifest=writer.finish(dict(original_candidate_manifest_sha256=file_hash(CANDIDATE/'manifest.json'),
            original_reference_completion_sha256=file_hash(REFERENCE/'complete.json'),
            independent_full_reference_output_readback_sha256=file_hash(REFERENCE_READER/'complete.json'),
            every_candidate_native_target_independently_reconstructed=True,
            actual_counts=dict(counts),all_original_named_held_stems=32953,
            no_private_generated_or_held_inputs_or_outputs_used_as_model_text=True,
            model_text_constructed_exclusively_from_screened_train_program_original_public_input_and_actual_public_stdout=True,
            target_origin='Actual retained original reference stdout, not publisher gold or generated teacher text',
            auxiliary_shared_language_head_task_no_parameter_or_teacher_addition=True,
            admitted_scope='Complete public-input execution-prediction CE only; no reward or human preference admission',
            independent_negative_exposure_search_and_semantic_independence_not_certified=True))
        writer=None
        check=CompleteCodePack(ADMITTED,'train','audit')
        if len(check)!=len(admitted_rows) or not len(check):raise ValueError('Nonempty fully rebuilt separate native pack required')
        for i,expected in enumerate(admitted_rows):
            if check.sequence(i)!=expected:raise ValueError('Every separately written native target/EOS mask must cold-read exactly')
        guard.check(full=True);complete_guard.check(full=True);FileBindings(source_result_bindings).check(full=True)
        atomic_json(OUT/'all-original-reference-result-bindings.json',source_result_bindings)
        evidence={**files,**complete_bindings,**source_result_bindings}
        evidence.update({str(p):file_hash(p) for p in [CANDIDATE/'manifest.json',OUT/'plan.json',
            OUT/'every-original-reference-and-native-target-readback.jsonl',OUT/'candidate-exposure-component-exclusions.json',
            OUT/'all-original-reference-result-bindings.json',ADMITTED/'manifest.json']})
        atomic_json(ADMITTED/'admission.json',dict(at=datetime.now(timezone.utc).isoformat(),
            dataset_manifest_sha256=file_hash(ADMITTED/'manifest.json'),admitted_to_training=True,
            admitted_scope='Public-input actual-reference-stdout prediction CE only',
            source_files_sha256=evidence,actual_native_counts=manifest['counts']['train'],
            every_original_reference_output_and_every_candidate_native_target_read_back=True,
            actual_all32953_named_held_stems_scanned_and_flagged_components_excluded=True,
            execution_reward_training_admitted=False,human_preference_training_admitted=False,
            no_original_candidate_or_held_source_modified=True,model_updates=0,goal_achieved=False))
        atomic_json(OUT/'complete.json',dict(at=datetime.now(timezone.utc).isoformat(),complete=True,
            every_original_reference_disposition_and_candidate_actual_native_target_rebuilt=True,
            actual_counts=dict(counts),admitted_native_train_counts=manifest['counts']['train'],
            every_separately_written_admitted_native_sequence_cold_read_exact=True,
            actual_original_held_stems=32953,blocked_source_components=len(blocked),
            original_candidate_remains_unadmitted_and_unchanged=True,
            dataset_admitted_to_auxiliary_CE_training=True,execution_reward_or_human_preference_admission=False,
            source_files_sha256={str(p):file_hash(p) for p in [OUT/'plan.json',ADMITTED/'manifest.json',ADMITTED/'admission.json']},
            models_constructed=0,model_updates=0,candidate_model_promoted=False,goal_achieved=False))
        print(json.dumps(dict(complete=str(OUT/'complete.json'),admitted_native_train_counts=manifest['counts']['train'],
            actual_candidate_counts=dict(counts),training_started=False)),flush=True)
    except BaseException as error:
        if writer is not None:writer.close()
        atomic_json(OUT/'failure.json',dict(error=repr(error),traceback=traceback.format_exc(),
            original_candidate_unmodified=True,model_updates=0,goal_achieved=False));raise


if __name__=='__main__':main()
