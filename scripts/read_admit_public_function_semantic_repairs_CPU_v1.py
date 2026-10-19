"""Rebuild the whole repair candidate and independently admit only exact views.

Every826original training source and every observed mutation is reconstructed.
Every observed mutant is freshly executed; accepted negatives and original
references run twice. Timing-dependent outcomes for excluded mutants are
retained as observations, never used to add a new row or select a favorable
benchmark population. Accepted native source/code/feedback/masks must rebuild
exactly. No neural model is constructed, trained or promoted.
"""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import gzip
import json
import os
from pathlib import Path
import shutil
import time
import traceback

from tokenizers import Tokenizer
from slm_lab.codenet_checkpoint_v1 import FileBindings
from slm_lab.codenet_token_pack_v1 import CompleteCodePack,FILES
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.documented_doctest_v1 import HARNESS
from slm_lab.exact_waiting_process_retirement_v1 import terminal
from slm_lab.local_source_closure_v2 import python_sources
from slm_lab.public_function_rollout_observation_v1 import observe
from slm_lab.public_function_semantic_repair_v1 import FORMAT,candidates,compact,counterexample,view,sha
from slm_lab.standalone_program_v1 import PYTHON_LAUNCHER

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'reports/public-function-semantic-repair-candidate-CPU-v1'
CANDIDATE=ROOT/'data/public-function-semantic-repair-candidate-v1'
ORIGINAL=ROOT/'data/direct-algorithm-function-contracts-admitted-v1'
OUT=ROOT/'reports/public-function-semantic-repair-v1-readback-admission'
ADMITTED=ROOT/'data/public-function-semantic-repair-admitted-v1'


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or OUT.exists() or ADMITTED.exists():raise ValueError('Fresh CPU-only whole source/native/runtime readback required')
    source_plan=json.loads((SOURCE/'plan.json').read_text())
    while not terminal(source_plan['owner']):
        if (SOURCE/'failure.json').exists():raise ValueError('Exact source collector failed; no partial corpus substitution')
        time.sleep(5)
    if (SOURCE/'failure.json').exists() or not (SOURCE/'complete.json').exists():raise ValueError('Exact complete original collector required')
    done=json.loads((SOURCE/'complete.json').read_text())
    if (not done.get('complete') or done['original_source_train_functions_inspected']!=826
            or source_plan['original_source_train_functions']!=826 or source_plan['maximum_mutations_per_function']!=16):
        raise ValueError('All original source functions and declared mutation coverage required')
    pack=CompleteCodePack(CANDIDATE,'train','audit');original_pack=CompleteCodePack(ORIGINAL,'train','training')
    original=list(map(json.loads,gzip.open(ORIGINAL/'original-rows.jsonl.gz','rt')))
    derived=list(map(json.loads,gzip.open(CANDIDATE/'original-rows.jsonl.gz','rt')))
    observations=list(map(json.loads,(SOURCE/'every-original-source-disposition-and-attempt.jsonl').read_text().splitlines()))
    if len(original_pack)!=826 or len(observations)!=826 or len(pack)!=len(derived) or any(r['split']!='train' for r in derived):
        raise ValueError('Complete826source inventory and only native train-role repair rows required')
    selected={r['original_native_train_position']:r for r in derived}
    if len(selected)!=len(derived) or [r['original_native_train_position'] for r in derived]!=sorted(selected):
        raise ValueError('Unique original-order complete repair rows required')
    tests=ROOT/'reports/public-function-semantic-repair-v1-final-tests.log'
    if '4 passed' not in tests.read_text():raise ValueError('Actual semantic mutation, native mask, held-role and public-runtime controls required')
    files={**source_plan['source_files_sha256'],**done['source_files_sha256']}
    for p in [SOURCE/'plan.json',SOURCE/'complete.json',tests,HARNESS,PYTHON_LAUNCHER,ROOT/'scripts/native_sandbox.py',
        *python_sources(ROOT,[Path(__file__),ROOT/'tests/test_public_function_semantic_repair_v1.py'])]:files[str(p)]=file_hash(p)
    guard=FileBindings(files);tokenizer=Tokenizer.from_file(str(CANDIDATE/'tokenizer.json'));OUT.mkdir();started=time.monotonic()
    atomic_json(OUT/'plan.json',dict(owner=dict(pid=os.getpid(),ticks=Path('/proc/self/stat').read_text().split(') ',1)[1].split()[19]),
        original_source_owner=source_plan['owner'],source_files_sha256=files,all826source_functions_required=True,
        all_observed_mutant_programs_reconstructed_and_freshly_executed=True,only_frozen_selected_candidate_rows_may_be_admitted=True,
        excluded_mutant_timing_outcomes_not_assumed_stable_or_coverage_optimal=True,threads=8,
        neural_forwards=0,neural_updates=0,goal_achieved=False))
    def work(item):
        position,index,source,stored=item
        if (stored['source_train_position']!=position or stored['original_global_index']!=index
                or stored['source_function_id']!=source['id'] or stored['source_group']!=source['group_id']
                or stored['original_source_role']!='train' or stored['original_complete_function_sha256']!=sha(source['response'])):
            raise ValueError('Every exact original source identity, role and positive program must reconstruct')
        references=[observe(source,source['response']) for _ in range(2)]
        candidate=selected.get(position);original_mutations=candidates(source,maximum=16)
        if len(stored['attempts'])>len(original_mutations):raise ValueError('No invented mutation attempt may be included')
        if candidate is not None and any(not r['all_original_examples_passed'] for r in references):
            raise ValueError('Every admitted original reference must freshly pass every public example twice')
        fresh_trials=[];rebuilt=None;fresh_runs=2
        for attempt,mutation in zip(stored['attempts'],original_mutations):
            if attempt['mutation']!={k:v for k,v in mutation.items() if k!='program'}:
                raise ValueError('Every observed single-body-mutation program must reconstruct from original source')
            fresh=[observe(source,mutation['program'])];fresh_runs+=1
            accepted=candidate is not None and mutation['mutation_position']==stored['selected_mutation_position']
            if accepted:
                fresh.append(observe(source,mutation['program']));fresh_runs+=1
                if [compact(r) for r in fresh]!=attempt['actual_mutant_runs']:
                    raise ValueError('Every selected actual public counterexample must reproduce twice in fresh interpreters')
                feedback=counterexample(source,fresh[0])
                if feedback is None:raise ValueError('Accepted bug must retain every original case and an actual failing function body')
                built,budget=view(source,mutation,feedback,tokenizer)
                if built is None or budget['disposition']!='complete_semantic_repair_candidate':raise ValueError('Complete positive code and native repair context must fit without clipping')
                row,ids,labels=built
                row.update(original_native_train_position=position,original_global_native_index=index,
                    original_source_pack_manifest_sha256=file_hash(ORIGINAL/'manifest.json'),
                    actual_reference_public_checks=stored['actual_reference_runs'],actual_mutant_public_checks=attempt['actual_mutant_runs'])
                expected={k:v for k,v in candidate.items() if k not in {'token_pack_index','complete_input_ids_sha256','target_tokens'}}
                if row!=expected:raise ValueError('Every original positive target, public feedback byte, role and rights field must rebuild')
                actual_ids,actual_labels=pack.sequence(candidate['token_pack_index'])
                if ids!=actual_ids or labels!=actual_labels:raise ValueError('Every native input/EOS byte, prompt mask and target denominator must rebuild')
                rebuilt=dict(repair_id=row['id'],original_source_id=source['id'],complete_native_context_tokens=len(ids),response_EOS_targets=sum(v!=-100 for v in labels),
                    selected_mutant_program_sha256=mutation['mutant_program_sha256'],all_original_public_examples_failed_as_recorded_twice=True)
            fresh_trials.append(dict(mutation_position=mutation['mutation_position'],mutant_program_sha256=mutation['mutant_program_sha256'],
                frozen_candidate_selected=accepted,fresh_public_runs=[compact(r) for r in fresh],
                original_first_observation_effect_matches=compact(fresh[0])==attempt['actual_mutant_runs'][0]))
        if (candidate is None)!=(rebuilt is None):raise ValueError('Every and only frozen complete candidate may be independently admitted')
        return dict(original_source_id=source['id'],source_train_position=position,original_global_index=index,
            actual_fresh_original_reference_runs=[compact(r) for r in references],all_observed_mutation_programs_rebuilt_and_reexecuted=fresh_trials,
            accepted_native_reconstruction=rebuilt,actual_fresh_program_runs=fresh_runs)
    try:
        items=[(p,int(index),original[int(index)],observations[p]) for p,index in enumerate(original_pack.positions)]
        count=accepted=targets=runs=changed_excluded=0
        with (OUT/'all-original-source-and-mutant-readbacks.jsonl').open('x') as journal,ThreadPoolExecutor(max_workers=8) as pool:
            for record in pool.map(work,items):
                journal.write(json.dumps(record)+'\n');journal.flush();count+=1;runs+=record['actual_fresh_program_runs']
                row=record['accepted_native_reconstruction']
                if row is not None:accepted+=1;targets+=row['response_EOS_targets']
                changed_excluded+=sum(not r['frozen_candidate_selected'] and not r['original_first_observation_effect_matches'] for r in record['all_observed_mutation_programs_rebuilt_and_reexecuted'])
                if count%25==0 or count==826:atomic_json(OUT/'progress.json',dict(original_train_sources_readback=count,planned=826,
                    complete_selected_repair_readbacks=accepted,actual_fresh_program_runs=runs,excluded_fresh_observations_differing=changed_excluded,
                    seconds=time.monotonic()-started,training_admitted=False,neural_updates=0,goal_achieved=False))
        if count!=826 or accepted!=len(pack) or accepted!=done['complete_repair_candidates'] or targets!=pack.manifest['counts']['train']['target_tokens']:
            raise ValueError('Every original source and every selected complete native target must reconcile')
        guard.check(full=True);ADMITTED.mkdir()
        for name in [*FILES,'manifest.json']:
            shutil.copyfile(CANDIDATE/name,ADMITTED/name)
            if file_hash(CANDIDATE/name)!=file_hash(ADMITTED/name):raise ValueError('Every candidate/native/rights byte must remain exact in the admitted copy')
        proof=OUT/'readback-complete.json'
        atomic_json(proof,dict(at=datetime.now(timezone.utc).isoformat(),complete=True,format=FORMAT,
            original_source_train_functions_readback=826,complete_semantic_repair_views=accepted,response_EOS_targets=targets,
            all_frozen_selected_native_bytes_masks_original_targets_rights_and_fresh_counterexamples_exact=True,
            all_observed_mutant_programs_reconstructed_and_reexecuted=True,actual_fresh_program_runs=runs,
            excluded_mutant_fresh_outcome_differences=changed_excluded,excluded_outcomes_not_used_to_add_rows_or_claim_optimal_coverage=True,
            same_bound_runtime_fresh_interpreters_not_independent_execution_implementation=True,
            original15validation_functions_never_enter_new_dataset=True,source_train_auxiliary_view_not_held_benchmark_or_human_feedback=True,
            source_files_sha256={**files,str(OUT/'all-original-source-and-mutant-readbacks.jsonl'):file_hash(OUT/'all-original-source-and-mutant-readbacks.jsonl')},
            neural_forwards=0,neural_updates=0,goal_achieved=False))
        atomic_json(ADMITTED/'admission.json',dict(admitted_to_training=True,dataset_manifest_sha256=file_hash(ADMITTED/'manifest.json'),
            independent_readback=str(proof),independent_readback_sha256=file_hash(proof),
            semantic_repair_SFT_admitted=True,human_preference_training_admitted=False,execution_RL_admitted=False,
            source_valid_or_test_records=0,new_neural_updates=0,goal_achieved=False))
        verified=CompleteCodePack(ADMITTED,'train','training')
        if len(verified)!=accepted:raise ValueError('Actual independently admitted native data gate must accept the complete selected corpus')
        atomic_json(OUT/'complete.json',dict(at=datetime.now(timezone.utc).isoformat(),complete=True,semantic_repair_SFT_admitted=True,
            original_source_train_functions_readback=826,complete_semantic_repair_views=accepted,response_EOS_targets=targets,
            actual_fresh_program_runs=runs,excluded_mutant_fresh_observation_differences=changed_excluded,
            new_neural_forwards=0,new_neural_updates=0,useful_model_gain_not_measured=True,
            source_files_sha256={str(p):file_hash(p) for p in [proof,ADMITTED/'admission.json',ADMITTED/'manifest.json']},goal_achieved=False))
        print(json.dumps(dict(complete=str(OUT/'complete.json'),admitted_repair_views=accepted,actual_fresh_program_runs=runs,neural_updates=0)),flush=True)
    except BaseException as error:
        atomic_json(OUT/'failure.json',dict(error=repr(error),traceback=traceback.format_exc(),goal_achieved=False));raise

if __name__=='__main__':main()
