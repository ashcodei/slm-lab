"""Rebuild all own14source-train trajectory pairs and rerun every public case.

No new neural generation or optimization. The same bound OS-confined runtime
is called in fresh interpreters; source/native/action reconstruction is separate
from the neural producer. Budget-exhausted actions never gain an invented EOS.
"""
from datetime import datetime,timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import traceback

from tokenizers import Tokenizer
from slm_lab.codenet_checkpoint_v1 import FileBindings
from slm_lab.codenet_token_pack_v1 import CompleteCodePack
from slm_lab.code_quality import instruction_code
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.exact_waiting_process_retirement_v1 import terminal
from slm_lab.local_source_closure_v2 import python_sources
from slm_lab.own_function_trajectory_preferences_v1 import native_pair,validate,FORMAT
from slm_lab.public_function_rollout_observation_v1 import observe

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'reports/completed-code-public-function-rollouts-CPU-v1'
OUT=ROOT/'reports/own-function-trajectory-preferences-v1-readback-admission'
DATA=ROOT/'data/own-function-trajectory-preferences-v1'
ORIGINAL=ROOT/'data/direct-algorithm-function-contracts-admitted-v1'


def effect(row):
    return {k:row[k] for k in ['status','syntax_valid','all_original_examples_observed','all_original_examples_passed',
        'original_public_example_count','passed_examples','entry_body_calls']+(['raw_results','example_passes'] if 'raw_results' in row else [])}


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or OUT.exists() or DATA.exists():raise ValueError('Fresh GPU-hidden whole actual-trajectory admission required')
    source_plan=json.loads((SOURCE/'plan.json').read_text());source_done=json.loads((SOURCE/'complete.json').read_text())
    if (not terminal(source_plan['owner']) or not source_done.get('complete') or not source_done.get('all_parameters_unchanged')
            or source_done.get('actual_unique_student_values')!=984399690
            or source_done.get('neural_optimizer_updates')!=0
            or not source_done.get('all14actual_original_prompt_and_generated_token_actions_retained')
            or (SOURCE/'failure.json').exists()):
        raise ValueError('Only the exact complete original own-model source-train diagnostic is eligible')
    tests=ROOT/'reports/own-function-trajectory-preferences-v1-final-tests.log'
    if '3 passed' not in tests.read_text():raise ValueError('Actual censored-action, source-role, all-mask and independent-admission controls required')
    files=dict(source_done['source_files_sha256'])
    paths=[SOURCE/'complete.json',tests,*python_sources(ROOT,[Path(__file__),ROOT/'tests/test_own_function_trajectory_preferences_v1.py'])]
    files.update({str(p):file_hash(p) for p in paths});guard=FileBindings(files)
    pack=CompleteCodePack(ORIGINAL,'train','audit');tokenizer=Tokenizer.from_file(str(ORIGINAL/'tokenizer.json'))
    rows=list(map(json.loads,gzip.open(ORIGINAL/'original-rows.jsonl.gz','rt')))
    selection=[]
    for position,index in enumerate(pack.positions):
        row=rows[int(index)];ids,labels=pack.sequence(position);start=next(i for i,v in enumerate(labels) if v!=-100)
        if len(ids)-start<=128 and start<=1024:selection.append((position,int(index),row,ids[:start],ids[start:]))
    actions=list(map(json.loads,(SOURCE/'all14-actual-native-token-actions.jsonl').open()))
    generated=list(map(json.loads,(SOURCE/'all14-generated-programs-and-public-executions.jsonl').open()))
    actual_selection=json.loads((SOURCE/'all14-original-source-train-records.json').read_text())
    if len(selection)!=14 or len(actions)!=14 or len(generated)!=14 or len(actual_selection)!=14:
        raise ValueError('Every original predeclared short source-train function and actual own trajectory must remain')
    OUT.mkdir();DATA.mkdir();me=dict(pid=os.getpid(),ticks=Path('/proc/self/stat').read_text().split(') ',1)[1].split()[19])
    atomic_json(OUT/'plan.json',dict(at=datetime.now(timezone.utc).isoformat(),owner=me,source_files_sha256=files,
        actual_original_source_train_trajectories=14,whole_original_file_hash_split_before_optimization=True,
        all_original_public_example_answers_visible_in_model_prompt=True,new_neural_forwards=0,neural_updates=0,goal_achieved=False))
    pairs=[];proof=[]
    try:
        for i,((position,index,row,prefix,chosen),event,delivery,stored) in enumerate(zip(selection,actions,generated,actual_selection)):
            if (row['split']!='train' or row['source_candidate']['role']!='train'
                    or stored!=dict(position=position,global_native_index=index,row=row,prompt_ids=prefix,complete_gold_ids=chosen,original_prefix_length=len(prefix))
                    or prefix!=event['original_input_token_ids'] or event['case_position']!=i
                    or event['complete_input_ids_sha256']!=hashlib.sha256(__import__('numpy').asarray(prefix,dtype='<u2').tobytes()).hexdigest()
                    or event['requested_native_action_budget']!=256 or event['actual_generated_token_ids']!=delivery['actual_generated_token_ids']
                    or delivery['original_source_id']!=row['id'] or delivery['original_train_position']!=position):
                raise ValueError('Every original source role/prompt prefix, action identity and original function must reconstruct')
            sampled=event['actual_generated_token_ids'];decoded=tokenizer.decode(sampled);program=instruction_code(decoded)
            if (decoded!=delivery['raw_decoded_response'] or program!=delivery['actual_generated_program']
                    or tokenizer.decode(chosen[:-1],skip_special_tokens=False)!=row['response']
                    or event['actual_native_action_count']!=len(sampled)
                    or event['actual_stopped_on_EOS']!=(sampled[-1]==2)):
                raise ValueError('Every exact original positive and actual failed decoded program/token byte must rebuild')
            positives=[observe(row,row['response']) for _ in range(2)]
            negatives=[observe(row,program) for _ in range(2)]
            if (any(not p['all_original_examples_passed'] for p in positives)
                    or any(n['all_original_examples_passed'] for n in negatives)
                    or any(effect(n)!=effect(delivery['actual_public_example_observations']) for n in negatives)):
                raise ValueError('Each original reference must freshly pass and the exact actual model trajectory must freshly reproduce its failure; no favorable subset')
            pair=native_pair(row,prefix,chosen,sampled,stopped=event['actual_stopped_on_EOS'],budget=256);validate(pair);pairs.append(pair)
            proof.append(dict(original_source_id=row['id'],actual_fresh_original_reference_runs=positives,
                actual_fresh_generated_program_runs=negatives,negative_program_sha256=hashlib.sha256(program.encode()).hexdigest()))
            atomic_json(OUT/'progress.json',dict(actual_complete_original_trajectory_readbacks=len(pairs),planned=14,model_updates=0,goal_achieved=False))
        control=[r for r in pairs if r['optimization_role']=='unoptimized_source_train_control']
        optimizing=[r for r in pairs if r['optimization_role']=='optimization_source_train']
        if not control or not optimizing:raise ValueError('Whole source-file groups must provide both optimization and unoptimized controls')
        guard.check(full=True)
        atomic_json(OUT/'all14-fresh-original-and-generated-executions.json',proof)
        proof_path=OUT/'readback-complete.json'
        atomic_json(proof_path,dict(at=datetime.now(timezone.utc).isoformat(),complete=True,
            all14_original_sources_native_actions_and_fresh_executions_rebuilt=True,actual_original_source_train_trajectories=14,
            actual_optimization_source_train_functions=len(optimizing),actual_unoptimized_source_train_controls=len(control),
            actual_censored_negative_trajectories=sum(r['rejected_is_budget_censored'] for r in pairs),
            every_negative_retains_only_actual_sampled_actions_with_no_invented_EOS=True,
            all_original_references_pass_and_all_actual_negative_programs_fail_twice=True,
            same_bound_runtime_fresh_interpreters_not_independent_runtime_implementation=True,
            original_public_example_answers_visible_in_model_prompt=True,no_new_neural_forward_or_optimizer_update=True,
            source_files_sha256={**files,str(OUT/'all14-fresh-original-and-generated-executions.json'):file_hash(OUT/'all14-fresh-original-and-generated-executions.json')},goal_achieved=False))
        pair_path=DATA/'all-original-function-trajectory-pairs.jsonl';pair_path.write_text(''.join(json.dumps(r)+'\n' for r in pairs))
        manifest=dict(format=FORMAT,original_source_train_trajectories=14,optimization_source_train_functions=len(optimizing),
            unoptimized_source_train_controls=len(control),pairs_sha256=file_hash(pair_path),
            whole_source_file_group_rule='sha256(original_source_group) first8 hex modulo4 ==0 is unoptimized source-train control',
            no_source_valid_or_test_records=True,no_invented_rejected_EOS=True,
            chosen_is_complete_original_public_source_not_neural_teacher=True,rejected_is_this_own_model_actual_failing_trajectory=True,
            source_files_sha256={**files,str(proof_path):file_hash(proof_path)},human_preferences=False,held_benchmark=False,goal_achieved=False)
        atomic_json(DATA/'manifest.json',manifest)
        atomic_json(DATA/'admission.json',dict(actual_trajectory_machine_preference_training_admitted=True,human_preferences=False,
            source_valid_or_test_records=0,manifest_sha256=file_hash(DATA/'manifest.json'),independent_readback=str(proof_path),
            independent_readback_sha256=file_hash(proof_path),original_passing_and_censored_failing_trajectories_not_standard_complete_answer_DPO=True,
            source_train_controls_forbidden_from_optimization=True,actual_neural_optimizer_updates=0,goal_achieved=False))
        atomic_json(OUT/'complete.json',dict(at=datetime.now(timezone.utc).isoformat(),complete=True,
            actual_trajectory_machine_preference_training_admitted=True,actual_source_train_trajectory_pairs=14,
            actual_optimization_source_train_functions=len(optimizing),actual_unoptimized_source_train_controls=len(control),
            actual_neural_optimizer_updates=0,actual_fresh_positive_and_negative_program_runs=56,
            source_files_sha256={str(DATA/p):file_hash(DATA/p) for p in ['manifest.json','admission.json','all-original-function-trajectory-pairs.jsonl']},
            machine_feedback_not_human_RLHF_or_independent_benchmark=True,goal_achieved=False))
        print(json.dumps(dict(complete=str(OUT/'complete.json'),source_train_trajectory_pairs=14,
            optimization_functions=len(optimizing),unoptimized_train_controls=len(control),new_neural_updates=0)),flush=True)
    except BaseException as error:
        atomic_json(OUT/'failure.json',dict(error=repr(error),traceback=traceback.format_exc(),goal_achieved=False));raise


if __name__=='__main__':main()
