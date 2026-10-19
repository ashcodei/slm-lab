"""Independently reconstruct all source/actions and rerun all public examples.

No model is constructed or trained. This calls the same bound OS-confined
runtime in fresh processes; it is not a second execution implementation or an
independent held benchmark. Both optimized and unoptimized train-pool groups
are reported without selecting a favorable function or arm.
"""
from datetime import datetime,timezone
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import time
import traceback

from tokenizers import Tokenizer
from slm_lab.codenet_checkpoint_v1 import FileBindings
from slm_lab.code_quality import instruction_code
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.exact_waiting_process_retirement_v1 import terminal
from slm_lab.local_source_closure_v2 import python_sources
from slm_lab.own_function_result_readback_v1 import effect,reconstruct,summarize,validate_training
from slm_lab.own_function_trajectory_preferences_v1 import OwnFunctionTrajectoryPreferences
from slm_lab.public_function_rollout_observation_v1 import observe

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'reports/own-function-trajectory-preferences-full-CPU-v4'
OUT=ROOT/'reports/own-function-training-results-CPU-v1-readback'
DATA=ROOT/'data/own-function-trajectory-preferences-v1'
ORIGINAL=ROOT/'reports/completed-code-public-function-rollouts-CPU-v1/all14-original-source-train-records.json'

def records(path):return list(map(json.loads,Path(path).read_text().splitlines()))

def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or OUT.exists():raise ValueError('Fresh CPU-only readback required')
    tests=ROOT/'reports/own-function-result-readback-v1-tests.log'
    if '4 passed' not in tests.read_text():raise ValueError('Actual source/action/stopping/control corruption tests required')
    runtime=json.loads((SOURCE/'plan.json').read_text());math_path=Path(runtime['original_math_plan'])
    if file_hash(math_path)!=runtime['original_math_plan_sha256']:raise ValueError('Original immutable experiment mathematics required')
    plan=json.loads(math_path.read_text());data=OwnFunctionTrajectoryPreferences(DATA,'audit')
    tokenizer_path=ROOT/'checkpoints/code-first-continuation-v3/standalone/tokenizer.json'
    files=dict(runtime['source_files_sha256']);files.update(data.source_files_sha256)
    for p in [tests,SOURCE/'plan.json',math_path,ORIGINAL,tokenizer_path,*python_sources(ROOT,[Path(__file__),ROOT/'tests/test_own_function_result_readback_v1.py'])]:files[str(p)]=file_hash(p)
    guard=FileBindings(files);tokenizer=Tokenizer.from_file(str(tokenizer_path));sources=json.loads(ORIGINAL.read_text())
    if len(sources)!=14 or [r['row']['id'] for r in sources]!=[r['id'] for r in data.rows]:raise ValueError('Every exact original source function required')
    order=[];options=plan['training_options']
    for epoch in range(options['epochs']):
        selected=list(data.optimization_positions);random.Random(options['seed']+epoch).shuffle(selected)
        order.extend([selected[i:i+options['global_batch_size']] for i in range(0,len(selected),options['global_batch_size'])])
    if order!=plan['actual_complete_source_epoch_batches'] or len(order)!=25:raise ValueError('All original optimization functions in all five epochs required')
    OUT.mkdir();atomic_json(OUT/'plan.json',dict(owner=dict(pid=os.getpid(),ticks=Path('/proc/self/stat').read_text().split(') ',1)[1].split()[19]),
        exact_live_producer=runtime['owner'],source_files_sha256=files,all14source_train_functions_required=True,
        new_neural_forwards=0,new_neural_updates=0,goal_achieved=False))
    outcomes={};proofs=[];new_runs=0
    try:
        for pair,source in zip(data.rows,sources):
            if (source['prompt_ids']!=pair['original_native_prompt_ids'] or source['complete_gold_ids']!=pair['chosen_original_complete_response_EOS_ids']
                    or tokenizer.decode(pair['chosen_original_complete_response_EOS_ids'][:-1],skip_special_tokens=False)!=source['row']['response']):
                raise ValueError('Every original complete positive source and true EOS must reconstruct')
            gold=[observe(source['row'],source['row']['response']) for _ in range(2)];new_runs+=2
            if any(not r['all_original_examples_passed'] for r in gold):raise ValueError('Every original reference must still pass every public example twice')
            proofs.append(dict(source_id=pair['id'],fresh_original_reference_runs=gold))
        for endpoint in ['initial','chosen_CE','trajectory_DPO']:
            done=SOURCE/('initial-complete.json' if endpoint=='initial' else endpoint+'/complete.json')
            while not done.exists():
                if (SOURCE/'failure.json').exists() or terminal(runtime['owner']):raise ValueError('Exact producer ended before every original matched arm completed')
                atomic_json(OUT/'progress.json',dict(waiting_for_exact_endpoint=endpoint,completed_endpoints=list(outcomes),new_program_runs=new_runs,goal_achieved=False));time.sleep(10)
            if endpoint=='initial':
                events=SOURCE/'initial-actual-native-actions.jsonl';delivery=SOURCE/'initial-actual-programs-and-public-executions.jsonl'
            else:
                folder=SOURCE/endpoint;events=folder/'all14-actual-native-actions.jsonl';delivery=folder/'all14-actual-generated-programs-and-public-executions.jsonl'
                metrics=folder/'actual-training-metrics.jsonl';validate_training(records(metrics),order,data.rows,endpoint,options)
                saved=json.loads(done.read_text())
                if (saved.get('complete') is not True or saved.get('actual_logical_updates')!=25
                        or saved.get('all10optimization_functions_in_each_of5epochs') is not True
                        or saved.get('every_exported_value_and_actual_complete_function_logits_cold_load_exact') is not True):
                    raise ValueError('Both complete full-size trained and cold-loaded arms required')
                for p in [metrics,folder/'actual-next-step-recovery.json',folder/'actor.pt',folder/'complete-private-state.pt']:
                    files[str(p)]=file_hash(p)
            for p in [done,events,delivery]:files[str(p)]=file_hash(p)
            actual=reconstruct(data.rows,records(events),records(delivery),tokenizer.decode,instruction_code)
            observations=[]
            for pair,source,row in zip(data.rows,sources,actual):
                fresh=[observe(source['row'],row['program']) for _ in range(2)];new_runs+=2
                if any(effect(r)!=row['producer_effect'] for r in fresh):raise ValueError('Every original public example outcome must reproduce twice')
                observations.append(dict(source_id=pair['id'],optimization_role=pair['optimization_role'],fresh_generated_program_runs=fresh,
                    program_sha256=hashlib.sha256(row['program'].encode()).hexdigest(),actual_generated_tokens=row['actual_generated_tokens'],
                    actual_stopped_on_EOS=row['actual_stopped_on_EOS']))
            outcomes[endpoint]=summarize(actual)
            atomic_json(OUT/(endpoint+'-fresh-public-executions.json'),observations)
            atomic_json(OUT/(endpoint+'-complete.json'),dict(complete=True,actual_all14_source_and_action_reconstructions=True,
                actual_fresh_generated_program_runs=28,actual_source_train_results=outcomes[endpoint],held_benchmark=False,goal_achieved=False))
        guard.check(full=True)
        with (OUT/'matched-results.csv').open('x',newline='') as stream:
            writer=csv.writer(stream);writer.writerow(['endpoint','role','functions','syntax_valid','all_public_examples_passed'])
            for endpoint,result in outcomes.items():
                for role in ['optimization_source_train','unoptimized_source_train_control']:
                    row=result[role];writer.writerow([endpoint,role,row['functions'],row['syntax_valid'],row['all_original_public_examples_passed']])
        atomic_json(OUT/'complete.json',dict(at=datetime.now(timezone.utc).isoformat(),complete=True,actual_matched_source_train_results=outcomes,
            actual_fresh_original_and_generated_program_runs=new_runs,all42actual_native_deliveries_and_stopping_boundaries_reconstructed=True,
            all_complete_original_training_positions_and_masks_reconstructed=True,new_neural_forwards=0,new_neural_updates=0,
            same_bound_runtime_fresh_processes_not_independent_execution_implementation=True,
            source_train_controls_not_independent_held_benchmark=True,public_expected_answers_visible_in_original_model_prompts=True,
            source_files_sha256={**files,**{str(p):file_hash(p) for p in OUT.glob('*') if p.is_file() and p.name!='progress.json'}},
            candidate_promoted=False,goal_achieved=False))
        print(json.dumps(dict(complete=str(OUT/'complete.json'),actual_matched_source_train_results=outcomes)),flush=True)
    except BaseException as error:
        atomic_json(OUT/'failure.json',dict(error=repr(error),traceback=traceback.format_exc(),goal_achieved=False));raise

if __name__=='__main__':main()
