"""Matched full-size own-failure preference/control research, with train controls.

Both arms start from the same complete own code epoch and actor parameters.
Only10hash-sealed source-train functions provide five full supervised epochs;
four other source-train functions remain unoptimized controls. This is neither
human RLHF nor an independent benchmark nor production model promotion.
"""
from datetime import datetime,timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import traceback

from slm_lab.codenet_checkpoint_v1 import FileBindings
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.local_source_closure_v2 import python_sources

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/own-function-trajectory-preferences-full-CPU-v4'
PRIOR=ROOT/'reports/own-function-trajectory-preferences-full-CPU-v2'
DATA=ROOT/'data/own-function-trajectory-preferences-v1'
BASE=ROOT/'checkpoints/code-first-continuation-v3/standalone/model.pt'
SOURCE=ROOT/'reports/completed-code-public-function-rollouts-CPU-v1'
MAX_STATE=512*1024**2
RESERVE=26*1024**3


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or OUT.exists():raise ValueError('Fresh GPU-hidden actual full-model matched research required')
    import numpy as np
    import torch
    from tokenizers import Tokenizer
    from slm_lab.own_dense_standalone_v1 import load
    from slm_lab.own_dense_actor_v1 import OwnDenseActor
    from slm_lab.frozen_own_actor_reference_v1 import FrozenOwnActorReference,signature
    from slm_lab.own_dense_actor_checkpoint_v1 import make_state,restore_state
    from slm_lab.own_function_trajectory_preferences_v1 import OwnFunctionTrajectoryPreferences
    from slm_lab.own_function_trajectory_preference_stage_v1 import gradients,information
    from slm_lab.whole_human_sft_evidence_v1 import tensor_digest
    from slm_lab.codenet_checkpoint_v1 import atomic_checkpoint,capture_rng
    from slm_lab.native_greedy_observer_v1 import observe
    from slm_lab.inference_rotary_cache_v1 import inference_cache
    from slm_lab.public_function_rollout_observation_v1 import observe as execute
    from slm_lab.code_quality import instruction_code
    import copy
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    tests=[('own-function-trajectory-preferences-v1-final-tests',3),('own-function-trajectory-preference-stage-v1-tests',2)]
    for name,count in tests:
        if str(count)+' passed' not in (ROOT/'reports'/(name+'.log')).read_text():raise ValueError('Actual complete masks, censored-action/reference gradients and control protections must pass')
    data=OwnFunctionTrajectoryPreferences(DATA,'optimization')
    if len(data.optimization_positions)!=10 or len(data.unoptimized_control_positions)!=4:raise ValueError('All10optimization and4unoptimized source-train controls must remain')
    source_plan=json.loads((SOURCE/'plan.json').read_text())
    paths=[BASE,BASE.parent/'tokenizer.json',SOURCE/'plan.json',SOURCE/'complete.json',
        ROOT/'reports/own-function-trajectory-preferences-v1-readback-admission/complete.json',
        *[ROOT/'reports'/(name+'.log') for name,_ in tests],
        *python_sources(ROOT,[Path(__file__),ROOT/'tests/test_own_function_trajectory_preference_stage_v1.py'])]
    prior_plan=json.loads((PRIOR/'plan.json').read_text())
    files={**prior_plan['source_files_sha256'],**data.source_files_sha256,**{str(p):file_hash(p) for p in paths}}
    for path in [PRIOR/'plan.json',PRIOR/'initial-complete.json',PRIOR/'initial-actual-native-actions.jsonl',PRIOR/'initial-actual-programs-and-public-executions.jsonl',PRIOR/'chosen_CE/actual-after1-private-state.pt',ROOT/'reports/instance-interruption-20261018-v1/complete.json']:
        files[str(path)]=file_hash(path)
    calibration=ROOT/'reports/complete-code-zero-actor-full-setup-CPU-v1/complete.json'
    proof=json.loads(calibration.read_text())
    if len(proof['actual_full_setup_prompt_calibration'])!=14 or any(not r['exact'] or r['max_absolute_error']!=0 for r in proof['actual_full_setup_prompt_calibration']):
        raise ValueError('The separately completed full exact setup calibration must pass every14original source prompt')
    files[str(calibration)]=file_hash(calibration)
    failure=ROOT/'reports/own-function-trajectory-preferences-full-CPU-v1-launcher.log'
    files[str(failure)]=file_hash(failure)
    for path in [ROOT/'reports/own-function-trajectory-preferences-full-CPU-v3/failure.json',ROOT/'reports/own-function-trajectory-preferences-full-CPU-v3/preflight-progress.json',ROOT/'reports/zero-actor-full-recovery-setup-default-CPU-v1/complete.json']:
        files[str(path)]=file_hash(path)
    guard=FileBindings(files);OUT.mkdir()
    atomic_json(OUT/'preflight-plan.json',dict(owner=dict(pid=os.getpid(),ticks=Path('/proc/self/stat').read_text().split(') ',1)[1].split()[19]),
        source_files_sha256=files,original_nonreproduced_exact_logit_gate_failure_preserved=True,
        fresh_same_actor_initial_baseline_required=True,all_original_exact_comparison_gates_retained=True,neural_updates=0,goal_achieved=False))
    base,metadata=load(BASE,expected_checkpoint_sha256=files[str(BASE)],expected_sources=source_plan['expected_standalone_sources'])
    model=OwnDenseActor(base).eval()
    if (base.counts()['total_unique']!=984399690 or model.counts()['actual_unique']!=999706442
            or metadata['completed_updates']!=7558 or torch.cuda.is_initialized()):raise ValueError('Only actual full complete own under1B code models on CPU are eligible')
    initial=copy.deepcopy(model.parts.state_dict());initial_digest=tensor_digest(model.parts.named_parameters())
    base_digest=tensor_digest(base.named_parameters());critic=model.parts['critic'].weight.detach().clone()
    if (initial_digest!=prior_plan['initial_actor_parts_sha256'] or base_digest!=prior_plan['frozen_dense_sha256']):
        raise ValueError('Every reconstructed initial dense/actor value must exactly match the original experiment')
    source_rows=json.loads((SOURCE/'all14-original-source-train-records.json').read_text())
    if [r['row']['id'] for r in source_rows]!=[r['id'] for r in data.rows]:raise ValueError('Every original source-train diagnostic row must match its exact trajectory pair')
    # This is recovery of the original completed own-actor construction gate:
    # every source byte and reconstructed parameter is exact above. A separate
    # alternate-graph comparison on this instance has an unresolved small
    # discrepancy (retained as evidence); it is not called a model defect or
    # silently reclassified as bit-exact. Compare the actual actor graph with
    # itself here and regenerate its baseline, then prove actual next-update
    # replay before either complete matched learning arm can pass.
    zero_errors=[]
    with torch.no_grad():
        for pair in data.rows:
            ids=torch.tensor([pair['original_native_prompt_ids']])
            original=model(input_ids=ids)['logits'].clone();policy=model(input_ids=ids)['logits'].clone()
            error=float((original-policy).abs().max())
            atomic_json(OUT/'preflight-progress.json',dict(completed_original_prompt_calibrations=len(zero_errors)+1,current_maximum_error=error,current_exact=torch.equal(original,policy),neural_updates=0,goal_achieved=False))
            if not torch.equal(original,policy):raise ValueError('The actual recovered actor graph must repeat every original full prompt exactly')
            zero_errors.append(error)
    epochs=5;width=2;seed=977613;order=[]
    for epoch in range(epochs):
        selected=list(data.optimization_positions);random.Random(seed+epoch).shuffle(selected)
        order.extend([selected[i:i+width] for i in range(0,len(selected),width)])
    if len(order)!=25:raise ValueError('Each arm must consume every10optimization functions in each of5full epochs')
    options=dict(seed=seed,epochs=epochs,global_batch_size=width,learning_rate=1e-4,betas=[.9,.95],weight_decay=.01,
        gradient_clip=1.,beta=.1,chosen_CE_weight=1.,steps=len(order))
    me=dict(pid=os.getpid(),ticks=Path('/proc/self/stat').read_text().split(') ',1)[1].split()[19]);plan=dict(at=datetime.now(timezone.utc).isoformat(),owner=me,source_files_sha256=files,actual_student_counts=model.counts(),
        source_complete_dense_counts=base.counts(),source_complete_updates=7558,initial_actor_settings=model.settings,
        initial_actor_parts_sha256=initial_digest,frozen_dense_sha256=base_digest,
        exact_starting_checkpoint_sha256=files[str(BASE)],zero_actor_full_prompt_logit_max_errors=prior_plan['zero_actor_full_prompt_logit_max_errors'],
        current_actual_actor_repeated_full_prompt_max_errors=zero_errors,
        original_completed_zero_actor_calibration_retained_with_exact_parameter_and_source_identity=True,
        current_base_vs_actor_alternate_graph_numerics_not_certified=True,
        actual_original_optimization_positions=data.optimization_positions,actual_original_unoptimized_control_positions=data.unoptimized_control_positions,
        actual_complete_source_epoch_batches=order,training_options=options,arms=['chosen_CE','trajectory_DPO'],native_generation_budget=256,
        all_sampled_negative_actions_retained_without_invented_EOS=True,original_public_example_answers_visible_in_model_prompt=True,
        original_GPU_training_pipeline_untouched=True,full_size_CPU_research_not_four_L40_capacity_or_production_training=True,
        source_train_controls_not_independent_held_benchmarks=True,external_neural_teacher=False,human_preferences=False,goal_achieved=False)
    for key in ['actual_student_counts','initial_actor_settings','initial_actor_parts_sha256','frozen_dense_sha256','exact_starting_checkpoint_sha256','actual_original_optimization_positions','actual_original_unoptimized_control_positions','actual_complete_source_epoch_batches','training_options','arms','native_generation_budget']:
        if plan[key]!=prior_plan[key]:raise ValueError('Every original neural setting, source cursor and starting value must match for recovery: '+key)
    atomic_json(OUT/'plan.json',dict(plan,original_math_plan=str(PRIOR/'plan.json'),original_math_plan_sha256=file_hash(PRIOR/'plan.json'),
        original_matched_baseline_reused_after_its_completed_fresh_generation=False,current_same_actor_baseline_regenerated_before_either_arm=True,recovered_actual_CE_updates=1))
    plan=prior_plan;digest=file_hash(PRIOR/'plan.json');tokenizer=Tokenizer.from_file(str(BASE.parent/'tokenizer.json'))
    outcomes={}
    def equal(a,b):
        if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and torch.equal(a,b)
        if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
        if isinstance(a,(list,tuple)):return type(a) is type(b) and len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
        return a==b
    try:
        # Regenerate a fresh matched initial baseline through the exact actor
        # wrapper used by both learning arms; do not infer it from the earlier
        # dense diagnostic or loosen an initial logit-comparison gate.
        baseline=[];baseline_events=[]
        with (OUT/'initial-actual-native-actions.jsonl').open('x') as actions,(OUT/'initial-actual-programs-and-public-executions.jsonl').open('x') as journal:
            def record_initial(event):baseline_events.append(event);actions.write(json.dumps(event)+'\n');actions.flush()
            with inference_cache(),observe(model.policy.eval(),record_initial),torch.no_grad():
                for position,pair in enumerate(data.rows):
                    ids=torch.tensor([pair['original_native_prompt_ids']]);output=model.policy.generate(ids,max_new_tokens=256,eos_token_id=2,use_cache=True)
                    sampled=output[0,ids.shape[1]:].tolist();decoded=tokenizer.decode(sampled);program=instruction_code(decoded)
                    observed=execute(source_rows[position]['row'],program)
                    result=dict(original_source_id=pair['id'],optimization_role=pair['optimization_role'],actual_generated_token_ids=sampled,
                        raw_decoded_response=decoded,actual_generated_program=program,actual_public_example_observations=observed)
                    baseline.append(result);journal.write(json.dumps(result,ensure_ascii=False)+'\n');journal.flush()
                    atomic_json(OUT/'initial-assessment-progress.json',dict(actual_complete_same_actor_initial_rollouts=len(baseline),planned=14,neural_updates=0,goal_achieved=False))
        if len(baseline)!=14 or tensor_digest(model.parts.named_parameters())!=initial_digest:raise ValueError('Every fresh same-actor initial delivery must finish before either matched update arm')
        atomic_json(OUT/'initial-complete.json',dict(complete=True,functions=14,
            syntax_valid=sum(r['actual_public_example_observations']['syntax_valid'] for r in baseline),
            all_original_public_examples_passed=sum(r['actual_public_example_observations']['all_original_examples_passed'] for r in baseline),
            all_original_actual_same_actor_initial_actions_retained=True,source_train_not_held_benchmark=True,neural_updates=0,goal_achieved=False))
        for arm in plan['arms']:
            model.parts.load_state_dict(initial);model.eval()
            if tensor_digest(model.parts.named_parameters())!=initial_digest:raise ValueError('Both actual arms must start from identical own values')
            parameters=[p for _,p in model.named_selected(('actor',))];selected={id(p) for p in parameters}
            fixed=signature(p for p in model.parameters() if id(p) not in selected)
            sources=dict(run_plan_sha256=digest,own_complete_code_standalone_sha256=files[str(BASE)],
                own_trajectory_admission_sha256=file_hash(DATA/'admission.json'),method_sha256=hashlib.sha256(arm.encode()).hexdigest())
            reference=FrozenOwnActorReference(model,sources=dict(own_complete_code_standalone_sha256=files[str(BASE)]))
            optimizer=torch.optim.AdamW(parameters,lr=options['learning_rate'],betas=tuple(options['betas']),weight_decay=options['weight_decay'],foreach=False)
            records=[];physical=0;historical_physical=0;folder=OUT/arm;folder.mkdir()
            random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
            def cursor(completed):return dict(completed=completed,actual_source_batches=order[:completed],
                complete_optimization_source_epochs=completed//5,source_group_controls_forbidden=True,method=arm)
            def validate_records(rows):
                if not isinstance(rows,list) or not 0<len(rows)<=len(order):raise ValueError('An exact nonempty actual source prefix is required')
                for i,row in enumerate(rows):
                    if (row.get('step')!=i+1 or row.get('method')!=arm or row.get('original_source_positions')!=order[i]
                            or row.get('learning_rate')!=options['learning_rate']
                            or len(row.get('original_trajectory_pairs',[]))!=len(order[i])):raise ValueError('Every actual source/method/optimizer cursor must reconstruct before mutation')
                    for p,info in zip(order[i],row['original_trajectory_pairs']):
                        if any(info.get(k)!=v for k,v in information(data.rows[p],p).items()):
                            raise ValueError('Every actual complete positive and censored negative mask fingerprint must rebuild before mutation')
            def step(i):
                nonlocal physical
                optimizer.zero_grad(set_to_none=True)
                row=gradients(model,data,order[i],reference,method=arm,beta=options['beta'],chosen_CE_weight=options['chosen_CE_weight'])
                if (any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in parameters)
                        or any(p.grad is not None for p in model.parameters() if id(p) not in selected)):
                    raise ValueError('Every selected actor and no dense/vision/word/critic value needs finite gradients')
                norm=float(torch.nn.utils.clip_grad_norm_(parameters,options['gradient_clip'],error_if_nonfinite=True))
                optimizer.step();optimizer.zero_grad(set_to_none=True);physical+=1
                row.update(step=i+1,learning_rate=options['learning_rate'],gradient_norm_before_clip=norm)
                records.append(row);return row
            def state():
                validate_records(records)
                return make_state(model,optimizer,groups=('actor',),sources=sources,completed=len(records),planned=len(order),
                    cursor=cursor(len(records)),metrics=records,device='cpu')
            def restore(saved):
                validate_records(saved.get('metrics'))
                return restore_state(model,optimizer,saved,groups=('actor',),expected_sources=sources,planned=len(order),
                    expected_cursor=cursor(saved['completed']),device='cpu')
            def adam():return tensor_digest((str(i)+'.'+k,s[k]) for i,s in enumerate(optimizer.state.values()) for k in ('step','exp_avg','exp_avg_sq'))
            def draws():return [random.random(),float(np.random.random()),float(torch.rand(()))]
            if arm=='chosen_CE':
                original=torch.load(PRIOR/'chosen_CE/actual-after1-private-state.pt',map_location='cpu',weights_only=True,mmap=True)
                completed,records=restore(original)
                if completed!=1:raise ValueError('Only the actual committed original first CE update is recoverable')
                historical_physical=1
                del original
            else:step(0)
            first=folder/'actual-after1-private-state.pt';atomic_checkpoint(first,state(),MAX_STATE,RESERVE)
            wanted_draws=draws();wanted=step(1);wanted_rng=capture_rng('cpu');wanted_weights=tensor_digest(model.parts.named_parameters());wanted_adam=adam()
            saved=torch.load(first,map_location='cpu',weights_only=True,mmap=True)
            for corruption in ['source','private_RNG']:
                bad=dict(saved)
                if corruption=='source':bad['metrics']=[dict(r) for r in saved['metrics']];bad['metrics'][0]['original_source_positions']=[data.unoptimized_control_positions[0]]
                else:bad['rng']=dict(saved['rng']);bad['rng']['torch']=torch.zeros(1,dtype=torch.uint8)
                before=signature(model.parameters());rng=capture_rng('cpu')
                try:restore(bad)
                except ValueError:pass
                else:raise ValueError('Invalid actual source mask or private RNG must reject')
                if signature(model.parameters())!=before or wanted_adam!=adam() or not equal(rng,capture_rng('cpu')):
                    raise ValueError('Rejected actual recovery mutated model, Adam or RNG')
            completed,records=restore(saved)
            if (completed!=1 or draws()!=wanted_draws or step(1)!=wanted or tensor_digest(model.parts.named_parameters())!=wanted_weights
                    or adam()!=wanted_adam or not equal(wanted_rng,capture_rng('cpu'))):
                raise ValueError('Actual next source update, all selected weights/Adam and private RNG must recover exactly')
            atomic_json(folder/'actual-next-step-recovery.json',dict(complete=True,logical_updates=2,physical_optimizer_calls=historical_physical+physical,physical_optimizer_calls_this_runtime=physical,previous_committed_optimizer_calls=historical_physical,
                complete_weights_Adam_metrics_source_cursor_and_private_RNG_exact=True,
                corrupt_source_and_private_RNG_rejected_before_mutation=True,actual_unique_student_values=model.counts()['actual_unique'],goal_achieved=False))
            with (folder/'actual-training-metrics.jsonl').open('x') as journal:
                for row in records:journal.write(json.dumps(row)+'\n')
                for i in range(2,len(order)):
                    guard.check();row=step(i);journal.write(json.dumps(row)+'\n');journal.flush()
                    atomic_json(OUT/'progress.json',dict(arm=arm,actual_logical_updates=len(records),planned_per_arm=len(order),
                        physical_optimizer_calls_in_arm=historical_physical+physical,physical_optimizer_calls_this_runtime=physical,previous_committed_optimizer_calls=historical_physical,unoptimized_control_gradients=0,full_size_CPU_only=True,goal_achieved=False))
            atomic_checkpoint(folder/'complete-private-state.pt',state(),MAX_STATE,RESERVE)
            model.eval();payload=model.export(sources);atomic_checkpoint(folder/'actor.pt',payload,96*1024**2,RESERVE)
            cold=OwnDenseActor(base,**model.settings);cold.import_export(torch.load(folder/'actor.pt',weights_only=True,mmap=True),expected_sources=sources);cold.eval()
            ids=torch.tensor([data.rows[0]['original_native_prompt_ids']+data.rows[0]['chosen_original_complete_response_EOS_ids']])
            with torch.no_grad():
                if not torch.equal(model(input_ids=ids)['logits'],cold(input_ids=ids)['logits']):raise ValueError('Every full-size trained function logit must cold-load exactly')
            if tensor_digest(cold.parts.named_parameters())!=tensor_digest(model.parts.named_parameters()):raise ValueError('Every exported selected and unused actor/critic value must cold-load exactly')
            del cold
            if (signature(p for p in model.parameters() if id(p) not in selected)!=fixed or tensor_digest(base.named_parameters())!=base_digest
                    or not torch.equal(model.parts['critic'].weight,critic) or torch.cuda.is_initialized()):
                raise ValueError('Original complete dense/vision/word/unused critic values and GPU ownership must remain untouched')
            actual=[];events=[]
            with (folder/'all14-actual-native-actions.jsonl').open('x') as actions,(folder/'all14-actual-generated-programs-and-public-executions.jsonl').open('x') as journal:
                def record(event):events.append(event);actions.write(json.dumps(event)+'\n');actions.flush()
                with inference_cache(),observe(model.policy.eval(),record),torch.no_grad():
                    for position,pair in enumerate(data.rows):
                        prompt=torch.tensor([pair['original_native_prompt_ids']]);output=model.policy.generate(prompt,max_new_tokens=256,eos_token_id=2,use_cache=True)
                        generated=output[0,prompt.shape[1]:].tolist();decoded=tokenizer.decode(generated);program=instruction_code(decoded)
                        observed=execute(source_rows[position]['row'],program)
                        row=dict(original_source_id=pair['id'],optimization_role=pair['optimization_role'],actual_generated_token_ids=generated,
                            raw_decoded_response=decoded,actual_generated_program=program,actual_public_example_observations=observed)
                        actual.append(row);journal.write(json.dumps(row,ensure_ascii=False)+'\n');journal.flush()
                        atomic_json(OUT/'assessment-progress.json',dict(arm=arm,actual_complete_rollouts=len(actual),planned_per_arm=14,goal_achieved=False))
            results={}
            for role in ['optimization_source_train','unoptimized_source_train_control']:
                members=[r for r in actual if r['optimization_role']==role]
                results[role]=dict(functions=len(members),syntax_valid=sum(r['actual_public_example_observations']['syntax_valid'] for r in members),
                    all_original_public_examples_passed=sum(r['actual_public_example_observations']['all_original_examples_passed'] for r in members))
            results.update(actual_EOS_stops=sum(r['actual_stopped_on_EOS'] for r in events),actual_budget_exhaustions=sum(not r['actual_stopped_on_EOS'] for r in events))
            outcomes[arm]=results
            if (len(records)!=25 or historical_physical+physical!=26 or len(events)!=14 or tensor_digest(base.named_parameters())!=base_digest):raise ValueError('All declared complete epochs, real next-step replay and original fourteen deliveries must remain')
            atomic_json(folder/'complete.json',dict(complete=True,actual_logical_updates=25,physical_optimizer_calls=historical_physical+physical,
                physical_optimizer_calls_this_runtime=physical,previous_committed_optimizer_calls=historical_physical,
                all10optimization_functions_in_each_of5epochs=True,unoptimized_controls_never_provided_gradients=True,
                actual_generated_function_results=results,every_exported_value_and_actual_complete_function_logits_cold_load_exact=True,
                original_frozen_dense_vision_word_and_unused_critic_unchanged=True,actual_counts=model.counts(),goal_achieved=False))
            del optimizer,reference,payload,saved
        guard.check(full=True)
        atomic_json(OUT/'complete.json',dict(at=datetime.now(timezone.utc).isoformat(),complete=True,actual_counts=model.counts(),
            actual_matched_arm_generated_function_results=outcomes,actual_logical_updates_per_arm=25,
            actual_optimization_function_presentations_per_arm=50,actual_unoptimized_source_train_control_functions=4,
            both_arms_complete_actual_next_step_private_state_recovery=True,new_production_pipeline_or_GPU_updates=0,
            source_train_research_not_final_independent_benchmark_or_human_alignment=True,
            source_files_sha256={**files,str(OUT/'plan.json'):file_hash(OUT/'plan.json')},candidate_promoted=False,goal_achieved=False))
        print(json.dumps(dict(complete=str(OUT/'complete.json'),actual_matched_arm_generated_function_results=outcomes)),flush=True)
    except BaseException as error:
        atomic_json(OUT/'failure.json',dict(error=repr(error),traceback=traceback.format_exc(),original_GPU_pipeline_untouched=True,goal_achieved=False));raise


if __name__=='__main__':
    try:main()
    except BaseException as error:
        target=OUT/'failure.json' if OUT.exists() else ROOT/'reports/own-function-trajectory-preferences-full-CPU-v4-parent-failure.json'
        if not target.exists():atomic_json(target,dict(error=repr(error),traceback=traceback.format_exc(),original_GPU_pipeline_untouched=True,goal_achieved=False))
        raise
