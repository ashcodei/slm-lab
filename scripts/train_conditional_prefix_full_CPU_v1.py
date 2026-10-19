"""Bounded real full-size CPU comparison of CE versus conditional execution RL.

Waits for the exact40-episode own-model diagnostic. Selects one original train
problem by a declared within-source reward-variation rule, then performs three
logical updates per arm with full original CE and full native action budgets.
Each arm additionally replays update2 from a complete private checkpoint.
This is not a complete source epoch, human alignment or GPU admission.
"""
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
REPORTS=ROOT/'reports/conditional-prefix-full-CPU-v1'
OUTPUT=ROOT/'checkpoints/conditional-prefix-full-CPU-v1'
QUEUE=ROOT/'reports/conditional-prefix-full-CPU-v1-queue.json'
PREVIOUS=ROOT/'reports/prefix-reward-density-full-CPU-v1'
ARMS=('anchor_CE','anchor_execution')


def same(identity):
    path=Path('/proc')/str(identity['pid'])/'stat'
    try:fields=path.read_text().split(') ',1)[1].split()
    except (FileNotFoundError,ProcessLookupError):return None
    return fields[0] if fields[19]==identity['ticks'] and fields[0] not in ['Z','X'] else None


def selected_source(diagnostic):
    initial=json.loads((PREVIOUS/'plan.json').read_text())
    groups=diagnostic['summary']['3:full_vocabulary_temperature1']['source_rewards']
    for identity in initial['selected_original_train_source_ids']:
        values=groups[identity]
        if len(values)!=4:raise ValueError('Every complete four-sample conditional source group required')
        if min(values)<max(values) and max(values)>0:return identity
    raise ValueError('The completed conditional diagnostic has no within-source varied actual reward group')


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or REPORTS.exists() or OUTPUT.exists() or QUEUE.exists():
        raise ValueError('Fresh CPU-hidden full-size conditional trial required')
    from slm_lab.corpus import file_hash
    from slm_lab.diverse_images import atomic_json
    from slm_lab.local_source_closure_v2 import python_sources
    me=dict(pid=os.getpid(),ticks=(Path('/proc')/str(os.getpid())/'stat').read_text().split(') ',1)[1].split()[19])
    previous_plan=json.loads((PREVIOUS/'plan.json').read_text());prior=previous_plan['owner']
    files=python_sources(ROOT,[Path(__file__),ROOT/'tests/test_conditional_trial_policy_CPU_v1.py',ROOT/'tests/test_admitted_public_prefix_episode_v1.py',ROOT/'tests/test_conditional_prefix_full_CPU_v1_driver.py'])
    files+=[ROOT/'reports/conditional-prefix-CPU-v1/complete.json',ROOT/'reports/conditional-trial-policy-CPU-v1-final-tests.log',ROOT/'reports/conditional-prefix-full-CPU-v1-driver-tests.log',
        ROOT/'reports/admitted-public-prefix-episode-v1-final-tests.log',PREVIOUS/'plan.json']
    bindings={str(p):file_hash(p) for p in files}
    atomic_json(QUEUE,dict(at=datetime.now(timezone.utc).isoformat(),owner=me,predecessor=prior,
        status='waiting for exact40-episode own-model diagnostic to finish',source_files_sha256=bindings,
        source_selection='First original source in the fixed four-source diagnostic order with four hinted stochastic samples and nonzero reward variation',
        planned_logical_updates_each_arm=3,planned_physical_updates_each_arm=4,CPU_only=True,
        automatic_retry=False,model_updates=0,goal_achieved=False))
    while same(prior):
        if (PREVIOUS/'failure.json').exists():raise ValueError('Predecessor neural diagnostic failed')
        time.sleep(2)
    if not (PREVIOUS/'complete.json').exists() or (PREVIOUS/'failure.json').exists():raise ValueError('Exact complete40-episode diagnostic required')
    import copy
    import random
    import numpy as np
    import torch
    from slm_lab.active_execution_input_bindings_v1 import input_bindings,merge_bindings
    from slm_lab.admitted_public_prefix_episode_v1 import AdmittedPublicPrefixEpisode
    from slm_lab.codenet_checkpoint_v1 import atomic_checkpoint,capture_rng
    from slm_lab.conditional_prefix_stage_CPU_v1 import ConditionalPrefixStage,trajectory_digest
    from slm_lab.conditional_trial_policy_CPU_v1 import export,load as load_trial,sources
    from slm_lab.expanded_execution_corpus_v1 import ExpandedExecutionCorpus
    from slm_lab.fast_joint_inference_v2 import load as load_base
    from slm_lab.frozen_own_actor_reference_v1 import FrozenOwnActorReference
    from slm_lab.native_prefix_rollout_CPU_v1 import sample
    from slm_lab.own_dense_actor_v1 import OwnDenseActor
    from slm_lab.public_reference_prefix_curriculum_v1 import view
    from slm_lab.whole_human_sft_evidence_v1 import tensor_digest
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    diagnostic=json.loads((PREVIOUS/'complete.json').read_text())
    proof=json.loads((ROOT/'reports/conditional-prefix-CPU-v1/complete.json').read_text())
    if (diagnostic['actual_neural_episodes']!=40 or not diagnostic['all_weights_unchanged']
            or not proof['next_conditional_update_weights_Adam_Python_NumPy_TorchCPU_RNG_and_actual_trajectory_replay_exact']):
        raise ValueError('Actual complete neural diagnostic and real conditional private recovery proof required')
    source_id=selected_source(diagnostic);corpus=ExpandedExecutionCorpus(ROOT)
    episode=AdmittedPublicPrefixEpisode(corpus,corpus.by_id[source_id],3,root=ROOT)
    inputs=dict(checkpoint=previous_plan['checkpoint'],checkpoint_sha256=previous_plan['checkpoint_sha256'],
        plan=str(ROOT/'reports/phase-joint-v3/plan.json'),
        plan_sha256=previous_plan['source_files_sha256'][str(ROOT/'reports/phase-joint-v3/plan.json')],arm='step-2048')
    base,_=load_base(inputs['checkpoint'],inputs['plan'],inputs['arm'],expected_checkpoint_sha256=inputs['checkpoint_sha256'],
        expected_plan_sha256=inputs['plan_sha256'],device='cpu')
    models={arm:OwnDenseActor(base,rank=16,seed=965417).eval() for arm in ARMS}
    initial_parts=tensor_digest(models[ARMS[0]].parts.named_parameters())
    if any(m.counts()['actual_unique']!=999706442 or tensor_digest(m.parts.named_parameters())!=initial_parts for m in models.values()):
        raise ValueError('Both exact full-size neutral own policy arms must start identically and strictly under1B')
    reference=FrozenOwnActorReference(models[ARMS[0]],sources=dict(own_step2048_checkpoint=inputs['checkpoint_sha256'],
        initial_neutral_actor_parts=initial_parts))
    for model in models.values():reference.assert_bound(model)
    base_digest=tensor_digest(base.named_parameters());ref_digest=tensor_digest(reference.parts.named_parameters())
    union={id(p):p for model in [*models.values(),reference] for p in model.parameters()}
    resident=sum(p.numel() for p in union.values())
    if resident!=984399690+3*15306752 or torch.cuda.is_initialized():raise ValueError('One shared own base and three explicit private branches on CPU only required')
    merge_bindings(bindings,input_bindings(ROOT,corpus))
    for p in [PREVIOUS/'complete.json',PREVIOUS/'actual-neural-episodes.jsonl',Path(inputs['checkpoint']),Path(inputs['plan'])]:
        merge_bindings(bindings,{str(p):file_hash(p)})
    for name,digest in bindings.items():
        if file_hash(Path(name))!=digest:raise ValueError('Bound CPU trial source changed before starting')
    REPORTS.mkdir();OUTPUT.mkdir();episode_binding=episode.binding()
    plan=dict(at=datetime.now(timezone.utc).isoformat(),owner=me,base_inputs=inputs,actual_base_updates=2048,
        is_complete_human_DPO_or_source_epoch=False,actor_settings=models[ARMS[0]].settings,actual_counts=models[ARMS[0]].counts(),
        initial_parts_sha256=initial_parts,own_frozen_starting_reference_binding=reference.binding(),
        actual_unique_resident_training_values=resident,actual_exported_student_values_each_arm=999706442,
        qualified_episode_binding=episode_binding,
        qualified_episode_binding_sha256=hashlib.sha256(json.dumps(episode_binding,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
        selected_original_train_source=source_id,source_selection= 'first fixed diagnostic source with complete within-source conditional reward variation',
        planned_updates_each_arm=3,physical_optimizer_calls_each_arm=4,local_fresh_group_size=4,
        source_full_reference_targets_each_logical_update=episode_binding['full_original_reference_EOS_targets'],
        source_full_action_budget=episode_binding['native_full_action_budget'],CPU_mechanics_horizon=None,
        learning_rate=1e-5,betas=[.9,.95],weight_decay=.01,anchor_weight=1.,KL_weight=.01,
        declared_CPU_cached_full_absolute_bound=.0001,training_seed=965517,
        paired_evaluation='one greedy and four fresh private-generator samples per unassisted/hinted condition, identical seeds at initial and both endpoints',
        source_files_sha256=bindings,CPU_threads=2,no_GPU_initialization=True,full_new_GPU_admission_still_required=True,
        no_new_human_labels=True,no_complete_source_epoch_claim=True,automatic_retry=False,candidate_promoted=False,goal_achieved=False)
    atomic_json(REPORTS/'plan.json',plan);digest=file_hash(REPORTS/'plan.json');native_sources=sources(plan,digest)
    from slm_lab.own_tokenizer_package_v1 import ensure
    ensure(ROOT/'data/tokenizer-corpus/tokenizer.json',OUTPUT,expected_sha256=file_hash(corpus.tokenizer_path),expected_vocab_size=32768)
    try:
        def evaluate(model,label,completed):
            model.eval();records=[]
            for quarters in [0,3]:
                ids,labels=corpus.sequence(episode.position);prepared=view(corpus.tokenizer,corpus.row(episode.position)['original'],ids,labels,quarters)
                for repetition in range(5):
                    sampler='greedy' if repetition==0 else 'full_vocabulary_temperature1';seed=965617+quarters*1009+repetition*9176
                    actual=sample(model.policy,corpus.tokenizer,prepared['native_rollout_prefix_ids'],prepared['public_reference_prefix_ids'],
                        horizon=prepared['complete_missing_suffix_training_budget'],sampler=sampler,seed=seed)
                    reward=corpus.score(episode.position,actual['complete_candidate_response'],stopped_on_eos=actual['stopped_on_generated_EOS'])
                    record=dict(stage=label,completed_trial_updates=completed,quarters=quarters,repetition=repetition,
                        actual_rollout=actual,actual_execution_reward=reward,public_training_source_not_held_capability=True)
                    records.append(record)
                    atomic_json(REPORTS/(label+'-actual-evaluation-prefix.json'),records)
                    print(json.dumps(dict(phase='training-source-evaluation',stage=label,quarters=quarters,repetition=repetition,
                        reward=reward['reward'],genuine_EOS=actual['stopped_on_generated_EOS'],completed_trial_updates=completed)),flush=True)
            atomic_json(REPORTS/(label+'-evaluation-complete.json'),dict(results=records,actual_neural_episodes=10,
                training_source_conditional_and_unassisted_diagnostic_only=True,model_updates=0,goal_achieved=False));return records
        initial_model=models[ARMS[0]];initial_batch=episode.batch('cpu')
        with torch.no_grad():
            a=initial_model(initial_batch['input_ids'])['logits'];b=models[ARMS[1]](initial_batch['input_ids'])['logits']
            if not torch.equal(a,b):raise ValueError('Both actual full-size initial policy computations must agree bit exactly')
            original=base(initial_batch['input_ids'])['logits'];delta=float((a-original).abs().max())
            if not torch.allclose(a,original,atol=.0001,rtol=1e-5):raise ValueError('Neutral initial actors must preserve the actual own partial foundation computation')
            del a,b,original
        atomic_json(REPORTS/'initial-full-size-native-parity.json',dict(two_initial_arms_bit_exact=True,
            neutral_actor_vs_own_base_logit_max_absolute_difference=delta,actual_exported_student_values=999706442,
            actual_CPU_only=True,model_updates=0,goal_achieved=False))
        atomic_checkpoint(OUTPUT/'initial-actor.pt',export(initial_model,plan,digest,'initial',0),96*1024**2,26*1024**3)
        before=evaluate(initial_model,'initial',0)
        def equal(a,b):
            if isinstance(a,torch.Tensor):return torch.equal(a,b)
            if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
            if isinstance(a,(list,tuple)):return type(a)is type(b) and len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
            return a==b
        after={};arm_metrics={}
        for arm in ARMS:
            model=models[arm];parameters=[p for _,p in model.named_selected(('actor',))]
            optimizer=torch.optim.AdamW(parameters,lr=1e-5,betas=(.9,.95),weight_decay=.01,foreach=False)
            stage=ConditionalPrefixStage(model,optimizer,episode,reference,method=arm,planned_updates=3,sources=native_sources,
                local_group_size=4,cache_absolute_bound=.0001)
            random.seed(965517);np.random.seed(965517);torch.manual_seed(965517)
            records=[]
            def step(label):
                record=stage.update();records.append(record)
                atomic_json(REPORTS/(arm+'-'+label+'-actual-native-step.json'),record)
                atomic_json(REPORTS/(arm+'-progress.json'),dict(actual_logical_updates=stage.completed,planned_updates=3,
                    actual_global_rewards=record['global_rewards'],full_original_reference_targets=record['source_reference_targets'],
                    original_reference_token_mean_CE=record['original_reference_token_mean_CE'],goal_achieved=False))
                print(json.dumps(dict(phase='actual-conditional-training',arm=arm,label=label,logical_update=stage.completed,
                    rewards=record['global_rewards'],advantages=record['global_advantages'])),flush=True);return record
            step('first');saved=OUTPUT/(arm+'-after-first.pt');atomic_checkpoint(saved,stage.state(),384*1024**2,26*1024**3)
            expected=step('expected-second');weights=model.export(native_sources);adam=copy.deepcopy(optimizer.state_dict());rng=capture_rng('cpu')
            raw=torch.load(saved,map_location='cpu',weights_only=True,mmap=True);bad=copy.deepcopy(raw);bad['episode_binding']['retained_reference_quarters']=2
            signature=[(id(p),p._version,p.data_ptr()) for p in model.parameters()]
            try:stage.resume(bad)
            except ValueError:pass
            else:raise ValueError('Wrong full-size source prefix must reject before mutation')
            if signature!=[(id(p),p._version,p.data_ptr()) for p in model.parameters()] or not equal(rng,capture_rng('cpu')):
                raise ValueError('Rejected full-size recovery changed model or RNG')
            stage.resume(raw);actual=step('replayed-second')
            if (not equal(weights,model.export(native_sources)) or not equal(adam,optimizer.state_dict())
                    or not equal(rng,capture_rng('cpu')) or trajectory_digest(expected['trajectories'])!=trajectory_digest(actual['trajectories'])):
                raise ValueError('Full-size next conditional update, Adam, private RNG and genuine suffix trajectories must replay exactly')
            step('third-final');atomic_checkpoint(OUTPUT/(arm+'-final-state.pt'),stage.state(),384*1024**2,26*1024**3)
            atomic_checkpoint(OUTPUT/(arm+'-actor.pt'),export(model,plan,digest,arm,3),96*1024**2,26*1024**3)
            if (len(optimizer.state)!=561 or any(float(s['step'])!=3 for s in optimizer.state.values())
                    or tensor_digest(model.parts.named_parameters())==initial_parts or tensor_digest(base.named_parameters())!=base_digest
                    or tensor_digest(reference.parts.named_parameters())!=ref_digest):
                raise ValueError('All561 actual actor Adam clocks, changed selected values and fixed own base/reference required')
            arm_metrics[arm]=dict(logical_updates=3,physical_optimizer_calls=4,all561_Adam_clocks=3,
                logical_reward_groups=[r['global_rewards'] for i,r in enumerate(records) if i!=1],
                logical_updates_with_varied_reward=sum(min(r['global_rewards'])<max(r['global_rewards']) for i,r in enumerate(records) if i!=1),
                next_update_private_weights_Adam_RNG_trajectories_exact=True,wrong_prefix_restore_rejected_before_mutation=True)
            after[arm]=evaluate(model,arm+'-final',3)
            cold,_=load_trial(OUTPUT/(arm+'-actor.pt'),REPORTS/'plan.json',arm,
                expected_checkpoint_sha256=file_hash(OUTPUT/(arm+'-actor.pt')),expected_plan_sha256=digest)
            probe=torch.tensor([episode.rollout_view()['native_rollout_prefix_ids'][:16]])
            pixels=torch.linspace(-1,1,3*224*224).reshape(1,3,224,224)
            with torch.no_grad():
                for image in [None,pixels]:
                    if not torch.equal(model(probe,pixel_values=image)['logits'],cold(probe,pixel_values=image)['logits']):
                        raise ValueError('Student-only full-size CPU native text/image cold reload must preserve exact logits')
            del cold
            atomic_json(REPORTS/(arm+'-complete.json'),dict(**arm_metrics[arm],student_only_native_text_and_synthetic_image_cold_reload_exact=True,
                actual_student_values=999706442,not_a_complete_human_policy_or_source_epoch=True,goal_achieved=False))
        if any(tensor_digest(models[arm].base.named_parameters())!=base_digest for arm in ARMS) or torch.cuda.is_initialized():
            raise ValueError('Both actual own bases must remain exact and CUDA must never initialize')
        for name,value in bindings.items():
            if file_hash(Path(name))!=value:raise ValueError('Bound full-size CPU experiment source changed')
        summary={}
        for label,items in [('initial',before),*((arm,after[arm]) for arm in ARMS)]:
            for quarter in [0,3]:
                for sampler in ['greedy','full_vocabulary_temperature1']:
                    rows=[r for r in items if r['quarters']==quarter and r['actual_rollout']['sampler']==sampler]
                    summary[label+':'+str(quarter)+':'+sampler]=dict(episodes=len(rows),positive_rewards=sum(r['actual_execution_reward']['reward']>0 for r in rows),
                        full_finite_case_credit_with_EOS=sum(r['actual_execution_reward']['reward']==1. for r in rows))
        atomic_json(REPORTS/'complete.json',dict(at=datetime.now(timezone.utc).isoformat(),complete=True,arms=arm_metrics,
            exported_student_values_each_arm=999706442,extra_frozen_own_reference_values=15306752,
            actual_all_arm_training_resident_unique_values=resident,own_base_and_private_reference_unchanged=True,
            source_id=source_id,qualified_one_problem_three_updates_not_complete_curriculum=True,
            actual_training_source_neural_evaluation_summary=summary,actual_CUDA_initializations=0,
            native_student_only_text_image_cold_reload_exact=True,full40_episode_diagnostic_sha256=file_hash(PREVIOUS/'complete.json'),
            full_new_GPU_capacity_and_complete_curriculum_still_required=True,
            no_new_human_labels_or_unassisted_held_capability_claim=True,
            independent_state_raw_generated_program_and_objective_readback_pending=True,candidate_promoted=False,goal_achieved=False))
        print(json.dumps(dict(complete=str(REPORTS/'complete.json'),arms=arm_metrics,summary=summary)),flush=True)
    except BaseException as error:
        atomic_json(REPORTS/'failure.json',dict(error=repr(error),traceback=traceback.format_exc(),automatic_retry=False,goal_achieved=False));raise


if __name__=='__main__':main()
