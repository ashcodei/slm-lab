"""Actual own full-size CPU generations, matched unassisted vs public prefixes.

This is a small training-source reward diagnostic on the retained step2048
foundation model, not a held capability benchmark or the later human-DPO actor.
No training or GPU initialization occurs. Every sampled suffix and original
execution result is retained; original public prefixes are labelled explicitly.
"""
from collections import defaultdict
from datetime import datetime,timezone
import hashlib
import json
import math
import os
from pathlib import Path
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/prefix-reward-density-full-CPU-v1'


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or os.environ.get('TOKENIZERS_PARALLELISM')!='false' or OUT.exists():
        raise ValueError('Fresh CPU-hidden single-tokenizer-thread own-model reward diagnostic required')
    import torch
    from slm_lab.active_execution_input_bindings_v1 import input_bindings,merge_bindings
    from slm_lab.corpus import file_hash
    from slm_lab.dense_foundation_inference_v1 import DenseNativeView
    from slm_lab.diverse_images import atomic_json
    from slm_lab.expanded_execution_corpus_v1 import ExpandedExecutionCorpus
    from slm_lab.fast_joint_inference_v2 import load
    from slm_lab.local_source_closure_v2 import python_sources
    from slm_lab.native_prefix_rollout_CPU_v1 import sample
    from slm_lab.public_reference_prefix_curriculum_v1 import view
    from slm_lab.whole_human_sft_evidence_v1 import tensor_digest
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    OUT.mkdir();started=time.monotonic();pid=os.getpid()
    fields=(Path('/proc')/str(pid)/'stat').read_text().split(') ',1)[1].split()
    owner=dict(pid=pid,ticks=fields[19],CUDA_VISIBLE_DEVICES='',torch_CPU_threads=2)
    atomic_json(OUT/'owner.json',dict(at=datetime.now(timezone.utc).isoformat(),owner=owner,model_updates=0,automatic_retry=False))
    try:
        proof_path=ROOT/'reports/phase-joint-step2048-readback-v1/complete.json';proof=json.loads(proof_path.read_text())
        phase_path=ROOT/'reports/phase-joint-v3/plan.json';phase=json.loads(phase_path.read_text())
        checkpoint=ROOT/'checkpoints/phase-joint-v3/step-2048/model.pt'
        prep_path=ROOT/'reports/public-reference-prefix-curriculum-v1/plan.json';prep=json.loads(prep_path.read_text())
        control_path=ROOT/'reports/public-reference-prefix-curriculum-v1/execution-controls.jsonl'
        controls=[json.loads(s) for s in control_path.open()]
        reader_path=ROOT/'reports/public-reference-prefix-curriculum-v1-independent-reader/complete.json'
        reader=json.loads(reader_path.read_text())
        if not reader['all32_prefix_only_controls_do_not_earn_full_credit'] or not proof['complete']:
            raise ValueError('Actual full-weight source readback and independent nontrivial public-prefix controls required')
        corpus=ExpandedExecutionCorpus(ROOT);groups=defaultdict(list)
        for source_id in prep['controls_selected_before_execution']:
            position=corpus.by_id[source_id];row=corpus.row(position)
            if row['response_EOS_targets']>512:continue
            stratum='documented' if row['kind']=='documented' else row['original']['difficulty']
            groups[stratum].append(position)
        selected=[]
        for stratum in ['documented','introductory','interview','competition']:
            if not groups[stratum]:raise ValueError('Every fixed train difficulty stratum must retain a complete reference within512 actions')
            selected.append(min(groups[stratum],key=lambda p:(corpus.row(p)['response_EOS_targets'],corpus.row(p)['id'])))
        selected_ids=[corpus.row(p)['id'] for p in selected]
        pairs={}
        for position in selected:
            row=corpus.row(position);ids,labels=corpus.sequence(position)
            prior=[r for r in controls if r['id']==row['id'] and r['retained_reference_quarters']==3]
            if len(prior)!=1 or not prior[0]['proposed_conditional_episode_nontrivial']:
                raise ValueError('Each supplied public prefix must already have a separately replayed nontrivial control')
            for quarter in [0,3]:
                value=view(corpus.tokenizer,row['original'],ids,labels,quarter)
                if value['complete_missing_suffix_training_budget']!=512:
                    raise ValueError('Both conditions require the identical complete512-token generation budget')
                pairs[(position,quarter)]=value
        tasks=[]
        for repetition in range(5):
            for position in selected:
                for quarter in [0,3]:
                    tasks.append(dict(index=len(tasks),position=position,source_id=corpus.row(position)['id'],
                        retained_reference_quarters=quarter,sampler='greedy' if repetition==0 else 'full_vocabulary_temperature1',
                        repetition=repetition,seed=964117+len(tasks)*1009))
        bindings=input_bindings(ROOT,corpus)
        files=[proof_path,phase_path,prep_path,control_path,reader_path,checkpoint,
            Path(phase['initial_dense_settings']['initializer']),ROOT/'reports/native-prefix-rollout-CPU-v1-corrected-tests.log']
        files+=python_sources(ROOT,[Path(__file__),ROOT/'tests/test_native_prefix_rollout_CPU_v1.py'])
        merge_bindings(bindings,{str(p):file_hash(p) for p in files})
        if bindings[str(checkpoint)]!=proof['source_files_sha256'][str(checkpoint)]:raise ValueError('Permanent own step2048 model changed')
        plan=dict(at=datetime.now(timezone.utc).isoformat(),owner=owner,
            checkpoint=str(checkpoint),checkpoint_sha256=bindings[str(checkpoint)],checkpoint_actual_updates=2048,
            own_unique_exported_values=984399690,selected_original_train_source_ids=selected_ids,
            source_selection='Within the earlier source-preselected16 train controls, choose smallest complete original target per difficulty stratum; never based on student scores',
            conditions=[dict(retained_reference_quarters=q,reference_hint_supplied=q>0) for q in [0,3]],
            matched_generation_budget=512,greedy_episodes=8,temperature1_stochastic_episodes=32,planned_actual_neural_episodes=40,
            tasks=tasks,source_files_sha256=bindings,inference_device_type='cpu',inference_precision='FP32',
            no_extra_probability_tolerance_admission=True,no_new_human_labels=True,
            not_the_future_human_DPO_starting_actor=True,not_a_held_or_unassisted_general_capability_benchmark=True,
            existing_live_GPU_and_queued_training_recipes_unchanged=True,automatic_retry=False,model_updates=0,goal_achieved=False)
        atomic_json(OUT/'plan.json',plan)
        assistant,metadata=load(checkpoint,phase_path,'step-2048',expected_checkpoint_sha256=bindings[str(checkpoint)],
            expected_plan_sha256=bindings[str(phase_path)],device='cpu')
        model=DenseNativeView(assistant).eval().requires_grad_(False)
        if sum(p.numel() for p in model.parameters())!=984399690 or torch.cuda.is_initialized():
            raise ValueError('Exactly the actual own under1B full model on CPU without CUDA initialization required')
        initial_digest=tensor_digest(model.named_parameters())
        atomic_json(OUT/'model-loaded.json',dict(at=datetime.now(timezone.utc).isoformat(),owner=owner,
            actual_unique_values=sum(p.numel() for p in model.parameters()),actual_model_parameter_sha256=initial_digest,
            checkpoint_sha256=bindings[str(checkpoint)],model_metadata=metadata,CUDA_initialized=False))
        records=[];ledger=OUT/'actual-neural-episodes.jsonl';last_progress=0.
        def progress(task,partial=None,force=False):
            nonlocal last_progress
            now=time.monotonic()
            if force or now-last_progress>15:
                value=dict(at=datetime.now(timezone.utc).isoformat(),owner=owner,completed_actual_neural_episodes=len(records),
                    planned_actual_neural_episodes=40,current_task=task,current_generation=partial,
                    elapsed_seconds=now-started,actual_generated_action_tokens=sum(r['actual_rollout']['actual_generated_action_tokens'] for r in records),
                    model_updates=0,goal_achieved=False)
                atomic_json(OUT/'progress.json',value);last_progress=now
                if force:print(json.dumps(value),flush=True)
        with ledger.open('x') as output:
            for task in tasks:
                value=pairs[(task['position'],task['retained_reference_quarters'])];progress(task,force=True)
                actual=sample(model,corpus.tokenizer,value['native_rollout_prefix_ids'],value['public_reference_prefix_ids'],
                    horizon=512,sampler=task['sampler'],seed=task['seed'],progress=lambda part:progress(task,part))
                reward=corpus.score(task['position'],actual['complete_candidate_response'],stopped_on_eos=actual['stopped_on_generated_EOS'])
                record=dict(task=task,actual_rollout=actual,actual_execution_reward=reward,
                    public_reference_prefix_tokens=value['retained_reference_tokens'],
                    complete_original_reference_EOS_targets=value['original_response_EOS_targets'],
                    original_source_native_input_sha256=value['original_native_input_sha256'],
                    conditional_training_source_diagnostic=True,unassisted_capability_score=False,model_updates=0)
                output.write(json.dumps(record,ensure_ascii=False,separators=(',',':'))+'\n');output.flush();os.fsync(output.fileno())
                records.append(record);progress(task,force=True)
        if tensor_digest(model.named_parameters())!=initial_digest or torch.cuda.is_initialized():
            raise ValueError('Actual complete own model values must remain unchanged and CUDA uninitialized')
        for name,digest in bindings.items():
            if file_hash(Path(name))!=digest:raise ValueError('Diagnostic source changed during neural sampling: '+name)
        summary={}
        for quarter in [0,3]:
            for sampler in ['greedy','full_vocabulary_temperature1']:
                items=[r for r in records if r['task']['retained_reference_quarters']==quarter and r['task']['sampler']==sampler]
                rewards=[r['actual_execution_reward']['reward'] for r in items]
                by_source=defaultdict(list)
                for item in items:by_source[item['task']['source_id']].append(item['actual_execution_reward']['reward'])
                summary[str(quarter)+':'+sampler]=dict(episodes=len(items),positive_rewards=sum(x>0 for x in rewards),
                    full_finite_case_credit_with_actual_EOS=sum(x==1. for x in rewards),mean_reward=math.fsum(rewards)/len(items),
                    genuine_EOS_deliveries=sum(r['actual_rollout']['stopped_on_generated_EOS'] for r in items),
                    source_groups_with_reward_variation=sum(min(v)<max(v) for v in by_source.values()),
                    source_rewards=dict(by_source))
        report=dict(at=datetime.now(timezone.utc).isoformat(),complete=True,owner=owner,actual_neural_episodes=len(records),
            actual_generated_action_tokens=sum(r['actual_rollout']['actual_generated_action_tokens'] for r in records),
            actual_own_model_values=984399690,actual_model_parameter_sha256=initial_digest,
            checkpoint_actual_updates=2048,summary=summary,plan_sha256=file_hash(OUT/'plan.json'),
            actual_neural_episodes_sha256=file_hash(ledger),
            cached_full_max_absolute_difference=max(r['actual_rollout']['cached_full_max_absolute_difference'] for r in records),
            all_weights_unchanged=True,actual_CUDA_initializations=0,model_updates=0,
            training_source_conditional_reward_diagnostic_not_a_held_general_capability_gain=True,
            not_an_RL_numerical_or_GPU_capacity_admission=True,
            source_files_sha256={str(p):file_hash(p) for p in [Path(__file__),OUT/'plan.json',OUT/'model-loaded.json',ledger]},
            candidate_promoted=False,goal_achieved=False)
        atomic_json(OUT/'complete.json',report);print(json.dumps(dict(complete=str(OUT/'complete.json'),summary=summary)),flush=True)
    except BaseException as error:
        atomic_json(OUT/'failure.json',dict(at=datetime.now(timezone.utc).isoformat(),owner=owner,type=type(error).__name__,
            message=str(error),traceback=traceback.format_exc(),automatic_retry=False,model_updates=0,goal_achieved=False));raise


if __name__=='__main__':main()
