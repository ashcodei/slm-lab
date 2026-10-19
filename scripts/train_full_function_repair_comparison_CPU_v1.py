"""Actual full-size, three-arm original-source/repair/code-weight comparison.

CPU research leaves the four-GPU production lineage untouched. All arms use
the same own7558-update standalone base and initial bounded actor, all826
original functions, target-matched791second views, and three complete epochs.
All three arms finish training before fixed baseline/final assessments start.
No result is automatically adopted and no globally novel method is claimed.
"""
from datetime import datetime,timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import time
import traceback

from slm_lab.codenet_checkpoint_v1 import FileBindings
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.local_source_closure_v2 import python_sources

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/full-function-repair-comparison-CPU-v1'
OUTPUT=ROOT/'checkpoints/full-function-repair-comparison-CPU-v1'
BASE=ROOT/'checkpoints/code-first-continuation-v3/standalone/model.pt'
PREP=ROOT/'reports/full-function-repair-comparison-sources-CPU-v1'
VALID=ROOT/'reports/source-function-validation-runtime-CPU-v1'
SOURCE=ROOT/'reports/completed-code-public-function-rollouts-CPU-v1/plan.json'
PROFILES=['plain_repeat','semantic_repair','weighted_semantic_repair']
MAX_STATE=512*1024**2;RESERVE=26*1024**3


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or OUT.exists() or OUTPUT.exists():
        raise ValueError('Fresh GPU-hidden CPU comparison required; no automatic overwrite/retry')
    lock=(ROOT/'reports/own-function-comparison-CPU.lock').open('a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    import copy
    import numpy as np
    import torch
    from slm_lab.async_durable_checkpoint_v1 import AsyncCheckpoint
    from slm_lab.codenet_checkpoint_v1 import atomic_checkpoint,capture_rng
    from slm_lab.CPU_frozen_cast_cache_v1 import CPUFrozenCastCache
    from slm_lab.function_repair_curriculum_v1 import FunctionRepairCurriculum,schedule
    from slm_lab.full_function_repair_stage_v1 import CodeWeights,FunctionRepairStage,gradients
    from slm_lab.function_repair_evaluation_CPU_v1 import source_validation,local_suites
    from slm_lab.own_dense_standalone_v1 import load
    from slm_lab.own_dense_actor_v1 import OwnDenseActor
    from slm_lab.whole_human_sft_evidence_v1 import tensor_digest
    torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    if len(os.sched_getaffinity(0))<8 or torch.cuda.is_initialized():raise ValueError('CPU-only four-thread research alongside original GPU drivers required')
    if shutil.disk_usage(ROOT).free<RESERVE+5*1024**3:raise ValueError('Complete bounded research snapshots must fit above shared persistent reserve')
    stat=dict(line.split()[:2] for line in Path('/sys/fs/cgroup/memory.stat').read_text().splitlines())
    live=sum(int(stat[k]) for k in ['anon','shmem','kernel']);limit=int(Path('/sys/fs/cgroup/memory.max').read_text())
    if live+32*1024**3+64*1024**3>limit:raise ValueError('32GiB CPU working allowance and64GiB other-job headroom required')
    tests={'full-function-repair-stage-v1-tests':5,'function-repair-evaluation-CPU-v1-tests':2,
        'full-function-repair-comparison-kernel-and-evaluation-v1-final-tests':7,
        'CPU-frozen-cast-cache-v1-tests':3,'function-repair-curriculum-v1-tests':3}
    for name,count in tests.items():
        if str(count)+' passed' not in (ROOT/'reports'/(name+'.log')).read_text():raise ValueError('Actual source, gradient, recovery, precision and cache controls must pass')
    data=FunctionRepairCurriculum(ROOT,'optimization');weights=CodeWeights(data)
    prepared=json.loads((PREP/'plan.json').read_text());entries=schedule(data.lengths,**prepared['actual_schedule_settings'])
    if (entries!=prepared['actual_complete_slot_batches'] or len(entries)!=306
            or data.identity(entries)!=prepared['actual_schedule_identity_sha256']):
        raise ValueError('All306already-prepared complete target-matched batches required')
    pulses=json.loads((ROOT/'reports/code-first-continuation-v3/plan.json').read_text())['pulse_sources']
    paths=[BASE,BASE.parent/'tokenizer.json',SOURCE,ROOT/'reports/code-first-continuation-v3/plan.json',
        PREP/'plan.json',PREP/'complete.json',VALID/'complete.json',VALID/'all15-fixed-native-validation-cases.json',
        *[ROOT/'reports'/(name+'.log') for name in tests],ROOT/'scripts/code_worker.py',ROOT/'scripts/native_sandbox.py',
        ROOT/'scripts/documented_doctest_raw_harness_v1.py',ROOT/'scripts/python_program_sandbox_v1.py',
        ROOT/'data/math-reasoning-v2/test.jsonl',ROOT/'data/math-reasoning-v2/manifest.json',
        Path(pulses['knowledge']).with_suffix('.audit.json')]
    for name in pulses.values():
        path=Path(name);paths.append(path)
        for row in map(json.loads,path.read_text().splitlines()):
            if row.get('image'):paths.append((path.parent/row['image']).resolve())
    paths.extend(python_sources(ROOT,[Path(__file__),ROOT/'tests/test_full_function_repair_stage_v1.py',
        ROOT/'tests/test_function_repair_evaluation_CPU_v1.py',ROOT/'scripts/code_worker.py']))
    files={**data.source_files_sha256,**weights.source_files_sha256,**{str(p):file_hash(p) for p in set(paths)}}
    guard=FileBindings(files);OUT.mkdir();OUTPUT.mkdir();shutil.copyfile(data.tokenizer_path,OUTPUT/'tokenizer.json')
    options=dict(learning_rate=.0001,betas=[.9,.95],weight_decay=.01,gradient_clip=1.,
        maximum_microbatch=4,maximum_microbatch_tokens=4096,seed=979113,CPU_threads=4,precision='CPU_BF16_cached',
        save_after_first_then_every=8,complete_source_epochs=3,matched_global_width=16)
    source=json.loads(SOURCE.read_text())
    base,metadata=load(BASE,expected_checkpoint_sha256=files[str(BASE)],expected_sources=source['expected_standalone_sources'])
    model=OwnDenseActor(base).eval();initial=copy.deepcopy(model.parts.state_dict());initial_digest=tensor_digest(model.parts.named_parameters())
    frozen_digest=tensor_digest(base.named_parameters());critic=model.parts['critic'].weight.detach().clone()
    if (model.counts()['actual_unique']!=999706442 or base.counts()['total_unique']!=984399690
            or metadata['completed_updates']!=7558):raise ValueError('Actual complete own under1B original7558-update student and bounded actor required')
    plan=dict(at=datetime.now(timezone.utc).isoformat(),owner=dict(pid=os.getpid(),ticks=Path('/proc/self/stat').read_text().split(') ',1)[1].split()[19]),
        source_files_sha256=files,actual_student_counts=model.counts(),initial_actor_parts_sha256=initial_digest,
        frozen_dense_sha256=frozen_digest,profiles=PROFILES,training_options=options,
        all_complete_slot_batches=entries,source_schedule_identity_sha256=data.identity(entries),
        actual_matched_source_budgets={arm:data.budget(arm,entries) for arm in ['plain_repeat','semantic_repair']},
        actual_updates_per_profile=306,pulse_sources=pulses,actual_cpu_affinity=sorted(os.sched_getaffinity(0)),
        cgroup_memory_limit=limit,preflight_live_anon_shmem_kernel_bytes=live,CPU_working_allowance=32*1024**3,
        every_original_target_and_true_EOS_kept=True,code_weight_only_third_arm=4.,
        capacity_must_pass_all_actual_full_size_profiles_before_training=True,
        original_source_validation_functions_not_optimized=15,
        fixed_assessments=dict(source_validation=15,source_validation_native_budget=1024,local_code=48,
            local_code_native_budget=512,knowledge=56,math=16,OCR=20,objects=15,photos=32,image_size=448),
        all_arm_training_finishes_before_any_baseline_or_final_validation=True,
        data_does_not_increase_original_target_exposure_in_repair_arms=True,
        source_public_examples_visible_in_prompts=True,not_human_RLHF_or_own_failure_RL=True,
        transient_cold_probe_additional_actor_and_critic_values=15306752,
        training_only_frozen_cast_values_and_bytes_reported_in_capacity=True,
        inference_actor_artifact_requires_own_base_not_self_contained_student=True,
        automatic_retry=False,automatic_promotion=False,original_GPU_pipeline_unmodified=True,goal_achieved=False)
    atomic_json(OUT/'plan.json',plan);plan_sha=file_hash(OUT/'plan.json')
    cache=CPUFrozenCastCache();cache.attach(model.policy)
    def sources(profile):return dict(comparison_plan_sha256=plan_sha,profile_sha256=hashlib.sha256(profile.encode()).hexdigest(),own_complete_model_sha256=files[str(BASE)])
    def reset(profile,mode,batches):
        model.parts.load_state_dict(initial);model.zero_grad(set_to_none=True)
        if tensor_digest(model.parts.named_parameters())!=initial_digest:raise ValueError('Every matched profile must start at identical actor values')
        random.seed(options['seed']);np.random.seed(options['seed']);torch.manual_seed(options['seed'])
        optimizer=torch.optim.AdamW([p for _,p in model.named_selected(('actor',))],lr=options['learning_rate'],
            betas=tuple(options['betas']),weight_decay=options['weight_decay'],foreach=False)
        return FunctionRepairStage(model,optimizer,data,batches,profile,options,sources(profile),weights,cache,mode)
    def equal(a,b):
        if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and torch.equal(a,b)
        if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
        if isinstance(a,(list,tuple)):return type(a) is type(b) and len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
        return a==b
    def adam(stage):return tensor_digest((str(i)+'.'+k,v[k]) for i,v in enumerate(stage.optimizer.state.values()) for k in ['step','exp_avg','exp_avg_sq'])
    def intact():
        cache.assert_current()
        if (tensor_digest(base.named_parameters())!=frozen_digest or not torch.equal(model.parts['critic'].weight,critic)
                or torch.cuda.is_initialized()):raise ValueError('Every original frozen value/unused critic and GPU ownership must stay exact')
    def cold(path,profile,slot):
        payload=torch.load(path,map_location='cpu',weights_only=True,mmap=True);other=OwnDenseActor(base,**model.settings).eval()
        other.import_export(payload,expected_sources=sources(profile));model.eval();start=len(cache.patched);cache.attach(other.policy)
        ids=torch.tensor([data.native('semantic_repair',slot)[0]],dtype=torch.long)
        try:
            with torch.no_grad(),torch.autocast('cpu',dtype=torch.bfloat16):
                wanted=model(input_ids=ids)['logits'];actual=other(input_ids=ids)['logits']
            if (not torch.equal(wanted,actual) or tensor_digest(model.parts.named_parameters())!=tensor_digest(other.parts.named_parameters())):
                raise ValueError('Actual cold full-context CPU BF16 logits and all actor/unused critic values must match exactly')
            return dict(complete=True,actual_complete_source_context=len(data.native('semantic_repair',slot)[0]),
                every_actor_and_unused_critic_value_exact=True,actual_CPU_BF16_function_logits_bit_exact=True,
                transient_extra_actor_and_critic_values=15306752,shared_frozen_base_not_duplicated=True)
        finally:
            for module,original in cache.patched[start:]:module.forward=original
            del cache.patched[start:]
    # Export the fixed initial graph now; no new validation is read for gradients.
    atomic_checkpoint(OUTPUT/'baseline-actor.pt',model.export(sources('baseline')),96*1024**2,RESERVE)
    capacity={};longest=max(range(len(data.slots)),key=lambda p:len(data.native('semantic_repair',p)[0]))
    try:
        for profile in PROFILES:
            stage=reset(profile,'capacity',entries[:2]);folder=OUT/('capacity-'+profile);folder.mkdir();times=[]
            atomic_json(OUT/'progress.json',dict(phase='capacity',profile=profile,completed=0,goal_achieved=False))
            start=time.perf_counter();stage.update();times.append(time.perf_counter()-start)
            first=OUTPUT/(profile+'-capacity-after1.pt');atomic_checkpoint(first,stage.state(),MAX_STATE,RESERVE)
            start=time.perf_counter();wanted=stage.update();times.append(time.perf_counter()-start)
            wanted_parts=tensor_digest(model.parts.named_parameters());wanted_adam=adam(stage);wanted_rng=capture_rng('cpu')
            saved=torch.load(first,map_location='cpu',weights_only=True,mmap=True)
            for corruption in ['source','private_RNG']:
                bad=dict(saved)
                if corruption=='source':bad['model']={**saved['model'],'sources':{**saved['model']['sources'],'comparison_plan_sha256':'0'*64}}
                else:bad['rng']={**saved['rng'],'torch':torch.zeros(1,dtype=torch.uint8)}
                before=stage.signature()
                try:stage.resume(bad)
                except ValueError:pass
                else:raise ValueError('Corrupt source or private RNG must reject before recovery mutation')
                if (tensor_digest(model.parts.named_parameters())!=wanted_parts or stage.signature()!=before
                        or adam(stage)!=wanted_adam or not equal(wanted_rng,capture_rng('cpu'))):
                    raise ValueError('Rejected full-size recovery mutated model, optimizer or private RNG')
            stage.resume(saved)
            if (stage.update()!=wanted or tensor_digest(model.parts.named_parameters())!=wanted_parts
                    or adam(stage)!=wanted_adam or not equal(wanted_rng,capture_rng('cpu'))):
                raise ValueError('Actual full-size next update, metrics, actor/Adam/private RNG must recover bit exactly')
            probes=[];arm='plain_repeat' if profile=='plain_repeat' else 'semantic_repair'
            selected=[max(range(len(data.slots)),key=lambda p:len(data.native(arm,p)[0])),next(batch[0] for batch in entries if len(batch)==1)]
            for slot in selected:
                stage.optimizer.zero_grad(set_to_none=True)
                record=gradients(model,data,[slot],profile,options,weights,amp=lambda:torch.autocast('cpu',dtype=torch.bfloat16))
                if (any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in stage.parameters)
                        or any(p.grad is not None for p in stage.fixed)):raise ValueError('Long/tail complete-source backward needs every/only selected finite gradient')
                probes.append(record);stage.optimizer.zero_grad(set_to_none=True)
            if tensor_digest(model.parts.named_parameters())!=wanted_parts:raise ValueError('Capacity-only backward probes must not add optimizer updates')
            payload=OUTPUT/(profile+'-capacity-actor.pt');atomic_checkpoint(payload,model.export(sources(profile)),96*1024**2,RESERVE)
            cold_proof=cold(payload,profile,longest);intact();guard.check(full=True)
            if len(stage.optimizer.state)!=561:raise ValueError('All561actual selected actor Adam states required')
            capacity[profile]=dict(complete=True,logical_updates=2,physical_optimizer_calls=3,
                actual_full_global_slot_batches=entries[:2],actual_unique_student_parameters=model.counts()['actual_unique'],
                actual_selected_Adam_tensors=len(stage.optimizer.state),first_two_actual_update_seconds=times,
                every_actor_Adam_metric_and_private_RNG_next_update_bit_exact=True,
                corrupt_source_and_private_RNG_rejected_before_mutation=True,
                backward_only_individual_long_and_tail_source_probes=probes,cold_export_import=cold_proof,
                original_dense_vision_word_and_critic_unchanged=True,cast_cache_inventory=cache.inventory(),
                production_updates=0,goal_achieved=False)
            atomic_json(folder/'complete.json',capacity[profile]);print(json.dumps(dict(capacity_profile_complete=profile,seconds=times)),flush=True)
            del saved,stage
        atomic_json(OUT/'capacity-complete.json',dict(complete=True,actual_profiles=capacity,production_updates=0,
            all_actual_full_size_source_gradient_next_step_recovery_and_cold_proofs_pass=True,goal_achieved=False))
        training={}
        for profile in PROFILES:
            stage=reset(profile,'train',entries);folder=OUT/profile;folder.mkdir();writer=AsyncCheckpoint();elapsed=[]
            try:
                with (folder/'actual-training-metrics.jsonl').open('x') as journal:
                    for i in range(len(entries)):
                        guard.check();writer.check();start=time.perf_counter();row=stage.update();elapsed.append(time.perf_counter()-start)
                        journal.write(json.dumps(row)+'\n');journal.flush()
                        if stage.completed==1 or stage.completed%options['save_after_first_then_every']==0 or stage.completed==len(entries):
                            os.fsync(journal.fileno());guard.check(full=True)
                            writer.submit(OUTPUT/(profile+'-resume.pt'),stage.state(),MAX_STATE,RESERVE)
                        atomic_json(OUT/'progress.json',dict(phase='training',profile=profile,actual_complete_updates=stage.completed,
                            planned_updates_per_profile=306,completed_profiles=list(training),last_original_target_mean_NLL=row['original_token_mean_NLL'],
                            last_update_seconds=elapsed[-1],capacity_all_profiles_passed=True,goal_achieved=False))
                        print(json.dumps(dict(profile=profile,step=stage.completed,planned=306,NLL=row['original_token_mean_NLL'],seconds=elapsed[-1])),flush=True)
                writer.flush();stage.validate(stage.metrics);intact()
                path=OUTPUT/(profile+'-actor.pt');atomic_checkpoint(path,model.export(sources(profile)),96*1024**2,RESERVE)
                proof=cold(path,profile,longest);guard.check(full=True)
                slots=[p for batch in entries for p in batch]
                if (len(slots)!=4851 or sum(r['response_EOS_targets'] for r in stage.metrics)!=2287110
                        or any(slots.count(p)!=3 for p in range(len(data.slots)))):
                    raise ValueError('All826original and791second-view source presentations exactly once per epoch required')
                training[profile]=dict(complete=True,actual_logical_updates=306,physical_optimizer_calls=306,
                    actual_complete_presentations=len(slots),original_response_EOS_targets=2287110,
                    original_source_validation_gradients=0,actual_unique_student_parameters=model.counts()['actual_unique'],
                    cold_export_import=proof,actual_update_seconds=elapsed,actor_checkpoint_sha256=file_hash(path),
                    complete_private_state_sha256=file_hash(OUTPUT/(profile+'-resume.pt')),candidate_promoted=False,goal_achieved=False)
                atomic_json(folder/'training-complete.json',training[profile])
            finally:writer.close()
            del stage
        atomic_json(OUT/'all-arms-training-complete.json',dict(complete=True,profiles=training,all_original_source_budgets_preserved=True,
            independent_quality_not_yet_measured=True,goal_achieved=False))
        evaluations={}
        for profile in ['baseline',*PROFILES]:
            path=OUTPUT/(profile+'-actor.pt');model.import_export(torch.load(path,map_location='cpu',weights_only=True,mmap=True),expected_sources=sources(profile));model.eval()
            atomic_json(OUT/'progress.json',dict(phase='evaluation',profile=profile,all_three_full_training_arms_complete=True,goal_achieved=False))
            folder=OUT/('evaluation-'+profile);folder.mkdir()
            valid=source_validation(model.policy,path,folder/'source-validation',ROOT)
            local=local_suites(model.policy,path,metadata,folder/'local',ROOT,pulses)
            evaluations[profile]=dict(source_validation_correct=valid['all_original_examples_passed'],source_validation_functions=15,
                local_code_correct=sum(sum(v['functional_passed'] for v in local['code-'+frame]['by_language'].values()) for frame in ['completion','instruction']),
                local_code_tasks=48,knowledge_generated_correct=sum(r['generated_correct'] for r in local['knowledge']['predictions']),knowledge_questions=56,
                math_correct=local['math'].get('correct'),math_examples=16,source_validation_report_sha256=file_hash(folder/'source-validation/complete.json'),
                all_local_reports_receipt_sha256=file_hash(folder/'local/complete.json'))
            intact();guard.check(full=True);atomic_json(folder/'complete.json',dict(complete=True,**evaluations[profile],candidate_promoted=False,goal_achieved=False))
        atomic_json(OUT/'complete.json',dict(at=datetime.now(timezone.utc).isoformat(),complete=True,training_profiles=training,matched_evaluations=evaluations,
            source_files_sha256=files,all_original_826functions_791views_three_epochs_per_arm=True,
            baseline_and_all_profiles_full_fixed_source_validation_and_local_suites_complete=True,
            source_validation_not_certified_all_history_independent=True,original_GPU_and_full_final_benchmark_pipeline_unchanged=True,
            no_automatic_promotion=True,goal_achieved=False))
    finally:cache.close()


if __name__=='__main__':
    try:main()
    except BaseException as error:
        target=OUT/'failure.json' if OUT.exists() else ROOT/'reports/full-function-repair-comparison-CPU-v1-parent-failure.json'
        atomic_json(target,dict(at=datetime.now(timezone.utc).isoformat(),error=repr(error),traceback=traceback.format_exc(),goal_achieved=False));raise
