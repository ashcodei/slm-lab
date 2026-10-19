"""Queue complete public reasoning warmup on the released two-thread CPU slot.

No GPU context/lease or canonical stage change. Full-size capacity, exact next
update recovery and cold output checks precede all483whole-source updates.
Baseline and final289reserved/15source/48code/24ordinaryPython/broad native
checks are mandatory. Verifier RL remains separate, pending GPU integration.
"""
from datetime import datetime,timezone
import fcntl
import gc
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
from slm_lab.exact_waiting_process_retirement_v1 import terminal
from slm_lab.public_pretraining_assessment_recovery_v1 import completed_training_after_observer_failure
from slm_lab.local_source_closure_v2 import python_sources
from slm_lab.public_reasoning_warmup_gate_v2 import ready

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/public-reasoning-warmup-CPU-v2'
OUTPUT=ROOT/'checkpoints/public-reasoning-warmup-CPU-v2'
QUEUE=ROOT/'reports/public-reasoning-warmup-CPU-v2-queue.json'
WAIT=ROOT/'reports/public-reasoning-warmup-CPU-v2-queue-progress.json'
EDGE=ROOT/'reports/public-edge-preference-comparison-CPU-v2'
EDGE_QUEUE=ROOT/'reports/public-edge-preference-comparison-CPU-v2-queue.json'
MATH=ROOT/'reports/public-pretraining-v1/plan.json'
RUNTIME=ROOT/'reports/public-pretraining-recovery-v1/plan.json'
BASE=ROOT/'checkpoints/public-pretraining-v1/trained/model.pt'
DATA=ROOT/'data/public-reasoning-curriculum-v1'
ADMITTED=ROOT/'reports/public-reasoning-curriculum-v1-readback-admission/complete.json'
MAX_STATE=384*1024**2
MAX_ACTOR=96*1024**2
RESERVE=40*1024**3


def read(path):return json.loads(Path(path).read_text())


def birth():return dict(pid=os.getpid(),ticks=Path('/proc/self/stat').read_text().split(') ',1)[1].split()[19])


def main():
    if (os.environ.get('CUDA_VISIBLE_DEVICES')!='' or os.nice(0)<19 or any(p.exists() for p in [OUT,OUTPUT,QUEUE])
            or any(os.environ.get(name)!='1' for name in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'])):
        raise ValueError('Fresh CUDA-hidden low-priority queued CPU warmup required')
    edge=read(EDGE_QUEUE);runtime=read(RUNTIME)
    if terminal(edge['owner']) or not completed_training_after_observer_failure(ROOT):raise ValueError('Capture exact actually live original predecessor births before queuing')
    logs=[('public-reasoning-group-gradients-v1-tests.log','9 passed'),('public-reasoning-stage-v1-tests.log','6 passed'),
        ('public-reasoning-reserved-evaluation-v1-tests.log','2 passed'),('public-reasoning-warmup-gate-v1-tests.log','2 passed'),
        ('function-repair-evaluation-CPU-v1-tests.log','2 passed')]
    if any(expected not in (ROOT/'reports'/name).read_text() for name,expected in logs):
        raise ValueError('All actual kernel/stage/native-reserved/gate/legacy-evaluation controls must pass')
    admitted=read(ADMITTED);files=dict(admitted['source_files_sha256']);files.update({str(ADMITTED):file_hash(ADMITTED),str(DATA/'admission.json'):file_hash(DATA/'admission.json')})
    paths=[MATH,RUNTIME,EDGE_QUEUE,*[ROOT/'reports'/name for name,_ in logs],
        ROOT/'scripts/assess_public_pretraining_interim_CPU_v2.py',ROOT/'scripts/python_function_observations_v1.py',
        ROOT/'scripts/code_worker.py',ROOT/'scripts/native_sandbox.py',
        ROOT/'reports/source-function-validation-runtime-CPU-v1/all15-fixed-native-validation-cases.json',
        ROOT/'data/math-reasoning-v2/test.jsonl',ROOT/'data/math-reasoning-v2/manifest.json',
        *python_sources(ROOT,[Path(__file__),ROOT/'tests/test_public_reasoning_stage_v1.py',
            ROOT/'tests/test_public_reasoning_reserved_evaluation_v1.py',ROOT/'tests/test_public_reasoning_warmup_gate_v1.py'])]
    pulses=read(MATH)['pulse_sources'];paths.append(Path(pulses['knowledge']).with_suffix('.audit.json'))
    for name in pulses.values():
        path=Path(name);paths.append(path)
        with path.open() as stream:
            for row in map(json.loads,stream):
                if row.get('image'):paths.append((path.parent/row['image']).resolve())
    files.update({str(path):file_hash(path) for path in set(paths)});guard=FileBindings(files);owner=birth()
    atomic_json(QUEUE,dict(owner=owner,source_files_sha256=files,exact_two_thread_predecessor=edge['owner'],public_pretraining_owner=runtime['owner'],
        initializer='complete unchanged own4447update public-pretrained dense model',CPU_threads_after_verified_release=2,CPU_nice=os.nice(0),
        complete_train_examples=2568,all12tasks_three_source_epochs=True,planned_updates=483,whole_source_presentations=7704,response_EOS_targets=57240,
        reserved_examples=289,reserved_native_action_budgets=[64,256],source_validation_functions=15,source_native_action_budget=1024,
        local_code_tasks=48,local_code_native_action_budget=512,ordinary_Python_programs=24,knowledge=56,math=16,OCR=20,objects=15,photos=32,image_side=448,
        actual_full_size_capacity_and_recovery_still_required=True,minimum_original_writer_and_disk_reserve_bytes=RESERVE,
        maximum_private_state_bytes=MAX_STATE,no_GPU_context_or_lease=True,no_original_queue_or_runtime_source_modified=True,
        original17family_human_RL_and_nine_endpoint_benchmark_scopes_preserved=True,verifier_RL_not_launched_by_this_queue=True,
        production_updates=0,automatic_retry=False,candidate_promoted=False,goal_achieved=False))
    lock=(ROOT/'reports/public-pretraining-interim-CPU.lock').open('a+')
    while True:
        guard.check();phase='waiting for exact existing CPU comparison and full public pretraining'
        if ready(ROOT,edge['owner'],runtime['owner']):
            try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:phase='waiting for released original two-thread CPU lock'
            else:
                stat=dict(line.split()[:2] for line in Path('/sys/fs/cgroup/memory.stat').read_text().splitlines())
                live=sum(int(stat[key]) for key in ['anon','shmem','kernel']);limit=int(Path('/sys/fs/cgroup/memory.max').read_text())
                if shutil.disk_usage(ROOT).free>=RESERVE+2*1024**3 and live+32*1024**3+64*1024**3<=limit:break
                fcntl.flock(lock,fcntl.LOCK_UN);phase='waiting for complete CPU working set and original persistent writer reserves'
        atomic_json(WAIT,dict(owner=owner,phase=phase,production_updates=0,goal_achieved=False));time.sleep(30)
    import copy
    import numpy as np
    import torch
    from slm_lab.async_durable_checkpoint_v1 import AsyncCheckpoint
    from slm_lab.codenet_checkpoint_v1 import atomic_checkpoint,capture_rng
    from slm_lab.CPU_frozen_cast_cache_v1 import CPUFrozenCastCache
    from slm_lab.function_repair_evaluation_CPU_v1 import source_validation,local_suites
    from slm_lab.parallel_function_repair_stage_v1 import value_digest
    from slm_lab.public_pretraining_native_v1 import load
    from slm_lab.public_reasoning_train_data_v1 import PublicReasoningTrainData
    from slm_lab.public_reasoning_stage_v1 import PublicReasoningStage
    from slm_lab.public_reasoning_reserved_evaluation_v1 import PublicReasoningReserved,evaluate
    from slm_lab.own_dense_actor_v1 import OwnDenseActor
    from slm_lab.whole_human_sft_evidence_v1 import tensor_digest
    from assess_public_pretraining_interim_CPU_v2 import ordinary_python
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    if torch.cuda.is_initialized() or len(os.sched_getaffinity(0))<8:raise ValueError('Original eight-core affinity and two CPU threads without CUDA required')
    data=PublicReasoningTrainData(DATA,ADMITTED);reserved=PublicReasoningReserved(DATA,ADMITTED)
    options=dict(learning_rate=1e-4,betas=[.9,.95],weight_decay=.01,gradient_clip=1.,precision='CPU_BF16_cached',epochs=3,width=16,seed=983113)
    entries=data.complete_schedule(epochs=3,width=16,seed=options['seed']);budget=data.budget(entries)
    if any(budget[key]!=value for key,value in dict(updates=483,source_presentations=7704,response_EOS_targets=57240,complete_context_tokens=735570).items()):
        raise ValueError('Complete exact original three-source-epoch budget required')
    files.update({str(BASE):file_hash(BASE),str(BASE.parent/'tokenizer.json'):file_hash(BASE.parent/'tokenizer.json'),str(EDGE/'complete.json'):file_hash(EDGE/'complete.json')})
    guard=FileBindings(files);base,metadata=load(BASE,MATH,'trained',expected_checkpoint_sha256=files[str(BASE)],expected_plan_sha256=files[str(MATH)],device='cpu')
    model=OwnDenseActor(base).eval()
    if (base.counts()['total_unique']!=984399690 or model.counts()['actual_unique']!=999706442
            or metadata['completed_updates']!=4447 or metadata['tokenizer_sha256']!=file_hash(DATA/'tokenizer.json')):
        raise ValueError('Every exact own complete under1B dense/head/actor value and tokenizer required')
    initial=copy.deepcopy(model.parts.state_dict());initial_digest=tensor_digest(model.parts.named_parameters())
    frozen_digest=tensor_digest(base.named_parameters());critic=model.parts['critic'].weight.detach().clone()
    longest=sorted(data.slots,key=lambda p:data.information(p)['complete_context_tokens'],reverse=True)
    capacity_entries=[longest[:16],next(list(entry) for entry in entries if len(entry)==8)]
    OUT.mkdir();OUTPUT.mkdir();shutil.copyfile(DATA/'tokenizer.json',OUTPUT/'tokenizer.json')
    plan=dict(at=datetime.now(timezone.utc).isoformat(),owner=owner,source_files_sha256=files,training_options=options,
        all_original_source_batches=entries,schedule_identity_sha256=data.identity(entries),complete_source_budget=budget,
        actual_capacity_batches=capacity_entries,actual_dense_counts=base.counts(),actual_student_counts=model.counts(),
        initial_actor_parts_sha256=initial_digest,frozen_dense_sha256=frozen_digest,baseline_and_final_full_scope=read(QUEUE),
        no_external_teacher_or_new_human_labels=True,source_reserved_only_not_historical_independence=True,
        no_automatic_model_adoption=True,candidate_promoted=False,goal_achieved=False)
    # The queue is immutable; its current bytes are a scope snapshot, not a new
    # binding to any mutable polling/progress file.
    atomic_json(OUT/'plan.json',plan);plan_sha=file_hash(OUT/'plan.json');cache=CPUFrozenCastCache();cache.attach(model.policy)
    def sources(profile):return dict(run_plan_sha256=plan_sha,profile_sha256=hashlib.sha256(profile.encode()).hexdigest(),own_complete_public_model_sha256=files[str(BASE)])
    def reset(mode,batches):
        model.parts.load_state_dict(initial);model.zero_grad(set_to_none=True)
        if tensor_digest(model.parts.named_parameters())!=initial_digest:raise ValueError('Exact same own starting actor required')
        random.seed(options['seed']);np.random.seed(options['seed']);torch.manual_seed(options['seed'])
        optimizer=torch.optim.AdamW([p for _,p in model.named_selected(('actor',))],lr=options['learning_rate'],betas=tuple(options['betas']),weight_decay=options['weight_decay'],foreach=False)
        return PublicReasoningStage(model,optimizer,data,batches,'whole_answer_warmup',options,sources('warmup'),mode=mode,cache=cache)
    def intact():
        cache.assert_current()
        if tensor_digest(base.named_parameters())!=frozen_digest or not torch.equal(model.parts['critic'].weight,critic) or torch.cuda.is_initialized():
            raise ValueError('Every original base/head/unused critic value and GPU isolation must remain exact')
    def cold(path,profile):
        other=OwnDenseActor(base,**model.settings).eval();other.import_export(torch.load(path,map_location='cpu',weights_only=True,mmap=True),expected_sources=sources(profile))
        first=len(cache.patched);cache.attach(other.policy);model.eval()
        try:
            ids=torch.tensor([data.sequence(longest[0])[0]])
            with torch.no_grad(),torch.autocast('cpu',dtype=torch.bfloat16):wanted=model(input_ids=ids)['logits'];actual=other(input_ids=ids)['logits']
            if not torch.equal(wanted,actual) or tensor_digest(model.parts.named_parameters())!=tensor_digest(other.parts.named_parameters()):raise ValueError('Actual whole longest-source cold logits and actor values must be bit exact')
            return dict(complete=True,actual_complete_source_context_tokens=ids.shape[1],CPU_BF16_logits_and_all_actor_values_bit_exact=True)
        finally:
            for module,original in cache.patched[first:]:module.forward=original
            del cache.patched[first:]
    try:
        atomic_checkpoint(OUTPUT/'baseline-actor.pt',model.export(sources('baseline')),MAX_ACTOR,RESERVE)
        stage=reset('capacity',capacity_entries);atomic_json(OUT/'progress.json',dict(phase='actual full-size capacity and recovery',goal_achieved=False))
        started=time.monotonic();stage.update();first_seconds=time.monotonic()-started
        first=OUTPUT/'capacity-after1.pt';atomic_checkpoint(first,stage.state(),MAX_STATE,RESERVE)
        wanted=stage.update();parts=tensor_digest(model.parts.named_parameters());adam=value_digest(stage.optimizer.state_dict());rng=value_digest(capture_rng('cpu'))
        saved=torch.load(first,map_location='cpu',weights_only=True,mmap=True)
        for kind in ['source_mask','RNG','options']:
            bad=copy.deepcopy(saved)
            if kind=='source_mask':bad['core']['metrics'][0]['complete_sources'][0]['response_EOS_targets']-=1
            elif kind=='RNG':bad['core']['rng']['torch']=torch.zeros(1,dtype=torch.uint8)
            else:bad['stage_binding']['options']['gradient_clip']=.5
            try:stage.resume(bad)
            except ValueError:pass
            else:raise ValueError('Corrupt source/RNG/options must reject before mutation')
            if tensor_digest(model.parts.named_parameters())!=parts or value_digest(stage.optimizer.state_dict())!=adam or value_digest(capture_rng('cpu'))!=rng:raise ValueError('Rejected recovery changed weights, Adam or private RNG')
        stage.resume(saved)
        if stage.update()!=wanted or tensor_digest(model.parts.named_parameters())!=parts or value_digest(stage.optimizer.state_dict())!=adam or value_digest(capture_rng('cpu'))!=rng:raise ValueError('Actual full-size next update must recover bit exactly')
        atomic_checkpoint(OUTPUT/'capacity-actor.pt',model.export(sources('warmup')),MAX_ACTOR,RESERVE)
        cold_proof=cold(OUTPUT/'capacity-actor.pt','warmup');intact();guard.check(full=True)
        if len(stage.optimizer.state)!=561:raise ValueError('All561actual selected actor Adam leaves required')
        atomic_json(OUT/'capacity-complete.json',dict(complete=True,actual_batches=capacity_entries,logical_updates=2,physical_optimizer_calls=3,all_selected_Adam_leaves=561,
            first_update_seconds=first_seconds,actual_next_update_actor_Adam_RNG_and_metrics_bit_exact=True,corrupt_source_RNG_options_rejected_before_mutation=True,
            cold=cold_proof,training_only_cast_inventory=cache.inventory(),production_updates=0,goal_achieved=False))
        del stage,saved,bad;gc.collect();stage=reset('train',entries);writer=AsyncCheckpoint()
        try:
            with (OUT/'actual-training-metrics.jsonl').open('x') as journal:
                for entry in entries:
                    guard.check();writer.check();started=time.monotonic();row=stage.update();seconds=time.monotonic()-started
                    journal.write(json.dumps(row)+'\n');journal.flush()
                    if stage.completed==1 or stage.completed%8==0 or stage.completed==len(entries):
                        os.fsync(journal.fileno());guard.check(full=True);writer.submit(OUTPUT/'resume.pt',stage.state(),MAX_STATE,RESERVE)
                    atomic_json(OUT/'progress.json',dict(phase='training',actual_complete_updates=stage.completed,planned_updates=483,last_original_token_mean_NLL=row['original_token_mean_NLL'],last_update_seconds=seconds,actual_full_size_capacity_passed=True,goal_achieved=False))
            writer.flush()
        finally:writer.close()
        stage.validate(stage.metrics);intact()
        if stage.completed!=483 or sum(row['response_EOS_targets'] for row in stage.metrics)!=57240:raise ValueError('Every original three-epoch source target must be consumed')
        atomic_checkpoint(OUTPUT/'warmup-actor.pt',model.export(sources('warmup')),MAX_ACTOR,RESERVE)
        training=dict(complete=True,actual_updates=483,actual_budget=budget,cold=cold(OUTPUT/'warmup-actor.pt','warmup'),checkpoint_sha256=file_hash(OUTPUT/'warmup-actor.pt'),private_resume_sha256=file_hash(OUTPUT/'resume.pt'))
        atomic_json(OUT/'training-complete.json',training);del stage;gc.collect();evaluations={}
        for profile in ['baseline','warmup']:
            path=OUTPUT/(profile+'-actor.pt');model.import_export(torch.load(path,map_location='cpu',weights_only=True,mmap=True),expected_sources=sources(profile));model.eval()
            folder=OUT/('evaluation-'+profile);folder.mkdir();atomic_json(OUT/'progress.json',dict(phase='full native evaluation',profile=profile,all483source_updates_complete=True,goal_achieved=False))
            reasoning=evaluate(model.policy,path,folder/'reserved-reasoning',reserved,checked=guard.check)
            valid=source_validation(model.policy,path,folder/'source-validation',ROOT)
            local=local_suites(model.policy,path,metadata,folder/'local',ROOT,pulses)
            ordinary=ordinary_python(local,folder/'ordinary-Python.json')
            evaluations[profile]=dict(reserved_reasoning_correct=reasoning['correct'],reserved_reasoning_examples=289,source_validation_correct=valid['all_original_examples_passed'],source_validation_functions=15,
                local_code_correct=sum(row['candidate'].get('functional_passed',False) for frame in ['completion','instruction'] for row in local['code-'+frame]['results']),local_code_tasks=48,
                ordinary_Python_correct=ordinary['functional_passed'],ordinary_Python_programs=24,knowledge_generated_correct=sum(row['generated_correct'] for row in local['knowledge']['predictions']),knowledge_questions=56,
                math_correct=local['math']['correct'],math_examples=16,all_original_OCR_object_photo_and_efficiency_checks_complete=True)
            intact();guard.check(full=True);atomic_json(folder/'complete.json',dict(complete=True,**evaluations[profile],candidate_promoted=False,goal_achieved=False))
        atomic_json(OUT/'complete.json',dict(complete=True,at=datetime.now(timezone.utc).isoformat(),owner=owner,source_files_sha256=files,training=training,matched_evaluations=evaluations,
            all2568train_sources_three_epochs_and_all289reserved_source_code_and_broad_native_checks_complete=True,
            original_full_public_human_RL_and_nine_endpoint_benchmark_scopes_unchanged=True,no_automatic_model_adoption=True,candidate_promoted=False,goal_achieved=False))
    finally:cache.close()


if __name__=='__main__':
    try:main()
    except BaseException as error:
        target=OUT/'failure.json' if OUT.exists() else ROOT/'reports/public-reasoning-warmup-CPU-v2-parent-failure.json'
        atomic_json(target,dict(at=datetime.now(timezone.utc).isoformat(),owner=birth(),error=repr(error),traceback=traceback.format_exc(),automatic_retry=False,original_training_and_full_benchmarks_unchanged=True,goal_achieved=False))
        raise
