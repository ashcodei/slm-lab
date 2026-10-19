"""Owned, gated four-L40 matched repair study after complete public pretraining.

All three arms start from the same own complete public-data checkpoint. This
optional research branch never changes the canonical human/RL initializer.
No GPU is claimed before the tested ten-controller handoff is admitted.
"""
import argparse
from datetime import datetime,timedelta,timezone
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time
import traceback

from slm_lab.codenet_checkpoint_v1 import FileBindings,atomic_checkpoint
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.function_repair_GPU_contract_v2 import (PROFILES,REPORT,QUEUE,OUTPUT,PUBLIC,RUNTIME,MODEL,NATIVE_COHORT,
    capacity_entries,handoff_admitted,public_ready,read,sources,training_options)
from slm_lab.local_source_closure_v2 import python_sources
import own_training_gpu_lease_v2 as owner

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/REPORT;DEST=ROOT/OUTPUT;MATH_PLAN=ROOT/PUBLIC/'plan.json'
PREP=ROOT/'reports/full-function-repair-comparison-sources-CPU-v1'
MAX_STATE=512*1024**2;MAX_ACTOR=96*1024**2;RESERVE=26*1024**3
TESTS={'parallel-function-repair-v1-final-tests':7,'parallel-function-repair-stage-v1-final-tests':4,
    'parallel-repair-durable-state-v1-tests':5,'function-repair-evaluation-GPU-v1-tests':3,
    'function-repair-GPU-controller-v1-tests':7}


def configure():
    owner.REPORTS=OUT;owner.resources.REPORTS=OUT


def factory(plan,device):
    """Actor initialization stays on CPU; CUDA uses a different RNG algorithm."""
    from slm_lab.public_pretraining_native_v1 import load
    from slm_lab.own_dense_actor_v1 import OwnDenseActor
    from slm_lab.whole_human_sft_evidence_v1 import tensor_digest
    base,metadata=load(ROOT/MODEL,MATH_PLAN,'trained',expected_checkpoint_sha256=plan['initializer']['checkpoint_sha256'],
        expected_plan_sha256=plan['initializer']['plan_sha256'],device='cpu')
    model=OwnDenseActor(base).eval()
    if (metadata['completed_updates']!=4447 or model.counts()!=plan['actual_student_counts']
            or tensor_digest(model.parts.named_parameters())!=plan['initial_actor_parts_sha256']
            or tensor_digest(base.named_parameters())!=plan['frozen_dense_sha256']):
        raise ValueError('Every fresh arm must start at the identical own complete public model and CPU-initialized actor')
    model=model.to(device);model.assert_shared();return model,metadata


def gpu_setup(index,visible):
    import torch
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=visible:raise ValueError('Exact declared physical GPU visibility required')
    torch.set_num_threads(1);torch.set_num_interop_threads(1);device=torch.device('cuda',index);torch.cuda.set_device(device)
    torch.cuda.set_per_process_memory_fraction(.85,device);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if torch.cuda.get_device_name(device)!='NVIDIA L40':raise ValueError('Actual original local L40 required')
    return device


def cold_export(stage,plan,path):
    """Shared frozen own base, fresh actor import; actual text/photo/word outputs."""
    import torch
    from slm_lab.own_dense_actor_v1 import OwnDenseActor
    from slm_lab.parallel_function_repair_stage_v1 import value_digest
    from slm_lab.whole_human_sft_evidence_v1 import tensor_digest
    error=None;proof=None;model=stage.model;device=stage.device
    try:
        if stage.rank==0:atomic_checkpoint(path,model.export(stage.sources),MAX_ACTOR,RESERVE)
    except Exception as exc:error=repr(exc)
    stage.coordinated(error,'native actor export')
    error=None
    try:
        bundle=ROOT/plan['cold_inputs']['path']
        if file_hash(bundle)!=plan['cold_inputs']['sha256']:raise ValueError('Actual admitted cold inputs changed')
        inputs=torch.load(bundle,map_location='cpu',weights_only=True,mmap=True)
        if value_digest(inputs)!=plan['cold_inputs']['semantic_sha256']:raise ValueError('All native cold tensor bytes must match')
        batches={n:{k:v.to(device) for k,v in b.items()} for n,b in inputs.items()}
        if batches['text']['input_ids'].shape[1]!=4096 or batches['photo']['pixel_values'].shape[-2:]!=(448,448):
            raise ValueError('Complete4096text and448photo cold probes required')
        def outputs(actor):
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                return [actor.policy(**batches[n])['logits'].cpu() for n in ['text','photo']]+[
                    actor.base.word_logits(batches['word']['word_pixels']).cpu()]
        model.eval();before=outputs(model);payload=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
        other=OwnDenseActor(model.base,**model.settings).eval();other.import_export(payload,expected_sources=stage.sources)
        after=outputs(other)
        if (any(not torch.equal(a,b) for a,b in zip(before,after))
                or tensor_digest(model.parts.named_parameters())!=tensor_digest(other.parts.named_parameters())
                or tensor_digest(model.base.named_parameters())!=stage.base_digest):
            raise ValueError('Every exported actor/unused critic and4096text448photo/real-word output must cold-load exactly')
        proof=dict(complete=True,every_actor_and_unused_critic_value_exact=True,
            actual4096text448photo_real_word_outputs_bit_exact=True,shared_unchanged_frozen_base=True,
            transient_additional_actor_and_critic_values=15306752,checkpoint_sha256=file_hash(path))
        del other,batches,before,after,payload;torch.cuda.empty_cache()
    except Exception as exc:error=repr(exc)
    stage.coordinated(error,'actual native cold import/forward')
    return proof


def distributed(profile,mode):
    if int(os.environ['WORLD_SIZE'])!=4 or int(os.environ['LOCAL_RANK'])!=int(os.environ['RANK']):
        raise ValueError('Actual four local replicas required')
    rank=int(os.environ['RANK']);plan=owner.registered(mode+'-'+profile+'-rank-'+str(rank))
    import random
    import numpy as np
    import torch
    from torch import distributed as dist
    from slm_lab.function_repair_curriculum_v1 import FunctionRepairCurriculum
    from slm_lab.full_function_repair_stage_v1 import CodeWeights
    from slm_lab.parallel_function_repair_stage_v1 import ParallelFunctionRepairStage
    from slm_lab.parallel_function_repair_v1 import local_backward
    from slm_lab.parallel_repair_experiment_v1 import prove_next_update,train
    from slm_lab.whole_human_sft_evidence_v1 import tensor_digest
    device=gpu_setup(rank,'0,1,2,3');dist.init_process_group('gloo',timeout=timedelta(minutes=45))
    try:
        guard=FileBindings(plan['source_files_sha256']);guard.check(full=True);model,_=factory(plan,device)
        data=FunctionRepairCurriculum(ROOT,'optimization');weights=CodeWeights(data);options=plan['training_options']
        entries=plan['capacity_complete_slot_batches'] if mode=='capacity' else plan['all_complete_slot_batches']
        optimizer=torch.optim.AdamW([p for _,p in model.named_selected(('actor',))],lr=options['learning_rate'],
            betas=tuple(options['betas']),weight_decay=options['weight_decay'],foreach=False)
        random.seed(options['seed']+rank);np.random.seed(options['seed']+rank)
        torch.random.default_generator.manual_seed(options['seed']+rank);torch.cuda.manual_seed(options['seed']+rank)
        stage=ParallelFunctionRepairStage(model,optimizer,data,entries,profile,options,sources(plan,file_hash(OUT/'plan.json'),profile),
            weights,rank=rank,world=4,mode=mode,precision='bf16')
        folder=OUT/(mode+'-'+profile);folder.mkdir(exist_ok=True);state_dir=DEST/(mode+'-'+profile)
        start=time.monotonic()
        if mode=='capacity':
            proof=prove_next_update(stage,state_dir,maximum_bytes=MAX_STATE,minimum_free_bytes=RESERVE)
            tail=stage.update()
            if len(entries[-1])!=1 or tail['actual_rank_source_shards']!=[entries[-1],[],[],[]]:
                raise ValueError('The actual single-source global tail must not duplicate examples on empty ranks')
            if len(optimizer.state)!=561:raise ValueError('All561 actual selected actor Adam states required')
            before=tensor_digest(model.parts.named_parameters());probes=[]
            arm='plain_repeat' if profile=='plain_repeat' else 'semantic_repair'
            longest=max(range(len(data.slots)),key=lambda p:len(data.native(arm,p)[0]))
            for slot in [longest,entries[-1][0]]:
                error=None
                try:
                    optimizer.zero_grad(set_to_none=True)
                    observation=local_backward(model,data,[slot],profile,options,weights,rank=0,world=1,
                        amp=lambda:torch.autocast('cuda',dtype=torch.bfloat16))
                    if (any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in stage.parameters)
                            or any(p.grad is not None for p in stage.fixed)):
                        raise ValueError('Actual longest/tail backward requires every/only selected finite gradient')
                    probes.append(observation);optimizer.zero_grad(set_to_none=True)
                except Exception as exc:error=repr(exc)
                stage.coordinated(error,'actual individual longest/tail source backward')
            if tensor_digest(model.parts.named_parameters())!=before:raise ValueError('Backward-only capacity must not update actor values')
            # Snapshot also proves unchanged full base and identical replicas/all Adam states.
            stage.snapshot();cold=cold_export(stage,plan,DEST/(profile+'-capacity-actor.pt'))
            result=dict(complete=True,physical_GPU=rank,actual_counts=model.counts(),actual_Adam_states=len(optimizer.state),
                logical_updates=3,physical_optimizer_calls=4,recovery=proof,true_global_tail_record=tail,
                individual_full_source_backward_probes=probes,cold_export_import=cold,
                production_updates=0,peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(device),seconds=time.monotonic()-start,goal_achieved=False)
            atomic_json(folder/('rank-'+str(rank)+'.json'),result)
        else:
            def preflight():
                error=None
                try:owner.owned(plan);guard.check()
                except Exception as exc:error=repr(exc)
                stage.coordinated(error,'owned source preflight')
            def progress(row,elapsed):
                atomic_json(OUT/'progress.json',dict(phase='training',profile=profile,actual_complete_updates=stage.completed,
                    planned_updates_per_profile=306,last_original_target_mean_NLL=row['original_token_mean_NLL'],
                    last_update_seconds=elapsed,all_actual_GPU_capacity_profiles_passed=True,goal_achieved=False))
                print(json.dumps(dict(profile=profile,step=stage.completed,planned=306,NLL=row['original_token_mean_NLL'],seconds=elapsed)),flush=True)
            result=train(stage,state_dir,folder/'training-metrics.jsonl',maximum_bytes=MAX_STATE,minimum_free_bytes=RESERVE,
                every=options['checkpoint_every'],preflight=preflight,progress=progress)
            slots=[p for batch in entries for p in batch]
            if (len(entries)!=306 or len(slots)!=4851 or sum(r['response_EOS_targets'] for r in stage.metrics)!=2287110
                    or any(slots.count(p)!=3 for p in range(len(data.slots)))):
                raise ValueError('All826original and791second-view presentations and every gold target required')
            cold=cold_export(stage,plan,DEST/(profile+'-actor.pt'))
            if rank==0:
                guard.check(full=True);atomic_json(folder/'complete.json',dict(complete=True,**result,
                    actual_complete_presentations=4851,original_response_EOS_targets=2287110,
                    original_source_validation_gradients=0,cold_export_import=cold,candidate_promoted=False,goal_achieved=False))
    finally:dist.destroy_process_group()


def assess(profile,gpu,suite):
    plan=owner.registered('assess-'+profile+'-'+suite+'-'+str(gpu));device=gpu_setup(0,str(gpu))
    import torch
    from slm_lab.function_repair_evaluation_GPU_v1 import source_shard,native_local
    FileBindings(plan['source_files_sha256']).check(full=True);model,metadata=factory(plan,device)
    inputs=read(OUT/'native-assessment-inputs.json');checkpoint=DEST/(profile+'-actor.pt')
    if file_hash(checkpoint)!=inputs[profile]['checkpoint_sha256']:raise ValueError('Exact declared frozen assessment actor required')
    model.import_export(torch.load(checkpoint,map_location='cpu',weights_only=True,mmap=True),
        expected_sources=sources(plan,file_hash(OUT/'plan.json'),profile));model.eval()
    folder=OUT/profile;folder.mkdir(exist_ok=True)
    if suite=='source':source_shard(model.policy,checkpoint,ROOT/NATIVE_COHORT,folder,rank=gpu,world=4,device=str(device))
    else:native_local(model.policy,metadata,checkpoint,folder,ROOT,plan['pulse_sources'],rank=gpu,suite=suite,device=str(device))


def prepare_sources():
    from slm_lab.function_repair_curriculum_v1 import FunctionRepairCurriculum,schedule
    from slm_lab.full_function_repair_stage_v1 import CodeWeights
    from slm_lab.function_repair_evaluation_GPU_v1 import validate_source_cohort
    data=FunctionRepairCurriculum(ROOT,'source_preparation');weights=CodeWeights(data);prepared=read(PREP/'plan.json')
    entries=schedule(data.lengths,epochs=3,width=16,seed=979113)
    if (entries!=prepared['actual_complete_slot_batches'] or data.identity(entries)!=prepared['actual_schedule_identity_sha256']
            or len(data.plain)!=826 or len(data.alternate)!=791):
        raise ValueError('All prepared complete original source batches must remain identical')
    capacity_entries(entries);validate_source_cohort(ROOT/NATIVE_COHORT)
    pulses=read(MATH_PLAN)['pulse_sources'];paths=[MATH_PLAN,ROOT/RUNTIME/'plan.json',PREP/'plan.json',PREP/'complete.json',
        ROOT/NATIVE_COHORT,ROOT/'reports/parallel-repair-recovery-CPU-v1/complete.json',
        ROOT/'data/math-reasoning-v2/test.jsonl',ROOT/'data/math-reasoning-v2/manifest.json',
        Path(pulses['knowledge']).with_suffix('.audit.json'),ROOT/'scripts/documented_doctest_raw_harness_v1.py',
        ROOT/'scripts/python_program_sandbox_v1.py',ROOT/'scripts/code_worker.py',ROOT/'scripts/native_sandbox.py']
    for name,count in TESTS.items():
        path=ROOT/'reports'/(name+'.log')
        if str(count)+' passed' not in path.read_text():raise ValueError('Actual experiment, source, observer and recovery controls must pass: '+name)
        paths.append(path)
    for name in pulses.values():
        path=Path(name);paths.append(path)
        for row in map(json.loads,path.read_text().splitlines()):
            if row.get('image'):paths.append((path.parent/row['image']).resolve())
    paths.append(ROOT/'reports/assessment-recovery-lineage-v1-tests.log')
    paths.extend(python_sources(ROOT,[Path(__file__),ROOT/'tests/test_function_repair_GPU_controller_v1.py',ROOT/'tests/test_assessment_recovery_lineage_v1.py']))
    files={**data.source_files_sha256,**weights.source_files_sha256,**owner.lease_sources(),**{str(p):file_hash(p) for p in set(paths)}}
    return data,entries,pulses,files


def parent():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or OUT.exists() or DEST.exists() or (ROOT/QUEUE).exists():
        raise ValueError('Fresh CUDA-hidden controller, queue and output paths required; no automatic overwrite/retry')
    data,entries,pulses,files=prepare_sources();guard=FileBindings(files);me=owner.resources.identity(os.getpid())
    predecessor=read(ROOT/'reports/public-pretraining-recovery-v1-queue.json')['owner'];OUT.mkdir()
    atomic_json(ROOT/QUEUE,dict(owner=me,source_files_sha256=files,exact_public_predecessor=predecessor,
        status='waiting for tested ten-controller handoff and full4447step public predecessor',GPU_claimed=False,
        profiles=PROFILES,actual_updates_per_profile=306,all_original_source_budgets_and_native_assessments_retained=True,
        canonical_human_and_RL_initializer_unchanged=True,automatic_retry=False,candidate_promoted=False,goal_achieved=False))
    while True:
        guard.check()
        if handoff_admitted(ROOT,me) and public_ready(ROOT,predecessor):break
        time.sleep(5)
    import torch
    from slm_lab.own_dense_actor_v1 import OwnDenseActor
    from slm_lab.public_pretraining_native_v1 import load
    from slm_lab.public_pretraining_data_v2 import PublicPretrainingData
    from slm_lab.parallel_function_repair_stage_v1 import value_digest
    from slm_lab.whole_human_sft_evidence_v1 import tensor_digest
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    model_sha=file_hash(ROOT/MODEL);math_sha=file_hash(MATH_PLAN)
    base,meta=load(ROOT/MODEL,MATH_PLAN,'trained',expected_checkpoint_sha256=model_sha,expected_plan_sha256=math_sha,device='cpu')
    model=OwnDenseActor(base).eval()
    if (meta['completed_updates']!=4447 or model.counts()['actual_unique']!=999706442 or torch.cuda.is_initialized()):
        raise ValueError('Own complete under1B public checkpoint and CUDA-hidden admission required')
    if shutil.disk_usage(ROOT).free<RESERVE+5*1024**3:raise ValueError('All bounded shared snapshots and native exports must fit above reserve')
    DEST.mkdir();shutil.copyfile((ROOT/MODEL).parent/'tokenizer.json',DEST/'tokenizer.json')
    # Select original training-side native capacity inputs, never held cases.
    public=read(MATH_PLAN);provider=PublicPretrainingData(ROOT,model.config)
    raw_name,raw_positions=public['capacity_global_batches'][0];photo_name,photo_positions=public['capacity_global_batches'][-2]
    word_position=next(i for i,r in enumerate(provider.retention.replay.words.rows) if r['byte_family']=='real_word')
    provider_guard=FileBindings(provider.source_files_sha256);provider_guard.check(full=True)
    raw=provider.batch(raw_name,[raw_positions[0]],'cpu');photo=provider.batch(photo_name,[photo_positions[0]],'cpu')
    word=provider.batch('words',[word_position],'cpu')
    native={name:{k:b[k] for k in ['input_ids','attention_mask','pixel_values'] if k in b} for name,b in [('text',raw),('photo',photo)]}
    native['word']={'word_pixels':word['word_pixels']};cold_path=DEST/'original-native-capacity-inputs.pt'
    provider_guard.check();atomic_checkpoint(cold_path,native,16*1024**2,RESERVE)
    native_sources=OUT/'cold-input-source-readback.json'
    atomic_json(native_sources,dict(complete=True,source_files_sha256=provider.source_files_sha256,
        all_original_provider_sources_checked_before_selection=True,selected_tensor_bundle_sha256=file_hash(cold_path),
        selected_tensor_semantic_sha256=value_digest(native),training_or_quality_updates=0,goal_achieved=False))
    cold_inputs=dict(path=str(cold_path.relative_to(ROOT)),sha256=file_hash(cold_path),semantic_sha256=value_digest(native),
        original_raw_family=raw_name,original_raw_position=raw_positions[0],original_photo_family=photo_name,
        original_photo_position=photo_positions[0],original_real_word_position=word_position,
        actual_provider_and_all_public_source_inputs_verified_once=True,source_readback_receipt_sha256=file_hash(native_sources),
        no_optimization_from_cold_inference_inputs=True)
    for path in [ROOT/MODEL,(ROOT/MODEL).parent/'tokenizer.json',DEST/'tokenizer.json',cold_path,native_sources,
            ROOT/'reports/assessment-recovery-lineage-v1/admission.json',ROOT/RUNTIME/'complete.json',
            ROOT/RUNTIME/'restored.json',ROOT/PUBLIC/'training-complete.json',ROOT/'reports/public-pretraining-complete-epoch-value-readback-CPU-v2/complete.json',ROOT/'reports/public-pretraining-recovery-v1/failure.json',ROOT/'reports/public-pretraining-recovery-v1/restored.json']:
        files[str(path)]=file_hash(path)
    guard=FileBindings(files);guard.check(full=True)
    plan=dict(at=datetime.now(timezone.utc).isoformat(),owner=me,waiter=owner.WAITER,
        terminal_waiter_receipt_sha256=owner.terminal_receipt_sha256(),source_files_sha256=files,
        expires_at_unix=time.time()+7*86400,initializer=dict(checkpoint=str(ROOT/MODEL),checkpoint_sha256=model_sha,plan_sha256=math_sha),
        actual_student_counts=model.counts(),initial_actor_parts_sha256=tensor_digest(model.parts.named_parameters()),
        frozen_dense_sha256=tensor_digest(base.named_parameters()),training_options=training_options(),profiles=PROFILES,
        all_complete_slot_batches=entries,capacity_complete_slot_batches=capacity_entries(entries),
        source_schedule_identity_sha256=data.identity(entries),actual_updates_per_profile=306,
        actual_matched_source_budgets={arm:data.budget(arm,entries) for arm in ['plain_repeat','semantic_repair']},
        pulse_sources=pulses,cold_inputs=cold_inputs,every_original_target_and_true_EOS_kept=True,code_weight_only_third_arm=4.,
        fixed_assessments=dict(source_validation=15,source_native_budget=1024,local_code=48,code_native_budget=512,
            knowledge=56,math=16,OCR=20,objects=15,photos=32,image_size=448),
        all_arm_training_finishes_before_any_baseline_or_final_validation=True,
        all_actual_full_size_capacity_profiles_must_pass_before_training=True,
        source_public_examples_visible_in_prompt=True,not_human_RLHF_or_execution_RL=True,
        canonical_human_RL_initializer_unchanged=True,automatic_retry=False,automatic_promotion=False,goal_achieved=False)
    with owner.LEASE.open('a+') as lease:
        fcntl.flock(lease.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB);owner.idle_original()
        used=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).splitlines()
        if len(used)!=4 or any(int(x)>128 for x in used):raise ValueError('All four original L40s must be idle before claiming this study')
        atomic_json(OUT/'plan.json',plan);atomic_json(OUT/'children.json',[])
        driver=str(Path(__file__).resolve());watcher=None;success=False
        def interrupted(signum,frame):raise KeyboardInterrupt('Owned GPU study interrupted')
        signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
        try:
            atomic_checkpoint(DEST/'baseline-actor.pt',model.export(sources(plan,file_hash(OUT/'plan.json'),'baseline')),MAX_ACTOR,RESERVE)
            del model,base,provider,raw,photo,word,native,data
            watcher=subprocess.Popen([str(ROOT/'.venv/bin/python'),'-u',driver,'--watchdog'],cwd=ROOT,
                stdout=(OUT/'watchdog.log').open('x'),stderr=subprocess.STDOUT,start_new_session=True,
                env={**os.environ,'CUDA_VISIBLE_DEVICES':''})
            def run(profile,mode):
                child=owner.launch(plan,['-u','-m','torch.distributed.run','--standalone','--nproc_per_node=4',driver,
                    '--distributed',mode,'--profile',profile],mode+'-'+profile+'-launcher','0,1,2,3')
                owner.wait_children(plan,[child])
            for profile in PROFILES:
                run(profile,'capacity')
                proofs=[read(OUT/('capacity-'+profile)/('rank-'+str(r)+'.json')) for r in range(4)]
                for rank,p in enumerate(proofs):
                    if (p['physical_GPU']!=rank or p['actual_counts']!=plan['actual_student_counts'] or p['actual_Adam_states']!=561
                            or p['logical_updates']!=3 or p['physical_optimizer_calls']!=4
                            or p['recovery']['exact_next_draws_global_update_actor_Adam_and_private_RNG'] is not True
                            or p['recovery']['invalid_source_and_private_RNG_rejected_before_mutation'] is not True
                            or p['recovery']['wrong_valid_peer_RNG_rejected_before_mutation'] is not True
                            or p['cold_export_import']['actual4096text448photo_real_word_outputs_bit_exact'] is not True):
                        raise ValueError('Every actual full-size four-GPU profile must pass source/gradient/recovery/cold gates')
            atomic_json(OUT/'capacity-complete.json',dict(complete=True,all_profiles=PROFILES,actual_original_L40s=4,
                actual_under1B_actor_and_all561Adam_leaves=True,all_actual_recovery_and_cold_gates_passed=True,production_updates=0,goal_achieved=False))
            for profile in PROFILES:run(profile,'train')
            training={p:read(OUT/('train-'+p)/'complete.json') for p in PROFILES}
            atomic_json(OUT/'all-arms-training-complete.json',dict(complete=True,profiles=training,all_original_source_budgets_preserved=True,goal_achieved=False))
            profiles=['baseline',*PROFILES]
            atomic_json(OUT/'native-assessment-inputs.json',{p:dict(checkpoint_sha256=file_hash(DEST/(p+'-actor.pt'))) for p in profiles})
            from slm_lab.function_repair_evaluation_GPU_v1 import reconcile_source,reconcile_local
            for profile in profiles:
                for suite in ['source','code','broad']:
                    children=[owner.launch(plan,['-u',driver,'--assess',profile,'--suite',suite,'--gpu',str(gpu)],
                        'assess-'+profile+'-'+suite+'-'+str(gpu),str(gpu)) for gpu in range(4)]
                    owner.wait_children(plan,children)
                folder=OUT/profile
                reconcile_source([folder/('source-validation-shard-'+str(r)+'.json') for r in range(4)],
                    ROOT/NATIVE_COHORT,DEST/(profile+'-actor.pt'),folder/'source-validation.json')
                reconcile_local(folder)
            guard.check(full=True);success=True
        except BaseException as error:
            atomic_json(OUT/'failure.json',dict(error=repr(error),traceback=traceback.format_exc(),automatic_retry=False,goal_achieved=False));raise
        finally:
            owner.restore_owned(plan,'Optional matched study ended; all exact registered GPU actors terminal')
            if watcher is not None:watcher.wait(timeout=20)
        if success:atomic_json(OUT/'complete.json',dict(complete=True,all_actual_GPU_capacity_and_three_matched_arms_complete=True,
            all_fifteen_source_and_full_local_text_image_code_assessments_complete=True,
            canonical_full_human_RL_and_nine_endpoint_assessment_pipeline_preserved=True,
            candidate_promoted=False,goal_achieved=False))


if __name__=='__main__':
    configure();parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--watchdog',action='store_true');parser.add_argument('--profile',choices=PROFILES)
    parser.add_argument('--distributed',choices=['capacity','train']);parser.add_argument('--assess',choices=['baseline',*PROFILES])
    parser.add_argument('--suite',choices=['source','code','broad']);parser.add_argument('--gpu',type=int,choices=range(4));args=parser.parse_args()
    if args.watchdog:owner.watchdog()
    elif args.distributed:distributed(args.profile,args.distributed)
    elif args.assess:assess(args.assess,args.gpu,args.suite)
    else:
        try:parent()
        except BaseException as error:
            atomic_json(ROOT/'reports/full-function-repair-comparison-GPU-v1-parent-failure.json',dict(error=repr(error),
                traceback=traceback.format_exc(),automatic_retry=False,goal_achieved=False));raise
