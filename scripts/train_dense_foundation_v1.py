"""Owned full dense-language training on every admitted book/code/human source.

This is a foundation-language comparison, not a final multimodal or human-RLHF
policy. Each exported student includes all heads and has984,399,690parameters.
KL arms temporarily hold our separate existing986Mmodel as a frozen training
reference; no external teacher weights or reference model enter the export.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time
from types import SimpleNamespace

import train_whole_human_SFT_v1 as legacy
from slm_lab.codenet_checkpoint_v1 import FileBindings, atomic_checkpoint
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json

owner = legacy.owner
ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT/'reports/dense-foundation-v1'
OUTPUT = ROOT/'checkpoints/dense-foundation-v1'
QUEUE = ROOT/'reports/dense-foundation-v1-queue.json'
PREVIOUS = ROOT/'reports/shared-words-v1'
PREVIOUS_QUEUE = ROOT/'reports/shared-words-v1-queue.json'
CPU_PROOF = ROOT/'reports/dense-foundation-full-CPU-v1'
MAX_STATE = 16*1024**3
MAX_MODEL = 5*1024**3
RESERVE = 26*1024**3


def configure():
    owner.REPORTS = REPORTS; owner.resources.REPORTS = REPORTS
    legacy.REPORTS = REPORTS; legacy.OUTPUT = OUTPUT


def previous_ended(identity):
    if owner.resources.same(identity): return False
    children = PREVIOUS/'children.json'
    if children.exists() and any(owner.resources.same(a) for a in json.loads(children.read_text())):
        return False
    if (not all((PREVIOUS/n).exists() for n in ('restored.json','complete.json','independent-comparison.json'))
            or (PREVIOUS/'failure.json').exists()):
        raise ValueError('Previous exact word owner/workers must finish successfully and restore original work')
    return True


def require_idle_devices():
    owner.idle_original()
    used = subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).splitlines()
    if len(used) != 4 or any(int(v.strip()) > 128 for v in used):
        raise ValueError('All four original local L40s must be idle')
    if shutil.disk_usage(ROOT).free < RESERVE+16*MAX_STATE:
        raise ValueError('Full four-arm dense states, capacity states and atomic writers reserve required')


def fixed_digest(model):
    from slm_lab.whole_human_sft_evidence_v1 import tensor_digest
    selected = {id(p) for _, p in model.language_named_parameters()}
    return tensor_digest((n,p) for n,p in model.named_parameters() if id(p) not in selected)


def signature(parameters):
    return [(id(p),p._version,p.data_ptr(),p.requires_grad) for p in parameters]


def exposure(data,setting):
    from slm_lab.codenet_stage_v1 import EpochOrder
    from slm_lab.dense_foundation_evidence_v1 import positions
    order=EpochOrder(len(data),setting['seed']);seen=[]
    for step in range(setting['steps']):seen.extend(positions(data,setting,order,step))
    if len(seen)!=len(data)*setting['epochs']:
        raise ValueError('Complete source epochs and unrepeated final partial update required')
    for epoch in range(setting['epochs']):
        if sorted(seen[epoch*len(data):(epoch+1)*len(data)])!=list(range(len(data))):
            raise ValueError('Every original source exactly once per declared pass required')
    return dict(complete_source_budget=data.budget(),actual_unique_sources=len(set(seen)),
        actual_exposures=len(seen),complete_passes=setting['epochs'],repeated_rounding_exposures=0,
        actual_last_update_sources=len(positions(data,setting,order,setting['steps']-1)))


def source_extremes(data):
    from slm_lab.dense_foundation_data_v1 import NAMES
    selected=[]
    for i,name in enumerate(NAMES):
        offset=0 if i==0 else data.ends[i-1]
        for key in ('complete_context_tokens','response_EOS_targets'):
            selected.append(offset+max(range(data.populations[name]),key=lambda p:data.information(offset+p)[key]))
    return list(dict.fromkeys(selected))


def held_evaluate(model,plan,arm,amp):
    from slm_lab.dense_foundation_held_v1 import DenseHeldSources,evaluate
    data=DenseHeldSources(ROOT,model.config,plan['education_context'])
    if data.populations!=plan['held_populations']:
        raise ValueError('Every original held source population must match the plan')
    return evaluate(model,data,REPORTS/arm/'held-foundation.json',amp,lambda:owner.owned(plan))


def worker(phase,arm,gpu):
    subject='capacity-'+str(gpu) if phase=='capacity' else ('initial-foundation' if phase=='initial' else 'train-'+arm)
    plan=legacy.registered(subject,gpu);torch,guard,amp=legacy.runtime(plan)
    import random
    import numpy as np
    from slm_lab.codenet_checkpoint_v1 import capture_rng
    from slm_lab.dense_foundation_data_v1 import DenseFoundationSources
    from slm_lab.dense_foundation_inference_v1 import fresh,own_reference,options,sources,export,load
    from slm_lab.dense_foundation_stage_v1 import DenseFoundationStage,accumulate_gradients
    from slm_lab.point_grounding_evaluation_v1 import state_digest
    from slm_lab.whole_human_sft_evidence_v1 import tensor_digest
    model,metadata=fresh(plan,'cuda:0')
    if tensor_digest(model.named_parameters())!=plan['initial_model_digest']:
        raise ValueError('Every arm must start from the exact own initial dense model, never disposable probe weights')
    if phase=='initial':
        atomic_checkpoint(OUTPUT/'initial/model.pt',export(model,plan,file_hash(REPORTS/'plan.json'),'initial',0),MAX_MODEL,RESERVE)
        held_evaluate(model,plan,'initial',amp);guard.check(full=True);return
    setting=options(plan,'rate_3e6_KL' if phase=='capacity' else arm)
    reference=own_reference(plan,'cuda:0') if setting['reference_KL_alpha'] else None
    data=DenseFoundationSources(ROOT,model.config,plan['education_context'])
    if data.populations!=plan['source_populations'] or data.budget()!=plan['source_exposure_budget']:
        raise ValueError('All original admitted sources and complete token masks must match the plan')
    random.seed(setting['seed']);np.random.seed(setting['seed']);torch.manual_seed(setting['seed'])
    optimizer=torch.optim.AdamW([p for _,p in model.language_named_parameters()],lr=setting['learning_rate'],
        betas=tuple(setting['betas']),weight_decay=setting['weight_decay'],foreach=False)
    kernel=DenseFoundationStage(model,optimizer,reference,data,setting,sources(plan,file_hash(REPORTS/'plan.json')),'cuda:0',amp)
    folder=OUTPUT/('capacity-'+str(gpu) if phase=='capacity' else arm)
    if phase=='capacity':folder.mkdir()
    elif set(p.name for p in folder.iterdir())!={'tokenizer.json'}:
        raise ValueError('Fresh tokenizer-only full dense arm required')
    unused=fixed_digest(model);fixed_versions=signature(kernel.fixed)
    reference_versions=signature(reference.parameters()) if reference is not None else None
    reference_digest=state_digest(reference) if reference is not None else None
    initial_language=tensor_digest(model.network.named_parameters())
    started=time.monotonic();extreme_records=[]
    def adam_digest():
        return tensor_digest((str(i)+'.'+k,state[k]) for i,state in enumerate(optimizer.state.values()) for k in ('step','exp_avg','exp_avg_sq'))
    if phase=='capacity':
        kernel.update();atomic_checkpoint(folder/'resume-step1.pt',kernel.state(),MAX_STATE,RESERVE)
        def draws():return [random.random(),float(np.random.random()),float(torch.rand(())),float(torch.rand((),device='cuda:0'))]
        expected_draws=draws();expected=kernel.update()
        learned=tensor_digest(model.network.named_parameters());moments=adam_digest()
        saved=torch.load(folder/'resume-step1.pt',map_location='cpu',weights_only=True,mmap=True)
        bad={**saved,'core':{**saved['core'],'rng':{**saved['core']['rng'],'cuda':torch.zeros(4,dtype=torch.uint8)}}}
        rng=capture_rng('cuda:0');versions=signature(model.parameters())
        try:kernel.resume(bad)
        except ValueError:pass
        else:raise ValueError('Corrupt private CUDA RNG must reject before mutation')
        if (versions!=signature(model.parameters()) or not torch.equal(rng['torch'],torch.get_rng_state())
                or not torch.equal(rng['cuda'],torch.cuda.get_rng_state()) or moments!=adam_digest()):
            raise ValueError('Invalid resume changed actual model/Adam/RNG')
        kernel.resume(saved)
        if (draws()!=expected_draws or kernel.update()!=expected
                or learned!=tensor_digest(model.network.named_parameters()) or moments!=adam_digest()):
            raise ValueError('Full dense next update/source cursor/Adam and all4RNG streams must recover exactly')
        atomic_checkpoint(folder/'resume-step2.pt',kernel.state(),MAX_STATE,RESERVE)
        # Actual full-length sources, including a4096-token book window. No
        # clipping, synthetic length stand-in or smaller production batch.
        for position in plan['capacity_source_extremes']:
            owner.owned(plan);optimizer.zero_grad(set_to_none=True)
            metric=accumulate_gradients(model,reference,data,[position],setting,'cuda:0',amp)
            if (any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in kernel.parameters)
                    or any(p.grad is not None for p in kernel.fixed)):
                raise ValueError('Every actual longest complete source must have finite full-language-only gradients')
            extreme_records.append(metric)
        optimizer.zero_grad(set_to_none=True);del saved,bad
    else:
        with (REPORTS/(arm+'-metrics.jsonl')).open('x') as journal:
            for _ in range(setting['steps']):
                owner.owned(plan);tick=time.monotonic();record=kernel.update()
                journal.write(json.dumps(record)+'\n');journal.flush()
                if kernel.completed==1 or kernel.completed%25==0 or kernel.completed==setting['steps']:
                    print(json.dumps(dict(**record,elapsed_seconds=time.monotonic()-started,step_seconds=time.monotonic()-tick)),flush=True)
                    atomic_json(REPORTS/(arm+'-progress.json'),dict(step=kernel.completed,planned_steps=setting['steps'],latest=record,goal_achieved=False))
                if kernel.completed%256==0 or kernel.completed==setting['steps']:
                    atomic_checkpoint(folder/'resume.pt',kernel.state(),MAX_STATE,RESERVE)
        atomic_checkpoint(folder/'model.pt',export(model,plan,file_hash(REPORTS/'plan.json'),arm,kernel.completed),MAX_MODEL,RESERVE)
    if (unused!=fixed_digest(model) or fixed_versions!=signature(kernel.fixed)
            or initial_language==tensor_digest(model.network.named_parameters())
            or (reference is not None and (reference_versions!=signature(reference.parameters()) or reference_digest!=state_digest(reference)))):
        raise ValueError('Every fixed vision/critic/head and own reference value/metadata must remain exact while dense language learns')
    common=dict(actual_counts=model.counts(),actual_dense_language_values=sum(p.numel() for p in kernel.parameters),
        actual_updates=kernel.completed,complete_Adam_states=len(optimizer.state),
        all_fixed_vision_critic_heads_and_own_reference_values_metadata_exact=True,
        separate_own_reference_loaded_for_training_only=reference is not None,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        seconds=time.monotonic()-started,candidate_promoted=False,goal_achieved=False)
    if phase=='capacity':
        atomic_json(REPORTS/f'capacity-{gpu}.json',dict(**common,physical_GPU=gpu,source_extremes=extreme_records,
            exact_next_update_and_corrupt_private_RNG_rejection=True,logical_disposable_updates=2,physical_optimizer_calls=3,
            retained_step1_sha256=file_hash(folder/'resume-step1.pt'),retained_step2_sha256=file_hash(folder/'resume-step2.pt'),
            production_updates=0));guard.check(full=True);return
    receipt=dict(**common,checkpoint_sha256=file_hash(folder/'model.pt'),resume_sha256=file_hash(folder/'resume.pt'),
        source_exposure_budget=plan['source_exposure_budget'])
    atomic_json(REPORTS/(arm+'-training-complete.json'),receipt)
    model.eval();batch,photo_id=legacy.probe_batch()
    with torch.no_grad(),amp():
        text=model(batch['input_ids'])['logits'].cpu()
        image=model(batch['input_ids'],batch['pixel_values'])['logits'].cpu()
        cache=model.network.prefill(batch['input_ids'],batch['pixel_values'])['cache']
        cached=model.network.decode(torch.tensor([[4]],device='cuda:0'),cache)['logits'].cpu()
    trained=tensor_digest(model.named_parameters())
    # The complete Adam state is already durable; release training-only memory
    # before loading a second full student for the actual native cold check.
    del kernel,optimizer,reference;torch.cuda.empty_cache()
    cold,_=load(folder/'model.pt',REPORTS/'plan.json',arm,expected_checkpoint_sha256=receipt['checkpoint_sha256'],
        expected_plan_sha256=file_hash(REPORTS/'plan.json'),device='cuda:0')
    with torch.no_grad(),amp():
        cold_text=cold(batch['input_ids'])['logits'].cpu();cold_image=cold(batch['input_ids'],batch['pixel_values'])['logits'].cpu()
        cold_cache=cold.network.prefill(batch['input_ids'],batch['pixel_values'])['cache']
        cold_cached=cold.network.decode(torch.tensor([[4]],device='cuda:0'),cold_cache)['logits'].cpu()
    if (not torch.equal(text,cold_text) or not torch.equal(image,cold_image) or not torch.equal(cached,cold_cached)
            or trained!=tensor_digest(cold.named_parameters())):
        raise ValueError('Every trained value and actual native text/image/cache cold reload must be exact')
    atomic_json(REPORTS/arm/'trained-shared-reload.json',dict(actual_full_trained_dense_native_cold_reload_exact=True,
        original_probe_photo_id=photo_id,all_actual_heads_counted=True,goal_achieved=False))
    del cold,cold_cache,cache;torch.cuda.empty_cache()
    held_evaluate(model,plan,arm,amp);guard.check(full=True)


def assess(arm,suite,gpu,shard):
    subject='assess-'+arm+'-'+suite+'-'+str(shard if shard is not None else gpu)
    plan=legacy.registered(subject,gpu);torch,guard,amp=legacy.runtime(plan)
    from slm_lab.dense_foundation_inference_v1 import load,DenseNativeView
    checkpoint=OUTPUT/arm/'model.pt'
    model,metadata=load(checkpoint,REPORTS/'plan.json',arm,expected_checkpoint_sha256=file_hash(checkpoint),
        expected_plan_sha256=file_hash(REPORTS/'plan.json'),device='cuda:0')
    legacy.native_suite(plan,SimpleNamespace(aligned=DenseNativeView(model)),metadata,arm,suite,gpu,guard,shard)


def bind_sources(tests):
    proof=json.loads((CPU_PROOF/'plan.json').read_text())
    files=dict(proof['source_files_sha256'])
    names=['scripts/train_dense_foundation_v1.py','scripts/review_dense_foundation_CPU_v1.py',
        'src/slm_lab/dense_foundation_held_v1.py','src/slm_lab/dense_foundation_evidence_v1.py',
        'tests/test_dense_foundation_held_v1.py','tests/test_dense_foundation_driver_v1.py',
        'tests/test_dense_foundation_evidence_v1.py','reports/dense-foundation-v1-full-source-contract.json',
        'reports/dense-foundation-v1-source-extremes.json',*tests]
    for name in names:
        path=ROOT/name;sha=file_hash(path)
        if str(path) in files and files[str(path)]!=sha:raise ValueError('Never reseal a changed bound source')
        files[str(path)]=sha
    for name in ('plan.json','complete.json','independent-complete-readback.json','resume-step1.pt','resume-step2.pt'):
        path=CPU_PROOF/name;files[str(path)]=file_hash(path)
    # Individually bind every token file actually consumed by this run, beyond
    # the frozen admission/provenance proof hashes inherited above.
    for directory in ('open-education-bookpack-v1','codenet-durable-pack-v2','APPS-verified-generation-v1','whole-human-alignment-v2'):
        for path in (ROOT/'data'/directory).iterdir():
            if path.is_file() and path.name not in ('progress.json','readback-progress.json'):
                sha=file_hash(path)
                if str(path) in files and files[str(path)]!=sha:raise ValueError('Original packed data changed since CPU evidence')
                files[str(path)]=sha
    return files


def parent():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('The waiting owner must explicitly hide every GPU')
    if any(p.exists() for p in (REPORTS,OUTPUT,QUEUE)):raise ValueError('Preserve prior foundation attempts; no automatic restart')
    tests={'reports/admitted-education-v1-tests.log':'3 passed','reports/dense-foundation-stage-v1-tests.log':'3 passed',
        'reports/dense-foundation-inference-v1-tests.log':'1 passed','reports/dense-foundation-held-v1-tests.log':'3 passed',
        'reports/dense-foundation-driver-v1-tests.log':'3 passed','reports/dense-foundation-evidence-v1-tests.log':'2 passed'}
    for name,passed in tests.items():
        if passed not in (ROOT/name).read_text():raise ValueError('Actual source/gradient/recovery/held/native/ownership checks required')
    proof=json.loads((CPU_PROOF/'complete.json').read_text())
    if not proof['actual_next_weights_Adam_all_CPU_RNG_cursor_metrics_exact'] or not proof['actual_trained_full_size_cold_reload_exact']:
        raise ValueError('Actual full984M mixed objective recovery and cold inference required')
    predecessor=json.loads(PREVIOUS_QUEUE.read_text())['owner'];me=owner.resources.identity(os.getpid())
    files=bind_sources(tests);queued_guard=FileBindings(files);deadline=time.monotonic()+72*3600
    while not previous_ended(predecessor):
        queued_guard.check()
        if time.monotonic()>deadline:raise ValueError('Exact original word owner still live; expire without touching GPUs')
        atomic_json(QUEUE,dict(status='waiting for exact shared-word owner/workers to finish successfully and restore original work',
            owner=me,predecessor=predecessor,source_files_sha256=files,observed_at=datetime.now(timezone.utc).isoformat(),
            production_model_updates=0,goal_achieved=False));time.sleep(10)
    queued_guard.check(full=True)
    for name in ('plan.json','complete.json','restored.json','independent-comparison.json'):
        path=PREVIOUS/name;files[str(path)]=file_hash(path)
    import torch
    from slm_lab.dense_foundation_data_v1 import DenseFoundationSources,NAMES
    from slm_lab.dense_foundation_held_v1 import DenseHeldSources
    from slm_lab.dense_foundation_inference_v1 import fresh
    from slm_lab.whole_human_sft_evidence_v1 import tensor_digest
    torch.set_num_threads(2)
    settings=dict(json.loads((CPU_PROOF/'plan.json').read_text())['initial_dense_settings'])
    model,_=fresh(settings,'cpu');data=DenseFoundationSources(ROOT,model.config,4096);held=DenseHeldSources(ROOT,model.config,4096)
    setting=dict(seed=949131,epochs=1,steps=math.ceil(len(data)/16),microbatch=1,accumulation=16,
        betas=[.9,.95],weight_decay=.01,gradient_clip=1.,coefficients=dict(education=1.,variety=1.,human=1.,program=4.))
    budget=exposure(data,setting);extremes=source_extremes(data)
    contract=json.loads((ROOT/'reports/dense-foundation-v1-full-source-contract.json').read_text())
    if (contract['complete_train_source_budget']!=budget['complete_source_budget']
            or {n:r['examples'] for n,r in contract['complete_held_source_budget'].items()}!=held.populations
            or [data.information(p) for p in extremes]!=json.loads((ROOT/'reports/dense-foundation-v1-source-extremes.json').read_text())['all_original_context_and_target_extremes']):
        raise ValueError('Complete original source contract, held populations and all source extremes must reconstruct')
    for name,prefix in [('education','open-education-bookpack-v1'),('code','codenet-durable-pack-v2'),
                        ('human','whole-human-alignment-v2'),('program','APPS-verified-generation-v1')]:
        settings[name+'_admission_sha256']=file_hash(ROOT/'data'/prefix/'admission.json')
    settings.update(initial_model_digest=tensor_digest(model.named_parameters()),education_context=4096,training_options=setting,
        source_populations=data.populations,source_exposure_budget=budget['complete_source_budget'],complete_exposure_audit=budget,
        capacity_source_extremes=extremes,held_populations=held.populations,
        arms=dict(rate_3e6_CE=dict(learning_rate=3e-6,reference_KL_alpha=0.,physical_GPU=0),
            rate_3e6_KL=dict(learning_rate=3e-6,reference_KL_alpha=.1,physical_GPU=1),
            rate_1e5_CE=dict(learning_rate=1e-5,reference_KL_alpha=0.,physical_GPU=2),
            rate_1e5_KL=dict(learning_rate=1e-5,reference_KL_alpha=.1,physical_GPU=3)))
    # Inference/stage options contain only optimizer/objective settings; GPU
    # placement is an ownership concern and is read separately in this driver.
    settings['arm_GPUs']={arm:spec.pop('physical_GPU') for arm,spec in settings['arms'].items()}
    del model
    guard=FileBindings(files)
    with (ROOT/'reports/code-residual-production-v4.gpu.lock').open('a+') as lease:
        fcntl.flock(lease.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        if not previous_ended(predecessor):raise ValueError('Previous exact owner became live')
        require_idle_devices();guard.check(full=True);REPORTS.mkdir();OUTPUT.mkdir()
        for arm in ['initial',*settings['arms']]:
            (REPORTS/arm).mkdir();(OUTPUT/arm).mkdir()
            shutil.copyfile(ROOT/'checkpoints/extended-curriculum-v1/joint/tokenizer.json',OUTPUT/arm/'tokenizer.json')
        plan=dict(**settings,created_at=datetime.now(timezone.utc).isoformat(),owner=me,waiter=owner.WAITER,
            source_files_sha256=files,expires_at_unix=time.time()+24*3600,automatic_retry=False,
            all_capacity_updates_discarded=True,external_teacher_weights=False,
            final_multimodal_or_genuine_human_RLHF_not_claimed=True,no_automatic_promotion=True,candidate_promoted=False,goal_achieved=False)
        atomic_json(REPORTS/'plan.json',plan);atomic_json(REPORTS/'children.json',[])
        def interrupted(signum,frame):raise KeyboardInterrupt('Owned dense foundation run interrupted')
        signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
        driver=str(Path(__file__).resolve())
        watcher=subprocess.Popen(['.venv/bin/python','-u',driver,'--watchdog'],cwd=ROOT,
            stdout=(REPORTS/'watchdog.log').open('x'),stderr=subprocess.STDOUT,start_new_session=True)
        def launch(args,subject,gpu):return owner.launch(plan,['-u',driver,*args],subject,str(gpu))
        def tasks(items):owner.wait_children(plan,[launch(args,subject,gpu) for args,subject,gpu in items])
        def native(arm):
            tasks([(['--assess',suite,'--arm',arm,'--gpu',str(i)],'assess-'+arm+'-'+suite+'-'+str(i),i)
                for i,suite in enumerate(('knowledge','code','cpp','vision'))])
            for suite in ('public','math'):
                tasks([(['--assess',suite,'--arm',arm,'--gpu',str(i),'--shard',str(i)],'assess-'+arm+'-'+suite+'-'+str(i),i) for i in range(4)])
            from slm_lab.full_math_shards_v1 import reconcile
            reconcile([REPORTS/arm/f'math-{i}.json' for i in range(4)],ROOT/'data/math-reasoning-v2/test.jsonl',
                OUTPUT/arm/'tokenizer.json',REPORTS/arm/'math-complete.json')
        try:
            owner.idle_original();os.kill(owner.WAITER['pid'],signal.SIGSTOP);until=time.monotonic()+5
            while (owner.resources.same(owner.WAITER) or {}).get('state') not in ('T','t'):
                if time.monotonic()>until:raise ValueError('Original idle waiter did not enter owned hold')
                time.sleep(.05)
            if (Path('/proc')/str(owner.WAITER['pid'])/'task'/str(owner.WAITER['pid'])/'children').read_text().strip():
                raise ValueError('Original waiter started real work; never preempt it')
            tasks([(['--worker','capacity','--gpu',str(i)],'capacity-'+str(i),i) for i in range(4)])
            capacity=[json.loads((REPORTS/f'capacity-{i}.json').read_text()) for i in range(4)]
            if not all(p['exact_next_update_and_corrupt_private_RNG_rejection'] and p['actual_dense_language_values']==931244544 for p in capacity):
                raise ValueError('All four full-size/full-length L40 capacity and exact recovery proofs required')
            atomic_json(REPORTS/'capacity.json',dict(passed=True,all_four_actual_L40s=True,proofs=capacity,production_updates=0,goal_achieved=False))
            tasks([(['--worker','initial','--gpu','0'],'initial-foundation',0)])
            native('initial')
            tasks([(['--worker','train','--arm',arm,'--gpu',str(plan['arm_GPUs'][arm])],'train-'+arm,plan['arm_GPUs'][arm]) for arm in plan['arms']])
            for arm in plan['arms']:native(arm)
            from slm_lab.dense_foundation_evidence_v1 import reconcile
            reconcile(ROOT,REPORTS,OUTPUT,data,held);guard.check(full=True)
            atomic_json(REPORTS/'complete.json',dict(status='Every full original dense-language epoch, held target and matched native suite completed; inspect measured quality',
                full_final_multimodal_and_RLHF_goal_achieved=False,candidate_promoted=False,goal_achieved=False))
        except BaseException as error:
            atomic_json(REPORTS/'failure.json',dict(error=repr(error),automatic_retry=False,goal_achieved=False));raise
        finally:
            owner.restore_owned(plan,'Dense foundation comparison ended; original work restored');watcher.wait(timeout=20)


if __name__=='__main__':
    configure();parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--watchdog',action='store_true');parser.add_argument('--worker',choices=['capacity','initial','train'])
    parser.add_argument('--assess',choices=['knowledge','code','cpp','vision','public','math'])
    parser.add_argument('--arm',default='initial');parser.add_argument('--gpu',type=int,choices=range(4),default=0)
    parser.add_argument('--shard',type=int,choices=range(4));args=parser.parse_args()
    if args.watchdog:owner.watchdog()
    elif args.worker:worker(args.worker,args.arm,args.gpu)
    elif args.assess:assess(args.arm,args.assess,args.gpu,args.shard)
    else:parent()
