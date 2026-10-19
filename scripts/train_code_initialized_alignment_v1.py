"""Owned full human/program CE +/- own-reference KL on the code-initialized model."""
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

import train_whole_human_SFT_v1 as legacy
from slm_lab.codenet_checkpoint_v1 import FileBindings, atomic_checkpoint
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json

owner = legacy.owner
ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT/'reports/code-initialized-alignment-v1'
OUTPUT = ROOT/'checkpoints/code-initialized-alignment-v1'
QUEUE = ROOT/'reports/code-initialized-alignment-v1-queue.json'
PREVIOUS = ROOT/'reports/whole-human-reward-v1'
PREVIOUS_QUEUE = ROOT/'reports/whole-human-reward-v1-queue.json'
CPU_PROOF = ROOT/'reports/own-aligned-assistant-v3-full-CPU-v1'
MAX_STATE = 224*1024**2
RESERVE = 26*1024**3


def configure():
    owner.REPORTS = REPORTS; owner.resources.REPORTS = REPORTS
    legacy.REPORTS = REPORTS; legacy.OUTPUT = OUTPUT


def streams(config):
    from slm_lab.whole_human_data_v2 import WholeHumanData
    from slm_lab.own_code_training_stage_v1 import PreparedCodeStream
    return dict(human=WholeHumanData(ROOT/'data/whole-human-alignment-v2','dialogue','train','optimization'),
        generation=PreparedCodeStream('generation',ROOT/'data/APPS-verified-generation-v1',ROOT/'checkpoints/whole-human-SFT-v1/initial/tokenizer.json',config))


def previous_ended(actor):
    if owner.resources.same(actor): return False
    if (PREVIOUS/'children.json').exists() and any(owner.resources.same(a) for a in json.loads((PREVIOUS/'children.json').read_text())):
        return False
    if not (PREVIOUS/'restored.json').exists() or not (PREVIOUS/'complete.json').exists() or (PREVIOUS/'failure.json').exists():
        raise ValueError('Prior actual human reward owner/workers must finish successfully and restore original work before GPU acquisition')
    return True


def require_idle_devices():
    owner.idle_original()
    used=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).splitlines()
    if len(used)!=4 or any(int(v.strip())>128 for v in used):raise ValueError('All actual original L40s must be idle')
    if shutil.disk_usage(ROOT).free<RESERVE+16*MAX_STATE:raise ValueError('Complete source/atomic optimizer reserve required')


def worker(phase,arm,gpu):
    subject = 'capacity-'+str(gpu) if phase=='capacity' else ('initial-human' if phase=='initial' else 'train-'+arm)
    plan = legacy.registered(subject,gpu); torch,guard,amp = legacy.runtime(plan)
    import random
    import numpy as np
    from slm_lab.codenet_checkpoint_v1 import capture_rng
    from slm_lab.code_initialized_alignment_inference_v1 import fresh,sources,options,export,load
    from slm_lab.code_initialized_alignment_stage_v1 import CodeInitializedAlignmentStage,source_batch,objectives,REFERENCE_ROUTES
    from slm_lab.whole_human_data_v2 import WholeHumanData
    from slm_lab.whole_human_sft_evidence_v1 import tensor_digest,fixed_digest,language_digest,evaluate_human
    model,metadata = fresh(plan,'cuda:0'); model.aligned.gradient_checkpointing = True
    if tensor_digest(model.parts.state_dict().items()) != plan['initial_parts_digest']:
        raise ValueError('Every worker must start at the same fresh code-initialized extension state')
    random.seed(plan['training_options']['seed']); np.random.seed(plan['training_options']['seed']); torch.manual_seed(plan['training_options']['seed'])
    if phase=='initial':
        atomic_checkpoint(OUTPUT/'initial/model.pt',export(model,plan,file_hash(REPORTS/'plan.json'),'initial',0),64*1024**2,RESERVE)
        valid = WholeHumanData(ROOT/'data/whole-human-alignment-v2','dialogue','valid','evaluation')
        evaluate_human(model.aligned,valid,REPORTS/'initial/human-validation.json',amp,lambda:owner.owned(plan))
        return
    data = streams(model.config); setting = options(plan,'mixed_reference_KL' if phase=='capacity' else arm)
    optimizer = torch.optim.AdamW(model.extension_parameters('language'),lr=setting['learning_rate'],
        betas=tuple(setting['betas']),weight_decay=setting['weight_decay'],foreach=False)
    kernel = CodeInitializedAlignmentStage(model,optimizer,data,setting,sources(plan,file_hash(REPORTS/'plan.json')),'cuda:0',amp)
    folder = OUTPUT/('capacity-'+str(gpu) if phase=='capacity' else arm)
    if phase=='capacity': folder.mkdir()
    elif set(p.name for p in folder.iterdir()) != {'tokenizer.json'}: raise ValueError('Fresh tokenizer-only trained arm required')
    batch,photo_id = legacy.probe_batch(); before = legacy.native_original(model,batch,amp)
    with torch.no_grad(),amp():
        initial_aligned = model.aligned(batch['input_ids'],batch['pixel_values'])['logits'].cpu()
        if not torch.equal(initial_aligned,before['code']): raise ValueError('Actual full GPU initial aligned image logits must exactly equal own trained code')
        cache = model.original.language.code.prefill(batch['input_ids'],batch['pixel_values'])['cache']
        cached_before = model.original.language.code.decode(torch.tensor([[4]],device='cuda:0'),cache)['logits'].cpu()
    fixed_before = fixed_digest(model); versions = legacy.signature(kernel.fixed); initial_language = language_digest(model)
    started = time.monotonic()
    if phase=='capacity':
        kernel.update(); atomic_checkpoint(folder/'resume.pt',kernel.state(),MAX_STATE,RESERVE)
        def draws(): return [random.random(),float(np.random.random()),float(torch.rand(())),float(torch.rand((),device='cuda:0'))]
        expected_draws = draws(); expected = kernel.update(); language = language_digest(model)
        def Adam_digest():
            return tensor_digest((str(i)+'.'+k,state[k]) for i,state in enumerate(optimizer.state.values()) for k in ('step','exp_avg','exp_avg_sq'))
        moments = Adam_digest(); saved = torch.load(folder/'resume.pt',map_location='cpu',weights_only=True,mmap=True)
        bad = {**saved,'core':{**saved['core'],'rng':{**saved['core']['rng'],'cuda':torch.zeros(4,dtype=torch.uint8)}}}
        rng = capture_rng('cuda:0'); signature = legacy.signature(model.parameters())
        try: kernel.resume(bad)
        except ValueError: pass
        else: raise ValueError('Corrupt private CUDA RNG must reject before mutation')
        if (signature != legacy.signature(model.parameters()) or not torch.equal(rng['torch'],torch.get_rng_state())
                or not torch.equal(rng['cuda'],torch.cuda.get_rng_state()) or moments != Adam_digest()):
            raise ValueError('Invalid private RNG restore changed live state')
        kernel.resume(saved)
        if draws()!=expected_draws or kernel.update()!=expected or language_digest(model)!=language or Adam_digest()!=moments:
            raise ValueError('Actual mixed-source full CUDA next-update weights/Adam/cursor/RNG must recover exactly')
        # Backward through actual full source extremes, never truncated stand-ins.
        human_positions = [max(range(len(data['human'])),key=lambda p:data['human'].rows[p]['native']['response']['length']),
            max(range(len(data['human'])),key=lambda p:data['human'].rows[p]['native']['response']['length']-data['human'].rows[p]['native']['response']['target_start'])]
        pack = data['generation'].pack
        generation_position = max(range(len(pack)),key=lambda p:int(pack.index[pack.positions[p]]['length']))
        extremes = []
        for name,position in [('human',p) for p in dict.fromkeys(human_positions)]+[('generation',generation_position)]:
            optimizer.zero_grad(set_to_none=True); source = source_batch(data,name,[position],'cuda:0'); inputs = {k:v for k,v in source.items() if k!='labels'}
            with amp():
                logits = model.aligned(**inputs)['logits']
                with torch.no_grad(): reference = model.route(REFERENCE_ROUTES[name])(**inputs)['logits']
            ce,kl,count = objectives(logits,reference,source['labels']); ((ce+.1*kl)/count).backward()
            if any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in kernel.parameters) or any(p.grad is not None for p in kernel.fixed):
                raise ValueError('Every actual longest whole source must have finite new-only CE/KL gradients')
            extremes.append(dict(stream=name,source_position=position,context_tokens=int(source['attention_mask'].sum()),response_EOS_targets=count))
        optimizer.zero_grad(set_to_none=True)
    else:
        with (REPORTS/(arm+'-metrics.jsonl')).open('x') as stream:
            for _ in range(setting['steps']):
                owner.owned(plan); tick=time.monotonic(); record=kernel.update();stream.write(json.dumps(record)+'\n');stream.flush()
                if kernel.completed==1 or kernel.completed%25==0 or kernel.completed==setting['steps']:
                    print(json.dumps(dict(**record,elapsed_seconds=time.monotonic()-started,step_seconds=time.monotonic()-tick)),flush=True)
                    atomic_json(REPORTS/(arm+'-progress.json'),dict(step=kernel.completed,planned_steps=setting['steps'],latest=record,goal_achieved=False))
                if kernel.completed%256==0 or kernel.completed==setting['steps']:
                    atomic_checkpoint(folder/'resume.pt',kernel.state(),MAX_STATE,RESERVE)
        atomic_checkpoint(folder/'model.pt',export(model,plan,file_hash(REPORTS/'plan.json'),arm,kernel.completed),64*1024**2,RESERVE)
    after = legacy.native_original(model,batch,amp)
    with torch.no_grad(),amp():
        cached_after = model.original.language.code.decode(torch.tensor([[4]],device='cuda:0'),cache)['logits'].cpu()
        aligned = model.aligned(batch['input_ids'],batch['pixel_values'])['logits'].cpu()
    if (fixed_before!=fixed_digest(model) or versions!=legacy.signature(kernel.fixed) or initial_language==language_digest(model)
            or any(not torch.equal(before[r],after[r]) for r in before) or not torch.equal(cached_before,cached_after)):
        raise ValueError('All original general/code/grounding/unused bytes/versions/cache must remain exact while new language learns')
    common = dict(actual_counts=model.counts(),actual_updates=kernel.completed,complete_Adam_states=len(optimizer.state),
        all_fixed_bytes_versions_and_original_native_routes_exact=True,pre_existing_original_code_cache_exact=True,
        original_native_probe_photo_id=photo_id,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(),elapsed_seconds=time.monotonic()-started,candidate_promoted=False,goal_achieved=False)
    if phase=='capacity':
        atomic_json(REPORTS/f'capacity-{gpu}.json',dict(**common,source_extremes=extremes,physical_GPU=gpu,
            exact_actual_next_update_and_invalid_private_RNG_rejection=True,logical_disposable_updates=2,physical_calls=3,
            resume_sha256=file_hash(folder/'resume.pt'),production_updates=0));return
    receipt = dict(**common,checkpoint_sha256=file_hash(folder/'model.pt'),resume_sha256=file_hash(folder/'resume.pt'),source_exposure_budget=plan['source_exposure_budget'])
    atomic_json(REPORTS/(arm+'-training-complete.json'),receipt)
    valid = WholeHumanData(ROOT/'data/whole-human-alignment-v2','dialogue','valid','evaluation')
    evaluate_human(model.aligned,valid,REPORTS/arm/'human-validation.json',amp,lambda:owner.owned(plan))
    cold,_ = load(folder/'model.pt',REPORTS/'plan.json',arm,expected_checkpoint_sha256=receipt['checkpoint_sha256'],expected_plan_sha256=file_hash(REPORTS/'plan.json'),device='cuda:0')
    with torch.no_grad(),amp(): cold_aligned = cold.aligned(batch['input_ids'],batch['pixel_values'])['logits'].cpu()
    if not torch.equal(aligned,cold_aligned) or language_digest(model)!=language_digest(cold) or any(not torch.equal(after[r],v) for r,v in legacy.native_original(cold,batch,amp).items()):
        raise ValueError('Actual full trained code-initialized inference reload must retain all aligned/original routes')
    atomic_json(REPORTS/arm/'trained-shared-reload.json',dict(actual_counts=cold.counts(),full_trained_aligned_and_original_3routes_exact=True,one_backbone_loaded=True,goal_achieved=False))
    guard.check(full=True)


def assess(arm,suite,gpu,shard):
    subject='assess-'+arm+'-'+suite+'-'+str(shard if shard is not None else gpu)
    plan=legacy.registered(subject,gpu);torch,guard,amp=legacy.runtime(plan)
    from slm_lab.code_initialized_alignment_inference_v1 import load
    checkpoint=OUTPUT/arm/'model.pt'
    model,metadata=load(checkpoint,REPORTS/'plan.json',arm,expected_checkpoint_sha256=file_hash(checkpoint),expected_plan_sha256=file_hash(REPORTS/'plan.json'),device='cuda:0')
    legacy.native_suite(plan,model,metadata,arm,suite,gpu,guard,shard)


def parent():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='': raise ValueError('CPU queue/owner must explicitly hide CUDA')
    if any(path.exists() for path in (REPORTS,OUTPUT,QUEUE)): raise ValueError('Preserve every previous queued or running experiment; no automatic restart')
    tests={'reports/own-aligned-assistant-v3-tests.log':'3 passed','reports/code-initialized-alignment-stage-v1-tests.log':'3 passed',
           'reports/code-initialized-alignment-inference-v1-tests.log':'1 passed','reports/code-initialized-alignment-driver-v1-tests.log':'3 passed'}
    for name,passed in tests.items():
        if passed not in (ROOT/name).read_text(): raise ValueError('Required actual architecture/objective/recovery/loader/resource checks missing')
    proof=json.loads((CPU_PROOF/'complete.json').read_text())
    if not proof['actual_next_update_all_new_tensors_Adam_RNG_cursor_and_metrics_exact'] or not proof['initial_full_code_image_human_and_program_logits_exact']:
        raise ValueError('Actual full-size code-initialized CPU retention/recovery proof required')
    predecessor=json.loads(PREVIOUS_QUEUE.read_text())['owner'];me=owner.resources.identity(os.getpid())
    names=['scripts/train_code_initialized_alignment_v1.py','src/slm_lab/own_aligned_assistant_v3.py',
        'src/slm_lab/code_initialized_alignment_stage_v1.py','src/slm_lab/code_initialized_alignment_inference_v1.py',
        'src/slm_lab/code_initialized_alignment_evidence_v1.py','tests/test_own_aligned_assistant_v3.py',
        'tests/test_code_initialized_alignment_stage_v1.py','tests/test_code_initialized_alignment_inference_v1.py',
        'tests/test_code_initialized_alignment_driver_v1.py',*tests]
    queued_files={str(ROOT/name):file_hash(ROOT/name) for name in names}
    for path in (CPU_PROOF/'plan.json',CPU_PROOF/'complete.json',CPU_PROOF/'resume.pt',CPU_PROOF/'trained-disposable-extensions.pt'):
        queued_files[str(path)]=file_hash(path)
    queued_guard=FileBindings(queued_files);deadline=time.monotonic()+20*3600
    while not previous_ended(predecessor):
        queued_guard.check()
        if time.monotonic()>deadline: raise ValueError('Preceding human-ranking work still live; queue expires without using GPUs')
        atomic_json(QUEUE,dict(status='waiting for exact full human-ranking owner/workers to finish and restore original work',owner=me,predecessor=predecessor,
            source_files_sha256=queued_files,observed_at=datetime.now(timezone.utc).isoformat(),model_updates=0,goal_achieved=False));time.sleep(10)
    queued_guard.check(full=True)
    old_path=ROOT/'reports/whole-human-SFT-v1/plan.json';old=json.loads(old_path.read_text());files=dict(old['source_files_sha256'])
    for path,sha in queued_files.items():
        if path in files and files[path]!=sha:raise ValueError('Never silently reseal a changed original source')
        files[path]=sha
    for path,sha in json.loads((CPU_PROOF/'plan.json').read_text())['source_files_sha256'].items():
        if path in files and files[path]!=sha:raise ValueError('Actual full CPU source binding changed; never reseal it')
        files[path]=sha
    for path in (old_path,PREVIOUS/'plan.json',PREVIOUS/'complete.json',PREVIOUS/'restored.json'):
        files[str(path)]=file_hash(path)
    from slm_lab.model import ModelConfig
    from slm_lab.code_initialized_alignment_evidence_v1 import budget
    from slm_lab.code_initialized_alignment_inference_v1 import fresh
    from slm_lab.whole_human_sft_evidence_v1 import tensor_digest
    data=streams(ModelConfig(**json.loads((ROOT/'configs/model.json').read_text())))
    settings=dict(initial_model_plan=old,actual_counts=proof['actual_counts'],
        human_source_admission_sha256=file_hash(ROOT/'data/whole-human-alignment-v2/admission.json'),program_source_admission_sha256=file_hash(ROOT/'data/APPS-verified-generation-v1/admission.json'),
        training_options=dict(seed=947431,steps=max(math.ceil(len(data['human'])/4),len(data['generation'])),human_per_update=4,programs_per_update=1,learning_rate=1e-5,betas=[.9,.95],weight_decay=.01,gradient_clip=1.),
        arms=dict(mixed_CE=dict(reference_KL_alpha=0.,physical_GPU=0),mixed_reference_KL=dict(reference_KL_alpha=.1,physical_GPU=1)))
    settings['source_exposure_budget']=budget(data,settings['training_options']);model,_=fresh(settings,'cpu')
    settings['initial_parts_digest']=tensor_digest(model.parts.state_dict().items());del model
    guard=FileBindings(files)
    with (ROOT/'reports/code-residual-production-v4.gpu.lock').open('a+') as lease:
        fcntl.flock(lease.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        if not previous_ended(predecessor):raise ValueError('Prior worker became live; never overlap owners')
        require_idle_devices()
        guard.check(full=True);REPORTS.mkdir();OUTPUT.mkdir()
        for arm in ['initial',*settings['arms']]:
            (REPORTS/arm).mkdir();(OUTPUT/arm).mkdir();shutil.copyfile(ROOT/'checkpoints/whole-human-SFT-v1/initial/tokenizer.json',OUTPUT/arm/'tokenizer.json')
        plan=dict(**settings,created_at=datetime.now(timezone.utc).isoformat(),owner=me,waiter=owner.WAITER,source_files_sha256=files,
            expires_at_unix=time.time()+12*3600,automatic_retry=False,all_capacity_updates_discarded=True,no_capability_based_policy_selection=True,candidate_promoted=False,goal_achieved=False)
        atomic_json(REPORTS/'plan.json',plan);atomic_json(REPORTS/'children.json',[])
        def interrupted(signum,frame):raise KeyboardInterrupt('Owned code-initialized alignment interrupted')
        signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
        driver=str(Path(__file__).resolve())
        watcher=subprocess.Popen(['.venv/bin/python','-u',driver,'--watchdog'],cwd=ROOT,stdout=(REPORTS/'watchdog.log').open('x'),stderr=subprocess.STDOUT,start_new_session=True)
        def launch(args,subject,gpu):return owner.launch(plan,['-u',driver,*args],subject,str(gpu))
        def tasks(items):owner.wait_children(plan,[launch(args,subject,gpu) for args,subject,gpu in items])
        try:
            owner.idle_original();os.kill(owner.WAITER['pid'],signal.SIGSTOP);until=time.monotonic()+5
            while (owner.resources.same(owner.WAITER) or {}).get('state') not in ('T','t'):
                if time.monotonic()>until:raise ValueError('Exact idle original waiter did not enter owned hold')
                time.sleep(.05)
            if (Path('/proc')/str(owner.WAITER['pid'])/'task'/str(owner.WAITER['pid'])/'children').read_text().strip():raise ValueError('Original waiter started work; never preempt it')
            tasks([(['--worker','capacity','--gpu',str(i)],'capacity-'+str(i),i) for i in range(4)])
            proofs=[json.loads((REPORTS/f'capacity-{i}.json').read_text()) for i in range(4)]
            if not all(p['exact_actual_next_update_and_invalid_private_RNG_rejection'] for p in proofs):raise ValueError('All four full source-extreme GPU recovery probes must pass')
            atomic_json(REPORTS/'capacity.json',dict(passed=True,all_four_original_L40s=True,proofs=proofs,production_updates=0,goal_achieved=False))
            jobs=[launch(['--worker','initial','--gpu','2'],'initial-human',2)]
            for arm,spec in plan['arms'].items():jobs.append(launch(['--worker','train','--arm',arm,'--gpu',str(spec['physical_GPU'])],'train-'+arm,spec['physical_GPU']))
            owner.wait_children(plan,jobs)
            work=[(arm,suite) for arm in ['initial',*plan['arms']] for suite in ('knowledge','code','cpp','vision')]
            for start in range(0,len(work),4):
                tasks([(['--assess',suite,'--arm',arm,'--gpu',str(gpu)],'assess-'+arm+'-'+suite+'-'+str(gpu),gpu) for gpu,(arm,suite) in enumerate(work[start:start+4])])
            from slm_lab.full_math_shards_v1 import reconcile as reconcile_math
            for arm in ['initial',*plan['arms']]:
                for suite in ('public','math'):
                    tasks([(['--assess',suite,'--arm',arm,'--gpu',str(i),'--shard',str(i)],'assess-'+arm+'-'+suite+'-'+str(i),i) for i in range(4)])
                reconcile_math([REPORTS/arm/f'math-{i}.json' for i in range(4)],ROOT/'data/math-reasoning-v2/test.jsonl',OUTPUT/arm/'tokenizer.json',REPORTS/arm/'math-complete.json')
            from slm_lab.code_initialized_alignment_evidence_v1 import reconcile
            reconcile(ROOT,REPORTS,OUTPUT,data);guard.check(full=True)
            atomic_json(REPORTS/'complete.json',dict(status='Both complete human/program code-initialized CE/KL arms and all matched native comparisons finished; inspect quality',candidate_promoted=False,goal_achieved=False))
        except BaseException as error:
            atomic_json(REPORTS/'failure.json',dict(error=repr(error),automatic_retry=False,goal_achieved=False));raise
        finally:
            owner.restore_owned(plan,'Code-initialized full-source comparison ended; original work restored');watcher.wait(timeout=20)


if __name__=='__main__':
    configure();parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--watchdog',action='store_true')
    parser.add_argument('--worker',choices=['capacity','initial','train']);parser.add_argument('--assess',choices=['knowledge','code','cpp','vision','public','math'])
    parser.add_argument('--arm',default='initial');parser.add_argument('--gpu',type=int,choices=range(4),default=0);parser.add_argument('--shard',type=int,choices=range(4))
    args=parser.parse_args()
    if args.watchdog:owner.watchdog()
    elif args.worker:worker(args.worker,args.arm,args.gpu)
    elif args.assess:assess(args.arm,args.assess,args.gpu,args.shard)
    else:parent()
