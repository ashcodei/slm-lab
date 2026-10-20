"""Queue full-language capacity/recovery after the exact coding GPU owner.

No CUDA context or lease is taken while waiting. The original four workers
remain protected. Registered GPU0 children have a22GiB allocator cap and
must preserve12GiB measured free memory; all probe learning is discarded.
"""
import argparse
from datetime import datetime,timezone
import fcntl
import gc
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

import probe_supplemental_public_code_GPU_v1 as original

ROOT=original.ROOT
PREVIOUS=Path('/tmp/slm-supplemental-public-code-full-GPU-v1')
OUT=Path('/tmp/slm-supplemental-dense-code-GPU-capacity-v1')
SCRIPT=Path(__file__).resolve()
LOCK=ROOT/'reports/supplemental-public-code-GPU0.lock'
RATES={'rate3e6':3e-6,'rate1e5':1e-5}
GIB=1024**3
original.OUT=OUT
sha,live,identity,write=original.sha,original.live,original.identity,original.write


def read(p):return json.loads(Path(p).read_text())


def previous_ready(actor):
    if live(actor):return False
    if (PREVIOUS/'failure.json').exists() or not (PREVIOUS/'complete.json').is_file():
        raise ValueError('Exact prior GPU owner ended without the complete matched experiment')
    done=read(PREVIOUS/'complete.json');released=read(PREVIOUS/'restored.json')
    if (not done['complete'] or not done['full_matched_assessments_complete']
            or not done['supplemental_GPU_contexts_released'] or done['full_source_training_updates_per_arm']!=230
            or not released['own_children_terminal'] or done['old_processes_signalled']!=0):
        raise ValueError('Complete source epochs, matched assessments and released own contexts are required')
    for endpoint in ['baseline','plain_tokens','eos_weight4']:
        j=read(PREVIOUS/endpoint/'assessment-complete.json')
        if (not j['complete'] or j['programs']!=48 or not j['all425reserved_views_scored']
                or not j['original_bounded_knowledge_math_OCR_object_and_grounding_checks_complete']
                or j['actual_registered_parameters']!=999706442):
            raise ValueError('Every original complete code, held and broad assessment must precede new CUDA')
    return True


def settings():
    return [dict(actor=dict(pid=p,birth=b),nice=os.getpriority(os.PRIO_PROCESS,p),
        scheduler=os.sched_getscheduler(p),affinity=sorted(os.sched_getaffinity(p))) for p,b in original.OLD_WORKERS]


def checked(plan):
    if (original.kernel_owner(LOCK)!=plan['owner']['pid'] or settings()!=plan['original_worker_settings']
            or shutil.disk_usage(OUT.parent).free<40*GIB):
        raise ValueError('Own exact lease, unchanged original worker settings and scratch reserve required')
    return original.guard(plan,read(OUT/'children.json'))


def stop(actor,plan):
    protected={p for p,_ in original.OLD_WORKERS}|{plan['owner']['pid'],plan['old_owner']['pid'],plan['predecessor']['pid']}
    original.stop_new(actor,protected=protected,script=SCRIPT)


def watchdog():
    plan=read(OUT/'plan.json')
    while not (OUT/'restored.json').exists():
        if not live(plan['owner']) or time.time()>plan['expires_at_unix']:
            for actor in read(OUT/'children.json'):stop(actor,plan)
            write('watchdog-cleanup.json',dict(own_registered_children_only=True,old_processes_signalled=0));return
        time.sleep(.2)


def child(profile):
    plan=read(OUT/'plan.json');me=identity(os.getpid())
    while me not in read(OUT/'children.json'):
        if not live(plan['owner']) or time.time()>plan['expires_at_unix']:raise ValueError('Unregistered child cannot allocate CUDA')
        time.sleep(.05)
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='0':raise ValueError('Only physicalGPU0 may be visible')
    checked(plan)
    import random
    import numpy as np
    import torch
    from slm_lab.codenet_checkpoint_v1 import atomic_checkpoint,capture_rng
    from slm_lab.own_dense_checkpoint_v1 import make_state,restore_state
    from slm_lab.own_native_query_v1 import inspect_package,load_package
    from slm_lab.parallel_function_repair_stage_v1 import value_digest
    from slm_lab.public_reasoned_code_train_data_v1 import PublicReasonedCodeSFTData
    from slm_lab.serial_dense_code_bf16_step_v1 import update
    from slm_lab.whole_human_sft_evidence_v1 import tensor_digest
    torch.set_num_threads(1);torch.set_num_interop_threads(1);os.sched_setaffinity(0,{20})
    data=PublicReasonedCodeSFTData(ROOT,purpose='capacity');entries=data.complete_schedule(epochs=1,width=16,seed=990231)
    if len(data)!=3675 or len(entries)!=230:raise ValueError('Every original admitted source and unchanged schedule required')
    native=plan['native'];package=inspect_package(native['checkpoint'],expected_sha256=native['checkpoint_sha256'],
        expected_tokenizer_sha256=native['tokenizer_sha256'],expected_sources=native['sources'])
    def fresh():
        with torch.random.fork_rng(devices=[]):model,_=load_package(package)
        if (model.counts()['total_unique']!=984399690 or len(model.language_named_parameters())!=317
                or sum(p.numel() for _,p in model.language_named_parameters())!=931244544):
            raise ValueError('Actual own984M native and every931M dense language value required')
        model.to('cuda:0')
        adam=torch.optim.AdamW([p for _,p in model.language_named_parameters()],lr=RATES[profile],
            betas=(.9,.95),eps=1e-8,weight_decay=.01,foreach=False,fused=False)
        return model,adam
    checked(plan);torch.cuda.set_device(0)
    total=torch.cuda.get_device_properties(0).total_memory;torch.cuda.set_per_process_memory_fraction(22*GIB/total,0)
    torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if torch.cuda.get_device_name(0)!='NVIDIA L40':raise ValueError('The original L40 is required')
    model,adam=fresh();indices=[]
    for field in ['complete_context_tokens','response_EOS_targets','original_prompt_positions']:
        p=max(range(len(data)),key=lambda p:data.information(p)[field]);i=next(i for i,e in enumerate(entries) if p in e)
        if i not in indices:indices.append(i)
    largest=max(range(len(entries)),key=lambda i:sum(data.information(p)['complete_context_tokens'] for p in entries[i]))
    for i in [largest,len(entries)-1]:
        if i not in indices:indices.append(i)
    if indices!=[137,61,95,229]:raise ValueError('All original source-extreme groups and genuine eleven-source tail required')
    groups=[entries[i] for i in indices];folder=OUT/profile;folder.mkdir();records=[]
    fixed_names={n for n,_ in model.language_named_parameters()}
    def fixed_hash(m):
        return value_digest(dict(network={n:t.detach().cpu() for n,t in m.network.state_dict().items()
            if n not in fixed_names and n!='lm_head.weight'},heads={n:t.detach().cpu() for n,t in m.heads.state_dict().items()}))
    fixed=fixed_hash(model);initial_language=tensor_digest(model.language_named_parameters())
    sources=dict(initial_native_sha256=native['checkpoint_sha256'],tokenizer_sha256=native['tokenizer_sha256'],
        public_manifest_sha256=data.manifest_sha256,public_admission_sha256=data.admission_sha256,
        capacity_plan_sha256=sha(OUT/'plan.json'),capacity_admission_sha256=sha(OUT/'admission.json'),
        discarded_capacity_only='9'*64)
    random.seed(1007719);np.random.seed(1007719);torch.manual_seed(1007719);torch.cuda.manual_seed(1007719)
    def step(i):return update(model,adam,data,groups[i],'plain_tokens',checked=lambda:checked(plan))
    def state(i):return make_state(model,adam,sources=sources,completed=i,planned=len(groups),
        cursor={'next_capacity_group':i},metrics=records,device='cuda:0')
    records.append(step(0));first=state(1)
    atomic_checkpoint(folder/'after-first.pt',first,16*GIB,40*GIB);first_hash=value_digest(first);del first
    records.append(step(1));expected=state(2);expected_hash=value_digest(expected)
    atomic_checkpoint(folder/'expected-second.pt',expected,16*GIB,40*GIB);expected_metrics=records.copy();del expected
    # A genuinely fresh graph/Adam avoids holding two full GPU moment sets.
    # Neither source snapshot is mutated or used as a production initializer.
    del model,adam;gc.collect();torch.cuda.empty_cache();checked(plan)
    model,adam=fresh();first=torch.load(folder/'after-first.pt',map_location='cpu',weights_only=True,mmap=True)
    if value_digest(first)!=first_hash:raise ValueError('Every original full recovery payload byte must remain exact')
    _,records=restore_state(model,adam,first,expected_sources=sources,planned=len(groups),
        expected_cursor={'next_capacity_group':1},device='cuda:0');del first
    records.append(step(1));replayed=state(2)
    if value_digest(replayed)!=expected_hash or records!=expected_metrics:
        raise ValueError('Every learned value, full317Adam states, private RNG and whole-source metric must replay exactly')
    del replayed
    for i in range(2,len(groups)):records.append(step(i))
    final=state(len(groups));atomic_checkpoint(folder/'final-capacity-state.pt',final,16*GIB,40*GIB)
    if (fixed_hash(model)!=fixed or tensor_digest(model.language_named_parameters())==initial_language
            or len(adam.state)!=317 or any(float(adam.state[p]['step'])!=len(groups) for _,p in model.language_named_parameters())):
        raise ValueError('Only all dense language weights/Adam clocks may learn; every vision/critic/auxiliary value stays exact')
    data.audit_provenance();checked(plan)
    write(profile+'/complete.json',dict(complete=True,profile=profile,learning_rate=RATES[profile],
        actual_unique_parameters=model.counts()['total_unique'],selected_full_language_values=931244544,
        actual_selected_Adam_states=len(adam.state),selected_original_global_batches=indices,
        original_groups=[[data.information(p) for p in g] for g in groups],actual_final_epoch_tail=len(groups[-1]),
        logical_updates=len(groups),physical_optimizer_calls=len(groups)+1,exact_fresh_native_Adam_RNG_next_update_recovery=True,
        every_fixed_vision_critic_auxiliary_value_unchanged=True,initial_language_sha256=initial_language,
        final_language_sha256=tensor_digest(model.language_named_parameters()),sources=sources,
        first_snapshot_sha256=sha(folder/'after-first.pt'),expected_second_sha256=sha(folder/'expected-second.pt'),
        final_capacity_state_sha256=sha(folder/'final-capacity-state.pt'),records=records,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        hard_allocator_limit_bytes=22*GIB,capacity_learning_discarded=True,production_updates=0,
        full_source_epoch_or_quality_gain_proved=False,candidate_promoted=False,goal_achieved=False))


def parent():
    if (OUT.exists() or os.environ.get('CUDA_VISIBLE_DEVICES')!='' or 'torch' in sys.modules
            or os.nice(0)!=0 or os.sched_getscheduler(0)!=os.SCHED_OTHER):
        raise ValueError('Fresh normal-priority non-neural CUDA-hidden queue owner required')
    from slm_lab.local_source_closure_v2 import python_sources
    tests=ROOT/'reports/serial-dense-code-bf16-step-v1-tests.log'
    if '4 passed' not in tests.read_text():raise ValueError('Actual full-language scalar-gradient and fresh-state recovery controls must pass')
    resources=ROOT/'reports/supplemental-dense-code-capacity-resources-v1-tests.log'
    if '2 passed' not in resources.read_text():raise ValueError('Actual predecessor/full-scope admission and own-only watchdog controls must pass')
    previous=read(PREVIOUS/'plan.json');native=read(original.NATIVE/'complete.json');old=read(original.OLD/'plan.json')['owner']
    paths=[*python_sources(ROOT,[SCRIPT]),tests,ROOT/'tests/test_serial_dense_code_bf16_step_v1.py',
        resources,ROOT/'tests/test_supplemental_dense_code_capacity_resources_v1.py',
        PREVIOUS/'plan.json',original.NATIVE/'complete.json']
    files={str(p):sha(p) for p in paths};OUT.mkdir();write('children.json',[])
    plan=dict(owner=identity(os.getpid()),predecessor=previous['owner'],old_owner=dict(pid=old['pid'],birth=old['ticks']),
        old_plan_sha256=sha(original.OLD/'plan.json'),target_uuid=previous['target_uuid'],expires_at_unix=time.time()+12*3600,
        native=native,source_files_sha256=files,original_worker_settings=settings(),rates=RATES,
        hard_own_allocator_limit_bytes=22*GIB,minimum_measured_GPU_free_bytes=12*GIB,
        original_source_population=3675,original_epoch_updates=230,capacity_selected_groups=[137,61,95,229],
        purpose='capacity_only_full_dense_language_learning_discarded',no_lease_or_CUDA_while_waiting=True,
        full_production_epoch_and_matched_quality_still_required=True,candidate_promoted=False,goal_achieved=False)
    write('plan.json',plan)
    try:
        while not previous_ready(plan['predecessor']):
            if time.time()>plan['expires_at_unix'] or any(sha(p)!=h for p,h in files.items()):
                raise ValueError('Exact queued runtime/source bytes and time budget must remain')
            write('progress.json',dict(phase='wait_exact_complete_matched_GPU_owner',predecessor=plan['predecessor'],
                actual_predecessor_confirmed_live=True,no_GPU_context_or_lease=True));time.sleep(10)
        memory=dict(line.split()[:2] for line in Path('/sys/fs/cgroup/memory.stat').read_text().splitlines())
        if (int(Path('/sys/fs/cgroup/memory.max').read_text())-sum(int(memory[k]) for k in ['anon','shmem','kernel'])<80*GIB
                or shutil.disk_usage(OUT.parent).free<168*GIB):
            raise ValueError('Complete actual full dense/Adam host and128GiB own/40GiB scratch funding required')
        GPUs,_=original.gpu_snapshot()
        if GPUs[0]['total_MiB']-GPUs[0]['used_MiB']<34*1024:
            raise ValueError('22GiB own allocator plus12GiB measured free margin must be funded before new CUDA')
        closed=[PREVIOUS/'complete.json',PREVIOUS/'restored.json']
        closed += [PREVIOUS/p/'assessment-complete.json' for p in ['baseline','plain_tokens','eos_weight4']]
        admission_files={**files,**{str(p):sha(p) for p in closed}}
        write('admission.json',dict(complete=True,source_files_sha256=admission_files,
            exact_predecessor_terminal_and_complete=True,all_original_assessments_preserved=True,capacity_only=True))
        with LOCK.open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);before=checked(plan);write('original-before.json',before)
            watcher=subprocess.Popen([sys.executable,'-u',str(SCRIPT),'--watchdog'],cwd=ROOT,
                stdout=(OUT/'watchdog.log').open('x'),stderr=subprocess.STDOUT,env={**os.environ,'CUDA_VISIBLE_DEVICES':''})
            children=[]
            try:
                for profile in RATES:
                    checked(plan)
                    with (OUT/(profile+'.log')).open('x') as log:
                        process=subprocess.Popen([sys.executable,'-u',str(SCRIPT),'--gpu-child',profile],cwd=ROOT,
                            stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'CUDA_VISIBLE_DEVICES':'0',
                            'CUBLAS_WORKSPACE_CONFIG':':4096:8','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'})
                        actor=identity(process.pid);children.append(actor);write('children.json',children)
                        write('progress.json',dict(phase='actual_full_dense_GPU_capacity',profile=profile,child=actor))
                        while process.poll() is None:checked(plan);time.sleep(1)
                        if process.returncode!=0:raise ValueError('Registered full dense capacity child failed; preserve without automatic retry')
                    until=time.monotonic()+20
                    while actor['pid'] in {a['pid'] for a in original.gpu_snapshot()[1]}:
                        checked(plan)
                        if time.monotonic()>until:raise ValueError('Own completed CUDA context has not released')
                        time.sleep(.2)
                if any(sha(p)!=h for p,h in admission_files.items()):raise ValueError('A queued/closed original source byte changed')
                after=checked(plan);write('original-after.json',after)
                write('complete.json',dict(complete=True,profiles=list(RATES),source_files_sha256=admission_files,
                    both_actual_full_dense_capacity_and_fresh_recovery_complete=True,old_processes_signalled=0,
                    original_step_before=before['old_step'],original_step_after=after['old_step'],
                    supplemental_GPU_contexts_released=True,original_worker_settings_preserved=True,
                    all_probe_learning_discarded=True,production_updates=0,full_source_training_and_quality_pending=True,goal_achieved=False))
            finally:
                for actor in children:stop(actor,plan)
                write('restored.json',dict(own_children_terminal=all(not live(a) for a in children),old_processes_signalled=0))
                watcher.wait(timeout=10)
    except BaseException as error:
        write('failure.json',dict(error=repr(error),traceback=traceback.format_exc(),partial_outputs_retained=True,
            no_automatic_retry=True,old_processes_signalled=0,goal_achieved=False));raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',action='store_true')
    parser.add_argument('--watchdog',action='store_true');parser.add_argument('--gpu-child',choices=list(RATES));args=parser.parse_args()
    if args.watchdog:watchdog()
    elif args.gpu_child:child(args.gpu_child)
    elif args.run:parent()
    else:parser.error('Explicit --run or owned child/watchdog mode required')
