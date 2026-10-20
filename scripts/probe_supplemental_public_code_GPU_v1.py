"""One separately bounded supplemental GPU capacity experiment, never production.

The exact existing four-rank owner keeps its original kernel flock and all
workers/settings. Only this experiment's registered children may be stopped.
Its CUDA worker is capped at12GiB and original phase/identity/free-memory
guards run throughout. Available memory is measured again before allocation.
"""
import argparse
import ctypes
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
OUT = Path('/tmp/slm-supplemental-public-code-GPU-capacity-v1')
OLD = ROOT / 'reports/broad-anchored-code-v7'
OLD_LOCK = ROOT / 'reports/code-residual-production-v4.gpu.lock'
NATIVE = Path('/tmp/slm-completed-joint-initializer-CPU-v3')
OLD_WORKERS = [(293164,'60395445'),(293165,'60395446'),(293166,'60395446'),(293167,'60395446')]
GIB = 1024**3


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def identity(pid):
    try: fields=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
    except FileNotFoundError:return None
    return None if fields[0] in ('Z','X') else dict(pid=int(pid),birth=fields[19])


def live(actor):return identity(actor['pid'])==actor


def write(name,value):
    path=OUT/name;pending=path.with_suffix(path.suffix+'.pending')
    with pending.open('w') as f:
        f.write(json.dumps(value,indent=2,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
    os.replace(pending,path)


def kernel_owner(path):
    st=Path(path).stat();device=os.major(st.st_dev),os.minor(st.st_dev),st.st_ino;result=[]
    for line in Path('/proc/locks').read_text().splitlines():
        fields=line.split()
        if len(fields)<8 or fields[1:4]!=['FLOCK','ADVISORY','WRITE']:continue
        major,minor,inode=fields[5].split(':')
        if (int(major,16),int(minor,16),int(inode))==device:result.append(int(fields[4]))
    if len(result)!=1:raise ValueError('The original actual exclusive kernel flock must have exactly one owner')
    return result[0]


def gpu_snapshot():
    raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.total,memory.used','--format=csv,noheader,nounits'],text=True)
    GPUs=[]
    for line in raw.splitlines():
        i,uuid,name,total,used=[v.strip() for v in line.split(',')]
        if name!='NVIDIA L40':raise ValueError('Only the existing original L40s may participate')
        GPUs.append(dict(index=int(i),uuid=uuid,total_MiB=int(total),used_MiB=int(used)))
    raw=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits'],text=True)
    apps=[]
    for line in raw.splitlines():
        uuid,pid,used=[v.strip() for v in line.split(',')]
        apps.append(dict(uuid=uuid,pid=int(pid),used_MiB=int(used)))
    return GPUs,apps


def last_step():
    with (OLD/'anchor_CE-metrics.jsonl').open('rb') as stream:
        stream.seek(0,2);stream.seek(max(0,stream.tell()-2*1024**2));raw=stream.read()
    return json.loads(next(line for line in reversed(raw.split(b'\n')[:-1]) if line))


def guard(plan,extras):
    if (not live(plan['owner']) or not live(plan['old_owner']) or time.time()>plan['expires_at_unix']
            or sha(OLD/'plan.json')!=plan['old_plan_sha256'] or kernel_owner(OLD_LOCK)!=plan['old_owner']['pid']
            or any(not live(dict(pid=p,birth=b)) for p,b in OLD_WORKERS)):
        raise ValueError('Every exact original live owner/worker, unchanged plan and original flock must remain')
    row=last_step()
    if row['method']!='anchor_CE' or row['step']+1000>=row['planned_source_updates']:
        raise ValueError('Only the unchanged original training phase with a substantial remaining source budget is eligible')
    GPUs,apps=gpu_snapshot()
    if len(GPUs)!=4 or GPUs[0]['index']!=0 or GPUs[0]['uuid']!=plan['target_uuid'] or GPUs[0]['total_MiB']<44*1024:
        raise ValueError('The exact original physicalGPU0 and four-device topology must remain')
    allowed={p for p,_ in OLD_WORKERS}|{a['pid'] for a in extras}
    if ({a['pid'] for a in apps}-allowed or not {p for p,_ in OLD_WORKERS}<={a['pid'] for a in apps}
            or any(identity(a['pid']) not in (None,a) for a in extras if a['pid'] in {v['pid'] for v in apps})
            or GPUs[0]['total_MiB']-GPUs[0]['used_MiB']<12*1024):
        raise ValueError('Only registered compute contexts and at least12GiB measured target-GPU free memory are allowed')
    return dict(at=datetime.now(timezone.utc).isoformat(),old_step=row['step'],GPUs=GPUs,compute_apps=apps,
        old_flock_owner=plan['old_owner']['pid'])


def stop_new(actor,*,protected,script):
    from slm_lab.exact_waiting_process_retirement_v1 import pidfd_open,send_term,syscall
    if actor['pid'] in protected:raise ValueError('An original/parent process can never be a cleanup target')
    if not live(actor):return
    arguments=Path(f'/proc/{actor["pid"]}/cmdline').read_bytes().decode().split('\0')
    if str(script) not in arguments or '--gpu-child' not in arguments:
        raise ValueError('Only the exact registered supplemental CUDA child may be stopped')
    fd=pidfd_open(actor['pid'])
    try:
        if not live(actor):return
        send_term(fd)
        until=time.monotonic()+8
        while live(actor) and time.monotonic()<until:time.sleep(.05)
        if live(actor):
            if hasattr(signal,'pidfd_send_signal'):signal.pidfd_send_signal(fd,signal.SIGKILL)
            else:syscall(424,ctypes.c_int(fd),ctypes.c_int(signal.SIGKILL),ctypes.c_void_p(None),ctypes.c_uint(0))
    finally:os.close(fd)


def watchdog():
    plan=json.loads((OUT/'plan.json').read_text());protected={p for p,_ in OLD_WORKERS}|{plan['old_owner']['pid'],plan['owner']['pid']}
    while not (OUT/'restored.json').exists():
        if not live(plan['owner']) or time.time()>plan['expires_at_unix']:
            for actor in json.loads((OUT/'children.json').read_text()):stop_new(actor,protected=protected,script=Path(__file__).resolve())
            write('watchdog-cleanup.json',dict(own_registered_children_only=True,old_processes_signalled=0));return
        time.sleep(.2)


def child(profile):
    plan=json.loads((OUT/'plan.json').read_text());me=identity(os.getpid())
    while me not in json.loads((OUT/'children.json').read_text()):
        if not live(plan['owner']) or time.time()>plan['expires_at_unix']:raise ValueError('Unregistered child cannot initialize CUDA')
        time.sleep(.05)
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='0':raise ValueError('Only the explicitly budgeted original physicalGPU0 is visible')
    def checked():return guard(plan,[a for a in json.loads((OUT/'children.json').read_text()) if live(a)])
    checked()
    import random
    import numpy as np
    import torch
    from slm_lab.codenet_checkpoint_v1 import capture_rng
    from slm_lab.own_dense_actor_v1 import OwnDenseActor
    from slm_lab.own_dense_actor_checkpoint_v1 import make_state,restore_state
    from slm_lab.own_native_query_v1 import inspect_package,load_package
    from slm_lab.parallel_function_repair_stage_v1 import value_digest
    from slm_lab.public_reasoned_code_train_data_v1 import PublicReasonedCodeSFTData
    from slm_lab.serial_public_code_bf16_step_v1 import update
    from slm_lab.whole_human_sft_evidence_v1 import tensor_digest
    torch.set_num_threads(1);torch.set_num_interop_threads(1);os.sched_setaffinity(0,{20})
    data=PublicReasonedCodeSFTData(ROOT,purpose='capacity');entries=data.complete_schedule(epochs=1,width=16,seed=990231)
    receipt=json.loads((NATIVE/'complete.json').read_text())
    if sha(data.folder/'tokenizer.json')!=receipt['tokenizer_sha256']:
        raise ValueError('Admitted data and own completed model must use the exact same tokenizer bytes')
    package=inspect_package(receipt['checkpoint'],expected_sha256=receipt['checkpoint_sha256'],
        expected_tokenizer_sha256=receipt['tokenizer_sha256'],expected_sources=receipt['sources'])
    with torch.random.fork_rng(devices=[]):base,_=load_package(package);model=OwnDenseActor(base,rank=16,seed=1003311).eval()
    if len(data)!=3675 or len(entries)!=230 or model.counts()['actual_unique']!=999706442:raise ValueError('Exact admitted full source and own999Mactor required')
    checked();torch.cuda.set_device(0)
    total=torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(12*GIB/total,0)
    torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if torch.cuda.get_device_name(0)!='NVIDIA L40':raise ValueError('Original L40 required')
    model.to('cuda:0');model.assert_shared();base_hash=tensor_digest(model.base.named_parameters())
    adam=torch.optim.AdamW([p for _,p in model.named_selected(('actor',))],lr=3e-5,betas=(.9,.95),eps=1e-8,weight_decay=.01,foreach=False,fused=False)
    selected_groups=[]
    for field in ['complete_context_tokens','response_EOS_targets','original_prompt_positions']:
        position=max(range(len(data)),key=lambda p:data.information(p)[field]);index=next(i for i,e in enumerate(entries) if position in e)
        if index not in selected_groups:selected_groups.append(index)
    largest=max(range(len(entries)),key=lambda i:sum(data.information(p)['complete_context_tokens'] for p in entries[i]))
    for i in [largest,len(entries)-1]:
        if i not in selected_groups:selected_groups.append(i)
    if len(selected_groups)<2:raise ValueError('Multiple whole original capacity groups and actual tail required')
    schedule=[entries[i] for i in selected_groups];records=[]
    sources=dict(initial_native_sha256=receipt['checkpoint_sha256'],tokenizer_sha256=receipt['tokenizer_sha256'],
        public_manifest_sha256=data.manifest_sha256,public_admission_sha256=data.admission_sha256,capacity_plan_sha256=sha(OUT/'plan.json'))
    random.seed(1003321);np.random.seed(1003321);torch.manual_seed(1003321);torch.cuda.manual_seed(1003321)
    folder=OUT/profile;folder.mkdir();write(profile+'/sources.json',dict(selected_original_global_batches=selected_groups,
        every_selected_source_information=[[data.information(p) for p in e] for e in schedule],complete_epoch_budget=data.budget(entries)))
    def step(i):
        row=update(model,adam,data,schedule[i],profile,checked=checked);records.append(row)
        write(profile+'/progress.json',dict(completed=len(records),planned=len(schedule),last=row,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),production_updates=0))
        return row
    first=step(0);state=make_state(model,adam,groups=('actor',),sources=sources,completed=1,planned=len(schedule),
        cursor={'next_capacity_entry':1},metrics=records,device='cuda:0');torch.save(state,folder/'after-first.pt')
    expected=step(1);expected_parts=tensor_digest(model.parts.named_parameters());expected_adam=value_digest(adam.state_dict());expected_rng=value_digest(capture_rng('cuda:0'))
    restore_state(model,adam,state,groups=('actor',),expected_sources=sources,planned=len(schedule),expected_cursor={'next_capacity_entry':1},device='cuda:0')
    records=[first]
    if step(1)!=expected or tensor_digest(model.parts.named_parameters())!=expected_parts or value_digest(adam.state_dict())!=expected_adam or value_digest(capture_rng('cuda:0'))!=expected_rng:
        raise ValueError('Exact next source/actor/Adam/private CUDA RNG recovery required')
    del state
    for i in range(2,len(schedule)):step(i)
    model.eval();checked();data.audit_provenance()
    if tensor_digest(model.base.named_parameters())!=base_hash:raise ValueError('Every original frozen native value must remain exact')
    result=dict(complete=True,profile=profile,actual_unique_parameters=model.counts()['actual_unique'],
        actual_selected_Adam_states=len(adam.state),actual_logical_updates=len(schedule),physical_optimizer_calls=len(schedule)+1,
        selected_original_global_batch_indices=selected_groups,every_capacity_source_preserved_whole=True,
        actual_final_epoch_tail=len(entries[-1]),exact_next_source_weights_Adam_RNG_recovery=True,
        original_frozen_native_values_unchanged=True,critic_is_initially_zero_not_a_trained_PPO_claim=True,
        full_source_provenance_rechecked=True,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        hard_new_allocator_limit_bytes=12*GIB,records=records,production_updates=0,
        full_epoch_or_model_quality_improvement_proved=False,goal_achieved=False)
    write(profile+'/complete.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='records'}),flush=True)


def parent():
    if OUT.exists() or os.environ.get('CUDA_VISIBLE_DEVICES')!='' or 'torch' in sys.modules:raise ValueError('Fresh CUDA-hidden non-neural parent required')
    from slm_lab.local_source_closure_v2 import python_sources
    old=json.loads((OLD/'plan.json').read_text());owner=dict(pid=old['owner']['pid'],birth=old['owner']['ticks'])
    GPUs,apps=gpu_snapshot();tests=ROOT/'reports/serial-public-code-bf16-step-v1-tests.log';assert '4 passed' in tests.read_text()
    resource_tests=ROOT/'reports/supplemental-public-code-resource-v1-tests.log';assert '2 passed' in resource_tests.read_text()
    files={str(p):sha(p) for p in [*python_sources(ROOT,[Path(__file__).resolve()]),tests,ROOT/'tests/test_serial_public_code_bf16_step_v1.py',
        resource_tests,ROOT/'tests/test_supplemental_public_code_resources_v1.py']}
    receipt=json.loads((NATIVE/'complete.json').read_text())
    if receipt.get('complete') is not True or receipt['actual_counts']['total_unique']!=984399690:
        raise ValueError('The actual completed own native source must remain exact')
    for p in [OLD/'plan.json',NATIVE/'complete.json',Path(receipt['checkpoint']),Path(receipt['checkpoint']).parent/'tokenizer.json',
            ROOT/'data/public-reasoned-code-SFT-v1/manifest.json',ROOT/'data/public-reasoned-code-SFT-v1/admission.json',
            ROOT/'data/public-reasoned-code-SFT-v1/tokenizer.json',ROOT/'reports/public-reasoned-code-pack-admission-v1/complete.json',
            ROOT/'reports/public-reasoned-code-train-data-CPU-v1/complete.json',ROOT/'reports/public-reasoned-code-stopping-plan-v1/complete.json']:
        files[str(p)]=sha(p)
    if files[receipt['checkpoint']]!=receipt['checkpoint_sha256']:
        raise ValueError('The complete source model checksum must match its independent receipt')
    for mode in ['anchor_CE','anchor_execution']:
        for rank in range(4):
            p=OLD/f'capacity-{mode}-{rank}.json';v=json.loads(p.read_text())
            if v['peak_reserved_bytes']>12*GIB or v['maximum_forced_action_horizon']!=3059:raise ValueError('Measured original full-source headroom differs')
            files[str(p)]=sha(p)
    if GPUs[0]['used_MiB']>16*1024:raise ValueError('Original target usage exceeds the separately budgeted preflight window')
    memory=dict(line.split()[:2] for line in Path('/sys/fs/cgroup/memory.stat').read_text().splitlines())
    available=int(Path('/sys/fs/cgroup/memory.max').read_text())-sum(int(memory[k]) for k in ['anon','shmem','kernel'])
    if available<80*GIB or __import__('shutil').disk_usage(OUT.parent).free<48*GIB:raise ValueError('Full own and other-job host/scratch budgets required')
    plan=dict(owner=identity(os.getpid()),old_owner=owner,old_plan_sha256=sha(OLD/'plan.json'),
        target_uuid=GPUs[0]['uuid'],expires_at_unix=time.time()+1200,source_files_sha256=files,
        old_global_flock_kept_by_its_original_owner=True,separate_supplemental_worker_memory_contract=True,
        new_allocator_cap_bytes=12*GIB,old_workers_and_settings_not_modified=True,capacity_only=True,production_updates=0,goal_achieved=False)
    before=guard(plan,[]);OUT.mkdir();write('plan.json',plan);write('children.json',[]);write('original-before.json',before)
    protected={p for p,_ in OLD_WORKERS}|{owner['pid'],os.getpid()}
    with (OUT/'supplemental-control.lock').open('x') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        watcher=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'--watchdog'],cwd=ROOT,
            stdout=(OUT/'watchdog.log').open('x'),stderr=subprocess.STDOUT,env={**os.environ,'CUDA_VISIBLE_DEVICES':''})
        children=[]
        try:
            for profile in ['plain_tokens','eos_weight4']:
                guard(plan,[])
                with (OUT/(profile+'.log')).open('x') as log:
                    process=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'--gpu-child',profile],cwd=ROOT,
                        stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'CUDA_VISIBLE_DEVICES':'0','CUBLAS_WORKSPACE_CONFIG':':4096:8',
                            'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'})
                    actor=identity(process.pid);children.append(actor);write('children.json',children)
                    while process.poll() is None:
                        guard(plan,[actor]);time.sleep(1)
                    if process.returncode!=0:raise ValueError('Supplemental capacity child failed; retain its exact log without retrying')
                    until=time.monotonic()+20
                    while actor['pid'] in {a['pid'] for a in gpu_snapshot()[1]}:
                        guard(plan,[actor])
                        if time.monotonic()>until:raise ValueError('Completed own CUDA context has not released; preserve evidence')
                        time.sleep(.2)
            if any(sha(p)!=h for p,h in files.items()):raise ValueError('Bound original/new runtime source changed')
            after=guard(plan,[]);write('original-after.json',after)
            write('complete.json',dict(complete=True,source_files_sha256=files,profiles=['plain_tokens','eos_weight4'],
                original_training_step_before=before['old_step'],original_training_step_after=after['old_step'],
                original_owner_flock_workers_and_sources_preserved=True,old_processes_signalled=0,
                supplemental_GPU_contexts_released=True,production_training_admitted=False,production_updates=0,
                no_full_epoch_or_quality_gain_claim=True,goal_achieved=False))
        except BaseException as error:
            write('failure.json',dict(error=repr(error),traceback=traceback.format_exc(),no_automatic_retry=True,partial_outputs_retained=True,goal_achieved=False));raise
        finally:
            for actor in children:stop_new(actor,protected=protected,script=Path(__file__).resolve())
            write('restored.json',dict(own_children_terminal=all(not live(a) for a in children),old_processes_signalled=0))
            watcher.wait(timeout=10)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',action='store_true');parser.add_argument('--watchdog',action='store_true')
    parser.add_argument('--gpu-child',choices=['plain_tokens','eos_weight4']);args=parser.parse_args()
    if args.watchdog:watchdog()
    elif args.gpu_child:child(args.gpu_child)
    elif args.run:parent()
    else:parser.error('Use explicit --run or a registered own child/watchdog mode')
