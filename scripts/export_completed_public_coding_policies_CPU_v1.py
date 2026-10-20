"""Deliver both completed coding policies without acquiring any GPU context.

The non-neural owner waits for each exact230-update receipt, then exports and
cold-checks its complete unfused policy. Separate fresh query processes are
forbidden from reading training/ancestor files or calling network services.
"""
import argparse
import contextlib
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
RUN = Path('/tmp/slm-supplemental-public-code-full-GPU-v1')
CAPACITY = Path('/tmp/slm-supplemental-public-code-GPU-capacity-v1')
OUT = Path('/tmp/slm-completed-public-coding-standalone-CPU-v1')
SCRIPT = Path(__file__).resolve()
PROFILES = ('plain_tokens','eos_weight4')
GIB = 1024**3


def read(path): return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def identity(pid):
    try: fields=Path('/proc',str(pid),'stat').read_text().rsplit(')',1)[1].split()
    except FileNotFoundError:return None
    return None if fields[0] in ('Z','X') else dict(pid=int(pid),birth=fields[19])


def write(name,value):
    path=OUT/name;pending=path.with_suffix(path.suffix+'.pending')
    with pending.open('x') as stream:
        stream.write(json.dumps(value,indent=2,allow_nan=False)+'\n');stream.flush();os.fsync(stream.fileno())
    os.replace(pending,path)


def checked(plan):
    if (identity(plan['owner']['pid'])!=plan['owner'] or time.time()>plan['expires_at_unix']
            or shutil.disk_usage(OUT.parent).free<40*GIB):
        raise ValueError('The exact live CPU owner, expiry and unchanged scratch reserve are required')


def bound_sources(run,profile):
    return dict(initial_native_sha256=run['native']['checkpoint_sha256'],tokenizer_sha256=run['native']['tokenizer_sha256'],
        public_manifest_sha256=run['data_manifest_sha256'],public_admission_sha256=run['data_admission_sha256'],
        run_plan_sha256=sha(RUN/'plan.json'),objective_sha256=hashlib.sha256(profile.encode()).hexdigest())


def export_child(profile):
    started=time.monotonic();plan=read(OUT/'plan.json');checked(plan)
    import torch
    from tokenizers import Tokenizer
    from slm_lab.codenet_checkpoint_v1 import FileBindings,atomic_checkpoint
    from slm_lab.completed_public_coding_policy_v1 import install_completed_policy
    from slm_lab.own_coding_policy_standalone_v1 import export,load
    from slm_lab.own_dense_actor_v1 import OwnDenseActor
    from slm_lab.own_native_query_v1 import inspect_package,load_package
    from slm_lab.public_reasoned_code_train_data_v1 import PublicReasonedCodeSFTData
    from slm_lab.whole_human_sft_evidence_v1 import tensor_digest
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    guard=FileBindings(plan['source_files_sha256']);guard.check(full=True)
    run=read(RUN/'plan.json');native=run['native'];proof=read(RUN/profile/'training-complete.json')
    source_receipts={str(RUN/profile/'training-complete.json'):sha(RUN/profile/'training-complete.json'),
        proof['final_state']['checkpoint']:proof['final_state']['checkpoint_sha256'],
        str(RUN/profile/'actual-updates.jsonl'):proof['actual_update_journal_sha256']}
    source_guard=FileBindings(source_receipts);source_guard.check(full=True)
    if proof['final_state']['completed']!=230 or Path(proof['final_state']['checkpoint']).name!='final-state.pt':
        raise ValueError('Only the complete final230-update state may be delivered')
    package=inspect_package(native['checkpoint'],expected_sha256=native['checkpoint_sha256'],
        expected_tokenizer_sha256=native['tokenizer_sha256'],expected_sources=native['sources'])
    with torch.random.fork_rng(devices=[]):
        base,_=load_package(package);model=OwnDenseActor(base,rank=16,seed=1003311).eval()
    rng=torch.get_rng_state().clone();data=PublicReasonedCodeSFTData(ROOT,purpose='optimization',views='both')
    state=torch.load(proof['final_state']['checkpoint'],map_location='cpu',weights_only=True,mmap=True)
    capacity=torch.load(CAPACITY/profile/'after-first.pt',map_location='cpu',weights_only=True,mmap=True)
    report=install_completed_policy(model,data,state,proof,profile=profile,expected_sources=bound_sources(run,profile),
        optimizer_options=dict(lr=3e-5,betas=(.9,.95),eps=1e-8,weight_decay=.01,foreach=False,fused=False),
        expected_capacity_rng=capacity['rng'])
    journal=[json.loads(line) for line in (RUN/profile/'actual-updates.jsonl').read_text().splitlines()]
    if (len(journal)!=230 or any(row!={'step':i+1,**state['metrics'][i]} for i,row in enumerate(journal))
            or not torch.equal(rng,torch.get_rng_state())):
        raise ValueError('Every original journal record and caller RNG must remain exact without training-state resume')
    folder=OUT/profile;folder.mkdir();write(profile+'/complete-training-readback.json',report)
    del state,capacity,journal
    shutil.copyfile(Path(native['checkpoint']).parent/'tokenizer.json',folder/'tokenizer.json')
    path=folder/'model.pt';bound=bound_sources(run,profile)
    payload=export(model,sources=bound,tokenizer_sha256=native['tokenizer_sha256'],completed_updates=230,
        provenance=dict(profile=profile,complete_original_epoch=True,source_final_state_sha256=proof['final_state']['checkpoint_sha256'],
            source_training_receipt_sha256=source_receipts[str(RUN/profile/'training-complete.json')],
            original_complete_training_budget=proof['original_complete_training_budget'],
            full_matched_quality_and_CUDA_BF16_cold_still_required=True,candidate_promoted=False))
    atomic_checkpoint(path,payload,5*GIB,40*GIB);del payload
    write(profile+'/progress.json',dict(phase='standalone_exported',completed_updates=230,checkpoint_sha256=sha(path)))
    cold,meta=load(path,expected_checkpoint_sha256=sha(path),expected_sources=bound)
    if tensor_digest(cold.named_parameters())!=tensor_digest(model.named_parameters()):
        raise ValueError('Every native/policy/critic value must cold-load exactly')
    tokenizer=Tokenizer.from_file(str(folder/'tokenizer.json'))
    short=torch.tensor([[1,*tokenizer.encode('User: Write a Python function to add two integers.\nAssistant:').ids]])
    full=torch.tensor([[1]+[4]*4094+[2]])
    photo=torch.linspace(-1,1,3*448*448).reshape(1,3,448,448)
    words=torch.linspace(-1,1,16*3*64*448).reshape(16,3,64,448)
    outputs={}
    with torch.no_grad():
        for name,ids,pixels in [('short_text',short,None),('full4096_synthetic_text',full,None),('photo448',short,photo)]:
            checked(plan);a,b=model(ids,pixel_values=pixels),cold(ids,pixel_values=pixels)
            if any(not torch.equal(a[k],b[k]) for k in ['logits','values']):
                raise ValueError('All complete original CPU text/photo/critic outputs must remain exact')
            outputs[name]={k:dict(shape=list(b[k].shape),sha256=tensor_digest([(k,b[k])])) for k in ['logits','values']}
            del a,b
            write(profile+'/progress.json',dict(phase='cold_outputs',completed=list(outputs)))
        a,b=model.base.word_logits(words),cold.base.word_logits(words)
        if list(a.shape)!=[16,140,257] or not torch.equal(a,b):raise ValueError('All complete16wholeword outputs must remain exact')
        outputs['whole_words']=dict(shape=list(b.shape),sha256=tensor_digest([('logits',b)]));del a,b
        for name,pixels in [('text',None),('photo',photo)]:
            a,b=model.generate(short,pixels,max_new_tokens=8,eos_token_id=2),cold.generate(short,pixels,max_new_tokens=8,eos_token_id=2)
            if not torch.equal(a,b):raise ValueError('Original cached greedy actions must remain exact')
            outputs[name+'_actions']=b[0,short.shape[1]:].tolist()
    data.audit_provenance();source_guard.check(full=True);guard.check(full=True);checked(plan)
    if torch.cuda.is_initialized() or not torch.equal(rng,torch.get_rng_state()):
        raise ValueError('No GPU context or caller RNG mutation is permitted for completed CPU delivery')
    write(profile+'/complete.json',dict(complete=True,profile=profile,checkpoint=str(path),checkpoint_sha256=sha(path),
        tokenizer_sha256=native['tokenizer_sha256'],sources=bound,completed_updates=230,
        actual_registered_parameters=meta['actual_counts']['actual_unique'],original_complete_epoch_reconciled=True,
        all_actual_values_and_full4096text448photo16wholeword_and_cached_actions_CPU_exact=True,
        synthetic_full_context_is_not_quality=True,unfused_operators_preserved=True,source_files_sha256=source_receipts,
        outputs=outputs,caller_RNG_unchanged=True,GPU_initialized=False,elapsed_seconds=time.monotonic()-started,
        matched_quality_and_CUDA_BF16_cold_pending=True,candidate_promoted=False,goal_achieved=False))


def query_child(profile):
    plan=read(OUT/'plan.json');checked(plan);proof=read(OUT/profile/'complete.json')
    forbidden=[str(RUN),str(Path(read(RUN/'plan.json')['native']['checkpoint']).parent.parent)]
    blocked=[];network=[]
    def audit(event,args):
        if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
            path=os.path.abspath(os.fsdecode(args[0]))
            if any(path==p or path.startswith(p+'/') for p in forbidden):
                blocked.append(path);raise RuntimeError('Standalone inference cannot read training or ancestor files')
        if event in ['socket.connect','socket.getaddrinfo']:
            network.append(event);raise RuntimeError('Standalone inference cannot call a network service')
    sys.addaudithook(audit)
    from slm_lab.own_coding_policy_standalone_v1 import main
    sys.argv=['own_coding_policy_standalone_v1','--checkpoint',proof['checkpoint'],'--prompt',
        'Write a Python function to add two integers.','--max-new-tokens','8','--precision','fp32','--threads','1','--json']
    buffer=io.StringIO()
    with contextlib.redirect_stdout(buffer):main()
    result=json.loads(buffer.getvalue())
    if (result['actual_generated_token_ids']!=proof['outputs']['text_actions']
            or result['checkpoint_sha256']!=proof['checkpoint_sha256'] or result['actual_registered_parameters']!=999706442):
        raise ValueError('The actual fresh standalone query must preserve every original action and exact package identity')
    write(profile+'/fresh-query.json',dict(complete=True,actual_fresh_query=result,blocked_ancestor_or_training_read_attempts=blocked,
        network_attempts=network,eight_action_query_not_quality=True,completed_epoch_updates=230,goal_achieved=False))


def parent():
    if OUT.exists() or os.environ.get('CUDA_VISIBLE_DEVICES')!='' or 'torch' in sys.modules:
        raise ValueError('Fresh non-neural CUDA-hidden completed-delivery owner required')
    os.nice(10);os.sched_setaffinity(0,{22})
    from slm_lab.local_source_closure_v2 import python_sources
    run=read(RUN/'plan.json')
    if identity(run['owner']['pid'])!=run['owner']:
        raise ValueError('The exact original full-source training owner must be live before this observer starts')
    tests=ROOT/'reports/completed-public-coding-policy-v1-tests.log'
    if '3 passed' not in tests.read_text():raise ValueError('The actual complete-epoch and corruption checks must pass first')
    memory=dict(line.split()[:2] for line in Path('/sys/fs/cgroup/memory.stat').read_text().splitlines())
    free=int(Path('/sys/fs/cgroup/memory.max').read_text())-sum(int(memory[k]) for k in ['anon','shmem','kernel'])
    if free<80*GIB or shutil.disk_usage(OUT.parent).free<64*GIB:
        raise ValueError('Complete real-model CPU and two-package budgets must remain funded')
    files={str(p):sha(p) for p in [*python_sources(ROOT,[SCRIPT]),tests,ROOT/'tests/test_completed_public_coding_policy_v1.py',RUN/'plan.json']}
    for profile in PROFILES:
        for path in [CAPACITY/profile/'after-first.pt',CAPACITY/profile/'complete.json']:files[str(path)]=sha(path)
    files[run['native']['checkpoint']]=run['native']['checkpoint_sha256']
    token=Path(run['native']['checkpoint']).parent/'tokenizer.json';files[str(token)]=run['native']['tokenizer_sha256']
    OUT.mkdir();plan=dict(owner=identity(os.getpid()),exact_training_owner=run['owner'],source_files_sha256=files,
        expires_at_unix=time.time()+8*3600,profiles=list(PROFILES),required_updates_per_arm=230,
        full_original_views=3675,CPU_affinity=[22],CPU_threads=1,own_nice=10,GPU_usage_permitted=False,
        old_processes_signalled=0,candidate_promoted=False,goal_achieved=False)
    write('plan.json',plan);write('children.json',[]);children=[];process=None
    try:
        for profile in PROFILES:
            while not (RUN/profile/'training-complete.json').exists():
                checked(plan)
                if identity(run['owner']['pid'])!=run['owner']:
                    raise ValueError('The exact original training owner ended without the required complete arm receipt')
                write('progress.json',dict(phase='wait_complete_original_epoch',profile=profile,
                    exact_training_owner_live=True,at=datetime.now(timezone.utc).isoformat()))
                time.sleep(10)
            for task in ['export','query']:
                checked(plan)
                with (OUT/(profile+'-'+task+'.log')).open('x') as log:
                    process=subprocess.Popen([sys.executable,'-u',str(SCRIPT),'--cpu-child',profile,'--task',task],
                        cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'CUDA_VISIBLE_DEVICES':'',
                            'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'})
                    actor=identity(process.pid);children.append(actor);write('children.json',children)
                    write('progress.json',dict(phase=task,profile=profile,child=actor))
                    while process.poll() is None:checked(plan);time.sleep(1)
                    if process.returncode!=0:raise ValueError('Own completed-delivery child failed; preserve evidence without retrying')
        if any(sha(p)!=h for p,h in files.items()):raise ValueError('An original or runtime source byte changed')
        write('complete.json',dict(complete=True,profiles=list(PROFILES),completed_updates_per_arm=230,
            original_whole_source_epochs_reconciled=True,both_actual_standalone_CPUcolds_and_fresh_queries_complete=True,
            no_GPU_context_taken=True,old_processes_signalled=0,full_quality_and_CUDA_BF16_cold_pending=True,
            candidate_promoted=False,goal_achieved=False))
    except BaseException as error:
        write('failure.json',dict(error=repr(error),traceback=traceback.format_exc(),partial_outputs_retained=True,no_automatic_retry=True,goal_achieved=False));raise
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            try:process.wait(timeout=10)
            except subprocess.TimeoutExpired:process.kill();process.wait(timeout=10)
        write('restored.json',dict(own_children_terminal=all(identity(a['pid'])!=a for a in children),old_processes_signalled=0))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',action='store_true')
    parser.add_argument('--cpu-child',choices=PROFILES);parser.add_argument('--task',choices=['export','query']);args=parser.parse_args()
    if args.cpu_child:
        if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU delivery cannot initialize a GPU')
        export_child(args.cpu_child) if args.task=='export' else query_child(args.cpu_child)
    elif args.run:parent()
    else:parser.error('Explicit --run or own CPU child mode required')
