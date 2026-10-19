"""Queue an owned original-attention experiment after the existing math owner.

New vision/head weights are an experiment, not the final general policy. No
external encoder or generated teacher is loaded. Existing GPU work is never
preempted; failures preserve evidence without automatic retry.
"""
import argparse
from datetime import datetime,timezone,timedelta
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

import assess_joint_java_comparison_v1 as resources
from slm_lab.codenet_checkpoint_v1 import FileBindings
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json

ROOT=Path(__file__).resolve().parents[1]
REPORTS=ROOT/'reports/point-grounding-joint-v1'
OUTPUT=ROOT/'checkpoints/point-grounding-joint-v1'
QUEUE=ROOT/'reports/point-grounding-joint-v1-queue.json'
HYPOTHESIS=REPORTS/'hypothesis.json'
DATA=ROOT/'data/localized-pointer-candidate-v1'
BASE=ROOT/'checkpoints/extended-curriculum-v1/joint/model.pt'
TOKEN=BASE.parent/'tokenizer.json'
MATH=ROOT/'reports/apps-full-language-GPU-math-v1'
MATH_QUEUE=ROOT/'reports/apps-full-language-GPU-math-v1-queue.json'
WAITER=dict(pid=2754743,ticks='27602681')
WAIT_STATE='waiting for finalized original paired real-photo questions'
MAXIMUM_SECONDS=3*60*60
RESERVE=26*1024**3
MAX_STATE=5*1024**3
resources.REPORTS=REPORTS


def idle_original():
    actor=resources.same(WAITER)
    if (not actor or actor['state']!='S'
            or (Path('/proc')/str(WAITER['pid'])/'task'/str(WAITER['pid'])/'children').read_text().strip()
            or json.loads((ROOT/'reports/dialogue-retention-v1/status.json').read_text())['status']!=WAIT_STATE):
        raise ValueError('Never preempt original work; exact sleeping childless photo-data waiter required')


def owned(plan):
    waiter=resources.same(plan['waiter'])
    if (not resources.same(plan['owner']) or not waiter or waiter['state'] not in ['T','t']
            or (REPORTS/'restored.json').exists() or time.time()>plan['expires_at_unix']):
        raise ValueError('Exact live owned point experiment, held original waiter and unexpired lease required')


def registered(subject):
    plan=json.loads((REPORTS/'plan.json').read_text());owned(plan)
    with (REPORTS/'children.lock').open('a+') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX)
        actors=json.loads((REPORTS/'children.json').read_text());actual=resources.identity(os.getpid())
        if not any(a['pid']==actual['pid'] and a['ticks']==actual['ticks'] for a in actors):
            actors.append(dict(**actual,subject=subject));atomic_json(REPORTS/'children.json',actors)
    return plan


def restore_owned(plan,reason):
    # Owner-finally and the deadline watchdog can reach cleanup together.
    # Serialize restoration and tolerate a worker exiting between identity
    # verification and SIGTERM. Only registered birth identities are stopped.
    with (REPORTS/'restore.lock').open('a+') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX)
        if (REPORTS/'restored.json').exists():
            return
        actors=json.loads((REPORTS/'children.json').read_text())
        for actor in actors:
            try:
                resources.stop_child(actor)
            except ProcessLookupError:
                pass
        deadline=time.monotonic()+10
        while any(resources.same(actor) for actor in actors):
            if time.monotonic()>deadline:
                raise ValueError('Registered point workers remain live; original waiter must stay held')
            time.sleep(.1)
        waiter=resources.same(plan['waiter'])
        if waiter and waiter['state'] in ['T','t']:
            try:
                os.kill(waiter['pid'],signal.SIGCONT)
            except ProcessLookupError:
                pass
        atomic_json(REPORTS/'restored.json',dict(at=datetime.now(timezone.utc).isoformat(),reason=reason,
            waiter=resources.same(plan['waiter']),all_registered_point_actors_terminal=True,goal_achieved=False))


def watchdog():
    plan=json.loads((REPORTS/'plan.json').read_text())
    while resources.same(plan['owner']) and not (REPORTS/'restored.json').exists():
        if time.time()>plan['expires_at_unix']:
            if resources.same(plan['owner']):
                os.kill(plan['owner']['pid'],signal.SIGTERM)
            restore_owned(plan,'Bounded point experiment lease expired; exact registered workers stopped')
            return
        time.sleep(.5)
    if not (REPORTS/'restored.json').exists():
        restore_owned(plan,'Point owner ended; registered workers stopped before original waiter restoration')


def load_head(plan,device,phase='baseline'):
    import torch
    from slm_lab.training import load_checkpoint
    from slm_lab.point_grounding_head_v1 import PointGroundingHead
    torch.manual_seed(plan['training_options']['seed'])
    native=BASE if phase=='baseline' else OUTPUT/'model.pt'
    raw,metadata=load_checkpoint(native,device,training_state=False)
    if metadata['tokenizer_sha256']!=plan['tokenizer_sha256']:
        raise ValueError('Own exact native base/tokenizer binding required')
    head=PointGroundingHead(raw,rank=256).to(device)
    if phase=='candidate':
        saved=torch.load(OUTPUT/'grounding-head.pt',map_location='cpu',weights_only=True)
        if (saved['native_checkpoint_sha256']!=file_hash(native) or saved['run_plan_sha256']!=file_hash(REPORTS/'plan.json')
                or saved['steps']!=1137 or saved['data_admission_sha256']!=file_hash(DATA/'admission.json')):
            raise ValueError('Actual final own native vision and exact additional head required')
        head.query_projection.load_state_dict(saved['query_projection']);head.patch_projection.load_state_dict(saved['patch_projection'])
    return head,metadata


def worker(phase):
    import random
    import numpy as np
    import torch
    from torch import distributed as dist
    from slm_lab.codenet_checkpoint_v1 import atomic_checkpoint
    from slm_lab.point_grounding_data_v1 import PointerData
    from slm_lab.point_grounding_evaluation_v1 import state_digest
    from slm_lab.point_grounding_stage_v1 import PointerStage,collective_check,cpu,frozen_digest,make_optimizer
    local=int(os.environ['LOCAL_RANK']);rank=int(os.environ['RANK'])
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='0,1,2,3' or int(os.environ['WORLD_SIZE'])!=4:
        raise ValueError('All four explicitly owned physical GPUs required')
    plan=registered(phase+'-rank-'+str(rank))
    guard=FileBindings({**plan['source_files_sha256'],str(REPORTS/'plan.json'):file_hash(REPORTS/'plan.json')})
    guard.check(full=True)
    data=PointerData(DATA,'train','optimization')
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    device=torch.device('cuda',local);torch.cuda.set_device(device)
    if torch.cuda.get_device_name(device)!='NVIDIA L40' or not torch.cuda.is_bf16_supported():
        raise ValueError('Actual original BF16 L40 population required')
    torch.cuda.set_per_process_memory_fraction(.85,device)
    dist.init_process_group('nccl',timeout=timedelta(seconds=180))
    try:
        head,metadata=load_head(plan,device);inventory=head.trainable_inventory()
        if (inventory['total_unique_parameters']!=706799104 or inventory['actual_trainable_parameters']!=52410880
                or inventory['actual_additional_head_parameters']!=786432):
            raise ValueError('Actual complete sub1B own grounding model inventory required')
        initial_state=state_digest(head);initial_frozen=frozen_digest(head)
        if initial_frozen!=plan['frozen_language_critic_sha256']:
            raise ValueError('Actual own original language/critic bytes must match full CPU proof')
        gathered=[None]*4;dist.all_gather_object(gathered,initial_state)
        if len(set(gathered))!=1:
            raise ValueError('Every actual worker must start with identical original model and seeded head')
        random.seed(plan['training_options']['seed']+rank);np.random.seed(plan['training_options']['seed']+rank);torch.manual_seed(plan['training_options']['seed']+rank)
        optimizer=make_optimizer(head,plan['training_options'])
        amp=lambda:torch.autocast('cuda',dtype=torch.bfloat16)
        sources=dict(run_plan_sha256=file_hash(REPORTS/'plan.json'),base_checkpoint_sha256=plan['base_checkpoint_sha256'],
            tokenizer_sha256=plan['tokenizer_sha256'],data_admission_sha256=file_hash(DATA/'admission.json'))
        kernel=PointerStage(head,optimizer,data,plan['training_options'],sources,device,amp)
        if kernel.steps!=1137 or len(data)!=36354:
            raise ValueError('Actual complete predeclared original query pass required')
        if phase=='capacity':
            kernel.update();kernel.validate_Adam();optimizer.zero_grad(set_to_none=True)
            longest=max(range(len(data)),key=lambda i:len(data.rows[i]['query_input_ids']))
            batch=collective_check(lambda:data.batch([longest,longest],device))
            content=data.rows[longest]['query_input_ids'][1:-1]
            synthetic=[1,*((content*34)[:32]),2]
            batch['input_ids']=torch.tensor([synthetic,synthetic],device=device)
            batch['attention_mask']=torch.ones_like(batch['input_ids'],dtype=torch.bool)
            with amp():
                loss=head(**batch)['loss']
            if not bool(torch.isfinite(loss)):
                raise ValueError('Actual full34token/448px capacity loss must be finite')
            loss.backward()
            if (any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in kernel.parameters)
                    or any(p.grad is not None for p in kernel.frozen) or frozen_digest(head)!=initial_frozen):
                raise ValueError('Actual full-shape capacity requires finite vision/head gradients and exact frozen text')
            torch.cuda.synchronize(device)
            proof=dict(rank=rank,GPU=torch.cuda.get_device_name(device),actual_inventory=inventory,
                allocated_complete_Adam_states=len(optimizer.state),actual_disposable_source_updates=1,
                actual_longest_source_query_tokens=len(content)+2,actual_capacity_query_tokens=34,
                capacity_query_is_synthetic_repetition_not_original_human_utterance=True,
                actual_capacity_image_shape=list(batch['pixel_values'].shape),actual_capacity_loss=float(loss.detach()),
                frozen_language_critic_sha256=initial_frozen,initial_state_sha256=initial_state,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(device),peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
                production_model_updates=0,model_weights_exported=False)
            proofs=[None]*4;dist.all_gather_object(proofs,proof)
            if rank==0:
                atomic_json(REPORTS/'capacity.json',dict(passed=True,actual_four_L40s=True,proofs=proofs,initial_state_sha256=initial_state,goal_achieved=False))
            return
        capacity=json.loads((REPORTS/'capacity.json').read_text())
        baseline=json.loads((REPORTS/'baseline/grounding/plan.json').read_text())
        if (not capacity['passed'] or not (REPORTS/'baseline-complete.json').exists()
                or initial_state!=capacity['initial_state_sha256'] or initial_state!=baseline['actual_model_state_sha256']):
            raise ValueError('Actual full GPU capacity and exact matching initial-head baseline required')
        started=time.monotonic()
        for step in range(kernel.steps):
            owned(plan);collective_check(guard.check);tick=time.monotonic();record=kernel.update();torch.cuda.synchronize(device)
            record['step_seconds']=time.monotonic()-tick
            if rank==0:
                with (REPORTS/'training-metrics.jsonl').open('a') as stream:
                    stream.write(json.dumps(record,separators=(',',':'))+'\n')
                if step<2 or (step+1)%8==0:
                    atomic_json(REPORTS/'status.json',dict(status='training own vision/head on original human attention',
                        observed_at=datetime.now(timezone.utc).isoformat(),actual_updates=step+1,planned_updates=kernel.steps,
                        actual_source_exposures=(step+1)*32,latest=record,goal_achieved=False))
                    print('original_human_attention_update',step+1,record['query_mean_nll'],flush=True)
            if step+1 in [1,16] or (step+1)%128==0 or step+1==kernel.steps:
                collective_check(lambda:guard.check(full=True))
                if collective_check(lambda:frozen_digest(head))!=initial_frozen:
                    raise ValueError('Every frozen own language/critic byte must remain identical')
                kernel.save(OUTPUT/'resume.pt',MAX_STATE,RESERVE+MAX_STATE)
                if rank==0:
                    atomic_json(REPORTS/'checkpoint-receipt.json',dict(steps=step+1,sha256=file_hash(OUTPUT/'resume.pt'),
                        bytes=(OUTPUT/'resume.pt').stat().st_size,full_model_Adam_worker_RNG_and_source_cursor=True,goal_achieved=False))
        collective_check(lambda:guard.check(full=True))
        if rank==0:
            final=dict(model=cpu(head.base.state_dict()),config=metadata['config'],stage='sft',
                training_kind='original_human_pointer_vision_head_v1',tokenizer_sha256=plan['tokenizer_sha256'],
                steps=kernel.completed,planned_steps=kernel.steps,training_options=plan['training_options'],
                source_checkpoint_sha256=plan['base_checkpoint_sha256'],run_plan_sha256=file_hash(REPORTS/'plan.json'),
                data_sha256=file_hash(DATA/'admission.json'),frozen_language_critic_sha256=initial_frozen,
                image_prompt_coordinates='letterbox input',truncated_sequences=0,candidate_promoted=False,goal_achieved=False)
            atomic_checkpoint(OUTPUT/'model.pt',final,4*1024**3,RESERVE)
            extra=dict(query_projection=cpu(head.query_projection.state_dict()),patch_projection=cpu(head.patch_projection.state_dict()),
                head_rank=256,steps=kernel.completed,native_checkpoint_sha256=file_hash(OUTPUT/'model.pt'),
                run_plan_sha256=file_hash(REPORTS/'plan.json'),data_admission_sha256=file_hash(DATA/'admission.json'),
                actual_inventory=inventory,goal_achieved=False)
            atomic_checkpoint(OUTPUT/'grounding-head.pt',extra,64*1024**2,RESERVE)
            atomic_json(REPORTS/'training-complete.json',dict(steps=kernel.completed,whole_original_query_pass=True,
                actual_source_exposures=kernel.completed*32,unique_original_queries=len(data),rounding_repeated_queries=30,
                native_checkpoint_sha256=file_hash(OUTPUT/'model.pt'),additional_head_sha256=file_hash(OUTPUT/'grounding-head.pt'),
                resumable_checkpoint_sha256=file_hash(OUTPUT/'resume.pt'),frozen_language_critic_sha256=initial_frozen,
                elapsed_seconds=time.monotonic()-started,candidate_promoted=False,goal_achieved=False))
        dist.barrier()
    finally:
        dist.destroy_process_group()


def assess(phase,suite):
    plan=registered(phase+'-suite-'+str(suite))
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=str(suite):
        raise ValueError('Exact isolated owned physical assessment GPU required')
    guard=FileBindings({**plan['source_files_sha256'],str(REPORTS/'plan.json'):file_hash(REPORTS/'plan.json')})
    guard.check(full=True)
    import torch
    torch.set_num_threads(1);torch.cuda.set_device(0);torch.cuda.set_per_process_memory_fraction(.85,0)
    folder=REPORTS/phase;folder.mkdir(exist_ok=True)
    checkpoint=BASE if phase=='baseline' else OUTPUT/'model.pt'
    if suite==0:
        from slm_lab.point_grounding_data_v1 import PointerData
        from slm_lab.point_grounding_evaluation_v1 import evaluate
        data=PointerData(DATA,'validation','evaluation');head,_=load_head(plan,'cuda:0',phase)
        prior=json.loads((DATA/'training-position-prior.json').read_text())
        with torch.autocast('cuda',dtype=torch.bfloat16):
            evaluate(head,data,prior,folder/'grounding',batch_size=4)
        if file_hash(folder/'grounding/control-pairs.jsonl')!=file_hash(DATA/'validation-control-pairs.jsonl'):
            raise ValueError('Every actual diagnostic control must equal its original presealed pairing')
    elif suite==1:
        from slm_lab.ocr_evaluation import evaluate
        evaluate(checkpoint,ROOT/'data/alignment-curriculum-v1/images.valid.jsonl',folder/'ocr.json',
            device='cuda:0',max_examples=256,max_new_tokens=128,image_size=448)
    elif suite==2:
        from slm_lab.object_evaluation import evaluate
        evaluate(checkpoint,ROOT/'data/object-qa-v1/valid.jsonl',folder/'objects.json',device='cuda:0',max_examples=180,max_new_tokens=32)
    else:
        from slm_lab.vision_evaluation import evaluate
        evaluate(checkpoint,ROOT/'data/alignment-curriculum-v1/contrastive.valid.jsonl',folder/'vision.json',
            device='cuda:0',max_examples=128,image_size=448)
    owned(plan);guard.check(full=True)
    atomic_json(folder/f'suite-{suite}-complete.json',dict(suite=suite,native_checkpoint_sha256=file_hash(checkpoint),
        phase=phase,run_plan_sha256=file_hash(REPORTS/'plan.json'),goal_achieved=False))


def wait_children(plan,children):
    while any(p.poll() is None for p in children):
        owned(plan)
        if any(p.poll() not in [None,0] for p in children):
            raise ValueError('Owned point child failed; preserve partial artifacts without automatic retry')
        time.sleep(2)
    if any(p.returncode!=0 for p in children):
        raise ValueError('Every owned point child must complete successfully')


def launch(plan,args,subject,visible):
    with (REPORTS/(subject+'.log')).open('xb') as stream:
        child=subprocess.Popen(['.venv/bin/python',*args],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True,
            env={**os.environ,'CUDA_VISIBLE_DEVICES':visible,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1',
                'TOKENIZERS_PARALLELISM':'false','CUBLAS_WORKSPACE_CONFIG':':4096:8'})
    actor=resources.identity(child.pid)
    if not actor:
        raise ValueError('Owned child ended before exact registration')
    with (REPORTS/'children.lock').open('a+') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX)
        actors=json.loads((REPORTS/'children.json').read_text());actors.append(dict(**actor,subject=subject))
        atomic_json(REPORTS/'children.json',actors)
    return child


def parent(files,predecessor):
    if (REPORTS/'plan.json').exists() or OUTPUT.exists():
        raise ValueError('Preserve earlier point run; no implicit restart')
    if resources.same(predecessor):
        raise ValueError('Existing exact math owner must actually end before acquisition')
    math_plan=json.loads((MATH/'plan.json').read_text())
    if any(math_plan['owner'][k]!=predecessor[k] for k in ['pid','ticks']):
        raise ValueError('Exact original math plan/owner identity required')
    if any(resources.same(a) for a in json.loads((MATH/'children.json').read_text())):
        raise ValueError('Every original registered math GPU worker must actually end')
    terminal=MATH/'comparison.json' if (MATH/'comparison.json').exists() else MATH/'failure.json'
    for path in [MATH/'plan.json',MATH/'restored.json',terminal]:
        files[str(path)]=file_hash(path)
    guard=FileBindings(files)
    hypothesis=json.loads(HYPOTHESIS.read_text());options=dict(hypothesis['proposed_fixed_training_options'],
        image_size=448,maximum_whole_query_tokens=34)
    with (ROOT/'reports/code-residual-production-v4.gpu.lock').open('a+') as lease:
        fcntl.flock(lease.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB);idle_original()
        memory=subprocess.run(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],
            capture_output=True,text=True,check=True).stdout.splitlines()
        if len(memory)!=4 or any(int(v.strip())>128 for v in memory):
            raise ValueError('All four point-experiment GPUs must actually be unused')
        guard.check(full=True)
        probe=json.loads((ROOT/'reports/point-grounding-full-CPU-v2/complete.json').read_text())
        plan=dict(created_at=datetime.now(timezone.utc).isoformat(),owner=resources.identity(os.getpid()),waiter=WAITER,
            predecessor=predecessor,predecessor_math_terminal_artifact=str(terminal),
            source_files_sha256=files,base_checkpoint_sha256=file_hash(BASE),tokenizer_sha256=file_hash(TOKEN),
            frozen_language_critic_sha256=probe['frozen_language_critic_SHA_after'],training_options=options,
            gradient_reduction='fixed_parameter_order_worker_average_v1',maximum_seconds=MAXIMUM_SECONDS,
            expires_at_unix=time.time()+MAXIMUM_SECONDS,automatic_retry=False,candidate_promoted=False,goal_achieved=False)
        OUTPUT.mkdir();shutil.copy2(TOKEN,OUTPUT/'tokenizer.json')
        atomic_json(REPORTS/'plan.json',plan);atomic_json(REPORTS/'children.json',[])
        def interrupted(signum,frame):
            raise KeyboardInterrupt('Owned point controller interrupted')
        signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
        watcher=subprocess.Popen(['.venv/bin/python','-u',str(Path(__file__).resolve()),'--watchdog'],cwd=ROOT,
            stdout=(REPORTS/'watchdog.log').open('xb'),stderr=subprocess.STDOUT,start_new_session=True)
        try:
            idle_original();os.kill(WAITER['pid'],signal.SIGSTOP);deadline=time.monotonic()+5
            while (resources.same(WAITER) or {}).get('state') not in ['T','t']:
                if time.monotonic()>deadline:
                    raise ValueError('Original idle waiter did not enter its owned hold')
                time.sleep(.05)
            if (Path('/proc')/str(WAITER['pid'])/'task'/str(WAITER['pid'])/'children').read_text().strip():
                raise ValueError('Original waiter began work before hold; do not launch point workers')
            driver=str(Path(__file__).resolve())
            train_args=['-m','torch.distributed.run','--standalone','--nnodes=1','--nproc-per-node=4',driver]
            child=launch(plan,[*train_args,'--worker','capacity'],'capacity','0,1,2,3');wait_children(plan,[child])
            for phase in ['baseline','candidate']:
                if phase=='candidate':
                    child=launch(plan,[*train_args,'--worker','train'],'training','0,1,2,3');wait_children(plan,[child])
                children=[launch(plan,['-u',driver,'--assess',phase,'--suite',str(i)],phase+'-'+str(i),str(i)) for i in range(4)]
                wait_children(plan,children)
                for i in range(4):
                    if not (REPORTS/phase/f'suite-{i}-complete.json').exists():
                        raise ValueError('Every actual native point/image assessment must complete')
                atomic_json(REPORTS/(phase+'-complete.json'),dict(completed_at=datetime.now(timezone.utc).isoformat(),
                    phase=phase,all_four_suites_complete=True,goal_achieved=False))
            from slm_lab.point_grounding_readback_v1 import reconcile
            reconcile(ROOT,REPORTS,OUTPUT,DATA)
            guard.check(full=True)
            atomic_json(REPORTS/'status.json',dict(status='original attention experiment complete; inspect matched results',
                candidate_promoted=False,goal_achieved=False))
        except BaseException as problem:
            atomic_json(REPORTS/'failure.json',dict(error=str(problem),automatic_retry=False,goal_achieved=False));raise
        finally:
            restore_owned(plan,'Owned original-attention experiment ended; original waiter restored')
            watcher.wait(timeout=20)


def queue():
    if QUEUE.exists() or (REPORTS/'plan.json').exists() or OUTPUT.exists():
        raise ValueError('Preserve queued or prior point experiment; no implicit duplicate')
    hypothesis=json.loads(HYPOTHESIS.read_text())
    from slm_lab.evaluation_provenance import capture
    files={str(ROOT/name):sha for name,sha in capture()['files_sha256'].items()}
    files.update(hypothesis['source_files_sha256'])
    admission=json.loads((DATA/'admission.json').read_text())
    files.update(admission['runtime_and_control_files_sha256'])
    source=json.loads((ROOT/'reports/localized-pointer-candidate-v1/independent-source-readback.json').read_text())
    files.update(source['source_candidate_files_sha256']);files.update(source['audit_files_sha256'])
    original=ROOT/'data/localized-pointer-source-v1'
    for name,receipt in json.loads((original/'collection.json').read_text())['sources'].items():
        files[str(original/name)]=receipt['sha256']
    names=['scripts/train_point_grounding_joint_v1.py','scripts/assess_joint_java_comparison_v1.py',
        'src/slm_lab/point_grounding_stage_v1.py','src/slm_lab/point_grounding_readback_v1.py',
        'src/slm_lab/apps_native_metrics_v1.py','src/slm_lab/localized_pointer_evidence_v1.py',
        'src/slm_lab/codenet_checkpoint_v1.py','tests/test_point_grounding_stage_v1.py',
        'tests/test_point_grounding_driver_v1.py','reports/point-grounding-stage-v1-tests.log',
        'reports/point-grounding-driver-v1-tests.log','data/alignment-curriculum-v1/images.valid.jsonl',
        'data/alignment-curriculum-v1/contrastive.valid.jsonl','data/object-qa-v1/valid.jsonl']
    for name in names:
        files[str(ROOT/name)]=file_hash(ROOT/name)
    for path in [HYPOTHESIS,MATH_QUEUE]:
        files[str(path)]=file_hash(path)
    for name in ['point-grounding-stage-v1-tests.log','point-grounding-driver-v1-tests.log']:
        if '2 passed' not in (ROOT/'reports'/name).read_text():
            raise ValueError('Actual collective recovery/source and active-predecessor rejection checks required')
    guard=FileBindings(files);predecessor=json.loads(MATH_QUEUE.read_text())['queue_owner']
    atomic_json(QUEUE,dict(queued_at=datetime.now(timezone.utc).isoformat(),queue_owner=resources.identity(os.getpid()),
        exact_predecessor=predecessor,source_files_sha256=files,
        purpose='Original full-query vision/head experiment after existing math GPU window ends',
        actual_model_updates=0,pending_not_started=True,automatic_retry=False,goal_achieved=False))
    while resources.same(predecessor):
        guard.check();time.sleep(5)
    while True:
        guard.check()
        if not resources.same(WAITER):
            raise ValueError('Original waiter ended; inspect a new resource window')
        try:
            idle_original();break
        except ValueError:
            time.sleep(5)
    parent(files,predecessor)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker',choices=['capacity','train']);parser.add_argument('--watchdog',action='store_true')
    parser.add_argument('--assess',choices=['baseline','candidate']);parser.add_argument('--suite',type=int,choices=[0,1,2,3])
    args=parser.parse_args()
    if args.watchdog:watchdog()
    elif args.worker:worker(args.worker)
    elif args.assess:
        if args.suite is None:raise ValueError('Exact isolated assessment suite required')
        assess(args.assess,args.suite)
    else:queue()
