"""Deliver a standalone policy plus its original frozen shared reward.

Wait non-neurally for the exact complete PPO owner and all its original full
assessments. A separate idle CPU exporter reads every original private source,
retains unfused policy factors, cold-checks all values and text/image/word/reward
outputs, and publishes one self-contained under1Bmodel with its tokenizer.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
RESERVE=40*1024**3


def read(p):return json.loads(Path(p).read_text())


def identity(pid):
    return dict(pid=pid,ticks=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[19])


def available_memory():
    values={k:int(v) for k,v in (line.split() for line in Path('/sys/fs/cgroup/memory.stat').read_text().splitlines())}
    return int(Path('/sys/fs/cgroup/memory.max').read_text())-sum(values[k] for k in ['anon','shmem','kernel'])


def worker(workspace):
    from slm_lab.exact_waiting_process_retirement_v1 import terminal
    from slm_lab.policy_reward_delivery_recipe_v1 import sha
    workspace=Path(workspace).resolve();out=workspace/'reports';plan=read(out/'plan.json')
    if (os.nice(0)!=19 or os.environ.get('CUDA_VISIBLE_DEVICES')!=''
            or any(os.environ.get(k)!='1' for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'])):
        raise ValueError('Only the registered CUDA-hidden nice19 one-thread delivery child is eligible')
    os.sched_setscheduler(0,os.SCHED_IDLE,os.sched_param(0));started=time.monotonic()
    import torch
    from slm_lab.codenet_checkpoint_v1 import FileBindings,atomic_checkpoint,capture_rng
    from slm_lab.diverse_images import atomic_json
    from slm_lab.native_parallel_ppo_export_v2 import materialize
    from slm_lab.own_policy_reward_standalone_v1 import export,load
    from slm_lab.parallel_function_repair_stage_v1 import value_digest
    import train_native_parallel_PPO_owned_GPU_v2 as original
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    guards=FileBindings(plan['source_files_sha256']);guards.check(full=True);last_source_check=[0.]
    def checked():
        if (terminal(plan['owner']) or time.monotonic()-started>6*3600 or os.nice(0)!=19
                or os.sched_getscheduler(0)!=os.SCHED_IDLE or torch.cuda.is_initialized()
                or available_memory()<64*1024**3 or min(shutil.disk_usage(ROOT).free,shutil.disk_usage(workspace).free)<RESERVE):
            raise ValueError('Exact live CPUowner, six-hour idle lifetime, no CUDA and all original writer/memory reserves must remain')
        # Whole frames have independent byte hashes. Check the complete source
        # map at phase boundaries and periodically, rather than statting every
        # predecessor path for every scalar in a long immutable history scan.
        if time.monotonic()-last_source_check[0]>=30:
            guards.check();last_source_check[0]=time.monotonic()
    atomic_json(out/'registered-child.json',dict(owner=identity(os.getpid()),parent=plan['owner'],GPU_contexts=0))
    try:
        checked();original.OUT=Path(plan['original_PPO_reports']);kernel=original.build(plan['PPO_plan'],torch.device('cpu'))
        source=materialize(kernel,plan['actual_original_private_artifacts'],expected_completed=plan['expected_completed_updates'],checked=checked)
        kernel.model.eval();before=value_digest(capture_rng('cpu'))
        folder=workspace/'package';folder.mkdir();tokenizer=folder/'tokenizer.json'
        shutil.copyfile(plan['PPO_plan']['initializer']['tokenizer'],tokenizer)
        packet=export(kernel.model,kernel.reward,sources=kernel.sources,
            provenance=dict(kind='complete-own-four-L40PPO-with-preserved-frozen-human-reward',
                original_PPO_plan_sha256=sha(Path(plan['original_PPO_reports'])/'plan.json'),
                all85786whole_original_prompts_once=True,all_original512action_capacity_and_private_replay_complete=True,
                all_original_before_after_assessments_complete=True,complete_original_private_readback=source,
                full_size_GPU_inference_of_this_new_file_not_replayed_by_CPUexport=True),
            completed_updates=10724,tokenizer_sha256=plan['PPO_plan']['tokenizer_sha256'])
        path=folder/'model.pt';atomic_checkpoint(path,packet,plan['storage_budget']['maximum_checkpoint_bytes'],RESERVE);del packet
        actual,metadata=load(path,expected_checkpoint_sha256=sha(path),expected_sources=kernel.sources)
        if metadata['actual_registered_parameters']!=999706442:raise ValueError('Every actual source policy/reward parameter must remain under1Bwith exactly one backbone')
        raw=torch.load(plan['PPO_plan']['cold_inputs']['path'],map_location='cpu',weights_only=True,mmap=True)
        if raw['text']['input_ids'].shape[1]!=4096 or tuple(raw['photo']['pixel_values'].shape[-2:])!=(448,448):
            raise ValueError('The complete original4096text/448photo source probes must remain unchanged')
        probes={}
        with torch.no_grad():
            for kind in ['text','photo']:
                checked();a,b=kernel.model(**raw[kind]),actual(**raw[kind])
                if any(not torch.equal(a[k],b[k]) for k in ['logits','values']):
                    raise ValueError('Every complete original unfused policy text/photo output and trained critic must cold-load exactly')
                probes[kind]=dict(actual_input_tokens=raw[kind]['input_ids'].shape[1],all_policy_logits_and_values_bit_exact=True)
            checked();a=kernel.model.base.word_logits(raw['word']['word_pixels']);b=actual.word_logits(raw['word']['word_pixels'])
            if tuple(a.shape)!=(16,140,257) or not torch.equal(a,b):raise ValueError('Every full original word/byte output must remain exact')
            position=max(range(len(kernel.data)),key=lambda p:kernel.data.rows[p]['native']['chosen']['length'])
            ids=torch.tensor([kernel.data.sequence(position,'chosen')[0]],dtype=torch.long)
            a,b=kernel.reward.score(ids),actual.preference_score(ids)
            if not torch.equal(a,b):raise ValueError('The actually trained original shared human reward must retain its exact encoder/head score')
            probes['human_reward']=dict(original_train_position=position,complete_source_tokens=ids.shape[1],original_and_cold_score=float(a),bit_exact=True)
        if before!=value_digest(capture_rng('cpu')):raise ValueError('Standalone reconstruction/export and complete cold probes must preserve CPU RNG')
        checked();guards.check(full=True)
        publication=Path(plan['published_folder']);pending=publication.with_name(publication.name+'.pending');pending.mkdir()
        files={};total=0
        for src,name in [(path,'model.pt'),(tokenizer,'tokenizer.json')]:
            total+=src.stat().st_size
            if total>plan['storage_budget']['maximum_publication_bytes']:raise ValueError('The full unfused checkpoint and tokenizer must fit without dropping a parameter')
            dest=pending/name
            with src.open('rb') as a,dest.open('xb') as b:
                while block:=a.read(1024**2):
                    checked()
                    if shutil.disk_usage(ROOT).free-len(block)<RESERVE:raise ValueError('The complete publication must retain the original40GiBfloor')
                    b.write(block)
                b.flush();os.fsync(b.fileno())
            if sha(dest)!=sha(src):raise ValueError('Every actual durable policy/reward/tokenizer byte must remain exact')
            files[name]=sha(dest)
        atomic_json(pending/'manifest.json',dict(format='own-standalone-policy-and-preserved-human-reward-delivery-v1',
            files_sha256=files,sources=kernel.sources,actual_registered_parameters=999706442,
            trained_reward_feature_basis_preserved=True,old_checkpoint_or_features_needed_for_inference=False,
            all_original_folded_native_and_benchmarks_preserved=True,model_quality_proved=False,goal_achieved=False))
        pending.rename(publication)
        # The separate CLI cold process loads only the durable model/tokenizer.
        # Release both full graphs first to keep within the declared host budget.
        import gc
        del actual,kernel;gc.collect();checked()
        smoke=out/'standalone-query-smoke.json'
        with smoke.open('x') as stream:
            result=subprocess.run([str(ROOT/'.venv/bin/python'),'-m','slm_lab.own_policy_reward_query_v1',
                '--checkpoint',str(publication/'model.pt'),'--expected-sha256',files['model.pt'],
                '--prompt','Write a Python function to add two integers.','--max-new-tokens','8','--threads','1','--json'],
                cwd=ROOT,stdout=stream,stderr=subprocess.PIPE,text=True,timeout=900,env={**os.environ,'CUDA_VISIBLE_DEVICES':''})
        if result.returncode!=0:raise ValueError('The actual separate-process native CLI failed: '+result.stderr[-3000:])
        smoke_result=read(smoke)
        if smoke_result.get('actual_registered_parameters')!=999706442 or not 1<=smoke_result.get('actual_native_actions',0)<=8:
            raise ValueError('The entire counted durable model must load and generate through the public CLI')
        checked();guards.check(full=True)
        atomic_json(out/'export-complete.json',dict(complete=True,owner=identity(os.getpid()),published_folder=str(publication),
            package_files_sha256=files,actual_registered_parameters=999706442,additional_reward_encoder_parameters=0,
            all_policy_reward_and_auxiliary_values_and_original_full_outputs_CPUcold_exact=True,actual_complete_probes=probes,
            actual_original_private_sources_checked=source,separate_actual_durable_CLI_completed=True,
            GPU_contexts=0,new_optimizer_updates=0,new_model_quality_gain=False,goal_achieved=False))
    except BaseException as exc:
        atomic_json(out/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc(),
            all_original_sources_and_partial_new_delivery_preserved=True,automatic_retry=False,goal_achieved=False));raise


def parent(args):
    from slm_lab.local_source_closure_v2 import python_sources
    from slm_lab.policy_reward_delivery_recipe_v1 import prepare,sha,STORAGE
    from slm_lab.reserved_atomic_metadata_v1 import write_json
    workspace=Path(args.workspace).absolute();PPO=Path(args.PPO).resolve()
    if (workspace.exists() or workspace.resolve()!=workspace or os.environ.get('CUDA_VISIBLE_DEVICES')!='' or 'torch' in sys.modules
            or os.nice(0)!=0 or os.sched_getscheduler(0)!=os.SCHED_OTHER
            or any(os.environ.get(k)!='1' for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'])):
        raise ValueError('A fresh normal CUDA-hidden non-neural exact-source observer must precede CPUdelivery')
    captured=read(PPO/'waiter-plan.json')['owner'];files={str(p):sha(p) for p in python_sources(ROOT,[Path(__file__)])}
    files[str(PPO/'waiter-plan.json')]=sha(PPO/'waiter-plan.json');workspace.mkdir();out=workspace/'reports';out.mkdir()
    def guard():
        if any(sha(p)!=h for p,h in files.items()):raise ValueError('Every exact original PPO owner and new standalone delivery byte must remain')
    def write(name,value):return write_json(out/name,value,root=workspace,minimum_free_bytes=RESERVE,
        maximum_metadata_bytes=STORAGE['maximum_metadata_bytes'],guard=guard)
    write('waiter-plan.json',dict(owner=identity(os.getpid()),source_files_sha256=files,exact_original_PPO_owner=captured,
        GPU_contexts=0,GPU_leases=0,new_training_updates=0,goal_achieved=False))
    deadline=time.monotonic()+35*86400;recipe=None
    while recipe is None:
        guard();recipe=prepare(ROOT,PPO=PPO,expected_owner=captured,runtime_sources=files)
        if recipe is not None:break
        write('waiting-progress.json',dict(waiting_for='exact complete original four-L40PPO owner and every original assessment',
            neural_work_started=False,GPU_contexts=0,new_training_updates=0))
        if time.monotonic()>deadline:raise ValueError('Source lifetime expired without relabelling any original process stopped')
        time.sleep(30)
    while (shutil.disk_usage(workspace).free<STORAGE['required_scratch_free_bytes']
            or shutil.disk_usage(ROOT).free<STORAGE['required_home_free_bytes'] or available_memory()<112*1024**3):
        guard();write('waiting-progress.json',dict(waiting_for='full48GiBscratch/45GiBhome/112GiBhost free delivery budget',neural_work_started=False))
        if time.monotonic()>deadline:raise ValueError('Full delivery funding wait expired without altering original workers')
        time.sleep(30)
    published=ROOT/'checkpoints/native-policy-reward-standalone-v1'
    if published.exists() or published.with_name(published.name+'.pending').exists():raise ValueError('Preserve every prior complete or partial standalone delivery')
    write('plan.json',dict(recipe,owner=identity(os.getpid()),published_folder=str(published)))
    with (out/'CPU-export.log').open('x') as log:
        child=subprocess.Popen(['nice','-n','19',str(ROOT/'.venv/bin/python'),'-u',str(Path(__file__).resolve()),
            '--workspace',str(workspace),'--export'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
            start_new_session=True,env={**os.environ,'CUDA_VISIBLE_DEVICES':''})
        write('child.json',dict(owner=identity(child.pid),exact_new_CPU_child_only=True))
        while child.poll() is None:guard();time.sleep(5)
        if child.returncode!=0:raise ValueError('The exact CPUdelivery child failed; preserve every original and new partial artifact')
    done=read(out/'export-complete.json')
    if done.get('complete') is not True or done.get('actual_registered_parameters')!=999706442:raise ValueError('Actual complete standalone delivery proof is required')
    write('complete.json',dict(complete=True,owner=identity(os.getpid()),actual_registered_parameters=999706442,
        deployed_policy_and_original_trained_shared_reward_preserved=True,published_folder=str(published),
        old_checkpoints_or_training_features_not_needed_for_inference=True,
        original_four_L40PPO_and_full_assessment_artifacts_unchanged=True,new_model_quality_gain=False,goal_achieved=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace',type=Path,default=Path('/tmp/slm-native-policy-reward-delivery-CPU-v1'))
    parser.add_argument('--PPO',type=Path,default=Path('/tmp/slm-native-parallel-PPO-owned-GPU-v2/reports'))
    parser.add_argument('--run',action='store_true');parser.add_argument('--export',action='store_true');args=parser.parse_args()
    if args.run:parent(args)
    elif args.export:worker(args.workspace)
    else:parser.error('Use explicit source-waiting --run or the registered owned CPU --export mode')
