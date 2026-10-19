"""Full current-native human reward learning after the complete joint run.

The non-neural observer waits for the exact original owner and every full
assessment. A fresh owned run then checks four real feature capacities,
encodes all original train/held responses in FP32, fits one complete head
epoch on CPU, cold-loads the head/native and publishes a small durable bundle.
PPO and useful own-output human alignment remain separate requirements.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
RESERVE = 40 * 1024**3


def read(p): return json.loads(Path(p).read_text())


def normal():
    if (os.nice(0) != 0 or os.sched_getscheduler(0) != os.SCHED_OTHER
            or any(os.environ.get(k) != '1' for k in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'])):
        raise ValueError('Only a normal nice0 single-thread new owned reward runtime is eligible')


def configure(workspace):
    global WORKSPACE, OUT, DEST, owner
    WORKSPACE = Path(workspace).resolve(); OUT = WORKSPACE/'reports'; DEST = WORKSPACE/'checkpoints'
    import own_training_gpu_lease_v2 as owner
    owner.REPORTS = OUT; owner.resources.REPORTS = OUT


def memory_available():
    values = {k:int(v) for k,v in (line.split() for line in Path('/sys/fs/cgroup/memory.stat').read_text().splitlines())}
    return int(Path('/sys/fs/cgroup/memory.max').read_text())-sum(values[k] for k in ['anon','shmem','kernel'])


def preflight(plan):
    normal(); owner.owned(plan)
    if (shutil.disk_usage(WORKSPACE).free < RESERVE or shutil.disk_usage(ROOT).free < RESERVE
            or memory_available() < 64*1024**3):
        raise ValueError('The actual writer reserves and64GiBother-worker memory headroom must remain')


def setup_device(physical):
    import torch
    normal(); torch.set_num_threads(1); torch.set_num_interop_threads(1); torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    if physical is None:
        if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or torch.cuda.is_initialized():
            raise ValueError('CPU reward fitting/cold work must keep CUDA hidden and uninitialized')
        return torch.device('cpu')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != str(physical): raise ValueError('Exact isolated original physical GPU visibility is required')
    torch.cuda.set_device(0); torch.cuda.set_per_process_memory_fraction(.85, 0)
    if torch.cuda.get_device_name(0) != 'NVIDIA L40': raise ValueError('Only an original physical L40 is eligible')
    return torch.device('cuda', 0)


def load_base(plan, device):
    from slm_lab.own_dense_standalone_v1 import load
    from slm_lab.whole_human_sft_evidence_v1 import tensor_digest
    spec = plan['initializer']; base, metadata = load(spec['checkpoint'],
        expected_checkpoint_sha256=spec['checkpoint_sha256'], expected_sources=spec['sources'])
    if (base.counts()['total_unique'] != 984399690 or base.config.hidden_size != 1536
            or metadata['completed_updates'] != 5362 or metadata['tokenizer_sha256'] != spec['tokenizer_sha256']
            or metadata['provenance'].get('completed_original_joint_source_epoch') is not True
            or sum(p.numel() for p in base.heads['reward'].parameters()) != 198272):
        raise ValueError('The fixed complete original5362update joint native and existing reward head are required')
    base.to(device).eval()
    sources = dict(native_checkpoint_sha256=spec['checkpoint_sha256'], human_manifest_sha256=plan['human_manifest_sha256'],
        human_admission_sha256=plan['human_admission_sha256'], frozen_native_encoder_sha256=tensor_digest(base.network.named_parameters()))
    return base, sources


def capacities(plan, sources):
    rows = [read(OUT/'capacity'/('rank-'+str(r)+'.json')) for r in range(4)]
    for r, value in enumerate(rows):
        if (value.get('format') != 'own-current-native-whole-human-FP32-feature-GPU-capacity-v1'
                or value.get('complete') is not True or value.get('physical_GPU') != r
                or value.get('actual_native_values') != 984399690 or value.get('actual_device_type') != 'cuda'
                or value.get('actual_device_index') != 0 or value.get('original_shard_rank') != r or value.get('original_world') != 4
                or value.get('sources') != sources or value.get('options') != plan['feature_options']
                or value.get('full_size_GPU_feature_capacity_passed') is not True
                or value.get('all_values_metadata_and_private_RNG_unchanged') is not True
                or value.get('maximum_padded_and_longest_whole_original_batches_checked') is not True):
            raise ValueError('Every actual original L40 must prove its unchanged complete feature-capacity inputs')
    return rows


def feature_worker(physical, capacity):
    from slm_lab.codenet_checkpoint_v1 import FileBindings, atomic_checkpoint
    from slm_lab.corpus import file_hash
    from slm_lab.diverse_images import atomic_json
    from slm_lab.native_human_reward_v1 import collect_features
    from slm_lab.native_reward_capacity_v1 import prove
    from slm_lab.whole_human_data_v2 import WholeHumanData
    subject = ('capacity-' if capacity else 'features-')+str(physical)
    plan = owner.registered(subject); preflight(plan); device = setup_device(physical)
    base, sources = load_base(plan, device); guard = FileBindings(plan['source_files_sha256'])
    train = WholeHumanData(plan['data'], 'preferences', 'train', 'optimization')
    if capacity:
        proof = prove(base, train, physical, 4, sources, plan['feature_options'], physical_GPU=physical, checked=lambda:preflight(plan))
        guard.check(full=True); preflight(plan); atomic_json(OUT/'capacity'/('rank-'+str(physical)+'.json'), proof); return
    capacities(plan, sources); held = WholeHumanData(plan['data'], 'preferences', 'valid', 'evaluation'); artifacts = {}
    for role, data in [('train', train), ('valid', held)]:
        def progress(row):
            if row['encoded_complete_responses'] % 1024 == 0 or row['encoded_complete_responses'] == row['planned_complete_responses']:
                atomic_json(OUT/'features'/('progress-'+str(physical)+'.json'), dict(role=role, **row))
        value = collect_features(base, data, physical, 4, sources, **plan['feature_options'], checked=lambda:preflight(plan), progress=progress)
        path = DEST/'features'/(role+'-'+str(physical)+'.pt')
        maximum = plan['storage_budget']['maximum_train_shard_bytes' if role == 'train' else 'maximum_held_shard_bytes']
        atomic_checkpoint(path, value, maximum, RESERVE)
        artifacts[role] = dict(path=str(path), sha256=file_hash(path), original_pairs=len(value['positions']),
            complete_context_tokens=value['complete_context_tokens'])
    guard.check(full=True); preflight(plan)
    atomic_json(OUT/'features'/('rank-'+str(physical)+'.json'), dict(complete=True, physical_GPU=physical, sources=sources,
        actual_native_values=984399690, all_original_train_and_held_responses_encoded=True, full_source_roles_preserved=True,
        artifacts=artifacts, model_optimizer_updates=0, goal_achieved=False))


def feature_inputs(plan, expected_sha):
    from slm_lab.codenet_checkpoint_v1 import FileBindings
    from slm_lab.corpus import file_hash
    import torch
    path = OUT/'feature-admission.json'
    if file_hash(path) != expected_sha: raise ValueError('The exact parent-admitted complete feature bytes must remain')
    receipt = read(path)
    guard = FileBindings({**plan['source_files_sha256'], str(path):expected_sha, **receipt['source_files_sha256']})
    shards = {role:[torch.load(row['artifacts'][role]['path'], map_location='cpu', weights_only=True, mmap=True)
        for row in receipt['all_four_complete_shards']] for role in ['train','valid']}
    return shards, receipt, guard


def fit_worker(expected_sha):
    from slm_lab.codenet_checkpoint_v1 import atomic_checkpoint
    from slm_lab.corpus import file_hash
    from slm_lab.diverse_images import atomic_json
    from slm_lab.native_reward_fitting_v1 import fit
    from slm_lab.whole_human_data_v2 import WholeHumanData
    plan = owner.registered('CPU-fit'); preflight(plan); device = setup_device(None)
    base, sources = load_base(plan, device); shards, admitted, guard = feature_inputs(plan, expected_sha)
    if admitted['sources'] != sources: raise ValueError('The complete actual native feature basis must agree in fitting')
    train = WholeHumanData(plan['data'], 'preferences', 'train', 'optimization')
    held = WholeHumanData(plan['data'], 'preferences', 'valid', 'evaluation')
    def progress(row):
        if row['step'] == 1 or row['step'] % 128 == 0 or row['step'] == plan['options']['steps']:
            atomic_json(OUT/'fit'/'progress.json', dict(actual_completed_updates=row['step'], planned_updates=plan['options']['steps'],
                actual_pair_mean_nll=row['pair_mean_nll'], full_epoch_pairs=85786, goal_achieved=False))
    def checkpoint(stage):
        slot = (stage.completed//256) % 2
        atomic_checkpoint(DEST/'fit'/('resume-slot-'+str(slot)+'.pt'), stage.state(), plan['storage_budget']['maximum_head_state_bytes'], RESERVE)
    result = fit(base, train, held, shards['train'], shards['valid'], sources, plan['options'],
        checked=lambda:preflight(plan), progress=progress, checkpoint=checkpoint)
    bundle = DEST/'fit'/'head-bundle.pt'
    atomic_checkpoint(bundle, result['bundle'], plan['storage_budget']['maximum_head_state_bytes'], RESERVE)
    for name in ['initial_held', 'final_held']:
        atomic_json(OUT/'fit'/(name+'.json'), result[name])
        if (OUT/'fit'/(name+'.json')).stat().st_size > plan['storage_budget']['maximum_held_report_bytes']:
            raise ValueError('Preserve every original held record; full report exceeded its declared budget')
    guard.check(full=True); preflight(plan)
    atomic_json(OUT/'fit'/'complete.json', dict(complete=True, sources=sources, bundle=str(bundle), bundle_sha256=file_hash(bundle),
        all85786original_human_pairs_once=True, actual_updates=result['bundle']['head']['actual_updates'],
        all4642initial_and_final_held_scores_complete=True, actual_head_recovery=result['actual_head_recovery'],
        actual_native_values=984399690, actual_head_values=198272, PPO_training_complete=False, goal_achieved=False))


def cold_worker(expected_features, expected_fit):
    import torch
    from slm_lab.codenet_checkpoint_v1 import atomic_checkpoint
    from slm_lab.corpus import file_hash
    from slm_lab.diverse_images import atomic_json
    from slm_lab.native_human_reward_v1 import NativeRewardFeatures, evaluate_reward, FrozenSharedNativeReward
    from slm_lab.native_reward_bundle_v1 import attach
    from slm_lab.own_dense_actor_v1 import OwnDenseActor
    from slm_lab.own_dense_standalone_v1 import export, load
    from slm_lab.whole_human_data_v2 import WholeHumanData
    from slm_lab.whole_human_sft_evidence_v1 import tensor_digest
    plan = owner.registered('CPU-cold'); preflight(plan); device = setup_device(None)
    if file_hash(OUT/'fit'/'complete.json') != expected_fit: raise ValueError('The exact full fitted-head receipt must remain')
    fitted = read(OUT/'fit'/'complete.json'); base, sources = load_base(plan, device)
    scorer = attach(base, fitted['bundle'], expected_sha256=fitted['bundle_sha256'], expected_sources=sources, expected_options=plan['options'])
    shards, admitted, guard = feature_inputs(plan, expected_features)
    held = WholeHumanData(plan['data'], 'preferences', 'valid', 'evaluation')
    values = NativeRewardFeatures(base, held, shards['valid'], sources)
    actual = evaluate_reward(base, values); original = read(OUT/'fit'/'final_held.json')
    if actual != original: raise ValueError('Every original held current-native head score must replay exactly from cold head bytes')
    native_sources = dict(sources, trained_human_reward_bundle_sha256=fitted['bundle_sha256'])
    folder = DEST/'native'; folder.mkdir(); shutil.copyfile(plan['initializer']['tokenizer'], folder/'tokenizer.json')
    checkpoint = folder/'model.pt'
    payload = export(base, sources=native_sources, provenance=dict(kind='own-complete-current-native-human-reward-head-v1',
        complete_original_joint_body_updates=5362, complete_human_head_updates=2681, all85786training_pairs_once=True,
        all4642initial_final_held_head_scores_complete=True, own_output_human_alignment_not_established=True),
        completed_updates=2681, tokenizer_sha256=plan['initializer']['tokenizer_sha256'])
    atomic_checkpoint(checkpoint, payload, plan['storage_budget']['maximum_native_export_bytes'], RESERVE)
    cold, metadata = load(checkpoint, expected_checkpoint_sha256=file_hash(checkpoint), expected_sources=native_sources)
    if tensor_digest(cold.named_parameters()) != tensor_digest(base.named_parameters()):
        raise ValueError('Every complete trained-reward native parameter must cold-load exactly')
    trained = torch.load(fitted['bundle'], map_location='cpu', weights_only=True, mmap=True)['head']
    cold_scorer = FrozenSharedNativeReward(cold, trained)
    policy = OwnDenseActor(cold, rank=16, seed=997331).eval()
    union = {id(p):p for m in [policy,cold_scorer] for p in m.parameters()}
    if policy.counts()['actual_unique'] != 999706442 or sum(p.numel() for p in union.values()) != 999706442:
        raise ValueError('The actually shared learned reward and policy/critic must retain exactly one under1Bnative graph')
    position = min(range(len(held)), key=lambda p: held.rows[p]['native']['chosen']['length'])
    ids = torch.tensor([held.sequence(position, 'chosen')[0]], dtype=torch.long)
    with torch.no_grad():
        a, b = scorer.score(ids), cold_scorer.score(ids)
        if not torch.equal(a,b) or not torch.equal(base(ids)['logits'],cold(ids)['logits']):
            raise ValueError('Actual complete original cold native text and shared human reward outputs must agree exactly')
    guard.check(full=True); preflight(plan)
    atomic_json(OUT/'cold.json', dict(complete=True, actual_native_values=984399690, actual_shared_policy_critic_and_reward_values=999706442,
        additional_reward_encoder_values=0, all_original4642cached_feature_head_scores_replayed=True,
        GPU_transformer_feature_computation_not_replayed_by_this_reader=True, all_native_parameters_cold_load_exact=True,
        actual_complete_original_text_and_shared_reward_outputs_exact=True, checkpoint=str(checkpoint), checkpoint_sha256=file_hash(checkpoint),
        sources=native_sources, tokenizer_sha256=metadata['tokenizer_sha256'], original_fixed_joint_language_vision_word_values_preserved=True,
        PPO_training_complete=False, own_output_human_alignment_or_high_end_quality_proved=False, goal_achieved=False))


def parent(args):
    from slm_lab.full_experiment_launch_queue_v1 import identity
    from slm_lab.local_source_closure_v2 import python_sources
    from slm_lab.native_reward_owned_recipe_v1 import prepare, sha
    from slm_lab.reserved_atomic_metadata_v1 import write_json
    normal(); workspace = Path(args.workspace).absolute(); joint = Path(args.joint).resolve()
    if workspace.exists() or workspace.resolve() != workspace or os.environ.get('CUDA_VISIBLE_DEVICES') != '' or 'torch' in sys.modules:
        raise ValueError('A fresh normal-scheduled CUDA-hidden non-neural source observer must precede owned reward work')
    captured = read(joint/'waiter-plan.json')['owner']; files = {str(p):sha(p) for p in python_sources(ROOT,[Path(__file__)])}
    files[str(joint/'waiter-plan.json')] = sha(joint/'waiter-plan.json')
    workspace.mkdir(); reports = workspace/'reports'; reports.mkdir()
    def guard():
        normal()
        if any(sha(p) != h for p,h in files.items()): raise ValueError('Every exact original joint owner and reward runtime byte must remain')
    def write(name,value):
        return write_json(reports/name,value,root=workspace,minimum_free_bytes=RESERVE,maximum_metadata_bytes=64*1024**2,guard=guard)
    write('waiter-plan.json',dict(owner=identity(os.getpid()),source_files_sha256=files,exact_original_joint_owner=captured,
        all_original_experiments_and_full_assessments_preserved=True,GPU_contexts=0,GPU_leases=0,head_updates=0,goal_achieved=False))
    deadline = time.monotonic()+35*86400
    while True:
        guard(); recipe = prepare(ROOT,joint=joint,expected_owner=captured,runtime_sources=files)
        if (recipe is not None and shutil.disk_usage(workspace).free >= recipe['storage_budget']['required_scratch_free_bytes']
                and shutil.disk_usage(ROOT).free >= recipe['storage_budget']['required_home_free_bytes'] and memory_available() >= 112*1024**3): break
        write('waiting-progress.json',dict(waiting_for='exact full original joint completion and complete physical reward budgets',
            GPU_work_started=False,head_updates=0,goal_achieved=False))
        if time.monotonic()>deadline: raise ValueError('Source observation lifetime expired; no original owner is relabelled stopped')
        time.sleep(30)
    configure(workspace)
    from slm_lab.codenet_checkpoint_v1 import FileBindings
    from slm_lab.diverse_images import atomic_json
    lease = owner.LEASE.open('a+')
    while True:
        try: fcntl.flock(lease.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB); break
        except BlockingIOError:
            write('waiting-progress.json',dict(waiting_for='actual original exclusive GPU lease',GPU_work_started=False,head_updates=0))
            if time.monotonic()>deadline: raise ValueError('GPU lease observation expired; no existing run is preempted')
            time.sleep(30)
    with lease:
        owner.idle_original(); guard()
        rows = subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.used','--format=csv,noheader,nounits'],text=True).splitlines()
        if len(rows)!=4 or any(r.rsplit(',',1)[0].strip()!='NVIDIA L40' or int(r.rsplit(',',1)[1])>128 for r in rows):
            raise ValueError('All four actual original L40s must be idle before new reward capacity work')
        published = ROOT/'checkpoints/native-human-reward-owned-v1'
        if published.exists() or published.with_name(published.name+'.pending').exists(): raise ValueError('Preserve every prior published or partial head experiment')
        bound = dict(recipe['source_files_sha256']); bound.update(owner.lease_sources())
        plan = dict(recipe,owner=identity(os.getpid()),source_files_sha256=bound,waiter=owner.WAITER,
            terminal_waiter_receipt_sha256=owner.terminal_receipt_sha256(),expires_at_unix=time.time()+7*86400,
            published_head_folder=str(published),candidate_promoted=False,goal_achieved=False)
        for p in [OUT/'capacity',OUT/'features',OUT/'fit',DEST/'features',DEST/'fit']:p.mkdir(parents=True)
        atomic_json(OUT/'plan.json',plan); atomic_json(OUT/'children.json',[]); source_guard=FileBindings(bound)
        driver=str(Path(__file__).resolve()); base=['-u',driver,'--workspace',str(WORKSPACE)]
        watcher=subprocess.Popen([str(ROOT/'.venv/bin/python'),*base,'--watchdog'],cwd=ROOT,stdout=(OUT/'watchdog.log').open('x'),
            stderr=subprocess.STDOUT,start_new_session=True,env={**os.environ,'CUDA_VISIBLE_DEVICES':''})
        def interrupted(signum,frame):raise KeyboardInterrupt('Owned current-native reward run interrupted')
        signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
        def tasks(mode):
            children=[owner.launch(plan,[*base,'--'+mode,'--gpu',str(r)],mode+'-'+str(r),str(r)) for r in range(4)]
            owner.wait_children(plan,children)
        success=False
        try:
            preflight(plan);tasks('capacity');tasks('features')
            artifacts=[read(OUT/'features'/('rank-'+str(r)+'.json')) for r in range(4)]; common=artifacts[0]['sources'];admitted_files={}
            for r,row in enumerate(artifacts):
                if (row.get('complete') is not True or row.get('physical_GPU')!=r or row.get('sources')!=common
                        or row.get('all_original_train_and_held_responses_encoded') is not True or row.get('full_source_roles_preserved') is not True):
                    raise ValueError('Every actual complete original feature shard and exact source role must remain')
                admitted_files[str(OUT/'features'/('rank-'+str(r)+'.json'))]=sha(OUT/'features'/('rank-'+str(r)+'.json'))
                for role,n in [('train',85786),('valid',4642)]:
                    item=row['artifacts'][role]
                    if item['original_pairs']!=len(range(r,n,4)) or sha(item['path'])!=item['sha256']:
                        raise ValueError('Every complete original train/held feature artifact must match its whole physical bytes')
                    admitted_files[item['path']]=item['sha256']
            atomic_json(OUT/'feature-admission.json',dict(complete=True,sources=common,all_four_complete_shards=artifacts,source_files_sha256=admitted_files))
            admission_sha=sha(OUT/'feature-admission.json')
            child=owner.launch(plan,[*base,'--fit','--feature-admission-sha',admission_sha],'CPU-fit','');owner.wait_children(plan,[child])
            fit_sha=sha(OUT/'fit'/'complete.json')
            child=owner.launch(plan,[*base,'--cold','--feature-admission-sha',admission_sha,'--fit-receipt-sha',fit_sha],'CPU-cold','');owner.wait_children(plan,[child])
            fitted=read(OUT/'fit'/'complete.json');cold=read(OUT/'cold.json');pending=published.with_name(published.name+'.pending');pending.mkdir()
            publication=[(Path(fitted['bundle']),'head-bundle.pt'),(OUT/'fit'/'initial_held.json','initial-held.json'),
                (OUT/'fit'/'final_held.json','final-held.json'),(OUT/'cold.json','cold.json')]
            published_files={};total=0
            for original,name in publication:
                total+=original.stat().st_size
                if total>plan['storage_budget']['maximum_published_bytes']:raise ValueError('Preserve full outputs; durable head publication exceeded its full budget')
                target=pending/name
                with original.open('rb') as src,target.open('xb') as dst:
                    while block:=src.read(1024**2):
                        preflight(plan)
                        if shutil.disk_usage(ROOT).free-len(block)<RESERVE:raise ValueError('Complete durable publication must preserve the original40GiBwriter floor')
                        dst.write(block)
                    dst.flush();os.fsync(dst.fileno())
                if sha(target)!=sha(original):raise ValueError('Every durable trained-head output byte must match')
                published_files[name]=sha(target)
            atomic_json(pending/'manifest.json',dict(format='own-complete-current-native-human-reward-publication-v1',
                initializer=plan['initializer'],sources=fitted['sources'],options=plan['options'],files_sha256=published_files,
                scratch_standalone=cold,head_attachment_requires_no_training_features=True,PPO_or_own_output_quality_not_proven=True,goal_achieved=False))
            pending.rename(published);source_guard.check(full=True);preflight(plan);success=True
        except BaseException as exc:
            atomic_json(OUT/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc(),automatic_retry=False,goal_achieved=False));raise
        finally:
            owner.restore_owned(plan,'Owned complete current-native human reward experiment ended');watcher.wait(timeout=20)
        if success:
            atomic_json(OUT/'complete.json',dict(complete=True,full85786pair_reward_head_epoch_and_all4642initial_final_held_scores_complete=True,
                all_four_original_L40_feature_capacities_and_complete_original_responses_checked=True,
                actual_native_values=984399690,actual_head_values=198272,actual_shared_policy_critic_and_reward_values=999706442,
                durable_head_publication=str(published),full_standalone_native_export_and_head_score_cold_complete=True,
                original_all_other_native_values_and_original_experiments_preserved=True,
                PPO_training_complete=False,fresh_own_output_human_and_high_end_all_round_quality_pending=True,candidate_promoted=False,goal_achieved=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace',type=Path,default=Path('/tmp/slm-native-human-reward-owned-GPU-v1'))
    parser.add_argument('--joint',type=Path,default=Path('/tmp/slm-complete-joint-owned-GPU-v1/reports'))
    parser.add_argument('--run',action='store_true');parser.add_argument('--watchdog',action='store_true')
    parser.add_argument('--capacity',action='store_true');parser.add_argument('--features',action='store_true')
    parser.add_argument('--fit',action='store_true');parser.add_argument('--cold',action='store_true')
    parser.add_argument('--gpu',type=int,choices=range(4));parser.add_argument('--feature-admission-sha');parser.add_argument('--fit-receipt-sha')
    args=parser.parse_args()
    if args.run:parent(args)
    else:
        configure(args.workspace)
        if args.watchdog:owner.watchdog()
        elif args.capacity and args.gpu is not None:feature_worker(args.gpu,True)
        elif args.features and args.gpu is not None:feature_worker(args.gpu,False)
        elif args.fit and args.feature_admission_sha:fit_worker(args.feature_admission_sha)
        elif args.cold and args.feature_admission_sha and args.fit_receipt_sha:cold_worker(args.feature_admission_sha,args.fit_receipt_sha)
        else:parser.error('Use explicit source-waiting --run or a registered owned worker mode')
