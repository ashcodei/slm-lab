"""Queue full original human-ranking features and reward-head comparison.

Wait for the exact existing SFT owner and every registered worker to terminate.
Use all four idle L40s only under a new lease; release them before CPU head
learning. No policy update, teacher model or new backbone is involved.
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

import train_point_grounding_joint_v1 as owner
from slm_lab.codenet_checkpoint_v1 import FileBindings, atomic_checkpoint
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT/'reports/whole-human-reward-v1'
OUTPUT = ROOT/'checkpoints/whole-human-reward-v1'
FEATURES = ROOT/'data/whole-human-reward-features-v1'
PREVIOUS = ROOT/'reports/whole-human-SFT-v1'
DATA = ROOT/'data/whole-human-alignment-v2'
PROBE = ROOT/'reports/whole-human-reward-full-CPU-v2'
QUEUE = ROOT/'reports/whole-human-reward-v1-queue.json'
RESERVE = 26*1024**3


def configure():
    owner.REPORTS = REPORTS; owner.resources.REPORTS = REPORTS


def predecessor_ended():
    plan = json.loads((PREVIOUS/'plan.json').read_text())
    if owner.resources.same(plan['owner']) or any(owner.resources.same(a) for a in json.loads((PREVIOUS/'children.json').read_text())):
        return False
    if not (PREVIOUS/'restored.json').exists():
        raise ValueError('Exact preceding owner and registered workers must end and restore the original waiter')
    if not (PREVIOUS/'complete.json').exists() or (PREVIOUS/'failure.json').exists():
        raise ValueError('Preceding full human SFT/native comparison did not complete; inspect its evidence before a new GPU run')
    return True


def feature_sources(plan):
    return dict(run_plan_sha256=file_hash(REPORTS/'plan.json'), source_manifest_sha256=plan['source_manifest_sha256'],
        source_admission_sha256=plan['source_admission_sha256'], frozen_encoder_sha256=plan['frozen_encoder_sha256'])


def worker(rank):
    subject = 'features-'+str(rank)
    plan = json.loads((REPORTS/'plan.json').read_text()); me = owner.resources.identity(os.getpid())
    if os.environ.get('CUDA_VISIBLE_DEVICES') != str(rank):
        raise ValueError('Exact owned isolated physical GPU required before CUDA initialization')
    while True:
        owner.owned(plan)
        if any(a['pid'] == me['pid'] and a['ticks'] == me['ticks'] and a['subject'] == subject
               for a in json.loads((REPORTS/'children.json').read_text())):
            break
        time.sleep(.05)
    import torch
    from slm_lab.point_grounding_evaluation_v1 import state_digest
    from slm_lab.whole_human_data_v2 import WholeHumanData
    from slm_lab.whole_human_reward_v1 import collect_features
    from slm_lab.whole_human_sft_inference_v1 import fresh
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True); torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(.85, 0)
    if torch.cuda.get_device_name(0) != 'NVIDIA L40':
        raise ValueError('Actual original L40 required')
    guard = FileBindings(plan['source_files_sha256']); guard.check(full=True)
    model, _ = fresh(plan['initial_model_plan'], 'cuda:0')
    if state_digest(model.original.language.base) != plan['frozen_encoder_sha256']:
        raise ValueError('Exact own frozen original-general encoder bytes required')
    fixed = [(id(p), p._version, p.data_ptr()) for p in model.parameters()]
    last_progress = [0.]
    for role, purpose in (('train', 'optimization'), ('valid', 'evaluation')):
        data = WholeHumanData(DATA, 'preferences', role, purpose)
        def checked():
            owner.owned(plan); guard.check()
        def progress(value):
            if time.monotonic()-last_progress[0] >= 10:
                last_progress[0] = time.monotonic()
                atomic_json(REPORTS/f'features-{rank}-progress.json', dict(role=role, rank=rank, **value, goal_achieved=False))
                print(role, value['encoded_complete_responses'], '/', value['planned_complete_responses'], flush=True)
        payload = collect_features(model, data, rank, 4, feature_sources(plan), device='cuda:0',
            amp=lambda:torch.autocast('cuda', dtype=torch.bfloat16), checked=checked, progress=progress,
            token_budget=4096, max_sequences=16)
        if fixed != [(id(p), p._version, p.data_ptr()) for p in model.parameters()] or any(p.grad is not None for p in model.parameters()):
            raise ValueError('Feature collection must leave every actual model tensor and gradient unchanged')
        atomic_checkpoint(FEATURES/f'{role}-{rank}.pt', payload, 384*1024**2, RESERVE)
    guard.check(full=True)
    if state_digest(model.original.language.base) != plan['frozen_encoder_sha256']:
        raise ValueError('Frozen encoder bytes changed during complete source feature collection')
    atomic_json(REPORTS/f'features-{rank}-complete.json', dict(rank=rank, all_weights_versions_pointers_fixed=True,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(), feature_sources=feature_sources(plan),
        files_sha256={f'{role}-{rank}.pt':file_hash(FEATURES/f'{role}-{rank}.pt') for role in ('train', 'valid')},
        model_updates=0, goal_achieved=False))


def learn_rewards(plan):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('Head learning and feature readback must run on CPU with hidden CUDA devices')
    import torch
    from slm_lab.whole_human_data_v2 import WholeHumanData
    from slm_lab.whole_human_reward_v1 import RewardFeatures, RewardHeadStage, evaluate_reward, FEATURE_ROUTE
    from slm_lab.whole_human_reward_inference_v1 import export as export_reward, attach as attach_reward, frozen_general_reward
    from slm_lab.whole_human_sft_evidence_v1 import tensor_digest
    from slm_lab.whole_human_sft_inference_v1 import fresh
    torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
    guard = FileBindings(plan['source_files_sha256']); guard.check(full=True)
    data = {role:WholeHumanData(DATA, 'preferences', role, purpose)
            for role, purpose in (('train', 'optimization'), ('valid', 'evaluation'))}
    all_hashes = {}
    for rank in range(4):
        receipt = json.loads((REPORTS/f'features-{rank}-complete.json').read_text())
        if receipt['feature_sources'] != feature_sources(plan) or not receipt['all_weights_versions_pointers_fixed']:
            raise ValueError('Every actual complete frozen feature worker receipt required')
        all_hashes.update(receipt['files_sha256'])
    if any(file_hash(FEATURES/name) != sha for name, sha in all_hashes.items()):
        raise ValueError('Every complete feature shard must retain its original worker bytes')
    features = {role:RewardFeatures(data[role], [torch.load(FEATURES/f'{role}-{rank}.pt', map_location='cpu',
        weights_only=True, mmap=True) for rank in range(4)], feature_sources(plan)) for role in data}
    atomic_json(FEATURES/'admission.json', dict(complete_original_train_pairs=len(data['train']),
        complete_original_held_pairs=len(data['valid']), feature_sources=feature_sources(plan),
        files_sha256=all_hashes, original_native_tokens_roles_fingerprints_and_full_coverage_reconstructed=True,
        numerical_GPU_feature_forwards_not_independently_replayed=True, feature_route=FEATURE_ROUTE,
        train_only_head_optimization_admitted=True, goal_achieved=False))
    admission_sha = file_hash(FEATURES/'admission.json')
    sources = {**feature_sources(plan), 'feature_admission_sha256':admission_sha}
    model, _ = fresh(plan['initial_model_plan'], 'cpu')
    def fixed_digest():
        chosen = {id(p) for p in model.extension_parameters('reward')}
        return tensor_digest((n, p) for n, p in model.named_parameters() if id(p) not in chosen)
    initial = {n:t.detach().clone() for n,t in model.parts['reward'].state_dict().items()}
    fixed = fixed_digest(); baseline = evaluate_reward(model, features['valid'])
    atomic_json(REPORTS/'initial-human-rankings.json', baseline)
    results = {'initial':baseline['overall']}; exports = {}
    for arm, spec in plan['arms'].items():
        model.parts['reward'].load_state_dict(initial)
        options = {**plan['training_options'], 'learning_rate':spec['learning_rate']}
        optimizer = torch.optim.AdamW(model.extension_parameters('reward'), lr=options['learning_rate'],
            betas=tuple(options['betas']), weight_decay=options['weight_decay'], foreach=False)
        stage = RewardHeadStage(model, optimizer, features['train'], options, sources)
        folder = OUTPUT/arm; folder.mkdir()
        for step in range(options['steps']):
            guard.check(); record = stage.update()
            if step == 0 or (step+1)%100 == 0:
                atomic_json(REPORTS/'head-progress.json', dict(arm=arm, planned_steps=options['steps'], latest=record, goal_achieved=False))
                print('actual_human_reward_head_update', arm, step+1, record['pair_mean_nll'], flush=True)
            if (step+1)%512 == 0 or step+1 == options['steps']:
                atomic_checkpoint(folder/'resume.pt', stage.state(), 64*1024**2, RESERVE)
        draws = [p for m in stage.metrics for p in m['source_positions']]
        if len(draws) != len(data['train']) or sorted(draws) != list(range(len(data['train']))) or fixed_digest() != fixed:
            raise ValueError('Exactly every human training pair once and all nonreward weights unchanged required')
        evaluation = evaluate_reward(model, features['valid']); results[arm] = evaluation['overall']
        atomic_json(REPORTS/(arm+'-human-rankings.json'), evaluation)
        payload = export_reward(model, plan, file_hash(REPORTS/'plan.json'), admission_sha, arm, stage.completed)
        atomic_checkpoint(folder/'model.pt', payload, 2*1024**2, RESERVE)
        trained = {n:t.detach().clone() for n,t in model.parts['reward'].state_dict().items()}
        model.parts['reward'].load_state_dict(initial)
        attachment = attach_reward(model, folder/'model.pt', REPORTS/'plan.json', FEATURES/'admission.json', arm,
            expected_checkpoint_sha256=file_hash(folder/'model.pt'), expected_plan_sha256=file_hash(REPORTS/'plan.json'),
            expected_feature_admission_sha256=admission_sha)
        if any(not torch.equal(t, model.parts['reward'].state_dict()[n]) for n,t in trained.items()):
            raise ValueError('Every trained reward head tensor must survive completed attachment')
        sequence, _ = data['valid'].sequence(0, 'chosen')
        with torch.no_grad():
            fresh_feature = frozen_general_reward(model, torch.tensor([sequence]))
        if not bool(torch.isfinite(fresh_feature).all()) or fixed_digest() != fixed:
            raise ValueError('Actual reloaded full-model reward route must be finite with all nonreward weights fixed')
        exports[arm] = dict(checkpoint_sha256=file_hash(folder/'model.pt'), resume_sha256=file_hash(folder/'resume.pt'),
                            every_original_train_pair_once=True, actual_Adam_states=len(optimizer.state),
                            trained_head_attachment=attachment, actual_finite_reloaded_full_model_reward=True)
    best = min(plan['arms'], key=lambda a:(results[a]['pair_mean_nll'], a))
    selected = best if results[best]['pair_mean_nll'] < results['initial']['pair_mean_nll'] else 'initial'
    guard.check(full=True)
    atomic_json(REPORTS/'complete.json', dict(status='Full original human-ranking frozen-general reward-head comparison complete',
        initial_and_trained_held_human_ranking_metrics=results, selected_arm=selected,
        selection='minimum complete held-human pair NLL; retain initial unless improved; no policy capability selection',
        trained_exports=exports, all_nonreward_model_weights_unchanged=True,
        genuine_publisher_human_rankings=True, fresh_human_judgments_on_own_rollouts=False,
        policy_updates=0, reward_policy_alignment_and_high_end_capability_not_established=True,
        candidate_promoted=False, goal_achieved=False))


def parent():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('CPU parent requires explicitly hidden CUDA devices')
    if REPORTS.exists() or OUTPUT.exists() or FEATURES.exists() or QUEUE.exists():
        raise ValueError('Preserve previous feature/training/queue artifacts; no automatic restart')
    if not (PROBE/'complete.json').exists():
        raise ValueError('Actual full-size CPU frozen-feature/reward recovery proof required')
    if '3 passed' not in (ROOT/'reports/whole-human-reward-v1-tests.log').read_text():
        raise ValueError('Focused whole-source/frozen-encoder/recovery tests required')
    if '3 passed' not in (ROOT/'reports/whole-human-reward-driver-v1-tests.log').read_text():
        raise ValueError('Exact predecessor and isolated-worker resource checks required')
    if '1 passed' not in (ROOT/'reports/whole-human-reward-inference-v1-tests.log').read_text():
        raise ValueError('Completed reward attachment and source/corruption rejection tests required')
    queued_files = {}
    for name in ('scripts/train_whole_human_reward_v1.py', 'src/slm_lab/whole_human_reward_v1.py',
                 'src/slm_lab/whole_human_reward_inference_v1.py', 'tests/test_whole_human_reward_v1.py',
                 'tests/test_whole_human_reward_driver_v1.py', 'tests/test_whole_human_reward_inference_v1.py',
                 'reports/whole-human-reward-v1-tests.log', 'reports/whole-human-reward-driver-v1-tests.log',
                 'reports/whole-human-reward-inference-v1-tests.log', 'scripts/probe_whole_human_reward_full_CPU_v2.py'):
        queued_files[str(ROOT/name)] = file_hash(ROOT/name)
    for path in (PREVIOUS/'plan.json', PROBE/'plan.json', PROBE/'complete.json', PROBE/'resume.pt'):
        queued_files[str(path)] = file_hash(path)
    queue_guard = FileBindings(queued_files)
    preceding_identity = json.loads((PREVIOUS/'plan.json').read_text())['owner']
    waiting_owner = owner.resources.identity(os.getpid()); waiting_since = time.monotonic()
    while not predecessor_ended():
        queue_guard.check()
        if time.monotonic()-waiting_since > 9*3600:
            raise ValueError('Preceding owner still running; queue expires without launching GPUs')
        atomic_json(QUEUE, dict(status='waiting for exact full-human-SFT owner and workers to end and restore original work',
            owner=waiting_owner, predecessor=preceding_identity, source_files_sha256=queued_files,
            observed_at=datetime.now(timezone.utc).isoformat(), model_updates=0, goal_achieved=False))
        time.sleep(10)
    queue_guard.check(full=True)
    old = json.loads((PREVIOUS/'plan.json').read_text()); probe = json.loads((PROBE/'complete.json').read_text())
    if (not probe['actual_next_update_all_reward_tensors_Adam_cursor_and_RNG_exact']
            or not probe['all_fixed_bytes_versions_pointers_original_native_routes_and_code_cache_exact']
            or not probe['all_capacity_updates_to_be_discarded']):
        raise ValueError('Every actual full-size frozen-feature and reward recovery condition required')
    files = dict(old['source_files_sha256'])
    for path, sha in queued_files.items():
        if path in files and files[path] != sha:
            raise ValueError('Never silently reseal a changed existing source binding')
        files[path] = sha
    for path in (PREVIOUS/'plan.json', PREVIOUS/'complete.json', PREVIOUS/'restored.json',
                 PREVIOUS/'independent-comparison.json', PROBE/'plan.json', PROBE/'complete.json', PROBE/'resume.pt'):
        files[str(path)] = file_hash(path)
    owner.idle_original()
    guard = FileBindings(files)
    with (ROOT/'reports/code-residual-production-v4.gpu.lock').open('a+') as lease:
        fcntl.flock(lease.fileno(), fcntl.LOCK_EX|fcntl.LOCK_NB)
        if not predecessor_ended(): raise ValueError('Prior work became live; never overlap GPU owners')
        owner.idle_original()
        memory = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True).splitlines()
        if len(memory) != 4 or any(int(v.strip()) > 128 for v in memory):
            raise ValueError('All four actual original GPUs must be idle')
        if shutil.disk_usage(ROOT).free < RESERVE+4*1024**3:
            raise ValueError('Complete source features and atomic checkpoint reserve required')
        guard.check(full=True); REPORTS.mkdir(); OUTPUT.mkdir(); FEATURES.mkdir()
        from slm_lab.whole_human_data_v2 import WholeHumanData
        count = len(WholeHumanData(DATA, 'preferences', 'train', 'optimization'))
        held_count = len(WholeHumanData(DATA, 'preferences', 'valid', 'evaluation'))
        plan = dict(created_at=datetime.now(timezone.utc).isoformat(), owner=waiting_owner, waiter=owner.WAITER,
            source_files_sha256=files, source_manifest_sha256=file_hash(DATA/'manifest.json'),
            source_admission_sha256=file_hash(DATA/'admission.json'), initial_model_plan=old,
            source_population=count, held_source_population=held_count,
            frozen_encoder_sha256=probe['frozen_original_general_encoder_sha256'],
            training_options=dict(seed=947331, steps=math.ceil(count/64), batch_size=64, betas=[.9, .95], weight_decay=.01, gradient_clip=1.),
            arms=dict(rate_1e4=dict(learning_rate=1e-4), rate_3e4=dict(learning_rate=3e-4), rate_1e3=dict(learning_rate=1e-3)),
            expires_at_unix=time.time()+4*3600, automatic_retry=False, all_capacity_updates_discarded=True,
            candidate_promoted=False, goal_achieved=False)
        atomic_json(REPORTS/'plan.json', plan); atomic_json(REPORTS/'children.json', [])
        def interrupted(signum, frame): raise KeyboardInterrupt('Owned frozen-feature collection interrupted')
        signal.signal(signal.SIGTERM, interrupted); signal.signal(signal.SIGINT, interrupted)
        driver = str(Path(__file__).resolve())
        watcher = subprocess.Popen(['.venv/bin/python', '-u', driver, '--watchdog'], cwd=ROOT,
            stdout=(REPORTS/'watchdog.log').open('xb'), stderr=subprocess.STDOUT, start_new_session=True)
        try:
            owner.idle_original(); os.kill(owner.WAITER['pid'], signal.SIGSTOP); deadline = time.monotonic()+5
            while (owner.resources.same(owner.WAITER) or {}).get('state') not in ('T', 't'):
                if time.monotonic() > deadline: raise ValueError('Exact idle original waiter did not enter owned hold')
                time.sleep(.05)
            if (Path('/proc')/str(owner.WAITER['pid'])/'task'/str(owner.WAITER['pid'])/'children').read_text().strip():
                raise ValueError('Original waiter started work; never preempt it')
            children = [owner.launch(plan, ['-u', driver, '--worker', str(rank)], 'features-'+str(rank), str(rank)) for rank in range(4)]
            owner.wait_children(plan, children); guard.check(full=True)
            atomic_json(REPORTS/'feature-collection-complete.json', dict(all_four_actual_frozen_feature_workers_ended=True, model_updates=0, goal_achieved=False))
        except BaseException as error:
            atomic_json(REPORTS/'failure.json', dict(phase='feature_collection', error=repr(error), automatic_retry=False, goal_achieved=False)); raise
        finally:
            owner.restore_owned(plan, 'Frozen general feature collection ended; release every GPU before CPU reward learning')
            watcher.wait(timeout=20)
    try:
        learn_rewards(plan)
    except BaseException as error:
        atomic_json(REPORTS/'failure.json', dict(phase='CPU_reward_learning', error=repr(error), automatic_retry=False, goal_achieved=False)); raise


if __name__ == '__main__':
    configure(); parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', type=int, choices=range(4)); parser.add_argument('--watchdog', action='store_true')
    args = parser.parse_args()
    if args.watchdog: owner.watchdog()
    elif args.worker is not None: worker(args.worker)
    else: parent()
