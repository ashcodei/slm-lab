"""Bounded full-source coding SFT beside the unchanged original four-GPU run.

This separate resource contract retains the original global flock, exact old
workers and their settings. One registered GPU0 child has a hard12GiB cap;
only its own registered children can be stopped. No current RM/PPO claim.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

import probe_supplemental_public_code_GPU_v1 as capacity

ROOT = capacity.ROOT
OUT = Path('/tmp/slm-supplemental-public-code-full-GPU-v1')
CAPACITY = capacity.OUT
SCRIPT = Path(__file__).resolve()
COMMON_LOCK = ROOT/'reports/supplemental-public-code-GPU0.lock'
PROFILES = ('plain_tokens', 'eos_weight4')
ENDPOINTS = ('baseline', *PROFILES)
GIB = 1024**3
capacity.OUT = OUT  # New process-local output; closed capacity sources stay unchanged.
sha, identity, live, guard, write = capacity.sha, capacity.identity, capacity.live, capacity.guard, capacity.write


def read(path): return json.loads(Path(path).read_text())


def protected(plan):
    return {p for p, _ in capacity.OLD_WORKERS}|{plan['old_owner']['pid'], plan['owner']['pid']}


def stop(actor, plan): capacity.stop_new(actor, protected=protected(plan), script=SCRIPT)


def checked(plan):
    if capacity.kernel_owner(COMMON_LOCK) != plan['owner']['pid']:
        raise ValueError('The actual supplemental kernel lock must stay with its exact owner')
    if shutil.disk_usage(OUT.parent).free < 40*GIB:
        raise ValueError('The unchanged scratch writer reserve must remain funded')
    # NVML can briefly retain a just-exited context. The original guard checks
    # any reused PID against its registered birth before accepting that entry.
    return guard(plan, read(OUT/'children.json'))


def watchdog():
    plan = read(OUT/'plan.json')
    while not (OUT/'restored.json').exists():
        if not live(plan['owner']) or time.time() > plan['expires_at_unix']:
            for actor in read(OUT/'children.json'): stop(actor, plan)
            write('watchdog-cleanup.json', dict(own_registered_children_only=True, old_processes_signalled=0))
            return
        time.sleep(.2)


def sources(plan, endpoint):
    import hashlib
    return dict(initial_native_sha256=plan['native']['checkpoint_sha256'],
        tokenizer_sha256=plan['native']['tokenizer_sha256'], public_manifest_sha256=plan['data_manifest_sha256'],
        public_admission_sha256=plan['data_admission_sha256'], run_plan_sha256=sha(OUT/'plan.json'),
        objective_sha256=hashlib.sha256(endpoint.encode()).hexdigest())


def initialize(plan, endpoint):
    import random
    import numpy as np
    import torch
    from slm_lab.own_dense_actor_v1 import OwnDenseActor
    from slm_lab.own_native_query_v1 import inspect_package, load_package
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    os.sched_setaffinity(0, {20})
    native = plan['native']
    package = inspect_package(native['checkpoint'], expected_sha256=native['checkpoint_sha256'],
        expected_tokenizer_sha256=native['tokenizer_sha256'], expected_sources=native['sources'])
    with torch.random.fork_rng(devices=[]):
        base, _ = load_package(package)
        model = OwnDenseActor(base, rank=16, seed=1003311).eval()
    checked(plan)
    torch.cuda.set_device(0)
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(12*GIB/total, 0)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if torch.cuda.get_device_name(0) != 'NVIDIA L40' or model.counts()['actual_unique'] != 999706442:
        raise ValueError('The exact original L40 and full own999M policy graph are required')
    model.to('cuda:0')
    random.seed(1003321)
    np.random.seed(1003321)
    torch.manual_seed(1003321)
    torch.cuda.manual_seed(1003321)
    return model


def train_child(plan, profile):
    import torch
    from slm_lab.codenet_checkpoint_v1 import atomic_checkpoint
    from slm_lab.own_dense_actor_checkpoint_v1 import make_state
    from slm_lab.public_reasoned_code_train_data_v1 import PublicReasonedCodeSFTData
    from slm_lab.serial_public_code_epoch_v1 import train
    from slm_lab.whole_human_sft_evidence_v1 import tensor_digest
    model = initialize(plan, profile)
    data = PublicReasonedCodeSFTData(ROOT, purpose='optimization', views='both')
    folder = OUT/profile
    folder.mkdir()
    bound = sources(plan, profile)
    initial = tensor_digest(model.parts.named_parameters())
    fixed = tensor_digest(model.base.named_parameters())
    adam = torch.optim.AdamW([p for _, p in model.named_selected(('actor',))], lr=3e-5,
        betas=(.9, .95), eps=1e-8, weight_decay=.01, foreach=False, fused=False)
    write(profile+'/initial.json', dict(actual_registered_parameters=model.counts()['actual_unique'],
        selected_policy_sha256=initial, frozen_native_sha256=fixed, sources=bound,
        initial_optimizer_states=len(adam.state), discarded_capacity_weights_not_loaded=True,
        critic_initially_zero_not_trained_PPO=True))
    journal = folder/'actual-updates.jsonl'
    with journal.open('x') as stream:
        def record(step, row):
            stream.write(json.dumps(dict(step=step, **row), allow_nan=False)+'\n')
            stream.flush()
            write(profile+'/progress.json', dict(phase='training', actual_complete_updates=step,
                planned_updates=230, last=row, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(), goal_achieved=False))
            print(json.dumps(dict(profile=profile, step=step, planned=230,
                train_NLL=row['original_token_mean_NLL'], EOS_NLL=row['EOS_token_mean_NLL'])), flush=True)
        def checkpoint(step, entries, rows):
            state = make_state(model, adam, groups=('actor',), sources=bound, completed=step,
                planned=len(entries), cursor={'next_global_batch':step}, metrics=rows, device='cuda:0')
            name = 'final-state.pt' if step == len(entries) else 'state-slot-'+str((step//16)%2)+'.pt'
            path = folder/name
            atomic_checkpoint(path, state, 384*1024**2, 40*GIB)
            write(profile+'/latest-state.json', dict(checkpoint=str(path), checkpoint_sha256=sha(path),
                completed=step, planned=len(entries), cursor=state['cursor']))
        result = train(model, adam, data, profile, checked=lambda:checked(plan), record=record, checkpoint=checkpoint)
        os.fsync(stream.fileno())
    if tensor_digest(model.base.named_parameters()) != fixed:
        raise ValueError('Every original frozen dense/vision/word/auxiliary value must remain exact')
    final = read(folder/'latest-state.json')
    if final['completed'] != 230 or Path(final['checkpoint']).name != 'final-state.pt':
        raise ValueError('A complete final original epoch checkpoint must be durable')
    result.update(initial_selected_policy_sha256=initial, frozen_native_sha256=fixed,
        final_selected_policy_sha256=tensor_digest(model.parts.named_parameters()), final_state=final,
        actual_selected_Adam_states=len(adam.state), actual_registered_parameters=model.counts()['actual_unique'],
        actual_update_journal_sha256=sha(journal), sources=bound,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        matched_assessments_pending=True, standalone_export_pending=True, no_new_human_or_PPO_updates=True)
    checked(plan)
    write(profile+'/training-complete.json', result)


def held_loss(model, data, plan, endpoint):
    import math
    import torch
    from slm_lab.codenet_token_pack_v1 import CompleteCodePack
    from slm_lab.complete_response_stopping_loss_v1 import complete_response_CE_sums
    held = CompleteCodePack(data.folder, 'validation', purpose='audit')
    if len(held) != 425: raise ValueError('All425 reserved views must remain evaluation-only')
    records = []
    with torch.no_grad():
        for i in range(len(held)):
            checked(plan)
            ids, labels = held.sequence(i)
            inputs = torch.tensor([ids], device='cuda:0')
            targets = torch.tensor([labels], device='cuda:0')
            mask = torch.ones_like(inputs, dtype=torch.bool)
            with torch.autocast('cuda', dtype=torch.bfloat16): logits = model(inputs)['logits']
            ce = complete_response_CE_sums(logits.float(), targets, mask, profile='plain_tokens')
            records.append(dict(reserved_local_position=i, original_pack_position=int(held.positions[i]),
                complete_input_token_ids=ids, original_response_EOS_labels=labels,
                CE_sum=float(ce['original_CE_sums'][0]), EOS_CE_sum=float(ce['EOS_CE_sums'][0]),
                response_EOS_targets=int(ce['original_target_counts'][0])))
            del logits, ce, inputs, targets, mask
    targets = sum(r['response_EOS_targets'] for r in records)
    if targets != held.manifest['counts']['validation']['target_tokens']:
        raise ValueError('Every held response/EOS target must be scored exactly once')
    write(endpoint+'/assessment/held-code-NLL.json', dict(complete=True, reserved_views=len(records),
        response_EOS_targets=targets, token_mean_NLL=math.fsum(r['CE_sum'] for r in records)/targets,
        EOS_token_mean_NLL=math.fsum(r['EOS_CE_sum'] for r in records)/len(records), records=records,
        reserved_optimization_updates=0, all_original_masks_and_targets_preserved=True, goal_achieved=False))


def assess_child(plan, endpoint):
    import torch
    from slm_lab.codenet_checkpoint_v1 import atomic_checkpoint
    from slm_lab.inference_rotary_cache_v1 import inference_cache
    from slm_lab.native_code_shards_v1 import evaluate_shard, reconcile
    from slm_lab.native_greedy_observer_v1 import observe, signature
    from slm_lab.public_reasoned_code_train_data_v1 import PublicReasonedCodeSFTData
    from slm_lab.whole_human_sft_evidence_v1 import tensor_digest
    model = initialize(plan, endpoint)
    initial = tensor_digest(model.parts.named_parameters())
    folder = OUT/endpoint
    if endpoint == 'baseline': folder.mkdir()
    if endpoint != 'baseline':
        result = read(folder/'training-complete.json')
        latest = result['final_state']
        if latest['completed'] != 230 or sha(latest['checkpoint']) != latest['checkpoint_sha256']:
            raise ValueError('Only the complete original230-update private state may be assessed')
        state = torch.load(latest['checkpoint'], map_location='cpu', weights_only=True, mmap=True)
        if state['completed'] != 230 or state['planned'] != 230 or state['cursor'] != {'next_global_batch':230}:
            raise ValueError('Exact full-source state cursor required')
        model.import_export(state['model'], expected_sources=sources(plan, endpoint))
        if initial != result['initial_selected_policy_sha256']:
            raise ValueError('Both arms and baseline must start from the identical fresh policy values')
        del state
    checkpoint = folder/'selected-model.pt'
    atomic_checkpoint(checkpoint, model.export(sources(plan, endpoint)), 128*1024**2, 40*GIB)
    shutil.copyfile(Path(plan['native']['checkpoint']).parent/'tokenizer.json', folder/'tokenizer.json')
    assessment = folder/'assessment'
    assessment.mkdir()
    model.eval()
    native = model.policy
    native.gradient_checkpointing = False
    before = signature(native)
    data = PublicReasonedCodeSFTData(ROOT, purpose='optimization')
    held_loss(native, data, plan, endpoint)
    metadata = dict(tokenizer_sha256=plan['native']['tokenizer_sha256'], training_options={'image_size':448})
    original_threads = torch.set_num_threads
    torch.set_num_threads = lambda count:None
    try:
        with inference_cache():
            for frame in ['completion', 'instruction']:
                journal = assessment/('code-'+frame+'-actual-actions.jsonl')
                with journal.open('x') as stream:
                    def record(row):
                        checked(plan)
                        stream.write(json.dumps(row)+'\n')
                        stream.flush()
                    with observe(native, record) as delivered:
                        envelope = evaluate_shard(native, metadata, checkpoint, assessment,
                            rank=0, world=1, frame=frame, device='cuda:0')
                    if len(delivered) != 24: raise ValueError('Every original native code delivery must retain its real action IDs/EOS')
                    os.fsync(stream.fileno())
                reconcile([envelope], assessment/('code-'+frame+'.json'))
            for name, options in [
                    ('knowledge', dict(max_examples=56, max_new_tokens=32)),
                    ('math', dict(max_examples=16, max_new_tokens=256)),
                    ('ocr', dict(max_examples=32, max_new_tokens=128, image_size=448)),
                    ('objects', dict(max_examples=32, max_new_tokens=32)),
                    ('vision', dict(max_examples=32, image_size=448))]:
                module = importlib.import_module('slm_lab.'+{'objects':'object'}.get(name, name)+'_evaluation')
                loader = module.load_checkpoint
                module.load_checkpoint = lambda *a, **kw:(native, metadata)
                try:
                    checked(plan)
                    path = ROOT/'data/math-reasoning-v2/test.jsonl' if name == 'math' else Path(plan['pulse_sources'][name])
                    module.evaluate(checkpoint, path, assessment/(name+'.json'), device='cuda:0', **options)
                finally: module.load_checkpoint = loader
    finally: torch.set_num_threads = original_threads
    if signature(native) != before or any(p.grad is not None for p in native.parameters()):
        raise ValueError('The complete baseline/candidate must remain unchanged throughout assessment')
    data.audit_provenance()
    checked(plan)
    write(endpoint+'/assessment-complete.json', dict(complete=True, endpoint=endpoint,
        selected_model_checkpoint=str(checkpoint), selected_model_sha256=sha(checkpoint),
        selected_checkpoint_requires_the_explicit_bound_native=True, standalone_export_pending=True,
        initial_selected_policy_sha256=initial, actual_registered_parameters=model.counts()['actual_unique'],
        every_original_code_program_and_action_retained=True, programs=48, all425reserved_views_scored=True,
        original_bounded_knowledge_math_OCR_object_and_grounding_checks_complete=True,
        all_actual_loaded_values_unchanged=True, inference_precision='BF16', inference_device='cuda',
        actual_training_updates=0 if endpoint == 'baseline' else 230,
        small_reused_development_suites_not_final_benchmarks=True, candidate_promoted=False, goal_achieved=False))


def child(endpoint, task):
    plan = read(OUT/'plan.json')
    me = identity(os.getpid())
    while me not in read(OUT/'children.json'):
        if not live(plan['owner']) or time.time() > plan['expires_at_unix']:
            raise ValueError('Only an exact registered live child may initialize CUDA')
        time.sleep(.05)
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '0':
        raise ValueError('Only the separately budgeted physicalGPU0 may be visible')
    checked(plan)
    if task == 'train' and endpoint in PROFILES: train_child(plan, endpoint)
    elif task == 'assess': assess_child(plan, endpoint)
    else: raise ValueError('An explicit complete training or assessment endpoint is required')


def parent():
    if OUT.exists() or os.environ.get('CUDA_VISIBLE_DEVICES') != '' or 'torch' in sys.modules:
        raise ValueError('Fresh CUDA-hidden non-neural owner required')
    from slm_lab.local_source_closure_v2 import python_sources
    capacity_result = read(CAPACITY/'complete.json')
    if not capacity_result['complete'] or not capacity_result['supplemental_GPU_contexts_released']:
        raise ValueError('Completed actual full-size source capacity and released contexts are required')
    if any(sha(p) != h for p, h in capacity_result['source_files_sha256'].items()):
        raise ValueError('Every closed capacity/input/source byte must remain unchanged')
    for profile in PROFILES:
        result = read(CAPACITY/profile/'complete.json')
        if (not result['complete'] or result['actual_unique_parameters'] != 999706442
                or result['actual_selected_Adam_states'] != 561 or not result['exact_next_source_weights_Adam_RNG_recovery']
                or result['selected_original_global_batch_indices'] != [137,61,95,229]
                or result['actual_final_epoch_tail'] != 11 or result['peak_reserved_bytes'] > 8*GIB):
            raise ValueError('Both complete original full-size capacity/recovery profiles are mandatory')
    tests = ROOT/'reports/serial-public-code-epoch-v1-tests.log'
    if '3 passed' not in tests.read_text(): raise ValueError('The real complete-epoch/corruption checks must pass before launch')
    resource_tests = ROOT/'reports/supplemental-public-code-epoch-resources-v1-tests.log'
    if '1 passed' not in resource_tests.read_text(): raise ValueError('The actual own-only full-epoch watchdog cleanup must pass')
    native = read(capacity.NATIVE/'complete.json')
    files = dict(capacity_result['source_files_sha256'])
    for path in [*python_sources(ROOT, [SCRIPT]), ROOT/'src/slm_lab/serial_public_code_epoch_v1.py',
            ROOT/'tests/test_serial_public_code_epoch_v1.py', tests, CAPACITY/'complete.json',
            ROOT/'tests/test_supplemental_public_code_epoch_resources_v1.py', resource_tests,
            *(CAPACITY/p/'complete.json' for p in PROFILES)]: files[str(path)] = sha(path)
    # Dynamic evaluator imports are explicitly bound too.
    from slm_lab.evaluation_provenance import SOURCE_NAMES
    for name in SOURCE_NAMES: files[str(ROOT/'src/slm_lab'/(name+'.py'))] = sha(ROOT/'src/slm_lab'/(name+'.py'))
    pulse = read(capacity.OLD/'plan.json')['pulse_sources']
    for path in [*map(Path, pulse.values()), ROOT/'data/protected-joint-v1-pulses/knowledge.audit.json',
            ROOT/'data/math-reasoning-v2/test.jsonl', ROOT/'data/math-reasoning-v2/manifest.json']:
        files[str(path)] = sha(path)
    memory = dict(line.split()[:2] for line in Path('/sys/fs/cgroup/memory.stat').read_text().splitlines())
    headroom = int(Path('/sys/fs/cgroup/memory.max').read_text())-sum(int(memory[k]) for k in ['anon','shmem','kernel'])
    if headroom < 80*GIB or shutil.disk_usage(OUT.parent).free < 64*GIB:
        raise ValueError('Actual full-model host headroom and full scratch/reserve budgets must remain funded')
    GPUs, _ = capacity.gpu_snapshot()
    if GPUs[0]['used_MiB'] > 16*1024: raise ValueError('Original target GPU must retain the measured capacity margin')
    old = read(capacity.OLD/'plan.json')['owner']
    plan = dict(owner=identity(os.getpid()), old_owner=dict(pid=old['pid'],birth=old['ticks']),
        old_plan_sha256=sha(capacity.OLD/'plan.json'), target_uuid=GPUs[0]['uuid'], expires_at_unix=time.time()+8*3600,
        source_files_sha256=files, native=native, pulse_sources=pulse,
        data_manifest_sha256=sha(ROOT/'data/public-reasoned-code-SFT-v1/manifest.json'),
        data_admission_sha256=sha(ROOT/'data/public-reasoned-code-SFT-v1/admission.json'),
        original_global_flock_workers_settings_and_sources_preserved=True, separate_bounded_supplemental_resource_contract=True,
        own_allocator_limit_bytes=12*GIB, own_CPU_affinity=[20], own_threads=1,
        profiles=list(PROFILES), full_source_views_per_arm=3675, response_EOS_targets_per_arm=3038462,
        complete_context_positions_per_arm=4797843, updates_per_arm=230, genuine_last_batch=11,
        epochs=1, width=16, source_seed=990231, lr=3e-5, betas=[.9,.95], eps=1e-8, weight_decay=.01, gradient_clip=1.,
        protocol='serial_microbatch1_BF16_forward_FP32_CE_weights_Adam',
        evaluation_endpoints=list(ENDPOINTS), code_actions_per_program=512, full_code_programs_per_endpoint=48,
        final_independent_full_benchmarks_and_standalone_export_still_required=True,
        new_human_or_PPO_updates=0, candidate_promoted=False, goal_achieved=False)
    with COMMON_LOCK.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        before = guard(plan, [])
        OUT.mkdir()
        write('plan.json', plan)
        write('children.json', [])
        write('original-before.json', before)
        watcher = subprocess.Popen([sys.executable,'-u',str(SCRIPT),'--watchdog'],cwd=ROOT,
            stdout=(OUT/'watchdog.log').open('x'),stderr=subprocess.STDOUT,env={**os.environ,'CUDA_VISIBLE_DEVICES':''})
        children = []
        try:
            jobs = [('train',p) for p in PROFILES]+[('assess',p) for p in ENDPOINTS]
            for task, endpoint in jobs:
                checked(plan)
                with (OUT/(task+'-'+endpoint+'.log')).open('x') as log:
                    process = subprocess.Popen([sys.executable,'-u',str(SCRIPT),'--gpu-child',endpoint,'--task',task],
                        cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'CUDA_VISIBLE_DEVICES':'0',
                            'CUBLAS_WORKSPACE_CONFIG':':4096:8','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'})
                    actor = identity(process.pid)
                    children.append(actor)
                    write('children.json', children)
                    write('progress.json', dict(phase=task, endpoint=endpoint, child=actor, at=datetime.now(timezone.utc).isoformat()))
                    while process.poll() is None:
                        checked(plan)
                        time.sleep(1)
                    if process.returncode != 0:
                        raise ValueError('The registered full-source child failed; retain evidence without automatic retry')
                    until = time.monotonic()+20
                    while actor['pid'] in {a['pid'] for a in capacity.gpu_snapshot()[1]}:
                        checked(plan)
                        if time.monotonic() > until: raise ValueError('Completed CUDA context has not released')
                        time.sleep(.2)
            initials = {read(OUT/p/'initial.json')['selected_policy_sha256'] for p in PROFILES}
            initials.add(read(OUT/'baseline/assessment-complete.json')['initial_selected_policy_sha256'])
            if len(initials) != 1 or any(sha(p) != h for p,h in files.items()):
                raise ValueError('Every original source byte and identical matched initialization must remain exact')
            after = checked(plan)
            write('original-after.json', after)
            write('complete.json', dict(complete=True, full_source_training_updates_per_arm=230,
                full_matched_assessments_complete=True, profiles=list(PROFILES),
                original_step_before=before['old_step'], original_step_after=after['old_step'],
                old_processes_signalled=0, all_original_sources_workers_and_settings_preserved=True,
                supplemental_GPU_contexts_released=True, standalone_export_and_full_final_benchmarks_pending=True,
                candidate_promoted=False, goal_achieved=False))
        except BaseException as error:
            write('failure.json', dict(error=repr(error), traceback=traceback.format_exc(),
                partial_outputs_retained=True, no_automatic_retry=True, goal_achieved=False))
            raise
        finally:
            for actor in children: stop(actor, plan)
            write('restored.json', dict(own_children_terminal=all(not live(a) for a in children), old_processes_signalled=0))
            watcher.wait(timeout=10)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--watchdog', action='store_true')
    parser.add_argument('--gpu-child', choices=ENDPOINTS)
    parser.add_argument('--task', choices=['train','assess'])
    args = parser.parse_args()
    if args.watchdog: watchdog()
    elif args.gpu_child: child(args.gpu_child, args.task)
    elif args.run: parent()
    else: parser.error('Explicit --run or registered child/watchdog mode required')
