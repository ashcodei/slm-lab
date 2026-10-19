"""Queue matched complete-source anchored code training after human-v4.

Each arm uses all four existing L40s, a frozen own starting-policy reference,
and every original selected coding source. No automatic promotion or retry.
"""
import argparse
from datetime import datetime, timedelta, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

import train_whole_human_SFT_v1 as legacy
from slm_lab.codenet_checkpoint_v1 import FileBindings
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT/'reports/broad-anchored-code-v1'
OUTPUT = ROOT/'checkpoints/broad-anchored-code-v1'
QUEUE = ROOT/'reports/broad-anchored-code-v1-queue.json'
COHORT = ROOT/'reports/broad-anchored-code-v1-held-APPS-cohort.json'
PREVIOUS = ROOT/'reports/phase-human-preference-v4'
PREVIOUS_QUEUE = ROOT/'reports/phase-human-preference-v4-queue.json'
MAX_STATE = 384*1024**2
RESERVE = 26*1024**3
ARMS = ['anchor_CE', 'anchor_execution']
owner = legacy.owner


def configure():
    owner.REPORTS = REPORTS; owner.resources.REPORTS = REPORTS
    legacy.REPORTS = REPORTS; legacy.OUTPUT = OUTPUT


def previous_ready(identity):
    if owner.resources.same(identity): return False
    children = PREVIOUS/'children.json'
    if children.exists() and any(owner.resources.same(p) for p in json.loads(children.read_text())): return False
    if ((PREVIOUS/'failure.json').exists() or any(not (PREVIOUS/name).exists() for name in ['plan.json','complete.json','restored.json'])):
        raise ValueError('Complete human experiment, native assessments and restoration required')
    plan = json.loads((PREVIOUS/'plan.json').read_text()); done = json.loads((PREVIOUS/'complete.json').read_text())
    if (done.get('all_original_human_train_pairs_once_each_arm') is not True
            or done.get('all_held_human_pairs_and_native_development_assessments_complete') is not True
            or done.get('actual_source_epoch_updates_per_arm') != plan['training_options']['steps']):
        raise ValueError('Every complete human source epoch and unchanged native assessment required')
    for arm in ['chosen_CE','human_DPO']:
        trained = json.loads((PREVIOUS/(arm+'-training-complete.json')).read_text())
        receipt = json.loads((PREVIOUS/('train-'+arm+'-resume-distributed-state.json')).read_text())
        if (trained.get('actual_updates') != plan['training_options']['steps'] or trained.get('all_original_human_pairs_once') is not True
                or receipt.get('completed') != plan['training_options']['steps']
                or {p['rank'] for p in receipt['all_four_private_states']} != set(range(4))):
            raise ValueError('Actual complete four-replica human states required')
    return True


def distributed(mode, arm):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '0,1,2,3' or int(os.environ['WORLD_SIZE']) != 4:
        raise ValueError('Only all four owned original GPUs may train')
    rank = int(os.environ['RANK']); plan = owner.registered(mode+'-'+arm+'-rank-'+str(rank))
    import copy
    import random
    import numpy as np
    import torch
    from torch import distributed as dist
    from slm_lab.broad_anchored_execution_stage_v1 import BroadAnchoredExecutionStage, semantic_digest
    from slm_lab.broad_anchored_policy_v1 import fresh, make_reference, sources, export
    from slm_lab.broad_code_capacity_v1 import stress, adam_digest, equal
    from slm_lab.codenet_checkpoint_v1 import atomic_checkpoint, capture_rng
    from slm_lab.expanded_execution_corpus_v1 import ExpandedExecutionCorpus
    from slm_lab.frozen_own_actor_reference_v1 import signature
    from slm_lab.whole_human_sft_evidence_v1 import tensor_digest
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    device = torch.device('cuda', rank); torch.cuda.set_device(device)
    torch.cuda.set_per_process_memory_fraction(.85, device)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    if torch.cuda.get_device_name(device) != 'NVIDIA L40': raise ValueError('Original local L40 required')
    dist.init_process_group('gloo', timeout=timedelta(minutes=45))
    started = time.monotonic()
    try:
        guard = FileBindings(plan['source_files_sha256']); guard.check(full=True)
        digest = file_hash(REPORTS/'plan.json'); model, _ = fresh(plan, device)
        reference = make_reference(model, plan); corpus = ExpandedExecutionCorpus(ROOT)
        if len(corpus) != 3721 or reference.resident_inventory(model) != plan['resident_training_inventory']:
            raise ValueError('Exact full source population and all resident/exported parameters required')
        options = plan['training_options']; parameters = [p for _,p in model.named_selected(('actor',))]
        optimizer = torch.optim.AdamW(parameters, lr=options['learning_rate'], betas=tuple(options['betas']),
            weight_decay=options['weight_decay'], foreach=False)
        stage = BroadAnchoredExecutionStage(model, optimizer, corpus, reference, root=ROOT, method=arm,
            seed=options['seed'], sources=sources(plan,digest), world=4, rank=rank,
            KL_weight=options['KL_weight'], anchor_weight=options['anchor_weight'],
            cache_absolute_bound=plan['cache_calibration']['bound'], cache_calibration_sha256=plan['cache_calibration']['sha256'])
        if len(stage.order) != 3721: raise ValueError('Every original coding source must be consumed')
        random.seed(options['seed']+rank); np.random.seed(options['seed']+rank); torch.manual_seed(options['seed']+rank)
        frozen = tensor_digest(model.base.named_parameters()); initial = tensor_digest(model.parts.named_parameters())
        critic = model.parts['critic'].weight.detach().clone(); journal = None
        journal_hash = hashlib.sha256(); journal_bytes = 0
        def save_private(label):
            if journal is not None: journal.flush(); os.fsync(journal.fileno())
            dist.barrier()
            generation = math.ceil(stage.completed/256) if label == 'resume' else None
            stem = 'resume-slot-'+str(generation%2) if generation is not None else label
            path = OUTPUT/f'{mode}-{arm}-{stem}-rank-{rank}.pt'
            atomic_checkpoint(path, stage.state(), MAX_STATE, RESERVE)
            proof = dict(rank=rank, path=str(path), sha256=file_hash(path), actual_completed=stage.completed,
                external_journal_prefix_bytes=journal_bytes, external_journal_prefix_sha256=journal_hash.hexdigest())
            if mode == 'capacity':
                names = ['first'] if label == 'step1' else ['first','replayed-second']
                proof['external_capacity_records_sha256'] = {str(REPORTS/f'capacity-{arm}-{name}-native-step.json'):
                    file_hash(REPORTS/f'capacity-{arm}-{name}-native-step.json') for name in names}
            proofs = [None]*4; dist.all_gather_object(proofs, proof)
            if len({(p['external_journal_prefix_bytes'],p['external_journal_prefix_sha256']) for p in proofs}) != 1:
                raise ValueError('Every private state must bind the same durable global trajectory journal prefix')
            if rank == 0:
                atomic_json(REPORTS/f'{mode}-{arm}-{label}-distributed-state.json', dict(all_four_private_states=proofs,
                    completed=stage.completed, committed_generation=generation,
                    complete_distributed_state_requires_every_private_file_and_external_journal_prefix=True,
                    complete_distributed_state_requires_capacity_record_files_if_present=True,
                    previous_generation_files_retained=label=='resume', goal_achieved=False))
            return path
        def step():
            owner.owned(plan)
            if stage.completed%32 == 0: guard.check()
            return stage.update()
        extreme_proofs = []
        if mode == 'capacity':
            first = step()
            if rank == 0: atomic_json(REPORTS/f'capacity-{arm}-first-native-step.json',first)
            saved = save_private('step1')
            def draws(): return [random.random(),float(np.random.random()),float(torch.rand(())),float(torch.rand((),device=device))]
            expected_draws = draws(); expected = step()
            expected_after_RNG = capture_rng(device)
            if rank == 0: atomic_json(REPORTS/f'capacity-{arm}-expected-second-native-step.json',expected)
            weights, moments = tensor_digest(model.parts.named_parameters()), adam_digest(optimizer)
            payload = torch.load(saved,map_location='cpu',weights_only=True,mmap=True)
            bad = copy.deepcopy(payload); bad['core']['rng']['cuda'] = torch.zeros(4,dtype=torch.uint8)
            before = signature(model.parameters()); RNG = capture_rng(device)
            try: stage.resume(bad)
            except ValueError: pass
            else: raise ValueError('Invalid private CUDA RNG must reject before mutation')
            if before != signature(model.parameters()) or moments != adam_digest(optimizer) or not equal(RNG,capture_rng(device)):
                raise ValueError('Invalid state changed policy, Adam or private RNG')
            stage.resume(payload)
            if draws() != expected_draws: raise ValueError('Every actual private Python/NumPy/Torch CPU/CUDA RNG draw must replay')
            actual = step()
            if (weights != tensor_digest(model.parts.named_parameters()) or moments != adam_digest(optimizer)
                    or semantic_digest(expected['trajectories']) != semantic_digest(actual['trajectories'])
                    or expected['original_reference_CE_sum'] != actual['original_reference_CE_sum']
                    or not equal(expected_after_RNG,capture_rng(device))):
                raise ValueError('Actual next group, complete anchor, weights and every Adam value must replay')
            if rank == 0: atomic_json(REPORTS/f'capacity-{arm}-replayed-second-native-step.json',actual)
            save_private('step2')
            for position in plan['capacity_original_source_positions']:
                error = None; proof = None
                try: proof = stress(model,optimizer,corpus,reference,position,cache_absolute_bound=plan['cache_calibration']['bound'])
                except Exception as exc: error = repr(exc)
                errors = [None]*4; dist.all_gather_object(errors,error)
                if any(e is not None for e in errors): raise ValueError('Coordinated full native-shape capacity failure: '+repr(errors))
                extreme_proofs.append(proof)
            atomic_json(REPORTS/f'capacity-{arm}-{rank}.json', dict(physical_GPU=rank,actual_counts=model.counts(),
                resident_training_inventory=reference.resident_inventory(model),actual_Adam_states=len(optimizer.state),
                actual_logical_updates=2,physical_optimizer_calls=3,exact_next_global_update_weights_Adam_actions_private_RNG=True,
                invalid_private_CUDA_RNG_rejected_before_mutation=True,complete_native_extremes=extreme_proofs,
                maximum_forced_action_horizon=max(p['actual_forced_action_tokens'] for p in extreme_proofs),
                peak_allocated_bytes=torch.cuda.max_memory_allocated(device),peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
                seconds=time.monotonic()-started,production_updates=0,goal_achieved=False))
        else:
            capacity = json.loads((REPORTS/'capacity.json').read_text())
            if capacity.get('passed') is not True: raise ValueError('Fresh full-profile all-worker admission required before production')
            if rank == 0 and arm == ARMS[0]:
                atomic_checkpoint(OUTPUT/'initial-actor.pt',export(model,plan,digest,'initial',0),96*1024**2,RESERVE)
            dist.barrier()
            if rank == 0: journal = (REPORTS/(arm+'-metrics.jsonl')).open('xb')
            try:
                while stage.completed < len(stage.order):
                    tick = time.monotonic(); record = step()
                    raw = (json.dumps(record,separators=(',',':'),allow_nan=False)+'\n').encode()
                    journal_hash.update(raw); journal_bytes += len(raw)
                    if journal is not None:
                        journal.write(raw); journal.flush()
                        if stage.completed == 1 or stage.completed%25 == 0 or stage.completed == len(stage.order):
                            atomic_json(REPORTS/(arm+'-progress.json'),dict(actual_updates=stage.completed,planned_updates=len(stage.order),
                                last_source_id=record['source_id'],global_rewards=record['global_rewards'],
                                original_reference_token_mean_CE=record['original_reference_token_mean_CE'],
                                last_update_seconds=time.monotonic()-tick,elapsed_seconds=time.monotonic()-started,goal_achieved=False))
                    if stage.completed%256 == 0 or stage.completed == len(stage.order): save_private('resume')
                if rank == 0:
                    atomic_checkpoint(OUTPUT/(arm+'-actor.pt'),export(model,plan,digest,arm,stage.completed),96*1024**2,RESERVE)
                    atomic_json(REPORTS/(arm+'-training-complete.json'),dict(actual_updates=stage.completed,
                        all_3721_original_coding_sources_once=True,complete_reference_supervision_every_update=True,
                        own_starting_human_policy_reference_frozen=True,human_preference_labels_added=0,
                        journal_sha256=journal_hash.hexdigest(),journal_bytes=journal_bytes,native_quality_pending=True,goal_achieved=False))
            finally:
                if journal is not None: journal.close()
        if (frozen != tensor_digest(model.base.named_parameters()) or not torch.equal(critic,model.parts['critic'].weight)
                or initial == tensor_digest(model.parts.named_parameters())):
            raise ValueError('Selected code actor must learn with unchanged own foundation and unused critic')
        reference.assert_bound(model)
        if tensor_digest(reference.parts.named_parameters()) != plan['own_reference_binding']['frozen_actor_parts_sha256']:
            raise ValueError('Every frozen own starting-policy adapter value must remain exact')
        guard.check(full=True); dist.barrier()
    finally:
        dist.destroy_process_group()


def assess(arm,gpu,suite):
    plan = legacy.registered('assess-'+arm+'-'+suite+'-'+str(gpu),gpu)
    import torch
    from slm_lab.broad_anchored_policy_v1 import load
    from slm_lab.inference_rotary_cache_v1 import inference_cache
    torch.set_num_threads(1); torch.cuda.set_device(0); torch.cuda.set_per_process_memory_fraction(.85,0)
    torch.use_deterministic_algorithms(True); torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    checkpoint=OUTPUT/(arm+'-actor.pt'); model,metadata=load(checkpoint,REPORTS/'plan.json',arm,
        expected_checkpoint_sha256=file_hash(checkpoint),expected_plan_sha256=file_hash(REPORTS/'plan.json'),device='cuda:0')
    guard=FileBindings(plan['source_files_sha256']); folder=REPORTS/arm; folder.mkdir(exist_ok=True)
    if suite=='preferences':
        from slm_lab.held_human_shards_v1 import evaluate_shard
        from slm_lab.whole_human_data_v2 import WholeHumanData
        data=WholeHumanData(ROOT/'data/whole-human-alignment-v2','preferences','valid','evaluation')
        provenance=dict(checkpoint_sha256=file_hash(checkpoint),run_plan_sha256=file_hash(REPORTS/'plan.json'),
            human_manifest_sha256=plan['human_manifest_sha256'],frozen_dense_sha256=plan['frozen_dense_sha256'],
            inference_device_type='cuda',inference_precision='float32')
        evaluate_shard(model,data,folder/f'held-preferences-shard-{gpu}.json',rank=gpu,world=4,
            provenance=provenance,beta=.1,checked=lambda:owner.owned(plan))
    elif suite=='apps':
        from tokenizers import Tokenizer
        from slm_lab.held_apps_native_shards_v1 import evaluate_shard
        if file_hash(OUTPUT/'tokenizer.json') != plan['tokenizer_sha256']: raise ValueError('Exact own tokenizer package required')
        with inference_cache():
            evaluate_shard(model.policy,Tokenizer.from_file(str(OUTPUT/'tokenizer.json')),checkpoint,
                ROOT/'data/public-apps-publisher-test-evaluation-v1',COHORT,folder/f'held-APPS-shard-{gpu}.json',
                rank=gpu,world=4,device='cuda:0',checked=lambda:owner.owned(plan))
    else:
        def evaluate(module,name,data=None,**kwargs):
            saved=module.load_checkpoint; module.load_checkpoint=lambda *a,**kw:(model.policy,metadata)
            try:
                owner.owned(plan)
                if data is None: module.evaluate(checkpoint,folder/name,device='cuda:0',**kwargs)
                else: module.evaluate(checkpoint,data,folder/name,device='cuda:0',**kwargs)
            finally: module.load_checkpoint=saved
        original_threads=torch.set_num_threads; torch.set_num_threads=lambda n:None
        try:
            with inference_cache():
                if suite=='code':
                    from slm_lab.native_code_shards_v1 import evaluate_shard
                    for frame in ['completion','instruction']: evaluate_shard(model.policy,metadata,checkpoint,folder,rank=gpu,world=4,frame=frame)
                elif gpu==0:
                    from slm_lab import knowledge_evaluation
                    evaluate(knowledge_evaluation,'knowledge.json',Path(plan['pulse_sources']['knowledge']),max_examples=56,max_new_tokens=32)
                elif gpu==1:
                    from slm_lab import math_evaluation
                    evaluate(math_evaluation,'math.json',ROOT/'data/math-reasoning-v2/test.jsonl',max_examples=16,max_new_tokens=256)
                elif gpu==2:
                    from slm_lab import ocr_evaluation,object_evaluation
                    evaluate(ocr_evaluation,'ocr.json',Path(plan['pulse_sources']['ocr']),max_examples=32,max_new_tokens=128,image_size=448)
                    evaluate(object_evaluation,'objects.json',Path(plan['pulse_sources']['objects']),max_examples=32,max_new_tokens=32)
                else:
                    from slm_lab import vision_evaluation
                    evaluate(vision_evaluation,'vision.json',Path(plan['pulse_sources']['vision']),max_examples=32,image_size=448)
        finally: torch.set_num_threads=original_threads
    guard.check(full=True)
    atomic_json(folder/f'assessment-{suite}-{gpu}-complete.json',dict(physical_GPU=gpu,actual_counts=model.counts(),
        checkpoint_sha256=file_hash(checkpoint),development_comparison_not_final_independent_benchmark=True,goal_achieved=False))


def parent():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or any(p.exists() for p in [REPORTS,OUTPUT,QUEUE,COHORT]):
        raise ValueError('Fresh CPU-hidden queue required; preserve prior attempts')
    tests={'frozen-own-actor-reference-v1-tests':2,'broad-anchored-policy-v1-tests':2,'held-apps-native-shards-v1-tests':3,
           'broad-anchored-code-v1-driver-tests':3,'own-tokenizer-package-v1-tests':2}
    for name,count in tests.items():
        if str(count)+' passed' not in (ROOT/'reports'/(name+'.log')).read_text(): raise ValueError('Actual source, recovery, native and owner checks required')
    proof_path=ROOT/'reports/broad-anchored-execution-four-CPU-v1/complete.json'; proof=json.loads(proof_path.read_text())
    shape_path=ROOT/'reports/broad-code-capacity-CPU-v1/complete.json'; shape=json.loads(shape_path.read_text())
    if (not proof['all_four_next_update_weights_Adam_actions_objectives_and_private_CPU_RNGs_exact']
            or not proof['own_reference_preserves_actual_trained_starting_actor'] or shape['actual_longest_horizon']!=3059):
        raise ValueError('Actual four-CPU trained-reference recovery and complete native longest-horizon backward required')
    previous=json.loads(PREVIOUS_QUEUE.read_text()); files=dict(previous['source_files_sha256'])
    for p in [proof,shape]:
        FileBindings(p['source_files_sha256']).check(full=True); files.update(p['source_files_sha256'])
    admission=ROOT/'data/apps-execution-rewards-v1/admission.json'; admitted=json.loads(admission.read_text())
    FileBindings(admitted['source_files_sha256']).check(full=True); files.update(admitted['source_files_sha256'])
    additions=[Path(__file__),proof_path,shape_path,admission,
        ROOT/'src/slm_lab/anchored_execution_step_v2.py',ROOT/'src/slm_lab/broad_anchored_execution_stage_v1.py',
        ROOT/'src/slm_lab/broad_anchored_policy_v1.py',ROOT/'src/slm_lab/broad_code_capacity_v1.py',
        ROOT/'src/slm_lab/frozen_own_actor_reference_v1.py',ROOT/'src/slm_lab/held_apps_native_shards_v1.py',
        ROOT/'src/slm_lab/expanded_execution_corpus_v1.py',ROOT/'reports/expanded-execution-corpus-v1/complete.json',
        ROOT/'tests/test_frozen_own_actor_reference_v1.py',ROOT/'tests/test_broad_anchored_policy_v1.py',
        ROOT/'tests/test_held_apps_native_shards_v1.py',ROOT/'tests/test_broad_anchored_code_v1_driver.py']
    additions += [ROOT/'reports'/(name+'.log') for name in tests]
    from slm_lab.held_apps_native_shards_v1 import select
    data=ROOT/'data/public-apps-publisher-test-evaluation-v1'
    records=list(map(json.loads,(data/'evaluation-records.jsonl').read_text().splitlines()))
    selected=select(records,64)
    atomic_json(COHORT,dict(selected_original_test_ids=[r['id'] for r in selected],generation_budget=1024,
        manifest_sha256=file_hash(data/'manifest.json'),records_sha256=file_hash(data/'evaluation-records.jsonl'),
        preselected_without_candidate_generations_or_scores=True,optimization_forbidden=True,official_APPS_score=False,goal_achieved=False))
    additions += [COHORT,data/'manifest.json',data/'evaluation-records.jsonl',data/'independent-evaluation-readback.json']
    for p in additions: files[str(p)]=file_hash(p)
    guard=FileBindings(files); me=owner.resources.identity(os.getpid()); deadline=time.monotonic()+14*24*3600
    atomic_json(QUEUE,dict(status='waiting for complete exact human-v4 source epochs, native assessments and restoration',
        owner=me,predecessor=previous['owner'],source_files_sha256=files,original_code_sources=3721,
        planned_updates_each_arm=3721,arms=ARMS,own_starting_policy_arm='human_DPO',
        held_APPS_original_problems=64,held_public_human_pairs=4642,production_updates=0,
        fresh_full_size_four_L40_capacity_still_required=True,automatic_retry=False,candidate_promoted=False,goal_achieved=False))
    while not previous_ready(previous['owner']):
        guard.check()
        if time.monotonic()>deadline: raise ValueError('Queue lease expired without preempting or restarting preceding work')
        time.sleep(5)
    import torch
    from slm_lab.phase_human_policy_v1 import load as load_human
    from slm_lab.broad_anchored_policy_v1 import reference_sources
    from slm_lab.code_probability_guard_v8 import load as load_calibration
    from slm_lab.expanded_execution_corpus_v1 import ExpandedExecutionCorpus
    from slm_lab.frozen_own_actor_reference_v1 import FrozenOwnActorReference
    from slm_lab.whole_human_sft_evidence_v1 import tensor_digest
    torch.set_num_threads(1)
    starting_plan=PREVIOUS/'plan.json'; starting_checkpoint=ROOT/'checkpoints/phase-human-preference-v4/human_DPO-actor.pt'
    human_plan=json.loads(starting_plan.read_text())
    inputs=dict(checkpoint=str(starting_checkpoint),plan=str(starting_plan),checkpoint_sha256=file_hash(starting_checkpoint),plan_sha256=file_hash(starting_plan))
    model,_=load_human(starting_checkpoint,starting_plan,'human_DPO',expected_checkpoint_sha256=inputs['checkpoint_sha256'],expected_plan_sha256=inputs['plan_sha256'])
    corpus=ExpandedExecutionCorpus(ROOT); reference=FrozenOwnActorReference(model,sources=reference_sources(dict(starting_policy_inputs=inputs)))
    if (model.counts()['actual_unique'] != 999706442 or len(corpus) != 3721
            or human_plan['human_train_pairs'] != 85786 or human_plan['training_options']['steps'] != 5362):
        raise ValueError('Actual complete full-size own human policy and every original coding source required')
    extremes=[]
    for key in ['complete_context_positions','prefix_positions','response_EOS_targets','full_training_action_budget']:
        position=max(range(len(corpus)),key=lambda p:corpus.rows[p][key])
        if position not in extremes: extremes.append(position)
    extra=max(range(len(corpus)),key=lambda p:corpus.rows[p]['prefix_positions']+corpus.rows[p]['full_training_action_budget'])
    if extra not in extremes: extremes.append(extra)
    for p in [starting_plan,starting_checkpoint,PREVIOUS/'complete.json',PREVIOUS/'restored.json']:
        files[str(p)]=file_hash(p)
    with (ROOT/'reports/code-residual-production-v4.gpu.lock').open('a+') as lease:
        fcntl.flock(lease.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB); owner.idle_original()
        used=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).splitlines()
        if len(used)!=4 or any(int(x)>128 for x in used): raise ValueError('All original GPUs must be idle before acquisition')
        if shutil.disk_usage(ROOT).free<RESERVE+32*MAX_STATE+1024**3: raise ValueError('Complete state generations and reserve must fit')
        guard=FileBindings(files); guard.check(full=True); REPORTS.mkdir(); OUTPUT.mkdir()
        from slm_lab.own_tokenizer_package_v1 import ensure
        tokenizer_package=ensure(ROOT/'data/tokenizer-corpus/tokenizer.json',OUTPUT,expected_sha256=human_plan['tokenizer_sha256'],expected_vocab_size=model.config.vocab_size)
        plan=dict(starting_policy_inputs=inputs,actual_counts=model.counts(),frozen_dense_sha256=tensor_digest(model.base.named_parameters()),
            starting_actor_parts_sha256=tensor_digest(model.parts.named_parameters()),own_reference_binding=reference.binding(),
            resident_training_inventory=reference.resident_inventory(model),tokenizer_sha256=human_plan['tokenizer_sha256'],tokenizer_package=tokenizer_package,
            APPS_execution_admission_sha256=file_hash(admission),documented_probe_admission_sha256=file_hash(ROOT/'reports/bounded-code-probes-v1/admission.json'),
            human_manifest_sha256=human_plan['human_manifest_sha256'],original_coding_sources=len(corpus),planned_steps=len(corpus),
            training_options=dict(seed=963117,learning_rate=1e-5,betas=[.9,.95],weight_decay=.01,KL_weight=.01,anchor_weight=1.),
            capacity_original_source_positions=extremes,cache_calibration=load_calibration(ROOT,model),
            pulse_sources=human_plan['pulse_sources'],arms=ARMS,each_arm_uses_all_four_GPUs_in_sequence=True,
            exported_policy_strictly_under_1B=True,reference_is_own_complete_starting_human_policy=True,
            all_original_selected_coding_prompts_once_per_arm=True,parallel_reference_presentations_per_source=4,
            native_assessment_precision_by_suite=dict(code='bf16_autocast',broad='bf16_autocast',apps='bf16_autocast',preferences='float32'),
            owner=me,waiter=owner.WAITER,source_files_sha256=files,expires_at_unix=time.time()+7*24*3600,
            human_preference_labels_added=0,automatic_retry=False,candidate_promoted=False,goal_achieved=False)
        del model,reference,corpus
        atomic_json(REPORTS/'plan.json',plan); atomic_json(REPORTS/'children.json',[])
        driver=str(Path(__file__).resolve())
        def interrupted(signum,frame): raise KeyboardInterrupt('Owned broad anchored code experiment interrupted')
        signal.signal(signal.SIGTERM,interrupted); signal.signal(signal.SIGINT,interrupted)
        watcher=subprocess.Popen(['.venv/bin/python','-u',driver,'--watchdog'],cwd=ROOT,
            stdout=(REPORTS/'watchdog.log').open('x'),stderr=subprocess.STDOUT,start_new_session=True,env={**os.environ,'CUDA_VISIBLE_DEVICES':''})
        def run(mode,arm):
            child=owner.launch(plan,['-u','-m','torch.distributed.run','--standalone','--nproc_per_node=4',driver,'--distributed',mode,'--arm',arm],mode+'-'+arm+'-launcher','0,1,2,3')
            owner.wait_children(plan,[child])
        try:
            owner.idle_original(); os.kill(owner.WAITER['pid'],signal.SIGSTOP); stop=time.monotonic()+5
            while (owner.resources.same(owner.WAITER) or {}).get('state') not in ['T','t']:
                if time.monotonic()>stop: raise ValueError('Exact original waiter hold failed')
                time.sleep(.05)
            for arm in ARMS:
                run('capacity',arm)
                proofs=[json.loads((REPORTS/f'capacity-{arm}-{r}.json').read_text()) for r in range(4)]
                if any(p['physical_GPU']!=r or p['actual_counts']!=plan['actual_counts'] or p['actual_Adam_states']!=561
                       or not p['exact_next_global_update_weights_Adam_actions_private_RNG'] or p['maximum_forced_action_horizon']!=3059 for r,p in enumerate(proofs)):
                    raise ValueError('Every full-size fresh four-L40 extreme/recovery control must pass')
            atomic_json(REPORTS/'capacity.json',dict(passed=True,arms=ARMS,production_updates=0,goal_achieved=False))
            for arm in ARMS: run('train',arm)
            for arm in ['initial',*ARMS]:
                for suite in ['code','broad','preferences','apps']:
                    children=[owner.launch(plan,['-u',driver,'--assess',arm,'--suite',suite,'--gpu',str(gpu)],'assess-'+arm+'-'+suite+'-'+str(gpu),str(gpu)) for gpu in range(4)]
                    owner.wait_children(plan,children)
                from slm_lab.native_code_shards_v1 import reconcile as code_reconcile
                from slm_lab.held_human_shards_v1 import reconcile as human_reconcile
                from slm_lab.held_apps_native_shards_v1 import reconcile as apps_reconcile
                from slm_lab.whole_human_data_v2 import WholeHumanData
                for frame in ['completion','instruction']: code_reconcile([REPORTS/arm/f'code-{frame}-shard-{r}.json' for r in range(4)],REPORTS/arm/f'code-{frame}.json')
                human_reconcile(WholeHumanData(ROOT/'data/whole-human-alignment-v2','preferences','valid','evaluation'),
                    [REPORTS/arm/f'held-preferences-shard-{r}.json' for r in range(4)],REPORTS/arm/'held-preferences.json')
                apps_reconcile([REPORTS/arm/f'held-APPS-shard-{r}.json' for r in range(4)],COHORT,REPORTS/arm/'held-APPS.json')
            guard.check(full=True)
            atomic_json(REPORTS/'complete.json',dict(all_3721_original_coding_sources_once_each_arm=True,
                matched_native_code_broad_human_preference_and_held_APPS_assessments_complete=True,
                fresh_own_output_human_judgments_and_final_independent_benchmarks_still_required=True,
                independent_state_and_generated_output_readbacks_pending=True,candidate_promoted=False,goal_achieved=False))
        except BaseException as error:
            atomic_json(REPORTS/'failure.json',dict(error=repr(error),automatic_retry=False,goal_achieved=False)); raise
        finally:
            owner.restore_owned(plan,'Broad anchored code experiment ended; original work restored'); watcher.wait(timeout=20)


if __name__=='__main__':
    configure(); parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--watchdog',action='store_true'); parser.add_argument('--distributed',choices=['capacity','train'])
    parser.add_argument('--arm',choices=ARMS); parser.add_argument('--assess',choices=['initial',*ARMS])
    parser.add_argument('--gpu',type=int,choices=range(4)); parser.add_argument('--suite',choices=['code','broad','preferences','apps'])
    args=parser.parse_args()
    if args.watchdog: owner.watchdog()
    elif args.distributed: distributed(args.distributed,args.arm)
    elif args.assess: assess(args.assess,args.gpu,args.suite)
    else: parent()
