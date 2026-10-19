"""Queued full-size matched CE/sequence/first-divergence preference research.

Wait for the exact two-thread interim assessment to finish and for complete
public pretraining. Reuse only that freed CPU allocation; acquire no GPU lease.
All77pairs/three epochs and all baseline/final15source/48code/broad checks remain.
The canonical full human/RL/export/benchmark pipeline is not modified or gated.
"""
from datetime import datetime, timezone
import fcntl
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import time
import traceback

from slm_lab.codenet_checkpoint_v1 import FileBindings
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.exact_waiting_process_retirement_v1 import terminal
from slm_lab.public_pretraining_assessment_recovery_v1 import completed_training_after_observer_failure
from slm_lab.function_repair_GPU_contract_v2 import public_ready
from slm_lab.local_source_closure_v2 import python_sources

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/public-edge-preference-comparison-CPU-v2'
OUTPUT = ROOT / 'checkpoints/public-edge-preference-comparison-CPU-v2'
QUEUE = ROOT / 'reports/public-edge-preference-comparison-CPU-v2-queue.json'
WAIT = ROOT / 'reports/public-edge-preference-comparison-CPU-v2-queue-progress.json'
ASSESS = ROOT / 'reports/public-pretraining-interim-CPU-v2'
MATH = ROOT / 'reports/public-pretraining-v1/plan.json'
RUNTIME = ROOT / 'reports/public-pretraining-recovery-v1/plan.json'
BASE = ROOT / 'checkpoints/public-pretraining-v1/trained/model.pt'
DATA = ROOT / 'data/public-edge-case-hard-negatives-admitted-v1'
ADMITTED = ROOT / 'reports/public-edge-case-hard-negatives-v1-readback-admission/complete.json'
TESTS = ROOT / 'reports/public-edge-preference-stage-v1-tests.log'
MAX_STATE = 512 * 1024**2
# Retain both the canonical writer's14GiB and its original26GiB disk reserve.
RESERVE = 40 * 1024**3
PROFILES = ['chosen_CE', 'sequence_DPO', 'branch_DPO']


def read(path):
    return json.loads(Path(path).read_text())


def birth():
    return dict(pid=os.getpid(), ticks=Path('/proc/self/stat').read_text().split(') ', 1)[1].split()[19])


def main():
    if (os.environ.get('CUDA_VISIBLE_DEVICES') != '' or os.nice(0) < 19
            or any(p.exists() for p in [OUT, OUTPUT, QUEUE])):
        raise ValueError('Fresh CUDA-hidden low-priority queued CPU comparison required')
    assessment = read(ASSESS / 'evaluation-plan.json')
    runtime = read(RUNTIME)
    if terminal(assessment['owner']) or not completed_training_after_observer_failure(ROOT):
        raise ValueError('Capture the actual currently live predecessors before admitting this CPU queue')
    if '8 passed' not in TESTS.read_text():
        raise ValueError('Actual native/source/gradient/FP32/BF16/recovery/cold controls must pass')
    admitted = read(ADMITTED)
    files = dict(admitted['source_files_sha256'])
    paths = [ADMITTED, TESTS, MATH, RUNTIME, ASSESS / 'evaluation-plan.json', ASSESS / 'captured-source.json',
             ROOT / 'reports/source-function-validation-runtime-CPU-v1/complete.json',
             ROOT / 'reports/source-function-validation-runtime-CPU-v1/all15-fixed-native-validation-cases.json',
             ROOT / 'reports/function-repair-evaluation-CPU-v1-tests.log',
             ROOT / 'scripts/code_worker.py', ROOT / 'scripts/native_sandbox.py',
             ROOT / 'data/math-reasoning-v2/test.jsonl', ROOT / 'data/math-reasoning-v2/manifest.json',
             *python_sources(ROOT, [Path(__file__), ROOT / 'tests/test_public_edge_preference_stage_v1.py'])]
    pulses = read(MATH)['pulse_sources']
    paths.append(Path(pulses['knowledge']).with_suffix('.audit.json'))
    for name in pulses.values():
        path = Path(name); paths.append(path)
        for row in map(json.loads, path.read_text().splitlines()):
            if row.get('image'):
                paths.append((path.parent / row['image']).resolve())
    files.update({str(p): file_hash(p) for p in set(paths)})
    guard = FileBindings(files)
    owner = birth()
    atomic_json(QUEUE, dict(owner=owner, source_files_sha256=files, public_pretraining_owner=runtime['owner'],
                exact_two_thread_assessment_owner=assessment['owner'], initializer_requires_all4447updates_and_native_controls=True,
                fixed_profiles=PROFILES, all_original77pairs_three_epochs_per_arm=True,
                expected_updates_per_arm=60, expected_pair_presentations_per_arm=231,
                matched_positive_response_EOS_targets=98106, matched_negative_response_EOS_observations=98109,
                fixed_local_assessments=dict(source_validation=15, source_native_budget=1024, local_code=48,
                                            code_native_budget=512, knowledge=56, math=16, OCR=20, objects=15, photos=32, image_size=448),
                CPU_threads_after_verified_release=2, CPU_nice=os.nice(0), minimum_other_writer_and_disk_reserve_bytes=RESERVE,
                no_GPU_lease_or_context=True, no_canonical_queue_or_source_changed=True,
                original_full_training_and_all_final_benchmarks_preserved=True, automatic_retry=False,
                actual_full_size_capacity_not_yet_passed=True, neural_updates=0, candidate_promoted=False, goal_achieved=False))
    lock = (ROOT / 'reports/public-pretraining-interim-CPU.lock').open('a+')
    while True:
        guard.check()
        phase = 'waiting for exact live interim assessment and full public predecessor'
        assessment_done = terminal(assessment['owner'])
        if assessment_done and (not (ASSESS / 'complete.json').exists() or (ASSESS / 'failure.json').exists()):
            raise ValueError('The exact interim assessment ended without its complete matched result')
        ready = public_ready(ROOT, runtime['owner']) if terminal(runtime['owner']) else False
        if assessment_done and ready:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                phase = 'waiting for verified two-thread CPU allocation lock'
            else:
                stat = dict(line.split()[:2] for line in Path('/sys/fs/cgroup/memory.stat').read_text().splitlines())
                live = sum(int(stat[k]) for k in ['anon', 'shmem', 'kernel'])
                limit = int(Path('/sys/fs/cgroup/memory.max').read_text())
                if shutil.disk_usage(ROOT).free >= RESERVE + 5 * 1024**3 and live + 32*1024**3 + 64*1024**3 <= limit:
                    break
                fcntl.flock(lock, fcntl.LOCK_UN)
                phase = 'waiting for full CPU working set and canonical writer reserves'
        atomic_json(WAIT, dict(owner=owner, phase=phase, neural_updates=0, goal_achieved=False))
        time.sleep(30)
    import copy
    import numpy as np
    import torch
    from slm_lab.async_durable_checkpoint_v1 import AsyncCheckpoint
    from slm_lab.codenet_checkpoint_v1 import atomic_checkpoint, capture_rng
    from slm_lab.CPU_frozen_cast_cache_v1 import CPUFrozenCastCache
    from slm_lab.public_edge_preference_stage_v1 import EdgePreferences, EdgePreferenceStage, schedule, gradients
    from slm_lab.function_repair_evaluation_CPU_v1 import source_validation, local_suites
    from slm_lab.public_pretraining_native_v1 import load
    from slm_lab.own_dense_actor_v1 import OwnDenseActor
    from slm_lab.whole_human_sft_evidence_v1 import tensor_digest
    torch.set_num_threads(2); torch.set_num_interop_threads(1); torch.use_deterministic_algorithms(True)
    if torch.cuda.is_initialized() or len(os.sched_getaffinity(0)) < 8:
        raise ValueError('Exactly two CPU threads on the existing eight-core affinity and no GPU required')
    data = EdgePreferences(DATA, ADMITTED)
    entries = schedule(len(data.rows)); budget = data.budget(entries)
    if (len(data.rows) != 77 or len(entries) != 60 or budget['presentations'] != 231
            or budget['chosen_response_EOS_targets'] != 98106 or budget['rejected_response_EOS_targets'] != 98109):
        raise ValueError('Every original77pair and all three complete epochs must remain in every arm')
    files.update({str(BASE): file_hash(BASE), str(BASE.parent / 'tokenizer.json'): file_hash(BASE.parent / 'tokenizer.json'),
                  str(ASSESS / 'complete.json'): file_hash(ASSESS / 'complete.json')})
    guard = FileBindings(files)
    base, metadata = load(BASE, MATH, 'trained', expected_checkpoint_sha256=files[str(BASE)], expected_plan_sha256=files[str(MATH)], device='cpu')
    model = OwnDenseActor(base).eval()
    if (base.counts()['total_unique'] != 984399690 or model.counts()['actual_unique'] != 999706442
            or metadata['completed_updates'] != 4447 or metadata['tokenizer_sha256'] != file_hash(data.tokenizer_path)):
        raise ValueError('Every actual own complete under1B model/head/actor value required')
    initial = copy.deepcopy(model.parts.state_dict()); initial_digest = tensor_digest(model.parts.named_parameters())
    frozen_digest = tensor_digest(base.named_parameters()); critic = model.parts['critic'].weight.detach().clone()
    options = dict(learning_rate=1e-4, betas=[.9,.95], weight_decay=.01, gradient_clip=1., beta=.1,
                   preference_weight=1., precision='CPU_BF16_cached', CPU_threads=2, seed=980113,
                   complete_source_epochs=3, matched_global_width=4, save_after_first_then_every=8)
    OUT.mkdir(); OUTPUT.mkdir(); shutil.copyfile(data.tokenizer_path, OUTPUT / 'tokenizer.json')
    longest = max(data.slots, key=lambda p: data.lengths[p])
    tail = next(list(row) for row in entries if len(row) == 1)
    capacity_entries = [[longest], tail]
    plan = dict(at=datetime.now(timezone.utc).isoformat(), owner=owner, source_files_sha256=files,
                profiles=PROFILES, all_complete_pair_batches=entries, source_schedule_identity_sha256=data.identity(entries),
                actual_updates_per_profile=60, matched_complete_budget=budget, training_options=options,
                actual_student_counts=model.counts(), complete_public_dense_counts=base.counts(),
                initial_actor_parts_sha256=initial_digest, frozen_dense_sha256=frozen_digest,
                actual_capacity_batches=capacity_entries, exact_original_pair_rows=77, original_positive_source_functions=23,
                fixed_assessments=read(QUEUE)['fixed_local_assessments'], pulse_sources=pulses,
                all_three_arm_training_finishes_before_any_new_evaluation=True,
                full_sequence_and_fork_contexts_observed_by_all_controls=True,
                baseline_and_all_arms_use_same_actual_actor_graph=True,
                bit_identity_to_distinct_base_forward_graph_not_certified=True,
                branch_gradient_only_at_first_divergent_next_token_with_full_chosen_CE=True,
                own_reference_private_values_excluded_from_student=True, external_teacher=False, human_preferences=False,
                on_policy_RL=False, independent_full_final_benchmarks_unchanged=True, candidate_promoted=False, goal_achieved=False)
    atomic_json(OUT / 'plan.json', plan); plan_sha = file_hash(OUT / 'plan.json')
    cache = CPUFrozenCastCache(); cache.attach(model.policy)
    def sources(profile):
        return dict(comparison_plan_sha256=plan_sha, profile_sha256=hashlib.sha256(profile.encode()).hexdigest(),
                    own_complete_public_model_sha256=files[str(BASE)])
    def reset(profile, mode, batches):
        model.parts.load_state_dict(initial); model.zero_grad(set_to_none=True)
        if tensor_digest(model.parts.named_parameters()) != initial_digest:
            raise ValueError('Every arm must start from the same complete own model and actor')
        random.seed(options['seed']); np.random.seed(options['seed']); torch.manual_seed(options['seed'])
        optimizer = torch.optim.AdamW([p for _,p in model.named_selected(('actor',))], lr=options['learning_rate'],
                                     betas=tuple(options['betas']), weight_decay=options['weight_decay'], foreach=False)
        return EdgePreferenceStage(model, optimizer, data, batches, profile, options, sources(profile), cache, mode)
    def equal(a,b):
        if isinstance(a,torch.Tensor): return isinstance(b,torch.Tensor) and torch.equal(a,b)
        if isinstance(a,dict): return isinstance(b,dict) and a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
        if isinstance(a,(tuple,list)): return type(a) is type(b) and len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
        return a==b
    def adam(stage):
        return tensor_digest((str(i)+'.'+k,v[k]) for i,v in enumerate(stage.optimizer.state.values()) for k in ['step','exp_avg','exp_avg_sq'])
    def intact():
        cache.assert_current()
        if tensor_digest(base.named_parameters()) != frozen_digest or not torch.equal(model.parts['critic'].weight,critic) or torch.cuda.is_initialized():
            raise ValueError('Every original base/head/unused critic and GPU isolation must remain exact')
    def cold(path, profile):
        other = OwnDenseActor(base, **model.settings).eval()
        other.import_export(torch.load(path,map_location='cpu',weights_only=True,mmap=True),expected_sources=sources(profile))
        start = len(cache.patched); cache.attach(other.policy); model.eval()
        try:
            ids = torch.tensor([data.native(profile,longest)[0]])
            with torch.no_grad(),torch.autocast('cpu',dtype=torch.bfloat16):
                wanted=model(input_ids=ids)['logits']; actual=other(input_ids=ids)['logits']
            if not torch.equal(wanted,actual) or tensor_digest(model.parts.named_parameters()) != tensor_digest(other.parts.named_parameters()):
                raise ValueError('Cold full-context actor values and CPU BF16 logits must match exactly')
            return dict(complete=True,actual_native_context=ids.shape[1],all_actor_and_critic_values_exact=True,
                        full_context_CPU_BF16_logits_bit_exact=True,temporary_extra_actor_and_critic_values=15306752,
                        complete_original_base_shared_once=True)
        finally:
            for module,original in cache.patched[start:]: module.forward=original
            del cache.patched[start:]
    atomic_checkpoint(OUTPUT/'baseline-actor.pt',model.export(sources('baseline')),96*1024**2,RESERVE)
    try:
        capacity={}
        for profile in PROFILES:
            stage=reset(profile,'capacity',capacity_entries); folder=OUT/('capacity-'+profile); folder.mkdir()
            atomic_json(OUT/'progress.json',dict(phase='capacity',profile=profile,goal_achieved=False))
            start=time.monotonic(); stage.update(); first_seconds=time.monotonic()-start
            first=OUTPUT/(profile+'-capacity-after1.pt'); atomic_checkpoint(first,stage.state(),MAX_STATE,RESERVE)
            wanted=stage.update(); weights=tensor_digest(model.parts.named_parameters()); wanted_adam=adam(stage); rng=capture_rng('cpu')
            saved=torch.load(first,map_location='cpu',weights_only=True,mmap=True)
            for corruption in ['target_mask','private_RNG']:
                bad=copy.deepcopy(saved)
                if corruption=='target_mask': bad['metrics'][0]['complete_pairs'][0]['chosen_response_EOS_targets']-=1
                else: bad['rng']['torch']=torch.zeros(1,dtype=torch.uint8)
                try: stage.resume(bad)
                except ValueError: pass
                else: raise ValueError('Corrupt complete target mask/cursor or private RNG must reject')
                if tensor_digest(model.parts.named_parameters())!=weights or adam(stage)!=wanted_adam or not equal(rng,capture_rng('cpu')):
                    raise ValueError('Rejected recovery changed model/Adam/private RNG')
            stage.resume(saved)
            if stage.update()!=wanted or tensor_digest(model.parts.named_parameters())!=weights or adam(stage)!=wanted_adam or not equal(rng,capture_rng('cpu')):
                raise ValueError('Actual full-size next source update, actor/Adam/private RNG must recover exactly')
            path=OUTPUT/(profile+'-capacity-actor.pt'); atomic_checkpoint(path,model.export(sources(profile)),96*1024**2,RESERVE)
            proof=cold(path,profile); intact(); guard.check(full=True)
            if len(stage.optimizer.state)!=561: raise ValueError('All561actual actor Adam leaves required')
            capacity[profile]=dict(complete=True,logical_updates=2,physical_optimizer_calls=3,
                first_actual_full_size_update_seconds=first_seconds,actual_capacity_batches=capacity_entries,
                all_selected_Adam_leaves=561,actual_next_update_weights_Adam_metrics_private_RNG_bit_exact=True,
                corrupt_target_mask_and_private_RNG_rejected_before_mutation=True,cold=proof,reference_inventory=stage.reference.resident_inventory(model),
                training_only_cast_inventory=cache.inventory(),production_updates=0,goal_achieved=False)
            atomic_json(folder/'complete.json',capacity[profile]); del saved,bad,stage; gc.collect()
        atomic_json(OUT/'capacity-complete.json',dict(complete=True,profiles=capacity,production_updates=0,goal_achieved=False))
        training={}
        for profile in PROFILES:
            stage=reset(profile,'train',entries); folder=OUT/profile; folder.mkdir(); writer=AsyncCheckpoint(); elapsed=[]
            try:
                with (folder/'actual-training-metrics.jsonl').open('x') as journal:
                    for i in range(len(entries)):
                        guard.check(); writer.check(); start=time.monotonic(); row=stage.update(); elapsed.append(time.monotonic()-start)
                        journal.write(json.dumps(row)+'\n'); journal.flush()
                        if stage.completed==1 or stage.completed%8==0 or stage.completed==len(entries):
                            os.fsync(journal.fileno()); guard.check(full=True)
                            writer.submit(OUTPUT/(profile+'-resume.pt'),stage.state(),MAX_STATE,RESERVE)
                        atomic_json(OUT/'progress.json',dict(phase='training',profile=profile,actual_complete_updates=stage.completed,
                            planned_updates_per_profile=60,completed_profiles=list(training),last_chosen_token_mean_NLL=row['chosen_token_mean_NLL'],
                            last_update_seconds=elapsed[-1],full_size_capacity_all_profiles_passed=True,goal_achieved=False))
                    writer.flush()
                stage.validate(stage.metrics); intact()
                if (sum(r['chosen_response_EOS_targets'] for r in stage.metrics)!=98106
                        or sum(r['rejected_response_EOS_targets'] for r in stage.metrics)!=98109
                        or any(sum(batch.count(p) for batch in entries)!=3 for p in data.slots)):
                    raise ValueError('Every original complete pair and EOS must be consumed in all three full epochs')
                path=OUTPUT/(profile+'-actor.pt'); atomic_checkpoint(path,model.export(sources(profile)),96*1024**2,RESERVE)
                training[profile]=dict(complete=True,actual_logical_updates=60,physical_optimizer_calls=60,complete_budget=budget,
                    cold=cold(path,profile),actor_checkpoint_sha256=file_hash(path),private_state_sha256=file_hash(OUTPUT/(profile+'-resume.pt')),
                    actual_update_seconds=elapsed,candidate_promoted=False,goal_achieved=False)
                atomic_json(folder/'training-complete.json',training[profile])
            finally: writer.close()
            del stage; gc.collect()
        atomic_json(OUT/'all-arms-training-complete.json',dict(complete=True,profiles=training,quality_not_yet_measured=True,goal_achieved=False))
        evaluations={}
        for profile in ['baseline',*PROFILES]:
            path=OUTPUT/(profile+'-actor.pt')
            model.import_export(torch.load(path,map_location='cpu',weights_only=True,mmap=True),expected_sources=sources(profile)); model.eval()
            atomic_json(OUT/'progress.json',dict(phase='evaluation',profile=profile,all_three_full_training_arms_complete=True,goal_achieved=False))
            folder=OUT/('evaluation-'+profile); folder.mkdir()
            valid=source_validation(model.policy,path,folder/'source-validation',ROOT)
            local=local_suites(model.policy,path,metadata,folder/'local',ROOT,pulses)
            evaluations[profile]=dict(source_validation_correct=valid['all_original_examples_passed'],source_validation_functions=15,
                local_code_correct=sum(r['candidate'].get('functional_passed',False) for frame in ['completion','instruction'] for r in local['code-'+frame]['results']),
                local_code_tasks=48,knowledge_generated_correct=sum(r['generated_correct'] for r in local['knowledge']['predictions']),
                knowledge_questions=56,math_correct=local['math']['correct'],math_examples=16,
                source_validation_report_sha256=file_hash(folder/'source-validation/complete.json'),
                all_local_reports_receipt_sha256=file_hash(folder/'local/complete.json'))
            intact(); guard.check(full=True)
            atomic_json(folder/'complete.json',dict(complete=True,**evaluations[profile],candidate_promoted=False,goal_achieved=False))
        atomic_json(OUT/'complete.json',dict(complete=True,at=datetime.now(timezone.utc).isoformat(),owner=owner,
            source_files_sha256=files,training_profiles=training,matched_evaluations=evaluations,
            all77machine_pairs_three_epochs_all_original_contexts_and_EOS_preserved=True,
            all_baseline_and_final15source48code_and_broad_local_suites_complete=True,
            original_full_public_human_RL_export_and_final_benchmarks_unmodified=True,
            no_automatic_model_adoption=True,candidate_promoted=False,goal_achieved=False))
    finally: cache.close()


if __name__=='__main__':
    try: main()
    except BaseException as error:
        target=OUT/'failure.json' if OUT.exists() else ROOT/'reports/public-edge-preference-comparison-CPU-v2-parent-failure.json'
        atomic_json(target,dict(at=datetime.now(timezone.utc).isoformat(),owner=birth(),error=repr(error),traceback=traceback.format_exc(),
                    automatic_retry=False,original_training_and_all_final_benchmarks_unmodified=True,goal_achieved=False))
        raise
