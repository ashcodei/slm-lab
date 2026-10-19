"""Record actual complete-source controller mechanics and pending full scopes."""
from copy import deepcopy
from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess

from record_visual_stage_and_first_repair_arm_v1 import atomic, read, sha

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/batched-reasoning-training-controller-v1-review'
PREVIOUS = ROOT / 'reports/native-suffix-group-RL-v1-review/review.json'


def observed(identity):
    try: fields = Path('/proc', str(identity['pid']), 'stat').read_text().split(') ', 1)[1].split()
    except FileNotFoundError: return dict(owner=identity, live=False)
    live = fields[19] == identity['ticks'] and fields[0] not in ['Z', 'X']
    return dict(owner=identity, live=live, actual_ticks=fields[19], state=fields[0],
        nice=os.getpriority(os.PRIO_PROCESS, identity['pid']) if live else None,
        scheduler=os.sched_getscheduler(identity['pid']) if live else None)


def latest(folder):
    for name in ['failure.json', 'complete.json', 'progress.json']:
        path = folder / name
        if path.exists(): return dict(observed_file=name, value=read(path))
    raise ValueError('Inspect the same actual source owner; missing observation is not terminal')


def main():
    if OUT.exists(): raise ValueError('Preserve all earlier reviews')
    tests = ROOT / 'reports/batched-reasoning-training-controller-v1-tests.log'
    if '1 passed' not in tests.read_text(): raise ValueError('The connected four-replica two-profile test must complete')
    proof_root = ROOT / 'reports/batched-reasoning-training-controller-v1-pytest'
    paths = sorted(proof_root.rglob('anchor_*-rank-*.json'))
    if len(paths) != 8: raise ValueError('All four actual replica proofs for both profiles are required')
    for path in paths:
        row = read(path)
        if (row['profile'] not in ['anchor_CE', 'anchor_RL'] or not 0 <= row['rank'] < 4
                or row['actual_small_research_parameters'] != 267718 or row['actual_native_parameters'] != 267430
                or not all(row[k] for k in ['exact_complete_and_interrupted_recovered_final_state',
                    'actual_replayed_uncommitted_update_actions_verifiers_logps_gradients_equal',
                    'root_only_error_retains_both_observed_updates_and_old_durable_prefix',
                    'peer_only_bad_readback_and_missing_or_stale_pins_rejected_before_mutation',
                    'all_failed_and_recovery_journals_retained', 'CPU_fixture_does_not_admit_production'])
                or row['production_updates'] != 0 or row['full_size_GPUcapacity_or_quality']):
            raise ValueError('Every actual own-small recovery/failure scope must remain explicit')
        cap, fresh = row['actual_capacity'], row['actual_fresh_readback']
        trained, recovered = row['actual_complete_training'], row['actual_recovered_training']
        if (cap['actual_physical_optimizer_calls'] != 4 or cap['actual_complete_capacity_updates'] != 3
                or fresh['new_optimizer_calls'] != 0 or fresh['new_generated_actions'] != 0
                or fresh['actual_Adam_states'] != 17 or not fresh['independent_complete_actor_cold']['complete_text_photo_word_outputs_bit_exact']
                or trained['actual_complete_updates'] != 3 or trained['actual_new_optimizer_calls'] != 3
                or trained['original_complete_training_budget']['source_presentations'] != 9
                or trained['actual_complete_journal']['records'] != 3 or not trained['actual_complete_journal']['complete']
                or recovered['actual_new_optimizer_calls'] != 2 or recovered['actual_attempt_started_at_update'] != 1
                or recovered['actual_complete_journal']['records'] != 2
                or trained['actual_final_actor_sha256'] != recovered['actual_final_actor_sha256']
                or trained['actual_final_Adam_sha256'] != recovered['actual_final_Adam_sha256']
                or trained['actual_final_private_RNG_sha256'] != recovered['actual_final_private_RNG_sha256']):
            raise ValueError('Actual fresh state/cold, complete training and exact preserved-attempt recovery must reconcile')
    at = datetime.now(timezone.utc).isoformat()
    human = read(ROOT / 'reports/phase-human-preference-v8/chosen_CE-progress.json')
    gpu_owners = [observed(dict(pid=pid, ticks=ticks)) for pid, ticks in
        [(151931, '55253729'), (151932, '55253730'), (151933, '55253731'), (151934, '55253731')]]
    if not all(o['live'] for o in gpu_owners): raise ValueError('Inspect the exact original GPUworker transition')
    source_roots = ['full-public-token-suffix-qualification-gzip-recovery-CPU-v2',
        'full-public-token-suffix-readback-gzip-recovery-CPU-v2', 'full-native-token-suffix-qualification-CPU-v1',
        'full-corrected-multimodal-batch-readback-CPU-v1', 'full-corrected-multimodal-source-admission-CPU-v1']
    source_observations = {name: latest(ROOT / 'reports' / name) for name in source_roots}
    source_owners = {name: observed(read(ROOT / 'reports' / name / 'plan.json')['owner']) for name in source_roots}
    source_progress = source_observations[source_roots[0]]['value']['actual_counts']['complete_original_sources']
    snapshot = dict(at=at, actual_original_human_progress=human, actual_original_GPUowners=gpu_owners,
        actual_source_observations=source_observations, actual_source_owners=source_owners,
        actual_GPU_rows=subprocess.check_output(['nvidia-smi', '--query-gpu=index,name,utilization.gpu,memory.used,memory.total',
            '--format=csv,noheader'], text=True), actual_disk_rows=subprocess.check_output(['df', '-B1', str(ROOT)], text=True),
        mutable_queues_progress_journals_and_child_registries_observed_not_source_bound=True,
        no_existing_neural_worker_signalled_reprioritized_restarted_or_changed=True,
        no_new_GPUcontext_job_or_lease=True,
        shared_cache_deletion_awaits_actual_reply=not (ROOT / 'reports/shared-pretrained-cache-reclamation-v1/user-approval.json').exists())
    prior = read(PREVIOUS)
    requirements = deepcopy(prior['requirements'])
    requirements['gpu_runtime']['evidence'] = (
        'A connected actual four-replica test passes both matched anchor_CE/anchor_RLprofiles. Each explicitly own-small '
        '267718parameter research/267430native CPUfixture runs3complete capacity groups/4physical updates, then a second '
        'fresh replica set with empty Adam independently restores every committed actor/17Adam/private RNG/source cursor '
        'without any new sample or optimizer call. Complete32text/32photo/64x64word cold is exact. New durable training '
        'then executes all3declared mechanics groups/9source presentations. A root-only observer failure after its second '
        'real Adam retains both actual journal observations and the old committed1update prefix. Fresh recovery uses the '
        'explicit retained pointer and a new journal; its two remaining updates exactly match the uninterrupted final '
        'weights/Adam/private RNG/all metrics, including actual replayed actions/verifiers/logps/unclipped gradients. '
        'Both old failed observations and replay observations remain. Peer-only invalid readback and missing/stale '
        'pin controls reject before mutation. Small CPUproofs cannot admit full production. Full actual CUDAcapacity, '
        'fresh four-GPUreadback, owned launch, complete production training/native export and assessments remain.')
    requirements['serious_full_model_training']['evidence'] = (
        f'The full4447update public epoch and3x306GPUrepair arms remain complete. Original human first arm is{human["actual_updates"]}/5362. '
        'The new controller retains the full proposed2568source/12task/7704presentation/57240answer-EOS/1926global-update '
        'three-epoch reasoning protocol and full64/256native action budgets; its actual new proof uses9presentations '
        'per small mechanics trial, not the full model or full source epoch. Both matched profiles pass fresh actual '
        'capacity state readback, complete fixture training and interrupted-attempt recovery. Across these checks each '
        'replica performs22small optimizer calls(88across4replicas), including repeated capacity and interrupted replay '
        'calls. New production updates:0. Original queued warmup, single-source RLprotocol,3721human-code anchors, '
        '4733stepvisual epoch and9final endpoints remain uncompleted requirements.')
    requirements['human_feedback']['evidence'] = (
        f'The unchanged canonical four-L40chosen-response arm is{human["actual_updates"]}/5362updates. All85786human-ranked '
        'pairs in each chosen_CE/human_DPOarm remain. Public preferences include other-model texts ranked by humans, '
        'not fresh human judgments on this student. New reasoning verifier training is machine feedback, not human RLHF. '
        'Full human comparison completion, fresh student feedback and final quality remain pending.')
    requirements['public_data']['evidence'] += (
        f' Latest original full code-source recovery observation:{source_progress}/3721. Its full fresh reader and new '
        'native-profile successor remain separate source gates. No original source/CEtarget or failed/trivial/abstaining '
        'disposition is removed. The full reasoning controller uses the existing independently admitted public source '
        'adapter; no external model weights or live teacher API is introduced.')
    for value in requirements.values():
        if value['status'] != 'incomplete_for_requested_final_model': raise ValueError('All ten final requirements remain incomplete')
        value['previous_evidence'] = str(PREVIOUS)
    if len(requirements) != 10: raise ValueError('All original final requirements are required')
    names = ['src/slm_lab/batched_reasoning_training_controller_v1.py',
        'tests/test_batched_reasoning_training_controller_v1.py',
        'scripts/run_batched_reasoning_training_controller_v1_tests.py',
        'reports/batched-reasoning-actual-capacity-cases-CPU-v1/complete.json',
        'reports/batched-reasoning-actual-capacity-cases-CPU-v1/actual-complete-source-capacity-plan.json']
    journals = sorted(proof_root.rglob('*.jsonl.gz'))
    # Closed CPUfixture journals are retained; no live source/production stream is bound.
    files = {str(p): sha(p) for p in [Path(__file__), PREVIOUS, tests, *paths, *journals, *[ROOT / n for n in names]]}
    classification = ('progress:actual fresh four-replica capacity-state readback and durable complete-source batched reasoning '
        'training controller; both matched own-small profiles preserve exact interrupted-update replay and complete final '
        'weights/Adam/private RNG; all failed journals retained;0production updates or quality gain')
    OUT.mkdir()
    atomic(OUT / 'runtime-snapshot.json', snapshot)
    atomic(OUT / 'complete.json', dict(complete=True, at=at, source_files_sha256=files,
        actual_connected_tests=1, actual_profiles=['anchor_CE', 'anchor_RL'], actual_replicas=4,
        actual_small_research_parameters=267718, actual_small_native_parameters=267430,
        actual_capacity_groups_per_profile=3, actual_capacity_physical_calls_per_profile_per_replica=4,
        actual_fresh_readback_optimizer_calls=0, actual_fixture_training_groups=3, actual_fixture_source_presentations=9,
        actual_interrupted_observed_updates=2, actual_prior_durable_prefix=1, actual_recovery_new_updates=2,
        actual_all_small_physical_optimizer_calls_per_replica=22, actual_all_small_replica_optimizer_calls=88,
        exact_uninterrupted_and_recovered_final_weights_Adam_private_RNG_metrics=True,
        exact_replayed_uncommitted_actions_verifiers_logps_unclipped_gradients=True,
        both_failed_and_recovery_complete_journals_retained=True,
        proposed_full_training_budget=read(ROOT / names[-1])['complete_training_budget'],
        actual_original_human_updates=human['actual_updates'], actual_original_source_progress=source_progress,
        no_production_admission_from_own_small_proofs=True, actual_new_production_updates=0,
        actual_new_model_quality_gain=False, all_ten_final_requirements_incomplete=True, goal_achieved=False))
    metrics = (
        '**The connected four-replica controller test passes both matched CE/RLprofiles.** Each uses the actual own '
        '267718parameter research/267430native CPUmodel. After the existing3group/4physical-update capacity run, a newly '
        'constructed stage with empty Adam independently reads all actor/17Adam/private RNG/cursor values from the exact '
        'external receipt and pinned tensor file. No new action or Adam call occurs during readback. A separate complete '
        '32text/32photo/64x64word actor cold check remains bit exact.\n\n'
        '**Complete fixture training and interrupted recovery agree exactly.** Three declared groups/9source presentations '
        'finish with a lossless fsynced journal and two-slot private state. A root-only observer failure after its second '
        'actual update leaves both observations in the failed-attempt journal and the first update committed. Recovery '
        'uses that exact retained pointer and a new journal; it repeats the uncommitted second update and finishes the '
        'third. Replayed actual actions/verifiers/logps/unclipped gradients match. Final weights/Adam/private RNG/all '
        'metrics match the uninterrupted run. Earlier failed and replayed observations are preserved.\n\n'
        'All four real replicas coordinate rejection of a peer-only wrong readback and missing/stale external pins before '
        'mutation. The explicit CPUmechanics opt-in never admits production. Across two profiles, each replica performs '
        '22small optimizer calls(88total actual replica calls), including capacity replay and the interrupted repetition. '
        '**These are mechanics checks, not trained984Mmodel improvements.**\n\n'
        'Full proposed reasoning training remains all2568sources/12tasks for3epochs:7704presentations,57240complete '
        'answer/EOS targets and1926global updates, with original64/256native generation budgets. The full epoch is '
        'divisible by4; the fixture single-source/three-empty tail remains an additional control. The old single-source '
        'RLprotocol, original warmup, all17public families/human/visual scopes and9final endpoints remain. The controller '
        'owns no GPUlease. A complete owned launcher, actual full-size GPUcapacity/fresh readback/training, native export '
        'and full quality assessments are still required.\n\n'
        f'Original four-L40human first arm: **{human["actual_updates"]}/5362updates**. Original code-source recovery: '
        f'**{source_progress}/3721**; full fresh controls and separate native-profile validation remain. No existing neural '
        'worker was changed and no new GPUcontext/job/lease was opened.\n\n'
        '**New production updates:0. Latest completed coding correctness remains1/48at each original study endpoint. '
        'No new model-quality gain is established; all ten final requirements remain incomplete.** The earlier shared '
        '32.1GiBcache cleanup question still awaits an actual user reply; no shared weights were loaded or deleted.\n')
    (OUT / 'metrics.md').write_text(metrics)
    for p in [OUT / 'runtime-snapshot.json', OUT / 'complete.json', OUT / 'metrics.md']: files[str(p)] = sha(p)
    atomic(OUT / 'review.json', dict(at=at, requirements=requirements, source_files_sha256=files,
        previous_full_scope_review=str(PREVIOUS), current_goal_turn_classification=classification, goal_achieved=False))
    audit_path = ROOT / 'reports/completion-audit.json'
    audit = read(audit_path)
    audit.update(at=at, updated_at=at, requirements=requirements, latest_full_scope_review=str(OUT / 'review.json'),
        latest_batched_reasoning_training_controller=str(OUT / 'complete.json'),
        next_batched_reasoning_training_controller=str(ROOT / 'src/slm_lab/batched_reasoning_training_controller_v1.py'),
        current_goal_turn_classification=classification, goal_achieved=False)
    atomic(audit_path, audit)
    entry = (f'Batched reasoning durable training controller, **{at}**: both matched profiles pass an actual four-CPU '
        'replica integration test with fresh capacity readback, complete declared fixture training and exact interrupted '
        'update/final-state recovery. Failed and replay journals remain. Full7704source/57240target reasoning scope is '
        f'preserved; full-size GPUlaunch/training is still pending. Original human first arm: {human["actual_updates"]}/5362. '
        '**No production accuracy gain; all ten final requirements remain incomplete.** '
        '[Evidence](reports/batched-reasoning-training-controller-v1-review/metrics.md), '
        '[full review](reports/batched-reasoning-training-controller-v1-review/review.json). Earlier entries are historical snapshots.\n\n')
    for name in ['README.md', 'RESEARCH.md']:
        p = ROOT / name
        p.write_text(entry + p.read_text())
    print(str(OUT / 'complete.json'))


if __name__ == '__main__': main()
