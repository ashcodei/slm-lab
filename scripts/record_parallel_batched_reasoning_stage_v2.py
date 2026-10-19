"""Record verified four-CPU-replica mechanics and the intact full scope."""
from copy import deepcopy
from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess

from record_visual_stage_and_first_repair_arm_v1 import read, sha, atomic
from slm_lab.exact_waiting_process_retirement_v1 import terminal

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/parallel-batched-reasoning-stage-v2-review'
PREVIOUS = ROOT / 'reports/complete-repair-study-and-batched-RL-v2-review/review.json'
HUMAN = ROOT / 'reports/phase-human-preference-v8'


def main():
    if OUT.exists(): raise ValueError('Preserve every earlier completed or failed review')
    tests = ROOT / 'reports/parallel-batched-reasoning-stage-v2-tests.log'
    if '7 passed' not in tests.read_text(): raise ValueError('All7complete source, actual neural/recovery/rejection checks must pass')
    base = ROOT / 'reports/parallel-batched-reasoning-stage-v2-pytest'
    proofs = sorted(base.rglob('anchor_*-rank-*.json'))
    if len(proofs) != 8: raise ValueError('Both matched profiles need all4actual own-small CPUrank proofs')
    actual = [read(p) for p in proofs]
    for profile in ['anchor_CE', 'anchor_RL']:
        rows = sorted((r for r in actual if r['profile'] == profile), key=lambda r: r['rank'])
        if ([r['rank'] for r in rows] != list(range(4)) or len({r['final_checkpoint_sha256'] for r in rows}) != 1
                or any(r['actual_student_values'] != 267334 or r['actual_unique_global_source_groups'] != 3
                    or r['actual_source_presentations'] != 9 or r['actual_ordinary_Adam_states'] != 17
                    or r['final_Adam_clocks'] != 3 or r['actual_physical_Adam_calls'] != 4
                    or r['genuine_final_local_source_count'] != int(r['rank'] == 0)
                    or not all(r[k] for k in ['root_serial_full_source_mean_matches',
                        'all_actual_actions_verifiers_unclipped_gradients_weights_Adam_private_RNG_replay_exact',
                        'peer_only_valid_RNG_swap_rejected', 'peer_only_nonfinite_gradient_forward_coordinated_before_Adam',
                        'complete_nonfinite_sample_observation_retained_on_every_rank',
                        'root_only_snapshot_error_coordinated_before_mutation'])
                    or r['distinct_private_global_RNGs_retained'] != 4 or r['production_updates'] != 0
                    or r['full_size_GPU_capacity_or_model_quality'] is not False for r in rows)):
            raise ValueError('All actual rank-private/shared/source/gradient/recovery observations must reconcile')
    singles = sorted(base.rglob('actual-single-replica-recovery.json'))
    budgets = list(base.rglob('actual-complete-source-budget.json'))
    if len(singles) != 2 or len(budgets) != 1: raise ValueError('Both serial cold recoveries and whole-source budget are required')
    budget = read(budgets[0])
    if any(budget[k] != v for k, v in dict(updates=642, source_presentations=2568,
            complete_context_tokens=245190, response_EOS_targets=19080,
            full_native_generation_horizon_sum=242112, original_train_task_families=12, optimizer_updates=0).items()):
        raise ValueError('Every original source and full positive/EOS/native horizon budget must be preserved')
    gap_path = ROOT / 'reports/batched-reasoning-recovery-guard-gaps-v1.json'
    gap = read(gap_path)
    if (not gap['original_v1_two_late_recovery_rejection_gaps_observed']
            or not gap['corrected_v2_both_rejected_before_mutation'] or gap['actual_total_small_test_Adam_calls'] != 8
            or gap['production_updates'] != 0 or len(gap['actual_observations']) != 4):
        raise ValueError('Actual old failures and corrected pre-mutation rejection must remain explicit')
    plan = read(HUMAN / 'plan.json')
    progress = read(HUMAN / 'chosen_CE-progress.json')
    children = [p for p in read(HUMAN / 'children.json') if not terminal(p)]
    if (terminal(plan['owner']) or not read(HUMAN / 'capacity.json')['passed'] or (HUMAN / 'failure.json').exists()
            or plan['human_train_pairs'] != 85786 or plan['training_options']['steps'] != 5362
            or not any(p['subject'].startswith('train-chosen_CE-') for p in children)
            or not 0 < progress['actual_updates'] < progress['planned_updates'] == 5362):
        raise ValueError('This observation expects the original intact first human arm to remain training')
    at = datetime.now(timezone.utc).isoformat()
    snapshot = dict(at=at, original_human_owner=plan['owner'], actual_live_human_children=children,
        actual_human_progress=progress, actual_GPU_rows=subprocess.check_output(['nvidia-smi',
            '--query-gpu=index,name,utilization.gpu,memory.used,memory.total', '--format=csv,noheader'], text=True),
        public_suffix_qualification_progress=read(ROOT / 'reports/full-public-token-suffix-qualification-CPU-v1/progress.json'),
        full_public_suffix_producer_live=not terminal(dict(pid=146084, ticks='55111340')),
        full_public_suffix_fresh_reader_live=not terminal(dict(pid=146633, ticks='55128888')),
        full_visual_pixel_readback_progress=read(ROOT / 'reports/full-multimodal-pixel-readback-CPU-v1/progress.json'),
        full_visual_pixel_reader_live=not terminal(dict(pid=106808, ticks='54228332')),
        full_visual_corrected_batch_reader_live=not terminal(dict(pid=108345, ticks='54257301')),
        full_visual_source_admission_live=not terminal(dict(pid=109189, ticks='54271186')),
        visual_source_only_preparation_live=not terminal(dict(pid=136092, ticks='54738417')),
        actual_free_bytes=os.statvfs(ROOT).f_bavail * os.statvfs(ROOT).f_frsize,
        no_existing_neural_worker_signalled_reprioritized_restarted_or_patched=True,
        no_competing_GPU_context_job_or_lease_added=True,
        mutable_progress_registries_queues_and_journals_copied_into_observation_not_bound=True)
    requirements = deepcopy(read(PREVIOUS)['requirements'])
    requirements['gpu_runtime']['evidence'] = (
        'The new internal4replica batched public-reasoning stage passes7own-small CPUchecks. Both matched CE/RLprofiles '
        'use4actual267334parameter replicas,3global source groups/9sources,17ordinary Adam states and a real1source/3empty '
        'tail. An independent serial calculation of all4first-step local kernels matches actual equal-source gradient SUM '
        'and selected weights. Actual durable recovery repeats every next native action/verifier/logp/unclipped-gradient '
        'digest/weight/Adam/private RNG exactly. All4distinct private RNGs/cursors are retained. Peer-only malformed or swapped '
        'RNG, a peer-only nonfinite gradient rescore and root-only snapshot errors reject before Adam or live mutation. '
        'Full actual failed-sample observations are retained in all4workers. Two realv1recovery gaps are reproduced: source '
        'guard and selected-parameter identity were checked only after partial recovery mutation; preservedv2rejects both '
        'before copying. Explicit control usesGloo; CUDAgradients requireNCCL. These CPUchecks do not prove full-size CUDA '
        'capacity, collective behavior, performance or model quality. The original human run keeps the exclusive4L40lease.')
    requirements['serious_full_model_training']['evidence'] += (
        f" Observed original chosen_CE progress is{progress['actual_updates']}/5362updates. The new batched stage has0production "
        'updates. A distinct four-source protocol preserves all2568original reasoning sources/12families,19080complete '
        'answer/EOS targets,245190context tokens and242112native horizon positions per epoch in642global updates. '
        'It does not complete the old single-source RLprotocol or replace the original queued7704source/57240target warmup. '
        'The new7checks repeat the prior6stage controls plus a recovery regression; they are not13independent tests.')
    requirements['human_feedback']['evidence'] += (
        f" Current first-arm progress is{progress['actual_updates']}/5362; full85786pairs per arm and held evaluations remain. "
        'The new strict reasoning reward is machine feedback and provides no new human preference judgments.')
    requirements['public_data']['evidence'] += (
        ' The distinct batched reasoning protocol preserves all admitted train rows and full64/256action budgets; '
        'the explicit2action deliberately initialized CPUfixture is never admitted as production training. '
        'All3721human-code suffix references/14884counterfactual controls and their separate fresh reader continue; '
        'no suffix training admission or student gain is inferred from live source counts.')
    for row in requirements.values(): row['previous_evidence'] = str(PREVIOUS)
    if len(requirements) != 10 or any(r['status'] != 'incomplete_for_requested_final_model' for r in requirements.values()):
        raise ValueError('All ten final requirements remain incomplete')
    names = ['src/slm_lab/parallel_batched_reasoning_stage_v1.py', 'src/slm_lab/parallel_batched_reasoning_stage_v2.py',
        'tests/test_parallel_batched_reasoning_stage_v1.py', 'tests/test_parallel_batched_reasoning_stage_v2.py',
        'scripts/run_parallel_batched_reasoning_stage_v1_tests.py', 'scripts/run_parallel_batched_reasoning_stage_v2_tests.py',
        'scripts/probe_batched_reasoning_recovery_guards_CPU_v1.py', 'reports/parallel-batched-reasoning-stage-v1-tests.log',
        'reports/parallel-batched-reasoning-stage-v1-tests-v2.log', 'reports/batched-reasoning-recovery-guard-gaps-v1.log',
        'src/slm_lab/batched_own_native_rollouts_v2.py', 'src/slm_lab/batched_reasoning_group_gradients_v2.py',
        'src/slm_lab/parallel_repair_durable_state_v1.py', 'src/slm_lab/parallel_function_repair_stage_v1.py']
    closed = [Path(__file__), PREVIOUS, gap_path, tests, HUMAN / 'plan.json', HUMAN / 'capacity.json',
        *[ROOT / n for n in names], *sorted(base.rglob('*.json')), *sorted(base.rglob('*.pt'))]
    files = {str(p): sha(p) for p in closed}
    classification = ('progress:connected batched public-reasoning stage passes7actual own-small checks, both4replica '
        'matched profiles, full-source mean, genuine empty tails and exact durable next-update/private replay; '
        'two real late-recovery rejection gaps reproduced and fixed in preservedv2;0new production RLupdates or quality gain')
    OUT.mkdir()
    atomic(OUT / 'runtime-snapshot.json', snapshot)
    atomic(OUT / 'complete.json', dict(complete=True, at=at, source_files_sha256=files,
        actual_new_stage_controls=7, actual_own_small_student_values=267334,
        actual_profiles=['anchor_CE', 'anchor_RL'], actual_replicas_per_profile=4,
        actual_logical_updates_per_profile=3, actual_physical_Adam_calls_per_replica_per_profile=4,
        actual_original_sources_per_profile=9, actual_Adam_states_per_replica=17,
        exact_next_native_actions_verifiers_unclipped_gradients_weights_Adam_private_RNG=True,
        genuine_single_source_three_empty_rank_tail=True, serial_full_source_mean_matches_actual_reduction=True,
        peer_only_nonfinite_failure_preserves_all_samples_and_aborts_before_Adam=True,
        original_v1_late_recovery_failures_preserved=True, v2pre_mutation_rejection_verified=True,
        complete_one_epoch_proposed_data_budget=budget, original_single_source_and_warmup_scopes_unwaived=True,
        actual_human_progress_updates=progress['actual_updates'], actual_human_planned_updates=5362,
        full_size_CUDAcapacity_collectives_throughput_durability_native_export_assessment_pending=True,
        new_production_RL_updates=0, new_model_quality_gain=False, all_ten_final_requirements_incomplete=True,
        no_existing_neural_worker_change_or_competing_GPU_job=True, goal_achieved=False))
    text = (
        f"The original human chosen-response training is **{progress['actual_updates']}/5362updates** at this snapshot. "
        'All85786public human-ranked pairs per arm and original complete assessments remain. The four L40workers continue unchanged.\n\n'
        '**Seven connected batched-RL stage checks pass.** Both matched CE/RLprofiles run four actual267334parameter CPUreplicas, '
        'three distinct global source groups/nine original sources,17ordinary Adam states and a genuine one-source/three-empty-rank tail. '
        'Independent serial local gradients verify the actual equal-source global mean. Durable recovery exactly repeats next private '
        'draws, emitted actions, verifier results, probabilities, unclipped-gradient digest, actor values, Adam and all private RNGs. '
        'Cold single-replica recovery also passes. The final shared files match all four replicas.\n\n'
        'Peer-only valid RNGswaps, malformed peer RNGs, a peer-only NaNgradient rescore and root-only serialization faults reject '
        'before an optimizer commit or live recovery mutation. Every failed sampled attempt remains in the coordinated observation. '
        '**Two actualv1late-recovery gaps are reproduced and preserved**: changed source guard and selected-parameter identities '
        'could reject after partial mutation. **V2checks both before copying**; actual before/after hashes and cursors are retained.\n\n'
        'The proposed four-source protocol preserves the complete2568train sources/12families and19080whole answer/EOS targets '
        'per epoch, with245190context tokens,242112native horizon positions and642global updates. Every source retains64or256 '
        'generation actions. The explicit two-action test fixture is CPUmechanics only. This new protocol does not complete the old '
        'single-source RLprotocol or replace its queued7704source/57240target supervised warmup.\n\n'
        '**No new production RLupdate or capability improvement is claimed.** Full-size four-GPU capacity, CUDAcollectives, memory, '
        'throughput, durable recovery, native export and full matched evaluation remain required. The complete original coding study '
        'still scores1/48functional at all four endpoints; no arm is promoted. Previous broad regressions and all original failed '
        'experiments remain recorded. Full human-code suffix qualification/fresh execution and visual source readers continue. '
        'The visual GPUrunner remains unlaunched. All ten requested final requirements remain incomplete.\n\n'
        'The initial test launch used the system Python without pytest and failed before importing any test; that log is retained. '
        'The project virtual environment passed the full v1six and v2seven controls; overlapping checks are not summed.\n')
    (OUT / 'metrics.md').write_text(text)
    for p in [OUT / 'complete.json', OUT / 'runtime-snapshot.json', OUT / 'metrics.md']: files[str(p)] = sha(p)
    atomic(OUT / 'review.json', dict(at=at, requirements=requirements, source_files_sha256=files,
        previous_full_scope_review=str(PREVIOUS), previous_goal_turn_classification=read(PREVIOUS)['current_goal_turn_classification'],
        current_goal_turn_classification=classification, goal_achieved=False))
    audit_path = ROOT / 'reports/completion-audit.json'
    audit = read(audit_path)
    audit.update(at=at, updated_at=at, requirements=requirements, latest_full_scope_review=str(OUT / 'review.json'),
        latest_parallel_batched_reasoning_stage=str(OUT / 'complete.json'),
        next_batched_reasoning_stage=str(ROOT / 'src/slm_lab/parallel_batched_reasoning_stage_v2.py'),
        current_goal_turn_classification=classification, goal_achieved=False)
    atomic(audit_path, audit)
    entry = (f'Batched reasoning stage, **{at}**: seven own-small CPUchecks pass, including both four-replica matched profiles, '
        'correct source averaging, genuine empty tails and exact next-action/gradient/Adam/private-RNG durable recovery. '
        'Two actual late-recovery gaps are preserved and rejected before mutation inv2. '
        f"Original human chosen-response training:{progress['actual_updates']}/5362updates. "
        '**No new production RLupdates or quality gains; all ten final requirements remain incomplete.** '
        '[Evidence](reports/parallel-batched-reasoning-stage-v2-review/metrics.md), '
        '[full review](reports/parallel-batched-reasoning-stage-v2-review/review.json). Earlier entries are historical snapshots.\n\n')
    for name in ['README.md', 'RESEARCH.md']:
        path = ROOT / name
        path.write_text(entry + path.read_text())
    print(str(OUT / 'complete.json'))


if __name__ == '__main__': main()
