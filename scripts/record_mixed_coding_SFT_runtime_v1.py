"""Record actual complete source/CPU mechanics, never a production quality gain."""
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import subprocess

from record_visual_stage_and_first_repair_arm_v1 import atomic, read, sha
from record_public_weak_code_admission_v1 import observe

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/mixed-coding-SFT-runtime-v1-review'
PREVIOUS = ROOT / 'reports/public-weak-code-admission-v1-review/review.json'


def main():
    if OUT.exists(): raise ValueError('Preserve all earlier closed reviews')
    logs = [ROOT / 'reports' / n for n in ['mixed-coding-SFT-stage-v1-tests.log',
        'mixed-coding-SFT-capacity-and-SUM-v1-tests.log', 'mixed-coding-SFT-capacity-controller-v1-tests.log']]
    for p, expected in zip(logs, [5, 2, 1]):
        if str(expected) + ' passed' not in p.read_text() or ' failed' in p.read_text():
            raise ValueError('Every actual source, serial/distributed gradient, recovery and connected controller check must finish')
    folders = [ROOT / 'reports' / n for n in ['mixed-coding-SFT-stage-v1-pytest',
        'mixed-coding-SFT-capacity-and-SUM-v1-pytest', 'mixed-coding-SFT-capacity-controller-v1-pytest']]
    def reports(pattern): return [read(p) for folder in folders for p in sorted(folder.rglob(pattern))]
    full = reports('actual-complete-mixed-source-readback.json')
    if len(full) != 1 or full[0]['actual_one_epoch_budget']['source_presentations'] != 28028:
        raise ValueError('The actual complete mixed source population must reconcile')
    serial = reports('actual-serial-gradient-*.json')
    distributed = reports('*-serial-SUM-rank-*.json')
    recovery = reports('*_tokens-rank-*.json') + reports('balanced_sources-rank-*.json')
    controllers = reports('*-controller-rank-*.json')
    plans = reports('actual-complete-source-capacity-plan.json')
    if (len(serial) != 2 or len(distributed) != 8 or len(recovery) != 8 or len(controllers) != 8 or len(plans) != 1
            or any(r['actual_maximum_four_replica_vs_independent_serial_gradient_error'] >= 2e-6 for r in distributed)
            or any(not r['actual_next_update_weights_Adam_private_RNG_bit_exact'] for r in recovery)
            or any(not r['exact_next_losses_unclipped_gradients_weights_Adam_private_RNG']
                or r['full_size_GPU_capacity_passed'] or r['production_updates'] != 0
                or r['actual_Adam_states'] != 17 or r['actual_physical_optimizer_calls'] != 4 for r in controllers)):
        raise ValueError('All actual profile/rank/serial/recovery/controller observations must reconcile')
    plan = plans[0]
    if plan['complete_training_budget'] != full[0]['actual_three_epoch_budget'] or plan['full_size_capacity_passed']:
        raise ValueError('The capacity source plan is complete selection only, never an actual GPUcapacity proof')
    at = datetime.now(timezone.utc).isoformat()
    activity_path = ROOT / 'reports/original-human-GPU-activity-CPU-v1/complete.json'
    activity = read(activity_path)
    if activity.get('complete') is not True or activity['actual_sample_count'] != 16:
        raise ValueError('The bounded read-only GPU observation must finish before closed review')
    human = read(ROOT / 'reports/phase-human-preference-v8/chosen_CE-progress.json')
    owners = [observe(dict(pid=p, ticks=b)) for p, b in [(151931, '55253729'), (151932, '55253730'),
        (151933, '55253731'), (151934, '55253731')]]
    if not all(o['live'] for o in owners): raise ValueError('Inspect the exact original neural-owner transition before review')
    snapshot = dict(at=at, actual_original_human_progress=human, actual_original_GPUowners=owners,
        actual_GPU_rows=subprocess.check_output(['nvidia-smi', '--query-gpu=index,name,utilization.gpu,memory.used,memory.total', '--format=csv,noheader'], text=True),
        actual_disk_rows=subprocess.check_output(['df', '-B1', str(ROOT)], text=True),
        no_existing_neural_worker_signalled_reprioritized_restarted_or_changed=True,
        new_tests_CUDA_hidden_nice19_single_thread_idle_CPU22=True, new_GPUcontexts_or_leases=0,
        closed_short_GPU_activity_observation=str(activity_path), actual_short_GPU_activity_summary=activity['actual_GPU_summary'],
        mutable_progress_queues_journals_and_registries_observed_not_bound=True,
        shared_cache_cleanup_actual_user_reply_pending=not (ROOT / 'reports/shared-pretrained-cache-reclamation-v1/user-approval.json').exists())
    requirements = deepcopy(read(PREVIOUS)['requirements'])
    additions = dict(
        public_data=' The complete mixed coding SFTdata interface retains21739explicit weak synthetic public code programs,3721original human code programs and2568procedural reasoning train rows. All28028sources/all1752actual one-epoch native tensor batches are independently enumerated with complete question/response/EOS masks and padding. One epoch retains14090525contexts/7105578responseEOS targets; three epochs retain5256groups/84084presentations/21316734targets. Synthetic generated tests remain forbidden as RLoracles; original reserved roles and all earlier exclusions remain.',
        serious_full_model_training=' A new complete-response four-replica supervised stage and connected durable-capacity controller are implemented and checked only with explicit own-small267334/267718CPUreplicas. Plain token weighting is compared with positive source-family weights having nominal whole-epoch weighted target mass50/40/10weak/human/procedural. Per-batch normalization does not guarantee those epoch loss fractions. Every original target remains. No full-size mixed SFTcapacity, training, export or quality gain has occurred.',
        gpu_runtime=' Eight meaningful new source/mechanics checks pass: independent serial vs padded-microbatch and actual four-rank gradients for both profiles, exact next update/Adam/all private RNGs and separate cold reload, real one-source three-empty-rank control, peer-only nonfinite gradient rejection/clearing and corrupt peer-state rejection before mutation. Full source maximum-context/prompt/target and actual global/rank-shard worst batches are selected intact with4rank rotations, all three genuine12source epoch tails and a separate zero-gradient control. The actual full-size GPUcapacity and owned production driver remain necessary.',
        any_code_correctness_and_efficiency=' The larger mixed SFTstage is an experimental next training method, not a coding correctness gain. Latest completed coding comparison remains1/48at every original study endpoint; full algorithmic correctness, efficiency, broad language/OS/network code and general/visual held improvements remain unproved.',
        human_feedback=f' Latest unchanged original four-L40human first-arm progress:{human["actual_updates"]}/5362. Original85786human-ranked pairs perarm remain; the weak public synthetic coding source is separate from human feedback and own-on-policy RL. Full original human comparison and fresh own-output human judgments remain incomplete.')
    for k, evidence in additions.items(): requirements[k]['evidence'] += evidence
    requirements['gpu_runtime']['evidence'] += (
        f' A separate read-only16sample NVML/process observation lasts{activity["actual_observation_seconds"]:.2f}seconds. '
        'All four original exact neural owners remain live with unchanged scheduler/nice settings and about6GiBallocated each. '
        'Their short-window average activity is18.31–30.31percent. NVMLactivity is not direct tensor-core occupancy and no kernel '
        'profiler/causal throughput conclusion or measured improvement is claimed; the complete raw observations remain.')
    if len(requirements) != 10 or any(r['status'] != 'incomplete_for_requested_final_model' for r in requirements.values()):
        raise ValueError('All ten full goal requirements remain incomplete')
    for r in requirements.values(): r['previous_evidence'] = str(PREVIOUS)
    names = ['src/slm_lab/mixed_coding_SFT_data_v1.py', 'src/slm_lab/parallel_mixed_coding_SFT_stage_v1.py',
        'src/slm_lab/mixed_coding_SFT_capacity_cases_v1.py', 'src/slm_lab/mixed_coding_SFT_capacity_controller_v1.py',
        'tests/test_parallel_mixed_coding_SFT_stage_v1.py', 'tests/test_mixed_coding_SFT_capacity_and_SUM_v1.py',
        'tests/test_mixed_coding_SFT_capacity_controller_v1.py', 'scripts/run_mixed_coding_SFT_stage_v1_tests.py',
        'scripts/run_mixed_coding_SFT_capacity_and_SUM_v1_tests.py', 'scripts/run_mixed_coding_SFT_capacity_controller_v1_tests.py',
        'scripts/observe_original_GPU_activity_CPU_v1.py', 'reports/original-human-GPU-activity-CPU-v1.log',
        'reports/mixed-coding-SFT-tests-observed-owners-v1.json', 'reports/mixed-coding-SFT-controller-tests-owner-v1.json']
    closed = [Path(__file__), PREVIOUS, activity_path, *logs, *[ROOT / n for n in names],
        *[p for folder in folders for p in sorted(folder.rglob('*')) if p.is_file() and p.suffix in ['.json', '.pt']]]
    files = {str(p): sha(p) for p in closed}
    classification = ('progress:complete28028source mixed coding CE interface and matched plain/source-balanced stage;'
        '8source/own-smallCPU serial-and-four-replica SUM/recovery/cold/controller checks;full-source worst batches selected;'
        '0new full-size GPUcapacity/production updates/quality gain')
    OUT.mkdir()
    atomic(OUT / 'runtime-snapshot.json', snapshot)
    maximum = max(r['actual_maximum_four_replica_vs_independent_serial_gradient_error'] for r in distributed)
    atomic(OUT / 'complete.json', dict(complete=True, at=at, source_files_sha256=files,
        actual_meaningful_checks=8, actual_one_epoch_budget=full[0]['actual_one_epoch_budget'],
        actual_three_epoch_budget=full[0]['actual_three_epoch_budget'], actual_four_replica_gradient_max_absolute_error=maximum,
        actual_capacity_source_plan=plan, actual_connected_small_controller_profiles=2,
        actual_own_small_controller_Adam_calls_per_replica=8, actual_own_small_controller_all_replica_Adam_calls=32,
        actual_original_human_updates=human['actual_updates'], new_full_size_GPUcapacity=False,
        new_full_size_production_updates=0, new_model_quality_gain=False, all_ten_final_requirements_incomplete=True, goal_achieved=False))
    (OUT / 'metrics.md').write_text(f'''**The next coding SFT stage has complete data transport and tested gradient and recovery mechanics.** It combines all 21,739 admitted weak public synthetic code programs, 3,721 original public programs and 2,568 procedural reasoning sources: **28,028 train examples, 14,090,525 context tokens and 7,105,578 response/EOS targets per epoch**. Every one of the 1,752 native batches is checked. The complete three-epoch schedule contains 5,256 updates, 84,084 presentations and 21,316,734 targets. These are planned source counts, not completed production updates.

**Both plain and source-balanced whole-response losses are implemented.** All targets stay. Balancing gives positive relative weights to the smaller original-code and procedural families. Nominal whole-epoch weighted target mass is 50% weak synthetic code, 40% original public code and 10% procedural reasoning. Per-batch normalization does not guarantee those fractions of actual epoch loss. Public precomputed Qwen-generated text remains explicit weak SFT data; no outside weights, live teacher API or generated-test RL oracle is used.

**Eight actual source and small CPU model checks pass.** The largest observed four-replica versus independent serial gradient difference is {maximum:.12g}. Actual next losses, unclipped gradients, weights, Adam states and private RNGs replay exactly. A separate own actor reloads text, photo and word outputs exactly. A nonfinite forward on one peer clears partial gradients before Adam. Corrupt peer RNGs, source cursors, targets, Adam clocks and unused critics reject before mutation. Three empty ranks have no forward. The connected capacity controller performs 32 small Adam calls across two profiles and four replicas, including replay. These are implementation checks, not full-size GPU capacity or quality benchmarks.

The source selector retains {len(plan['selected_intact_original_batches'])} intact worst training batches and {len(plan['capacity_entries'])} four-rank capacity cases. They include the actual 12-source tails and a separate one-source control for empty ranks. Full 999,706,442 research / 984,399,690 native parameter GPU capacity, production training, standalone export and complete matched coding, general, vision and preference evaluation still must run. All earlier source scopes and nine final endpoints remain.

The unchanged original four-L40 human first arm is **{human['actual_updates']}/5,362 updates**. A separate 37-second read-only observation records 16 samples per GPU: average NVML activity was 18.31–30.31%, with about 6 GiB allocated each. This short observation does not measure tensor-core occupancy or establish the cause of idle time.

**No new full-size training or quality gain is established. Latest completed coding correctness remains 1/48 across the original coding study. Syntax rose from 14/48 to 30/48 in the semantic-repair variant, while overall functional correctness stayed flat. All ten final requirements remain incomplete.**
''')
    for p in [OUT / 'complete.json', OUT / 'runtime-snapshot.json', OUT / 'metrics.md']: files[str(p)] = sha(p)
    atomic(OUT / 'review.json', dict(at=at, requirements=requirements, source_files_sha256=files,
        previous_full_scope_review=str(PREVIOUS), current_goal_turn_classification=classification, goal_achieved=False))
    path = ROOT / 'reports/completion-audit.json'
    audit = read(path)
    audit.update(at=at, updated_at=at, requirements=requirements, latest_full_scope_review=str(OUT / 'review.json'),
        latest_mixed_coding_SFT_runtime=str(OUT / 'complete.json'), next_mixed_coding_SFT_stage=str(ROOT / names[1]),
        current_goal_turn_classification=classification, goal_achieved=False)
    atomic(path, audit)
    entry = (f'Complete mixed coding SFT mechanics, **{at}**: 28,028 complete train sources and 7,105,578 response/EOS targets per epoch; '
        'all 1,752 native batches checked. Plain and source-balanced four-replica gradients, durable next-update replay, peer faults and separate cold loads pass '
        'eight source and small CPU model checks. Full-source worst capacity batches are selected. '
        f'Original human first arm: {human["actual_updates"]}/5,362. **No new full-size GPU capacity, training or quality gain; all ten final requirements remain incomplete.** '
        '[Evidence](reports/mixed-coding-SFT-runtime-v1-review/metrics.md), [review](reports/mixed-coding-SFT-runtime-v1-review/review.json). Earlier entries are historical snapshots.\n\n')
    for name in ['README.md', 'RESEARCH.md']:
        p = ROOT / name
        p.write_text(entry + p.read_text())
    print(str(OUT / 'complete.json'))


if __name__ == '__main__': main()
