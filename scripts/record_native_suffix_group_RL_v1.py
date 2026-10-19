"""Close actual native-code gradient controls without claiming student gains."""
from copy import deepcopy
from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess

from record_visual_stage_and_first_repair_arm_v1 import atomic, read, sha

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/native-suffix-group-RL-v1-review'
PREVIOUS = ROOT / 'reports/native-code-oracle-and-gzip-recovery-v1-review/review.json'


def observe(identity):
    try:
        fields = Path('/proc', str(identity['pid']), 'stat').read_text().split(') ', 1)[1].split()
    except FileNotFoundError:
        return dict(owner=identity, live=False)
    live = fields[19] == identity['ticks'] and fields[0] not in ['Z', 'X']
    return dict(owner=identity, live=live, actual_ticks=fields[19], state=fields[0],
        nice=os.getpriority(os.PRIO_PROCESS, identity['pid']) if live else None,
        scheduler=os.sched_getscheduler(identity['pid']) if live else None)


def main():
    if OUT.exists(): raise ValueError('Preserve every earlier review')
    tests = ROOT / 'reports/batched-native-suffix-group-gradients-v1-tests.log'
    if '7 passed in 80.65s' not in tests.read_text(): raise ValueError('All seven actual connected controls must pass')
    proofs_root = ROOT / 'reports/batched-native-suffix-group-gradients-v1-pytest'
    paths = sorted(proofs_root.rglob('actual-native-source-group-gradient-control.json'))
    proofs = [read(p) for p in paths]
    if len(proofs) != 2 or {p['only_EOS_fixture'] for p in proofs} != {False, True}:
        raise ValueError('Actual mixed-reward and all-failure controls are required')
    for proof in proofs:
        if (proof['actual_small_fixture_parameters'] != 267334 or not proof['deliberately_initialized_source_specific_fixture']
                or proof['actual_all_original_CE_targets'] != 68 or proof['CPU_mechanics_horizon'] != 5
                or proof['complete_original_native_horizon'] != 512 or not proof['original_base_and_critic_unchanged']
                or not proof['exact_paired_private_actions_probabilities_execution_outcomes_and_full_CE']
                or proof['production_updates'] != 0 or proof['unassisted_coding_score']):
            raise ValueError('Exact source-specific small fixture and unconditional whole CE boundaries must remain explicit')
        expected = [0.] * 8 if proof['only_EOS_fixture'] else [0., 0., 1., 1., 1., 1., 1., 1.]
        if proof['actual_native_rewards'] != expected: raise ValueError('Retain every actual reward including all failures')
    recovery_path = next(proofs_root.rglob('actual-next-native-code-group-recovery.json'))
    recovery = read(recovery_path)
    if (not recovery['exact_actual_actions_verdict_semantics_logps_unclipped_gradients_weights_Adam_private_RNG']
            or recovery['actual_physical_test_Adam_calls'] != 3 or recovery['actual_logical_trial_updates'] != 2
            or not recovery['every_actual_execution_timing_and_worker_observation_retained']
            or not recovery['timing_and_thread_ID_diagnostics_not_claimed_equal'] or recovery['production_updates'] != 0):
        raise ValueError('The actual next update must replay all code semantics/gradients/state/RNG, with timings retained separately')
    scheduler_path = ROOT / 'reports/private-source-execution-thread-scheduler-v1.json'
    scheduler = read(scheduler_path)
    if (not scheduler['complete'] or scheduler['actual_normal_main_before'] != scheduler['actual_normal_main_after']
            or not all(w['actual_nice'] == 19 and w['actual_scheduler'] == 5
                for w in scheduler['actual_private_source_worker_observations'])):
        raise ValueError('Fresh normal parent isolation and actual idle source thread scheduling must hold')
    qualification = ROOT / 'reports/native-suffix-CPU-mechanics-qualification-v1/closed-qualified-controls.json'
    source_proof = read(qualification)
    if (source_proof['source_position'] != 188 or source_proof['actual_source_attempts'] != 1
            or source_proof['scope'] != 'CPU-mechanics' or source_proof['complete_source_population_qualified']
            or source_proof['model_generations'] != 0 or source_proof['model_updates'] != 0):
        raise ValueError('The one-source control never admits the full training population')
    native_root = ROOT / 'reports/full-native-token-suffix-qualification-CPU-v1'
    native_plan = read(native_root / 'plan.json')
    native_owner = observe(native_plan['owner'])
    if (not native_owner['live'] or native_owner['nice'] != 19 or native_owner['scheduler'] != 5
            or (native_root / 'failure.json').exists() or native_plan['all_original_sources'] != 3721
            or native_plan['all_original_views'] != 14884 or native_plan['all_original_response_EOS_targets'] != 584218):
        raise ValueError('The actual full-source native-profile successor must retain its exact idle source scope')
    at = datetime.now(timezone.utc).isoformat()
    human_root = ROOT / 'reports/phase-human-preference-v8'
    human = read(human_root / 'chosen_CE-progress.json')
    workers = [dict(pid=151931, ticks='55253729'), dict(pid=151932, ticks='55253730'),
        dict(pid=151933, ticks='55253731'), dict(pid=151934, ticks='55253731')]
    live_workers = [observe(w) for w in workers]
    if not all(w['live'] for w in live_workers): raise ValueError('Inspect the exact original production worker transition')
    producer = ROOT / 'reports/full-public-token-suffix-qualification-gzip-recovery-CPU-v2'
    reader = ROOT / 'reports/full-public-token-suffix-readback-gzip-recovery-CPU-v2'
    producer_value = read(producer / ('complete.json' if (producer / 'complete.json').exists() else 'progress.json'))
    reader_value = read(reader / ('complete.json' if (reader / 'complete.json').exists() else 'progress.json'))
    batch = ROOT / 'reports/full-corrected-multimodal-batch-readback-CPU-v1'
    cache = ROOT / 'reports/shared-pretrained-cache-reclamation-v1'
    snapshot = dict(at=at, original_human_first_arm=human, actual_original_neural_owners=live_workers,
        actual_GPU_rows=subprocess.check_output(['nvidia-smi', '--query-gpu=index,name,utilization.gpu,memory.used,memory.total',
            '--format=csv,noheader'], text=True), actual_disk_rows=subprocess.check_output(['df', '-B1', str(ROOT)], text=True),
        full_original_source_producer=producer_value, full_original_fresh_reader=reader_value,
        actual_native_profile_owner=native_owner, native_profile_observation=read(native_root / 'progress.json'),
        full_visual_batch_observation=read(batch / ('complete.json' if (batch / 'complete.json').exists() else 'progress.json')),
        full_visual_source_admission_exists=(ROOT / 'reports/full-multimodal-corrected-source-index-CPU-v1/source-training-admission.json').exists(),
        shared_pretrained_cache_user_reply_receipt_exists=(cache / 'user-approval.json').exists(),
        shared_pretrained_cache_deletion_applied=(cache / 'complete.json').exists(),
        no_existing_neural_worker_signalled_reprioritized_restarted_or_patched=True,
        no_new_GPU_context_job_or_lease=True, all_mutable_observations_snapshotted_not_source_bound=True)
    prior = read(PREVIOUS)
    requirements = deepcopy(prior['requirements'])
    evidence = {
        'own_architecture': 'Own scratch-origin text/image/word foundation remains. The new code-gradient kernel adds no learned parameters and uses the own starting actor as its fixed reference. No outside pretrained weights or live neural teacher were loaded.',
        'public_data': 'All17original families,3721whole human programs and14884native suffix views remain. Complete per-level masks preserve2287522context tokens/584218CE-EOS targets. Original source recovery and its entire18605fresh controls remain required. A new native-profile source-only successor waits for that exact full reader; every failed/trivial/abstaining view retains its CEanchor. It runs two actual native gold/prefix repeats for each eligible view and retains the two old legacy controls with their original provenance. No source or production RLadmission is inferred from this queue.',
        'under_1b_parameters': 'The existing exported dense student has984399690values and research actor/critic999706442. The actual new mechanics fixture has267334parameters. An own fixed reference has separately declared training-only private values, excluded from export. Full-size GPUinventory/capacity remains required for the new kernel.',
        'text_and_images': 'Independent physical/native pixel readback is complete for195121paths/302627views/91585whole-word views,0end-file-state differences. The complete4733step batch reader and separate source admission continue. The prepared complete visual GPUcontinuation remains unlaunched; no new visual quality result is established.',
        'human_feedback': f'The original four-L40chosen-response arm is{human["actual_updates"]}/5362updates. Each original chosen_CE/human_DPOarm retains85786human-ranked pairs. These include other-model texts ranked by public human annotators; they are not fresh judgments on this model. Machine execution rewards in the new kernel are not human RLHF.',
        'serious_full_model_training': 'The full4447update public epoch and all3x306GPUrepair updates remain complete and source/state verified. The original human stage continues on all4L40s. New code-gradient checks perform only5physical small-fixture Adamcalls(4logical test updates); they perform0production updates. Full new-profile source readback/admission, four-rank stage, actual full-size capacity/recovery/training/export and quality remain.',
        'high_end_llm_capability': 'NOT ACHIEVED. No evidence supports GPT6-class or excellent all-round capability. Source-specific deliberately initialized small-fixture rewards and lower token loss are not production-model accuracy.',
        'broad_independent_benchmarks': 'The completed local broad producer scores remain weak: generated knowledge2/1/2/2of56, math0/16, OCR0/20, objects0/15, photo retrieval1/32I2Tand2/32T2I. No new independent broad forward or gain is claimed. All original16961QA/3088AI2Dx3/600MBPP/4642human/3429wordx3and9final endpoints remain required.',
        'gpu_runtime': 'Seven connected own-small code-gradient checks pass. Eight real private full-vocabulary samples receive raw native execution verdicts while unconditional full-solution CEbackprop runs. Actual intervals overlap; this is not a throughput comparison. A fresh normal nice0parent stays unchanged while its newly created scoring threads become nice19/SCHED_IDLE. Missing verdicts abstain from the policy reward; actual NaNrescore/worker errors retain all sampled outcomes and clear partial gradients. Exact next actions/verdict semantics/logps/unclipped gradients/weights/Adam/private RNGreplay passes. Actual full-size CUDA equivalence, timing, four-rank recovery and production RLremain unproved. The existing human run retains the exclusive GPUlease.',
        'any_code_correctness_and_efficiency': 'Completed coding-study endpoints remain14/21/30/25syntax of48but1/48functional at all four endpoints. New mixed-reward fixture results0,0,1,1,1,1,1,1and all-EOS0x8come from a deliberately programmed five-token transition chain with63public reference tokens and a5token CPUmechanics horizon; the original production budget stays512and all68original CE/EOS targets remain. These are controlled mechanics, not unassisted coding or a full-model improvement.'}
    if set(evidence) != set(requirements): raise ValueError('Retain all ten final requirements')
    for name, value in requirements.items():
        if value['status'] != 'incomplete_for_requested_final_model': raise ValueError('No final requirement is complete')
        value['evidence'], value['previous_evidence'] = evidence[name], str(PREVIOUS)
    names = ['src/slm_lab/qualified_native_token_suffix_episode_v1.py', 'src/slm_lab/batched_native_suffix_group_gradients_v1.py',
        'src/slm_lab/source_execution_threads_v1.py', 'tests/test_batched_native_suffix_group_gradients_v1.py',
        'scripts/run_batched_native_suffix_group_gradients_v1_tests.py', 'scripts/qualify_native_suffix_CPU_mechanics_v1.py',
        'scripts/probe_private_execution_thread_scheduler_v1.py', 'scripts/qualify_full_native_token_suffixes_CPU_v1.py',
        'reports/native-suffix-mechanics-candidates-v1.json', 'reports/native-suffix-mechanics-candidates-v2.json',
        'reports/native-suffix-CPU-mechanics-qualification-v1/all-attempted-source-controls.jsonl',
        'reports/private-source-execution-thread-scheduler-v1.log',
        'reports/full-native-token-suffix-qualification-v1-launch-readback.json',
        'reports/full-native-token-suffix-qualification-v1-initial-launch-observation-failure.json']
    closed = [Path(__file__), PREVIOUS, tests, qualification, scheduler_path, recovery_path, *paths,
        native_root / 'plan.json', *[ROOT / name for name in names]]
    files = {str(p): sha(p) for p in closed}
    classification = 'progress:7actual own-small native code-gradient/recovery/failure controls pass; full original CE remains despite failed policy samples; exact native rewards affect gradients; one idle full-source native-profile successor queued;0new production RLupdates or quality gain'
    OUT.mkdir()
    atomic(OUT / 'runtime-snapshot.json', snapshot)
    atomic(OUT / 'complete.json', dict(complete=True, at=at, source_files_sha256=files,
        actual_connected_native_code_gradient_controls=7, actual_small_fixture_parameters=267334,
        actual_source_position=188, actual_original_CE_EOS_targets=68, actual_CPU_mechanics_horizon=5,
        unchanged_original_native_action_horizon=512, deliberately_source_initialized_fixture=True,
        actual_mixed_fixture_rewards=[0.,0.,1.,1.,1.,1.,1.,1.], actual_all_EOS_fixture_rewards=[0.] * 8,
        exact_next_actions_verdict_semantics_probabilities_unclipped_gradients_weights_Adam_private_RNG=True,
        actual_test_physical_Adam_calls=5, actual_test_logical_updates=4, actual_production_RL_updates=0,
        full_original3721sources14884views584218targets_retained=True,
        actual_full_native_source_owner=native_plan['owner'], full_native_source_profile_qualified=False,
        full_native_source_profile_independent_reader_and_admission_still_required=True,
        original_human_first_arm_updates=human['actual_updates'], original_human_first_arm_planned=5362,
        actual_full_size_GPUcapacity_speed_or_quality_not_established=True, new_model_quality_gain=False,
        all_ten_final_requirements_incomplete=True, goal_achieved=False))
    metrics = (
        '**Seven native-code gradient integration tests pass.** Eight real private full-vocabulary samples use the exact '
        'native program scorer. In the mixed fixture rewards are `[0,0,1,1,1,1,1,1]`; in the all-EOS fixture all eight fail. '
        'Execution rewards change actual RLgradients in the mixed case. All-zero advantages leave the matched gradients '
        'unchanged. Missing verdicts remain missing and abstain from the policy term. All68original reference/EOS targets '
        'remain active independently of policy outcomes.\n\n'
        '**This is a deliberately initialized267334parameter source-specific fixture**, using63public reference tokens '
        'and a5token CPUmechanics horizon. The complete original context is496tokens and its production horizon remains512. '
        'It is not an unassisted coding result or a production checkpoint. The test directly programs a five-token '
        'transition chain to make both real passing and failing samples observable. Five physical small-fixture Adamcalls '
        '(four logical test updates) occur; new production RLupdates are0.\n\n'
        'Checkpoint recovery reproduces the next actual actions, execution verdict semantics, cached/full/gradient logps, '
        'unclipped gradients, weights, Adam and private RNGexactly. Actual worker timings/IDs remain in both observations; '
        'they are not claimed equal. Nonfinite rescores and execution exceptions preserve every attempted sample/outcome '
        'and clear partial gradients. Fixed base values and unused critic remain unchanged.\n\n'
        'CPUscoring and full-CEbackprop intervals actually overlap in both fixture cases. A separate fresh normal-parent '
        'probe verifies only newly created source threads become idle/nice19; the parent remains normal/nice0. This is '
        '**no measured GPUspeedup**. Full-size CUDAcapacity, four-rank integration, production recovery/training/export '
        'and quality are still required.\n\n'
        'A new exact idle source-only successor waits for the entire existing3721source/14884view independent reader. '
        'For each legacy-eligible native view it retains two actual new gold/prefix runs and the two old legacy prefix '
        'observations with their original provenance. Failed/trivial/abstaining views remain CE-only; all original '
        '584218whole-solution targets remain. The new source profile is not yet qualified, independently read or admitted. '
        'The initial launcher sampled scheduling before its exec chain was known to complete; that observation failed. '
        'The same original child was then verified idle/nice19onCPU22without relaunch or worker changes.\n\n'
        f'The original four-L40human chosen-response arm is **{human["actual_updates"]}/5362updates**. All85786pairs per arm, '
        '17public families, complete reasoning/visual scopes and9final endpoints remain. Source recovery has observed '
        f'**{producer_value["actual_counts"]["complete_original_sources"]}/3721sources**; full fresh controls still remain. '
        'The complete visual batch/admission chain continues after the closed195121path pixel readback.\n\n'
        '**No production-model quality gain is established. Functional coding remains1/48at every completed coding-study '
        'endpoint. All ten requested final requirements remain incomplete.** The shared32.1GiBpretrained-cache removal '
        'still requires the pending actual user reply because those caches are outside this project.\n')
    (OUT / 'metrics.md').write_text(metrics)
    for p in [OUT / 'complete.json', OUT / 'runtime-snapshot.json', OUT / 'metrics.md']: files[str(p)] = sha(p)
    atomic(OUT / 'review.json', dict(at=at, requirements=requirements, source_files_sha256=files,
        previous_full_scope_review=str(PREVIOUS), current_goal_turn_classification=classification, goal_achieved=False))
    path = ROOT / 'reports/completion-audit.json'
    audit = read(path)
    audit.update(at=at, updated_at=at, requirements=requirements, latest_full_scope_review=str(OUT / 'review.json'),
        latest_native_code_suffix_group_RL=str(OUT / 'complete.json'),
        next_native_code_gradient_kernel=str(ROOT / 'src/slm_lab/batched_native_suffix_group_gradients_v1.py'),
        live_full_native_code_profile_source_directory=str(native_root),
        current_goal_turn_classification=classification, goal_achieved=False)
    atomic(path, audit)
    entry = (f'Native-code execution-guided gradient integration, **{at}**: seven own-small tests pass. Actual native rewards '
        'affect policy gradients while all original reference targets remain active. Exact next-action/verdict/gradient/'
        'Adam/private-RNG recovery and failed-group cleanup pass. The source-specific fixture is deliberately initialized; '
        '**no production-model accuracy gain is established**. A complete-source native-profile successor is queued. '
        f'Original human first arm: {human["actual_updates"]}/5362updates. **All ten final requirements remain incomplete.** '
        '[Evidence](reports/native-suffix-group-RL-v1-review/metrics.md), '
        '[full review](reports/native-suffix-group-RL-v1-review/review.json). Earlier entries are historical snapshots.\n\n')
    for name in ['README.md', 'RESEARCH.md']:
        p = ROOT / name
        p.write_text(entry + p.read_text())
    print(str(OUT / 'complete.json'))


if __name__ == '__main__': main()
