"""Close the actual launch/export/native controls and full pending objective."""
from copy import deepcopy
from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess

from record_visual_stage_and_first_repair_arm_v1 import atomic, read, sha

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/reasoning-aligned-GPU-readiness-v1-review'
PREVIOUS = ROOT / 'reports/batched-reasoning-training-controller-v1-review/review.json'


def observe(owner):
    try: fields = Path('/proc', str(owner['pid']), 'stat').read_text().split(') ', 1)[1].split()
    except FileNotFoundError: return dict(owner=owner, live=False)
    live = fields[19] == owner['ticks'] and fields[0] not in ['Z', 'X']
    return dict(owner=owner, live=live, state=fields[0], actual_ticks=fields[19],
        nice=os.getpriority(os.PRIO_PROCESS, owner['pid']) if live else None,
        scheduler=os.sched_getscheduler(owner['pid']) if live else None)


def main():
    if OUT.exists(): raise ValueError('Preserve all earlier complete or failed reviews')
    tests = ROOT / 'reports/reasoning-aligned-GPU-v3-tests.log'
    guards = ROOT / 'reports/reasoning-native-position-export-guard-v1-tests.log'
    if '9 passed' not in tests.read_text() or '2 passed' not in guards.read_text():
        raise ValueError('All11actual corrected source/native/export/position controls must pass')
    proofs_root = ROOT / 'reports/reasoning-aligned-GPU-v3-pytest'
    export_paths = sorted(proofs_root.rglob('actual-trained-reasoning-native-export.json'))
    if len(export_paths) != 2: raise ValueError('Both real matched trained-source export controls are required')
    for path in export_paths:
        p = read(path)
        if (p['actual_small_research_values'] != 267718 or p['actual_small_native_values'] != 267430
                or p['actual_completed_source_groups'] != 3 or not p['actual_all_native_values_and_text_photo_word_cold_exact']
                or not p['actual_original_dense_critic_preserved'] or p['original_dense_critic_was_nonzero']
                or not p['source_model_and_private_RNG_unchanged'] or not p['explicit_canonical_native_position_protocol']
                or not p['previous_uninstalled_CPU_capacity_position_protocol_not_relabelled'] or p['production_updates'] != 0):
            raise ValueError('Exact own-small trained-source exports and canonical native geometry must remain explicit')
    reserved_path = next(proofs_root.rglob('actual-complete-reserved-native-transport.json'))
    reserved = read(reserved_path)
    if (reserved['actual_original_reserved_sources'] != 289 or reserved['actual_task_families'] != 12
            or reserved['actual_native_greedy_neural_deliveries'] != 289 or not reserved['deliberately_initialized_EOS_only_fixture']
            or reserved['actual_fixture_correct'] != 0 or not reserved['every_original_source_freshly_regraded']
            or not reserved['all_actual_unchanged_whole_prompts_64or256budgets_and_true_EOS_retained']):
        raise ValueError('All real289small-fixture native transports and unchanged original budgets must reconcile')
    guard_paths = sorted((ROOT / 'reports/reasoning-native-position-export-guard-v1-pytest').rglob('actual-old-position-native-export-rejection.json'))
    if len(guard_paths) != 2 or any(not read(p)['actual_old_position_actor_rejected_before_fold_or_packaging'] for p in guard_paths):
        raise ValueError('Both actual old-position sources must reject before native packaging')
    prep = ROOT / 'reports/batched-reasoning-aligned-preparation-v1'
    prep_plan = read(prep / 'plan.json')
    prep_live = observe(prep_plan['owner'])
    if not prep_live['live'] or prep_live['nice'] != 19 or prep_live['scheduler'] != 5 or (prep / 'failure.json').exists():
        raise ValueError('The actual new idle source preparer must be live and unchanged')
    producer = ROOT / 'reports/full-public-token-suffix-qualification-gzip-recovery-CPU-v2'
    producer_done = read(producer / 'complete.json')
    stream = Path(producer_done['complete_control_file'])
    if (not producer_done['complete'] or producer_done['actual_counts']['complete_original_sources'] != 3721
            or producer_done['actual_counts']['prefix_control_runs'] != 14884
            or sha(stream) != producer_done['actual_all18605_runtime_controls_sha256']):
        raise ValueError('The entire closed original18605control stream must retain its actual bytes')
    at = datetime.now(timezone.utc).isoformat()
    human = read(ROOT / 'reports/phase-human-preference-v8/chosen_CE-progress.json')
    workers = [observe(dict(pid=pid, ticks=birth)) for pid, birth in [(151931, '55253729'), (151932, '55253730'),
        (151933, '55253731'), (151934, '55253731')]]
    if not all(p['live'] for p in workers): raise ValueError('Inspect the exact original neural-worker transition')
    reader = ROOT / 'reports/full-public-token-suffix-readback-gzip-recovery-CPU-v2'
    native = ROOT / 'reports/full-native-token-suffix-qualification-CPU-v1'
    snapshot = dict(at=at, actual_original_human_progress=human, actual_original_GPUowners=workers,
        actual_original_source_producer=observe(producer_done['owner']), complete_original_source_counts=producer_done['actual_counts'],
        actual_original_fresh_reader=observe(read(reader / 'plan.json')['owner']),
        actual_original_fresh_reader_observation=read(reader / ('complete.json' if (reader / 'complete.json').exists() else 'progress.json')),
        actual_native_profile_source_owner=observe(read(native / 'plan.json')['owner']),
        actual_native_profile_source_observation=read(native / 'progress.json'), actual_new_source_preparer=prep_live,
        actual_new_source_preparer_observation=read(prep / 'progress.json'),
        actual_GPU_rows=subprocess.check_output(['nvidia-smi', '--query-gpu=index,name,utilization.gpu,memory.used,memory.total',
            '--format=csv,noheader'], text=True), actual_disk_rows=subprocess.check_output(['df', '-B1', str(ROOT)], text=True),
        new_GPUdriver_output_exists=(ROOT / 'reports/batched-reasoning-aligned-GPU-v1').exists(),
        new_GPUdriver_checkpoint_output_exists=(ROOT / 'checkpoints/batched-reasoning-aligned-GPU-v1').exists(),
        mutable_registries_progress_and_live_observation_journals_snapshotted_not_bound=True,
        no_existing_neural_worker_changed_or_new_GPUcontext_job_or_lease=True,
        shared_cache_actual_user_reply_pending=not (ROOT / 'reports/shared-pretrained-cache-reclamation-v1/user-approval.json').exists())
    prior = read(PREVIOUS)
    requirements = deepcopy(prior['requirements'])
    requirements['own_architecture']['evidence'] += (
        ' A new complete-source reasoning export merges only the own learned actor into a new complete984Mnative, '
        'adds0parameters and preserves original image/word/auxiliary heads and the original dense critic. Actual '
        'small exports use267718research/267430native values; the original dense critic is zero in this fixture. '
        'No external weights/teacher/API is introduced. FP32actor-to-fold arithmetic differs and is recorded, not '
        'assumed identical to hot policy generation or quality.')
    requirements['gpu_runtime']['evidence'] = (
        'A complete owned normal nice0four-L40driver is implemented and UNLAUNCHED. Both matched CE/RLarms use the '
        'same completed fixed aligned canonical native, all2568train sources/12tasks/3epochs,8private attempts per '
        'source and full64/256horizons. Each arm requires actual109global capacity updates(110physical including '
        'replay), a fresh four-replica full-state/cold readback and all1926production updates, followed by complete '
        'native export/cold and all7original full assessment suites plus all289reserved reasoning cases. All '
        'original canonical/source scopes,9final endpoints and exclusive ownership remain required. Eleven actual '
        'own-small source/export/native/launch checks pass, including actual saved trained parts from both profiles, '
        'complete parameter and text/photo/word cold equality, actual289greedy EOS-only fixture deliveries and '
        'fresh strict regrading,4real launch-entry rejection controls, source-only import and exact full recipe. '
        'Two extra actual source controls reject the observed old geometry before export. No actual full-size '
        'CUDAcapacity/runtime/production/export/quality is inferred. A new idle source preparer cannot launch GPUwork.')
    requirements['public_data']['evidence'] += (
        ' The full original code-source producer now completes all3721sources/14884prefix controls; all18605closed '
        'reference/prefix observations retain their recorded SHA.3710gold references earn1;11do not.448trivial '
        'prefixes remain rejected;12955views are only producer proposals. Every original CEtarget and failure '
        'remains. Full independent fresh controls and separate native-profile qualification/admission remain pending.')
    requirements['text_and_images']['evidence'] += (
        ' An actual small native export exposes a geometry mismatch: its uninstalled source word interpolation '
        'does not cold-match the canonical deterministic standalone loader. The original failed v2logs/checkpoints '
        'remain. Exportv2requires the exact canonical vision/word interpolation methods; two old-profile sources '
        'reject before packaging. The corrected small fixture explicitly installs that original canonical native '
        'protocol and text/photo/FP32word cold then matches bit exactly. This does not waive previous numeric gates '
        'or establish full-size visual capacity/quality. The initial false nonzero-critic fixture assertion also '
        'remains preserved; the real source fixture critic is zero.')
    requirements['human_feedback']['evidence'] = (
        f'The unchanged canonical human chosen-response stage is{human["actual_updates"]}/5362updates on all4L40s. '
        'Each original human arm retains85786pairs. Public other-model responses ranked by humans remain distinct '
        'from fresh human judgments on this student and from new machine-verifier rewards. Full original human '
        'completion and held quality remain pending.')
    requirements['serious_full_model_training']['evidence'] += (
        f' Latest original human first-arm observation:{human["actual_updates"]}/5362. New GPUreasoning driver is '
        'implemented but unlaunched; its source-only preparer waits for the exact full original recipe. New production '
        'updates and model-quality gains remain0. Full complete-source training/native export/full candidate '
        'assessments are retained requirements, not completed by small-fixture checks.')
    for value in requirements.values():
        if value['status'] != 'incomplete_for_requested_final_model': raise ValueError('All ten final requirements remain incomplete')
        value['previous_evidence'] = str(PREVIOUS)
    if len(requirements) != 10: raise ValueError('Retain all ten final requirements')
    names = ['scripts/train_batched_reasoning_aligned_GPU_v1.py', 'scripts/prepare_batched_reasoning_aligned_CPU_v1.py',
        'src/slm_lab/reasoning_native_export_v1.py', 'src/slm_lab/reasoning_native_export_v2.py',
        'src/slm_lab/reasoning_reserved_shards_v1.py', 'tests/test_reasoning_aligned_GPU_v1.py',
        'tests/test_reasoning_aligned_GPU_v2.py', 'tests/test_reasoning_aligned_GPU_v3.py',
        'tests/test_reasoning_native_position_export_guard_v1.py', 'scripts/run_reasoning_aligned_GPU_v1_tests.py',
        'scripts/run_reasoning_aligned_GPU_v2_export_tests.py', 'scripts/run_reasoning_aligned_GPU_v3_tests.py',
        'scripts/run_reasoning_native_position_export_guard_v1_tests.py', 'reports/reasoning-aligned-GPU-v1-tests.log',
        'reports/reasoning-aligned-GPU-v2-export-tests.log', 'reports/batched-reasoning-aligned-preparation-v1-launch.json']
    files = {str(p): sha(p) for p in [Path(__file__), PREVIOUS, tests, guards, *export_paths, reserved_path, *guard_paths,
        prep / 'plan.json', producer / 'complete.json', stream, *[ROOT / n for n in names]]}
    classification = ('progress:complete owned matched full-source reasoning GPUdriver/export/evaluation implemented; '
        '11own-small/source checks pass; actual canonical-position cold mismatch and failed assumptions preserved and '
        'gated; original3721source control producer completes; idle non-neural preparation queued;0production updates or quality gain')
    OUT.mkdir()
    atomic(OUT / 'runtime-snapshot.json', snapshot)
    atomic(OUT / 'complete.json', dict(complete=True, at=at, source_files_sha256=files,
        actual_corrected_native_source_launch_controls=9, actual_old_geometry_export_rejection_controls=2,
        actual_small_trained_export_profiles=['anchor_CE', 'anchor_RL'], actual_small_research_values=267718,
        actual_small_native_values=267430, actual_complete_reserved_fixture_deliveries=289,
        reserved_fixture_deliberately_EOS_only_not_a_quality_score=True,
        all_actual_small_native_values_text_photo_word_cold_exact=True,
        both_original_failed_export_test_versions_preserved=True,
        old_uninstalled_geometry_rejected_before_native_packaging=True,
        full_owned_GPUdriver_implemented_not_launched=True, actual_source_only_preparer=prep_plan['owner'],
        all3721original_source18605controls_producer_complete=True,
        actual_original_source_counts=producer_done['actual_counts'], entire_independent_source_reader_still_required=True,
        actual_original_human_updates=human['actual_updates'], actual_new_production_updates=0,
        actual_new_model_quality_gain=False, all_ten_final_requirements_incomplete=True, goal_achieved=False))
    metrics = (
        '**The complete matched reasoning GPUdriver is implemented and unlaunched.** Both arms begin at the same '
        'fixed fully completed aligned own native. Each retains all2568sources/12tasks for3epochs:7704presentations, '
        '57240answer/EOS targets,1926updates and full64/256native action budgets,8private attempts per source. '
        'Each requires full109step capacity/110physical updates, separate fresh4replica state/cold readback, complete '
        'production training, complete984Mnative export/cold and all original7full assessment suites. All289source-reserved '
        'reasoning examples are also evaluated at baseline and both candidates with actual raw native actions. '
        'The old single-source RL/warmup and9canonical endpoints remain separate requirements.\n\n'
        '**Eleven actual source/native/export checks pass:**9corrected controls and2old-position rejection controls. '
        'Both prior trained267718parameter CPUactor fixtures export267430native values with all values and text/photo/'
        'FP32word outputs cold-exact. Their real original dense critic is zero and remains. Merging changes arithmetic '
        'order; actual small hot-to-fold errors are retained and hot generation/quality parity is not assumed. '
        'The native generator actually processes all289reserved inputs in an explicitly programmed EOS-only small '
        'fixture, with all12tasks/full64or256budgets/original prompts retained and independently regraded. Its0correct '
        'score is a transport control, not student capability. Four launch-gate controls reject inherited idle '
        'scheduling/live preparation/incomplete original pipeline/disk shortfall before lease or model output.\n\n'
        '**Actual export failures remain.** The first test assumed a nonzero original dense critic; the actual fixture '
        'has zero. The next real export reveals non-bit-exact word outputs because an old uninstalled position operator '
        'differs from the canonical deterministic standalone operator. Exportv2requires the exact original canonical '
        'native geometry before packaging, and two actual old-profile sources reject unchanged. Corrected fixtures '
        'explicitly install that canonical protocol and preserve exact equality. No numeric tolerance was widened and '
        'the older capacity geometry is not retroactively relabelled. Both failed versions/logs/artifacts remain.\n\n'
        '**The original full code-source producer completes:**3721references and14884prefix controls; its complete gzip '
        'SHA is verified.3710gold references earn1,11do not;448trivial prefixes remain rejected.12955views are producer '
        'proposals only. Every source/whole CEtarget/failure remains, and the full independent fresh reader plus separate '
        'native-profile qualification/admission are still required.\n\n'
        f'Original human first arm: **{human["actual_updates"]}/5362updates**. The new preparation owner is actually live '
        'at nice19/SCHED_IDLE and cannot import neural runtime or launch GPUwork. A later fresh normal nice0parent must '
        'hold the exclusive GPUlease and fit both complete candidates, private generations, full evaluations and40GiBreserve. '
        'Existing neural workers remain unchanged.\n\n'
        '**New production updates:0; no model-quality gain is established. Latest completed coding remains1/48at '
        'every original coding-study endpoint. All ten requested final requirements remain incomplete.** Shared cache '
        'reclamation still awaits the earlier actual user reply; no outside pretrained weights were used or deleted.\n')
    (OUT / 'metrics.md').write_text(metrics)
    for p in [OUT / 'complete.json', OUT / 'runtime-snapshot.json', OUT / 'metrics.md']: files[str(p)] = sha(p)
    atomic(OUT / 'review.json', dict(at=at, requirements=requirements, source_files_sha256=files,
        previous_full_scope_review=str(PREVIOUS), current_goal_turn_classification=classification, goal_achieved=False))
    audit_path = ROOT / 'reports/completion-audit.json'
    audit = read(audit_path)
    audit.update(at=at, updated_at=at, requirements=requirements, latest_full_scope_review=str(OUT / 'review.json'),
        latest_owned_reasoning_GPU_readiness=str(OUT / 'complete.json'),
        next_owned_reasoning_GPUdriver=str(ROOT / 'scripts/train_batched_reasoning_aligned_GPU_v1.py'),
        live_owned_reasoning_source_preparation_directory=str(prep), current_goal_turn_classification=classification, goal_achieved=False)
    atomic(audit_path, audit)
    entry = (f'Owned full-source reasoning experiment ready for source preparation, **{at}**:11own-small/native/source '
        'checks pass; full matched CE/RLdriver retains complete7704source/57240target training and full original '
        'assessments. The actual native geometry mismatch is preserved and blocked before export; no full-size GPU '
        'run is launched. All3721original code-source controls finish; the complete fresh reader remains. '
        f'Original human first arm: {human["actual_updates"]}/5362updates. **No production accuracy gain; all ten final '
        'requirements remain incomplete.** [Evidence](reports/reasoning-aligned-GPU-readiness-v1-review/metrics.md), '
        '[full review](reports/reasoning-aligned-GPU-readiness-v1-review/review.json). Earlier entries are historical snapshots.\n\n')
    for name in ['README.md', 'RESEARCH.md']:
        p = ROOT / name
        p.write_text(entry + p.read_text())
    print(str(OUT / 'complete.json'))


if __name__ == '__main__': main()
