"""Record connected coding controller/export checks and unlaunched full scope."""
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import subprocess

from record_visual_stage_and_first_repair_arm_v1 import atomic, read, sha
from record_public_weak_code_admission_v1 import observe

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/mixed-coding-SFT-owned-readiness-v1-review'
PREVIOUS = ROOT / 'reports/mixed-coding-SFT-runtime-v1-review/review.json'
PROOFS = ROOT / 'reports/mixed-coding-SFT-training-and-export-v1-pytest/test_actual_four_replica_capac0'
PREP = ROOT / 'reports/mixed-coding-SFT-aligned-preparation-v1'
BINDINGS = ROOT / 'reports/mixed-coding-SFT-source-bindings-CPU-v1'


def main():
    if OUT.exists(): raise ValueError('Preserve all earlier closed reviews')
    logs = [ROOT / 'reports/mixed-coding-SFT-training-and-export-v1-tests.log', ROOT / 'reports/mixed-coding-SFT-owned-driver-v1-tests.log']
    if '1 passed' not in logs[0].read_text() or '8 passed' not in logs[1].read_text():
        raise ValueError('All actual connected training/recovery/native export and owned-driver checks must finish')
    rows = [read(PROOFS / (p + '-rank-' + str(r) + '.json')) for p in ['plain_tokens', 'balanced_sources'] for r in range(4)]
    for row in rows:
        cap, fresh, trained, recovered, native = [row[k] for k in ['actual_capacity', 'actual_fresh_readback', 'actual_complete_training', 'actual_recovered_training', 'actual_native_export']]
        if (row['actual_small_research_parameters'] != 267718 or row['actual_native_parameters'] != 267430
                or not all(row[k] for k in ['exact_complete_and_interrupted_recovered_final_state',
                    'actual_replayed_uncommitted_losses_and_gradients_equal',
                    'root_only_error_retains_both_observed_updates_and_old_durable_prefix',
                    'peer_bad_readback_stale_actor_orphan_directory_and_stale_pins_rejected_before_mutation',
                    'all_failed_and_recovery_journals_retained', 'CPU_fixture_does_not_admit_production'])
                or cap['actual_physical_optimizer_calls'] != 4 or fresh['new_optimizer_calls'] != 0
                or fresh['initial_actor_sha256'] == fresh['actual_actor_sha256']
                or trained['actual_new_optimizer_calls'] != 3 or trained['actual_complete_journal']['records'] != 3
                or trained['original_complete_training_budget']['source_presentations'] != 18
                or trained['original_complete_training_budget']['response_EOS_targets'] != 568
                or recovered['actual_attempt_started_at_update'] != 1 or recovered['actual_new_optimizer_calls'] != 2
                or any(trained[k] != recovered[k] for k in ['actual_final_actor_sha256', 'actual_final_Adam_sha256', 'actual_final_private_RNG_sha256'])
                or not all(native[k] for k in ['all_original_native_values_and_text_photo_word_cold_exact',
                    'source_model_and_private_RNG_unchanged', 'original_nonzero_dense_critic_preserved_and_unused_research_critic_excluded'])
                or row['production_updates'] != 0 or row['full_model_GPUcapacity_training_or_quality']):
            raise ValueError('Every actual source/update/private recovery/native cold observation must reconcile')
    if ((BINDINGS / 'failure.json').exists() or not read(BINDINGS / 'complete.json')['complete']
            or observe(read(BINDINGS / 'plan.json')['owner'])['live']):
        raise ValueError('The complete actual mixed-source binding snapshot owner must be terminal')
    bindings = read(BINDINGS / 'complete.json')
    if bindings['complete_source_population'] != 28028 or bindings['models_constructed'] != 0:
        raise ValueError('The actual full source snapshot contains no neural model')
    prep = read(PREP / 'plan.json')
    prep_owner = observe(prep['owner'])
    if (not prep_owner['live'] or prep_owner['nice'] != 19 or prep_owner['scheduler'] != 5
            or prep['full_training_budget'] != bindings['full_training_budget']
            or prep['actual_capacity_updates_per_profile'] != 53 or not prep['GPU_workers_never_launched_by_this_preparation']):
        raise ValueError('The exact new source-only preparer must be actually live and retain the complete declared scope')
    block_path = ROOT / 'reports/batched-reasoning-aligned-GPU-v1-known-optimizer-block-v1.json'
    block = read(block_path)
    if block['launch_permitted'] or sha(block['affected_unlaunched_driver']) != block['affected_driver_sha256']:
        raise ValueError('The preserved unlaunched old reasoning optimizer mismatch remains explicit')
    at = datetime.now(timezone.utc).isoformat()
    human = read(ROOT / 'reports/phase-human-preference-v8/chosen_CE-progress.json')
    owners = [observe(dict(pid=p, ticks=b)) for p, b in [(151931, '55253729'), (151932, '55253730'),
        (151933, '55253731'), (151934, '55253731')]]
    if not all(o['live'] for o in owners): raise ValueError('Inspect the exact original neural-owner transition')
    snapshot = dict(at=at, actual_original_human_progress=human, actual_original_GPUowners=owners,
        actual_new_source_preparer=prep_owner, actual_terminal_source_binding_owner=observe(read(BINDINGS / 'plan.json')['owner']),
        actual_GPU_rows=subprocess.check_output(['nvidia-smi', '--query-gpu=index,name,utilization.gpu,memory.used,memory.total', '--format=csv,noheader'], text=True),
        actual_disk_rows=subprocess.check_output(['df', '-B1', str(ROOT)], text=True),
        no_existing_neural_worker_signalled_reprioritized_restarted_or_changed=True,
        new_checks_CUDA_hidden_nice19_single_thread_idle_CPU22=True, new_GPUcontexts_or_leases=0,
        mutable_queues_progress_journals_and_registries_observed_not_bound=True,
        shared_cache_cleanup_actual_user_reply_pending=not (ROOT / 'reports/shared-pretrained-cache-reclamation-v1/user-approval.json').exists())
    requirements = deepcopy(read(PREVIOUS)['requirements'])
    requirements['gpu_runtime']['evidence'] += (
        ' A connected actual four-replica CPUcheck now covers both plain_tokens/balanced_sources from complete capacity '
        'through fresh capacity state readback, full declared small training, root-only failure, pinned recovery and separate '
        'native export/cold. Initial actor hashes prevent learned capacity factors being reused with fresh Adam. Nonempty '
        'uncommitted directories cannot be overwritten. A second eight-check suite consumes the new owned driver ordinary '
        'AdamWfactory in actual source-bound stage updates for both profiles, rejects old Adam before update, and exercises '
        'normal-scheduler/preparer/canonical/disk launch gates before any lease/model/output. The older unlaunched reasoning '
        'driver v1 has a verified Adam/AdamW mismatch and must not launch; its frozen source/preparer stay untouched and '
        'a versioned correction is still required. The new mixed coding driver uses explicit AdamW foreachFalse/fusedFalse.')
    requirements['serious_full_model_training']['evidence'] += (
        f' The unchanged original four-L40human first arm is{human["actual_updates"]}/5362. The new mixed coding continuation '
        'now has an owned full driver and an actually live idle source-only preparer waiting for the exact original complete '
        'canonical/source recipe. It preserves both matched complete5256step/84084presentation/21316734target schedules, '
        '53full-source capacity groups plus one replay perprofile, independent four-replica capacity readbacks, native export '
        'and all7full original suites plus all289reserved reasoning cases. No actual full-size mixed coding GPUcapacity, '
        'training, standalone export or new quality result has run. The fixed completed canonical102step machineDPO native '
        'is the future initializer, not a selected quality winner. All earlier17families/nine final endpoints remain.')
    requirements['own_architecture']['evidence'] += (
        ' Actual small coding actor export nonmutatingly folds into the single original native backbone, preserves the '
        'original nonzero dense critic and all image/word/auxiliary heads, excludes the unused zero research critic, and '
        'requires the canonical native position protocol. Every native value and complete text/photo/word cold output is '
        'bit exact. Hot-to-folded arithmetic differences are measured, never assumed equivalent generation. Public '
        'precomputed Qwen-generated weak SFTtext remains explicit; no outside weights or live teacher API are loaded.')
    requirements['public_data']['evidence'] += (
        ' A separate terminal data-only owner reconstructs the complete actual mixed source identity and all53intact '
        'capacity cases, then closes every consumed weak/original-code/procedural file binding. This snapshot opens '
        'no model/forward/GPUcontext. Full28028train sources, all native4096maximum contexts and complete EOS targets '
        'remain; synthetic generated tests stay forbidden as RLoracles.')
    for r in requirements.values():
        if r['status'] != 'incomplete_for_requested_final_model': raise ValueError('All ten final requirements remain incomplete')
        r['previous_evidence'] = str(PREVIOUS)
    if len(requirements) != 10: raise ValueError('Retain every original final requirement')
    names = ['src/slm_lab/mixed_coding_SFT_training_controller_v1.py', 'src/slm_lab/mixed_coding_SFT_native_export_v1.py',
        'scripts/train_mixed_coding_SFT_aligned_GPU_v1.py', 'scripts/prepare_mixed_coding_SFT_aligned_CPU_v1.py',
        'scripts/snapshot_mixed_coding_SFT_sources_CPU_v1.py', 'tests/test_mixed_coding_SFT_training_and_export_v1.py',
        'tests/test_mixed_coding_SFT_owned_driver_v1.py', 'scripts/run_mixed_coding_SFT_training_and_export_v1_tests.py',
        'scripts/run_mixed_coding_SFT_owned_driver_v1_tests.py', 'reports/mixed-coding-SFT-training-and-export-v1-owner.json',
        'reports/mixed-coding-SFT-source-bindings-CPU-v1.log', 'reports/mixed-coding-SFT-aligned-preparation-v1-launch.json']
    closed = [Path(__file__), PREVIOUS, *logs, block_path, PREP / 'plan.json', BINDINGS / 'plan.json', BINDINGS / 'complete.json',
        *[ROOT / n for n in names], *[p for p in PROOFS.rglob('*') if p.is_file() and p.suffix in ['.json', '.pt']],
        *[p for p in (ROOT / 'reports/mixed-coding-SFT-owned-driver-v1-pytest').rglob('*.json')]]
    files = {str(p): sha(p) for p in closed}
    classification = ('progress:owned complete mixed coding SFTdriver/source-only preparation;9actual connected '
        'smallCPU training/recovery/native-export/optimizer/launch-gate checks;terminal complete-source binding '
        'snapshot;old unlaunched reasoning optimizer mismatch preserved and flagged against launch;'
        '0new full-size GPUcapacity/production updates/quality gain')
    OUT.mkdir()
    atomic(OUT / 'runtime-snapshot.json', snapshot)
    atomic(OUT / 'complete.json', dict(complete=True, at=at, source_files_sha256=files, actual_meaningful_checks=9,
        actual_connected_profiles=2, actual_four_replica_Adam_calls_including_capacity_replay_and_failed_attempts=88,
        actual_driver_factory_small_AdamW_updates=2, actual_complete_source_bindings=bindings,
        actual_full_training_budget=prep['full_training_budget'], actual_new_source_preparer=prep_owner,
        actual_native_export_observations={r['profile']: r['actual_native_export'] for r in rows if r['rank'] == 0},
        actual_original_human_updates=human['actual_updates'], full_size_new_GPUcapacity=False,
        full_size_new_production_updates=0, new_model_quality_gain=False, all_ten_final_requirements_incomplete=True, goal_achieved=False))
    (OUT / 'metrics.md').write_text(f'''**The complete coding continuation is implemented and queued for source preparation; GPU training has not started.** Both `plain_tokens` and `balanced_sources` retain three complete epochs: 5,256 updates, 84,084 source presentations and 21,316,734 response/EOS targets per profile. Full capacity uses 53 intact source groups plus one recovery replay per profile. All seven original full evaluation suites and all 289 reserved reasoning cases remain.

**Nine actual checks pass.** Four small CPU replicas execute both profiles through capacity, fresh capacity readback, complete declared training, a root-only failure after the second Adam step, pinned recovery, standalone export and independent reload. Both failed journal observations survive, while the durable prefix remains at one update; recovery reproduces the next loss, gradient, weights, Adam and private RNG exactly. A separate eight-check suite uses the actual driver optimizer factory in real stage updates and checks launch rejection before any lease/model/output.

The connected test performs 88 small Adam calls across four replicas and two profiles, including capacity replay and interrupted attempts. These are implementation checks on 267,718 research / 267,430 native parameter fixtures, not full-model capacity or quality. Native reload preserves every parameter and text, photo and word output exactly, including the original nonzero dense critic. Hot-to-folded FP32 arithmetic differences are recorded; equal generation is not assumed.

**An older unlaunched reasoning driver has an optimizer mismatch:** it creates Adam while its stage requires AdamW. Its frozen source and live source preparer remain untouched. The separate diagnostic records that this version must not be launched until a corrected version is prepared. The new coding driver explicitly uses AdamW with `foreach=False` and `fused=False`; the test consumes that optimizer in real updates.

A terminal data-only snapshot reconstructs all 28,028 admitted train sources and the complete capacity plan, closing the actual consumed source file bindings. Public precomputed synthetic code text remains weak SFT data; generated tests are not an independent correctness or RL oracle. No outside pretrained weights or live teacher API are loaded.

The new idle source preparer is actually live and waits for the exact complete canonical pipeline and source recipe. A later fresh normal-scheduled process must own all four idle L40s, sufficient disk and memory, full GPU capacity/readback, complete training, export and assessment. No existing neural worker was changed. The original human first arm is **{human['actual_updates']}/5,362 updates**.

**No new full-size training or measured capability gain is established. Latest completed coding correctness remains 1/48; all ten final goal requirements remain incomplete.**
''')
    for p in [OUT / 'complete.json', OUT / 'runtime-snapshot.json', OUT / 'metrics.md']: files[str(p)] = sha(p)
    atomic(OUT / 'review.json', dict(at=at, requirements=requirements, source_files_sha256=files,
        previous_full_scope_review=str(PREVIOUS), current_goal_turn_classification=classification, goal_achieved=False))
    path = ROOT / 'reports/completion-audit.json'
    audit = read(path)
    audit.update(at=at, updated_at=at, requirements=requirements, latest_full_scope_review=str(OUT / 'review.json'),
        latest_mixed_coding_SFT_owned_readiness=str(OUT / 'complete.json'),
        next_owned_mixed_coding_SFT_GPU_driver=str(ROOT / 'scripts/train_mixed_coding_SFT_aligned_GPU_v1.py'),
        live_mixed_coding_SFT_source_preparer_directory=str(PREP), known_unlaunched_reasoning_optimizer_block=str(block_path),
        current_goal_turn_classification=classification, goal_achieved=False)
    atomic(path, audit)
    entry = (f'Owned coding SFT continuation, **{at}**: matched full 5,256-step schedules and all original full assessments are implemented; '
        'nine connected small CPU/controller/export/optimizer/launch checks pass. Complete source bindings are closed and the new idle source preparer is live. '
        'The older unlaunched reasoning driver has a documented Adam/AdamW mismatch requiring a corrected version before launch. '
        f'Original human first arm: {human["actual_updates"]}/5,362. **No new full-model GPU capacity, training or quality gain; all ten final requirements remain incomplete.** '
        '[Evidence](reports/mixed-coding-SFT-owned-readiness-v1-review/metrics.md), [review](reports/mixed-coding-SFT-owned-readiness-v1-review/review.json). Earlier entries are historical snapshots.\n\n')
    for name in ['README.md', 'RESEARCH.md']:
        p = ROOT / name
        p.write_text(entry + p.read_text())
    print(str(OUT / 'complete.json'))


if __name__ == '__main__': main()
