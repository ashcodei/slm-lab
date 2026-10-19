"""Close V2 source-validation/update evidence without changing the project README."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/native-suffix-validation-v2-review'
TESTS = ROOT / 'reports/native-suffix-validation-stage-v2-pytest'
LOG = ROOT / 'reports/native-suffix-validation-stage-v2-tests.log'
QUEUE = ROOT / 'reports/native-suffix-validation-four-replicas-v2-queue'
CONTROL = ROOT / 'reports/native-suffix-CPU-mechanics-qualification-v1/closed-qualified-controls.json'


def read(path): return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as stream: return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False); stream.write('\n')


def process(pid, birth):
    try:
        f = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
    except FileNotFoundError:
        return dict(pid=pid, expected_ticks=birth, terminal=True)
    return dict(pid=pid, ticks=f[19], state=f[0], same_birth=f[19] == birth,
        terminal=f[19] != birth or f[0] in ['Z', 'X'], nice=int(f[16]), scheduler=os.sched_getscheduler(pid))


def main():
    os.nice(19); os.sched_setaffinity(0, {22}); os.sched_setscheduler(0, os.SCHED_IDLE, os.sched_param(0))
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or '12 passed' not in LOG.read_text():
        raise ValueError('All twelve actual V2 controls must pass in the CUDA-hidden owned CPU trial')
    if not process(180812, '56636391')['terminal']:
        raise ValueError('The exact V2 test process must be terminal before closing its evidence')
    costs = sorted({p.resolve() for p in TESTS.glob('test_actual_recorded_full_nati*/actual-recorded-source-validation-cost.json')})
    learned = sorted({p.resolve() for p in TESTS.glob('test_actual_new_stage_updates*/actual-v2-native-recovery-*.json')})
    if len(costs) != 1 or len(learned) != 2:
        raise ValueError('The actual paired source-validation measurement and both neural recovery profiles are required')
    cost = read(costs[0]); rows = [read(p) for p in learned]
    if (cost['observed_read_character_reduction'] <= .5 or {r['profile'] for r in rows} != {'anchor_CE', 'anchor_execution'}
            or any(r['actual_small_parameters'] != 267334 or r['physical_optimizer_calls'] != 3
                or r['original_human_sources'] != 1 or r['original_response_EOS_targets'] != 136
                or not r['exact_next_actions_actual_worker_semantics_logps_gradients_weights_AdamW'] for r in rows)):
        raise ValueError('The exact observed small CPU scope and every real next-update control must remain explicit')
    queue = read(QUEUE / 'plan.json')
    queue_state = process(181290, '56684248')
    if queue_state['terminal'] or (QUEUE / 'failure.json').exists():
        raise ValueError('The exact full four-replica successor must remain successfully waiting or running')
    from slm_lab.local_source_closure_v2 import python_sources
    seeds = [Path(__file__), ROOT / 'src/slm_lab/native_suffix_validation_stage_v2.py',
        ROOT / 'tests/test_native_suffix_validation_stage_v2.py', ROOT / 'scripts/run_native_suffix_validation_stage_v2_tests.py']
    files = {str(p): sha(p) for p in [*python_sources(ROOT, seeds), CONTROL, LOG, *costs, *learned, QUEUE / 'plan.json']}
    for mapping in [queue['source_files_sha256'], read(CONTROL)['source_files_sha256']]:
        for p, h in mapping.items():
            if p in files and files[p] != h: raise ValueError('Conflicting original source/proof identities')
            files[p] = h
    for p, h in files.items():
        if sha(p) != h: raise ValueError('An actual original source/runtime/proof byte differs: ' + p)
    audit_path = ROOT / 'reports/completion-audit.json'; audit = read(audit_path)
    if len(audit['requirements']) != 10 or any(r['status'] != 'incomplete_for_requested_final_model' for r in audit['requirements'].values()):
        raise ValueError('Every original final capability requirement remains incomplete')
    neural = [process(pid, birth) for pid, birth in [(151931, '55253729'), (151932, '55253730'),
        (151933, '55253731'), (151934, '55253731')]]
    if any(p['terminal'] or p['nice'] != 19 or p['scheduler'] != os.SCHED_IDLE for p in neural):
        raise ValueError('The original four exact neural workers and scheduling must remain unchanged')
    progress = read(ROOT / 'reports/phase-human-preference-v8/chosen_CE-progress.json')
    old = process(178844, '56558134')
    old['full_original_V1_trial_passed'] = old['terminal'] and '9 passed' in (ROOT / 'reports/parallel-native-suffix-stage-v1-tests.log').read_text()
    at = datetime.now(timezone.utc).isoformat()
    runtime = dict(at=at, original_neural_workers=neural, original_native_CPU_trial=old,
        actual_full_V2_successor=queue_state, human_CE_actual_updates=progress['actual_updates'],
        human_CE_planned_updates=progress['planned_updates'], observed_free_bytes=shutil.disk_usage(ROOT).free,
        full_native_source_data_admitted=(ROOT / 'data/native-human-code-suffix-v1/admission.json').exists(),
        live_state_is_snapshot_only_not_a_source_binding=True)
    review = dict(complete=True, at=at, previous_goal_turn_classification='progress',
        previous_turn_evidence='Created and pushed the private user-owned repository, shortened README, and measured actual source-validator read reduction.',
        this_goal_turn_classification='progress', source_files_sha256=files, independently_verified_bindings=len(files),
        actual_tests=12, actual_source_validation_comparison=cost, actual_small_CPU_optimizer_calls=6,
        actual_small_parameters=267334, actual_full_vocabulary=32768, actual_group_size=8,
        explicit_fixture_action_horizon=5, original_native_action_horizon_unchanged=512,
        one_repeated_original_human_program_not_broad_independent_learning=True,
        actual_both_profiles_next_native_actions_execution_semantics_gradients_weights_and_AdamW_replay=True,
        real_additional_source_same_size_same_timestamp_mutation_rejected=True,
        independent_thread_cannot_borrow_checked_source_context=True,
        all_source_control_masks_program_bytes_native_EOS_and_abstentions_retained=True,
        original_3721_human_sources_four_levels_44652_presentations_preserved=True,
        actual_full_V2_four_replica_trial_still_pending=True, current_runtime=runtime,
        new_full_size_GPUcapacity=False, new_full_size_GPUtraining_updates=0, new_accuracy_gain=False,
        latest_complete_matched_native_code_correctness='1/48 baseline and each prior repair arm',
        full_final_requirement_statuses={k:v['status'] for k,v in audit['requirements'].items()}, goal_achieved=False,
        remaining=['Complete the queued original-scope four-replica V2 trial',
            'Finish full-source native qualification, fresh controls and data admission',
            'Measure remaining fresh-episode initialization cost without weakening source checks',
            'Complete full-size capacity, training, standalone export and all original assessments',
            'Demonstrate broad multimodal, human-aligned and coding correctness gains toward the full requested capability'])
    readme_before = (ROOT / 'README.md').read_bytes()
    OUT.mkdir(); write(OUT / 'runtime-snapshot.json', runtime); write(OUT / 'review.json', review)
    old_cost, new_cost = cost['old_validation'], cost['new_validation']
    note = (f"Native-code source-validation optimization — {at}\n\n"
        f"Twelve actual controls pass. The same recorded eight-attempt native worker group passes both validators. "
        f"Measured reads fall from {old_cost['read_character_bytes']:,} to {new_cost['read_character_bytes']:,} bytes "
        f"({cost['observed_read_character_reduction']:.2%} reduction). Shared-CPU wall times are {old_cost['wall_seconds']:.2f}s "
        f"and {new_cost['wall_seconds']:.2f}s; this is not an isolated hardware or complete-training throughput benchmark. "
        "The complete guard remains at validation-batch boundaries; exact source/control reconstruction, emitted native actions, "
        "complete CE masks, worker program bytes, true EOS, rewards and abstentions remain checked. Actual extra source mutation "
        "with identical size/timestamp rejects, and another thread cannot borrow the active source-validation context.\n\n"
        "Both own-small CPU CE/execution profiles also perform actual updates and reproduce the next native actions, real worker "
        "semantics, gradients, weights and AdamW after recovery. Six physical optimizer calls use the 267,334-parameter, full32768vocabulary "
        "source-specific fixture, one repeated human program and explicit five-action mechanics horizon. The original native horizon "
        "remains512. This is not unassisted coding improvement. The full original four-replica successor is actually queued behind "
        "the exact existing CPU trials; its result is still pending. Fresh-execution episode initialization retains its original complete guard cost.\n\n"
        f"Original human CE arm at this snapshot: {progress['actual_updates']:,}/5,362. All original four worker births remain live. "
        "No new full-size GPU work or accuracy gain is claimed. All ten final requirements remain incomplete.\n")
    (OUT / 'metrics.md').write_text(note)
    write(OUT / 'complete.json', dict(complete=True, review_sha256=sha(OUT / 'review.json'),
        runtime_sha256=sha(OUT / 'runtime-snapshot.json'), metrics_sha256=sha(OUT / 'metrics.md'), goal_achieved=False))
    audit.update(at=at, goal_achieved=False, latest_full_scope_review=str(OUT / 'review.json'),
        latest_native_suffix_source_validation=str(OUT / 'review.json'),
        candidate_native_suffix_validation_stage=str(ROOT / 'src/slm_lab/native_suffix_validation_stage_v2.py'),
        live_native_suffix_four_replica_successor=str(QUEUE))
    pending = audit_path.with_suffix('.json.validation-v2-pending'); write(pending, audit); pending.replace(audit_path)
    research = ROOT / 'RESEARCH.md'
    research.write_text(note + '\n' + research.read_text())
    if (ROOT / 'README.md').read_bytes() != readme_before:
        raise ValueError('The concise project README must remain untouched by progress recording')
    print(json.dumps(dict(complete=True, tests=12, independently_verified_bindings=len(files),
        read_character_reduction=cost['observed_read_character_reduction'], human_CE_updates=progress['actual_updates'], goal_achieved=False)))


if __name__ == '__main__': main()
