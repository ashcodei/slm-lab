"""Close actual native-code stage mechanics; keep the full capability goal open."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/parallel-native-suffix-stage-v1-review'
TESTS = ROOT / 'reports/parallel-native-suffix-stage-v1-pytest'
LOG = ROOT / 'reports/parallel-native-suffix-stage-v1-tests.log'
CONTROL = ROOT / 'reports/native-suffix-CPU-mechanics-qualification-v1/closed-qualified-controls.json'


def sha(path):
    with Path(path).open('rb') as stream: return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path): return json.loads(Path(path).read_text())


def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False); stream.write('\n')


def live(pid, birth):
    f = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
    if f[19] != birth or f[0] in ['Z', 'X']: raise ValueError('The exact observed worker must remain live')
    return dict(pid=pid, ticks=f[19], state=f[0], nice=int(f[16]), scheduler=os.sched_getscheduler(pid))


def main():
    os.nice(19); os.sched_setaffinity(0, {22}); os.sched_setscheduler(0, os.SCHED_IDLE, os.sched_param(0))
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or '9 passed' not in LOG.read_text():
        raise ValueError('All nine actual native stage checks must pass in the CUDA-hidden CPU run')
    test_owner = Path('/proc/178844/stat')
    if test_owner.exists():
        state = test_owner.read_text().rsplit(')', 1)[1].split()
        if state[19] == '56558134' and state[0] not in ['Z', 'X']:
            raise ValueError('The exact tests must actually be terminal before their evidence is closed')
    singles = list(TESTS.glob('test_actual_single_replica_com*/actual-single-replica-native-recovery.json'))
    singles = sorted({p.resolve() for p in singles})
    four = sorted(TESTS.glob('test_actual_four_replicas_nati*/anchor_*-rank-*.json'))
    four = sorted({p.resolve() for p in four})
    ce = sorted({p.resolve() for p in TESTS.glob('test_unqualified_execution_vie*/actual-unqualified-view-full-CE.json')})
    pairs = sorted({p.resolve() for p in TESTS.glob('test_actual_CE_execution_pair_*/actual-paired-native-stage-policy-effect.json')})
    full = sorted({p.resolve() for p in TESTS.glob('test_full_original_schedule_re*/actual-full-native-source-schedule.json')})
    if [len(singles), len(four), len(ce), len(pairs), len(full)] != [2, 8, 2, 1, 1]:
        raise ValueError('All actual single/four-replica, CE-fallback, paired-policy and full-source schedule evidence is required')
    rows = [read(p) for p in four]
    actual_counts = {r['actual_small_parameters'] for r in rows}
    if len(actual_counts) != 1:
        raise ValueError('Every actual own-small replica must use the same observed parameter count')
    actual_count = next(iter(actual_counts))
    for profile in ['anchor_CE', 'anchor_execution']:
        selected = [r for r in rows if r['profile'] == profile]
        if (len(selected) != 4 or len({r['actual_final_shared_checkpoint_sha256'] for r in selected}) != 1
                or any(r['distinct_original_human_sources'] != 1 or r['source_presentations'] != 9
                    or r['original_response_EOS_targets'] != 612 or r['physical_optimizer_calls'] != 4
                    or not r['actual_serial_full_source_mean_matches'] for r in selected)):
            raise ValueError('The exact actual repeated-one-human-source fixture and all four replicas must remain explicit')
    from slm_lab.local_source_closure_v2 import python_sources
    seeds = [Path(__file__), ROOT / 'src/slm_lab/parallel_native_suffix_stage_v1.py',
        ROOT / 'tests/test_parallel_native_suffix_stage_v1.py', ROOT / 'scripts/run_parallel_native_suffix_stage_v1_tests.py']
    files = {str(p): sha(p) for p in [*python_sources(ROOT, seeds), CONTROL, LOG, *singles, *four, *ce, *pairs, *full]}
    for p, h in read(CONTROL)['source_files_sha256'].items():
        if p in files and files[p] != h: raise ValueError('A closed original source proof conflicts with this actual stage')
        files[p] = h
    for p, h in files.items():
        if sha(p) != h: raise ValueError('Every actual original-source/runtime/evidence byte must match: ' + p)
    audit_path = ROOT / 'reports/completion-audit.json'
    audit = read(audit_path)
    if len(audit['requirements']) != 10 or any(v['status'] != 'incomplete_for_requested_final_model' for v in audit['requirements'].values()):
        raise ValueError('All ten original final requirements remain incomplete')
    at = datetime.now(timezone.utc).isoformat()
    workers = [live(pid, birth) for pid, birth in [(151931, '55253729'), (151932, '55253730'),
        (151933, '55253731'), (151934, '55253731')]]
    progress = read(ROOT / 'reports/phase-human-preference-v8/chosen_CE-progress.json')
    pending = [live(pid, birth) for pid, birth in [(158779, '55770044'), (162287, '55947633'),
        (166298, '56079802'), (167672, '56126055')]]
    runtime = dict(at=at, original_neural_workers=workers, exact_native_source_chain=pending,
        human_CE_actual_updates=progress['actual_updates'], human_CE_planned_updates=progress['planned_updates'],
        full_native_source_data_admitted=(ROOT / 'data/native-human-code-suffix-v1/admission.json').exists(),
        observed_free_bytes=shutil.disk_usage(ROOT).free, snapshot_only_not_an_immutable_live_progress_binding=True)
    review = dict(complete=True, at=at, goal_achieved=False, previous_goal_turn_classification='progress',
        previous_turn_evidence='Corrected the unlaunched reasoning Adam/AdamW mismatch; nine actual driver checks and 176 SHA bindings verified.',
        this_goal_turn_classification='progress', source_files_sha256=files, independently_verified_bindings=len(files),
        new_stage=str(ROOT / 'src/slm_lab/parallel_native_suffix_stage_v1.py'), actual_tests=9,
        actual_own_small_parameters=actual_count, actual_full_vocabulary=32768, actual_group_size=8, explicit_fixture_horizon=5,
        actual_fixture_original_native_horizon=512, original_human_source_position=188,
        repeated_one_human_source_fixture_not_full_source_learning=True,
        actual_physical_small_AdamW_calls=47,
        actual_native_execution_rewards_change_the_matched_policy_gradient_and_actor=True,
        actual_four_replica_source_mean_matches_independent_serial_gradients=True,
        actual_shared_state_and_private_rng_recovery=True, complete_raw_timings_retained_but_not_replay_equal=True,
        actual_CE_only_views_keep_all_original_targets=True, actual_single_source_three_empty_rank_tail=True,
        peer_only_failure_preserves_all_native_actions_and_actual_execution_before_optimizer=True,
        full_source_schedule_updates=11172, full_source_schedule_presentations=44652,
        full_original_human_sources=3721, all_original_levels=[4, 16, 64, 0],
        full_original_response_EOS_targets=7010616, actual_full_source_schedule_empty_tails=12,
        production_native_action_budgets_never_shortened=True,
        current_runtime=runtime, new_full_size_GPUcapacity=False, new_full_size_GPUtraining_updates=0,
        new_unassisted_code_or_broad_multimodal_quality_gain=False,
        latest_completed_native_code_correctness='1/48 baseline and every prior repair arm; no new full-model result',
        remaining=['Complete full source qualification and independent readback/data admission',
            'Measure and reduce repeated source-validation I/O before full-size use',
            'Implement and run source-bound full-size capacity, recovery, complete training and native export',
            'Run all original code, efficiency, knowledge, image, word and human-preference assessments',
            'Demonstrate the requested high-end all-round capability from the sub-1B own model'],
        all_original_final_requirement_statuses={k:v['status'] for k,v in audit['requirements'].items()})
    OUT.mkdir()
    write(OUT / 'runtime-snapshot.json', runtime); write(OUT / 'review.json', review)
    note = (f"Native-code parallel execution training mechanics — {at}\n\n"
        "Nine actual source/own-small CPU tests pass. Both matched profiles run fresh eight-attempt native code groups, "
        "full unconditional original CE, own-starting-policy reference, source-mean gradient SUM and ordinary actor-only AdamW. "
        "Actual bounded worker rewards change the matched gradient and learned actor. An independently computed serial gradient "
        "matches the four-replica source mean. Complete shared checkpoint/private RNG recovery, genuine empty ranks, CE-only "
        "unqualified views and peer-only failure before optimizer commit pass. All raw execution timings remain; only execution semantics "
        f"are claimed replay-equal. The {actual_count:,}-parameter fixture repeats one original human program and has a five-action test horizon "
        "against an unchanged full native horizon of512. This is not an unassisted model-quality result.\n\n"
        "The source-only full schedule retains all3,721 human sources, all4levels, 44,652 presentations, 7,010,616 complete original "
        "answer/EOS targets, 11,172 updates and12real one-source/three-empty-rank tails. Full source admission, GPU capacity, "
        "production training/export and all original assessments remain required. Repeated source-validation I/O needs measurement/optimization.\n\n"
        f"Original human CE arm at this snapshot: {progress['actual_updates']:,}/5,362, all four original worker births still live. "
        "No new full-size GPU work or accuracy improvement is claimed. All ten final requirements remain incomplete.\n")
    (OUT / 'metrics.md').write_text(note)
    write(OUT / 'complete.json', dict(complete=True, review_sha256=sha(OUT / 'review.json'),
        metrics_sha256=sha(OUT / 'metrics.md'), runtime_sha256=sha(OUT / 'runtime-snapshot.json'), goal_achieved=False))
    audit.update(at=at, goal_achieved=False, latest_full_scope_review=str(OUT / 'review.json'),
        latest_parallel_native_code_suffix_stage=str(OUT / 'review.json'),
        next_parallel_native_code_suffix_stage=str(ROOT / 'src/slm_lab/parallel_native_suffix_stage_v1.py'))
    temp = audit_path.with_suffix('.json.native-stage-pending'); write(temp, audit); temp.replace(audit_path)
    heading = (f"Native-code execution training stage, **{at}**: nine actual small CPU/source checks pass, including four-replica "
        "gradient/recovery/fault controls, true worker-reward policy effects and full original CE fallback. The fixture repeats one human source; "
        "the full3,721source/44,652presentation schedule is preserved separately. "
        f"Original human first arm: {progress['actual_updates']:,}/5,362. "
        "**No new full-size GPU training or quality gain; all ten final requirements remain incomplete.** "
        "[Evidence](reports/parallel-native-suffix-stage-v1-review/metrics.md), [review](reports/parallel-native-suffix-stage-v1-review/review.json). "
        "Earlier entries are historical snapshots.\n\n")
    for path in [ROOT / 'README.md', ROOT / 'RESEARCH.md']: path.write_text(heading + path.read_text())
    print(json.dumps(dict(complete=True, tests=9, verified_bindings=len(files), human_CE_updates=progress['actual_updates'], goal_achieved=False)))


if __name__ == '__main__': main()
