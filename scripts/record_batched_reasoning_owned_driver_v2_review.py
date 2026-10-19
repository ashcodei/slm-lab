"""Close the optimizer correction and actual live-state evidence, without NN work."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/batched-reasoning-owned-driver-v2-review'
CHECKS = ROOT / 'reports/batched-reasoning-owned-driver-v2/complete.json'
PREP = ROOT / 'reports/batched-reasoning-aligned-preparation-v2'


def read(path): return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as stream: return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False); stream.write('\n')


def process(pid, expected):
    fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
    if fields[19] != expected or fields[0] in ['Z', 'X']:
        raise ValueError('Only the exact actual live process birth can establish a verified wait')
    return dict(pid=pid, ticks=fields[19], state=fields[0], nice=int(fields[16]),
        scheduler=os.sched_getscheduler(pid), affinity=sorted(os.sched_getaffinity(pid)))


def main():
    os.nice(19); os.sched_setaffinity(0, {22}); os.sched_setscheduler(0, os.SCHED_IDLE, os.sched_param(0))
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('Review is source-only and CUDA hidden')
    checks = read(CHECKS)
    if not checks['complete'] or checks['pytest_exit_code'] != 0 or checks['actual_small_CPU_optimizer_calls'] != 6:
        raise ValueError('The actual complete new driver integration tests must pass')
    plan = read(PREP / 'plan.json')
    if (PREP / 'failure.json').exists() or plan['owner'] != dict(pid=177810, ticks='56522852'):
        raise ValueError('The exact new source-only preparer must remain successful and live')
    diagnostic = ROOT / 'reports/batched-reasoning-aligned-GPU-v1-known-optimizer-block-v1.json'
    blocked = read(diagnostic)
    if sha(blocked['affected_unlaunched_driver']) != blocked['affected_driver_sha256']:
        raise ValueError('The original frozen bad driver must remain unchanged')
    files = {**checks['source_files_sha256'], **checks['actual_proofs_sha256'], **plan['source_files_sha256']}
    for path in [Path(__file__), CHECKS, PREP / 'plan.json', diagnostic,
            ROOT / 'reports/batched-reasoning-aligned-preparation-v2-launch.json',
            Path(blocked['affected_unlaunched_driver']), ROOT / 'reports/batched-reasoning-owned-driver-v2-tests.log']:
        files[str(path)] = sha(path)
    for name, digest in files.items():
        if sha(name) != digest: raise ValueError('Every immutable closed source/proof byte must match: ' + name)
    proofs = [read(p) for p in checks['actual_proofs_sha256']]
    original = [process(pid, ticks) for pid, ticks in [(151931, '55253729'), (151932, '55253730'),
        (151933, '55253731'), (151934, '55253731')]]
    if any(p['nice'] != 19 or p['scheduler'] != os.SCHED_IDLE for p in original):
        raise ValueError('Original workers must retain their original scheduling')
    source_waiters = [process(pid, ticks) for pid, ticks in [(136092, '54738417'), (165313, '56047070'),
        (176058, '56470709'), (177810, '56522852')]]
    progress = read(ROOT / 'reports/phase-human-preference-v8/chosen_CE-progress.json')
    at = datetime.now(timezone.utc).isoformat()
    runtime = dict(at=at, original_neural_workers=original, actual_source_waiters=source_waiters,
        chosen_CE_actual_updates=progress['actual_updates'], chosen_CE_planned_updates=progress['planned_updates'],
        observed_free_bytes=shutil.disk_usage(ROOT).free, snapshot_only_not_an_immutable_progress_binding=True)
    audit_path = ROOT / 'reports/completion-audit.json'
    audit = read(audit_path)
    if len(audit['requirements']) != 10 or any(r['status'] != 'incomplete_for_requested_final_model' for r in audit['requirements'].values()):
        raise ValueError('All ten original final requirements remain incomplete; never narrow completion')
    git_identity_ready = all(subprocess.run(['git', 'config', '--get', key], capture_output=True).returncode == 0
        for key in ['user.name', 'user.email'])
    review = dict(at=at, complete=True, goal_achieved=False, previous_goal_turn_classification='progress',
        previous_turn_evidence='Created the reviewable Git index, exclusion rules and persistent requested commit workflow; no model quality change.',
        this_goal_turn_classification='progress', concrete_change='Corrected the unlaunched reasoning optimizer in a new frozen version; actual stage update and recovery pass.',
        source_files_sha256=files, independently_SHA_verified_bindings=len(files), test_count=9,
        actual_small_CPU_optimizer_calls=6, actual_own_small_parameters=267718,
        actual_logical_updates_per_profile=2, actual_profiles=[p['profile'] for p in proofs],
        actual_full_vocabulary=32768, actual_mechanics_horizon=2, actual_group_size=8,
        actual_next_actions_verifiers_weights_and_AdamW_recovery_exact=True,
        original_dense_base_unchanged=True, old_Adam_rejected_before_update=True,
        all_four_original_neural_process_births_live_unchanged=True,
        new_preparation_is_source_only_and_waiting=True, current_runtime=runtime,
        production_full_native_action_horizons=[64, 256], full_proposed_budget=plan['full_reasoning_budget'],
        original_17_families_85786_human_pairs_3721_human_code_sources_and_nine_endpoints_preserved=True,
        new_full_size_GPU_capacity=False, new_full_size_GPU_training_updates=0, new_model_quality_gain=False,
        latest_completed_matched_native_code_correctness='1/48 before and all three repair arms; no net gain',
        full_scope_requirements={k: v['status'] for k, v in audit['requirements'].items()},
        commit_pending_missing_user_identity=not git_identity_ready,
        remaining=['Finish original canonical full training and all original evaluations',
            'Finish full visual and native-code source preparation/admission',
            'Provide the required actual disk/memory and idle four-GPU headroom',
            'Run actual full-size capacity, readback, complete training, native export and original full assessments',
            'Demonstrate broad, multimodal, human-aligned and coding correctness gains toward the full requested capability'])
    OUT.mkdir()
    write(OUT / 'runtime-snapshot.json', runtime)
    write(OUT / 'review.json', review)
    text = (f"Reasoning launcher optimizer correction — {at}\n\n"
        "The previously unlaunched Adam launcher is preserved. Version 2 constructs ordinary actor-only AdamW, "
        "with foreach/fused disabled and explicit algorithm metadata. Nine checks pass. Both own-small CPU CE/RL profiles "
        "use the actual launcher optimizer: six physical updates total, two logical updates per profile, exact next-action/verifier/weight/AdamW replay, "
        "and unchanged original dense base. The fixture has 267,718 parameters, full 32,768 vocabulary, eight attempts and an explicitly shortened two-action horizon.\n\n"
        "Production keeps the complete 7,704 presentations, 57,240 answer/EOS targets, 735,570 context tokens, "
        "1,926 updates per profile, full 64/256 native horizons, and every original assessment. The new source-only preparer "
        "waits for the exact live original preparation process; no neural runtime/GPU launch occurs there.\n\n"
        f"Current original human CE arm: {progress['actual_updates']:,}/{progress['planned_updates']:,}; all four exact original GPU workers remain live. "
        f"{len(files)} immutable source/proof bindings were independently SHA verified.\n\n"
        "No new full-size GPU capacity, full-size training, or quality gain is established by this correction. "
        "Latest complete native code correctness remains 1/48 across baseline and all three repair arms. "
        "All ten original final requirements remain incomplete. Commits await the user's author identity.\n")
    (OUT / 'metrics.md').write_text(text)
    write(OUT / 'complete.json', dict(complete=True, review_sha256=sha(OUT / 'review.json'),
        runtime_snapshot_sha256=sha(OUT / 'runtime-snapshot.json'), metrics_sha256=sha(OUT / 'metrics.md'), goal_achieved=False))
    audit['at'] = at
    audit['goal_achieved'] = False
    audit['latest_full_scope_review'] = str(OUT / 'review.json')
    audit['latest_reasoning_owned_optimizer_correction'] = str(OUT / 'review.json')
    audit['previous_owned_reasoning_GPUdriver'] = audit['next_owned_reasoning_GPUdriver']
    audit['next_owned_reasoning_GPUdriver'] = str(ROOT / 'scripts/train_batched_reasoning_aligned_GPU_v2.py')
    audit['previous_owned_reasoning_source_preparation_directory'] = audit['live_owned_reasoning_source_preparation_directory']
    audit['live_owned_reasoning_source_preparation_directory'] = str(PREP)
    pending = audit_path.with_suffix('.json.reasoning-v2-pending')
    write(pending, audit); pending.replace(audit_path)
    note = (f"Reasoning launcher optimizer corrected, **{at}**: nine actual small CPU/driver/source checks pass; "
        "both CE/RL profiles consume actor-only AdamW and recover exact next actions and optimizer state. "
        "The corrected source-only preparer is live; full production sources, horizons and assessments remain. "
        f"Original human first arm: {progress['actual_updates']:,}/5,362. "
        "**No new full-model GPU capacity, training or quality gain; all ten final requirements remain incomplete.** "
        "[Evidence](reports/batched-reasoning-owned-driver-v2-review/metrics.md), "
        "[review](reports/batched-reasoning-owned-driver-v2-review/review.json). Earlier entries are historical snapshots.\n\n")
    for path in [ROOT / 'README.md', ROOT / 'RESEARCH.md']:
        path.write_text(note + path.read_text())
    print(json.dumps(dict(complete=True, independently_verified_bindings=len(files), tests=9,
        actual_small_CPU_updates=6, human_CE_updates=progress['actual_updates'], goal_achieved=False)))


if __name__ == '__main__': main()
