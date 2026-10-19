"""Record indexed source transport, the whole curriculum and pending admission."""
from copy import deepcopy
from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess

from record_visual_stage_and_first_repair_arm_v1 import atomic, read, sha

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/native-suffix-indexed-data-readiness-v1-review'
PREVIOUS = ROOT / 'reports/independent-native-suffix-readiness-v1-review/review.json'


def observe(owner):
    try: fields = Path('/proc', str(owner['pid']), 'stat').read_text().split(') ', 1)[1].split()
    except FileNotFoundError: return dict(owner=owner, live=False)
    live = fields[19] == owner['ticks'] and fields[0] not in ['Z', 'X']
    return dict(owner=owner, live=live, actual_ticks=fields[19], state=fields[0],
        nice=os.getpriority(os.PRIO_PROCESS, owner['pid']) if live else None,
        scheduler=os.sched_getscheduler(owner['pid']) if live else None)


def main():
    if OUT.exists(): raise ValueError('Preserve every earlier complete or failed review')
    tests = ROOT / 'reports/native-suffix-indexed-data-v1-tests.log'
    if '8 passed in 27.22s' not in tests.read_text(): raise ValueError('All eight actual data/source/transport controls must pass')
    proof_root = ROOT / 'reports/native-suffix-indexed-data-v1-pytest'
    schedules = sorted(proof_root.rglob('actual-complete-source-view-schedule.json'))
    if len(schedules) != 2 or {read(p)['epochs'] for p in schedules} != {1, 3}:
        raise ValueError('Both complete source schedules must retain every original view and target')
    for path in schedules:
        p = read(path)
        epochs = p['epochs']
        if (p['actual_complete_original_sources'] != 3721 or p['actual_complete_original_views_per_epoch'] != 14884
                or p['actual_updates'] != 3724 * epochs or p['actual_source_presentations'] != 14884 * epochs
                or p['actual_original_context_tokens'] != 9150088 * epochs
                or p['actual_original_response_EOS_targets'] != 2336872 * epochs
                or p['actual_full_generation_horizon_sum'] != 7695421 * epochs
                or p['actual_single_source_three_empty_rank_training_steps'] != 4 * epochs
                or not p['only_source_schedule_selection_not_native_profile_admission']):
            raise ValueError('All whole original source budgets, native horizons and genuine level tails must reconcile')
    transport_path = next(proof_root.rglob('actual-public-proof-indexed-transport.json'))
    transport = read(transport_path)
    if not transport['actual_closed_public_source_proof_retained_byte_LF_and_value_exact'] or transport['actual_random_read_order'] != [2,0,1,2]:
        raise ValueError('Every actual original public proof byte/value and independent random member must survive')
    adapter_path = next(proof_root.rglob('actual-one-source-CPU-data-adapter.json'))
    adapter = read(adapter_path)
    if (adapter['actual_budget']['source_presentations'] != 9 or adapter['actual_budget']['distinct_original_human_sources'] != 1
            or adapter['actual_budget']['original_response_EOS_targets'] != 612
            or adapter['full_original_native_horizon'] != 512 or not adapter['CPU_fixture_admission_forbidden']):
        raise ValueError('The CPU fixture repeats one actual human source; it cannot claim nine independent sources or production admission')
    at = datetime.now(timezone.utc).isoformat()
    folder = ROOT / 'reports/native-human-code-suffix-data-admission-CPU-v1'
    plan = read(folder / 'plan.json')
    worker = observe(plan['owner'])
    if not worker['live'] or worker['nice'] != 19 or worker['scheduler'] != 5 or (folder / 'failure.json').exists():
        raise ValueError('The exact new idle source admission worker must actually be live')
    human = read(ROOT / 'reports/phase-human-preference-v8/chosen_CE-progress.json')
    GPUowners = [observe(dict(pid=pid, ticks=birth)) for pid, birth in [(151931, '55253729'), (151932, '55253730'),
        (151933, '55253731'), (151934, '55253731')]]
    if not all(p['live'] for p in GPUowners): raise ValueError('Inspect the exact original neural-worker transition')
    sources = {}
    for name in ['full-public-token-suffix-readback-gzip-recovery-CPU-v2', 'full-native-token-suffix-qualification-CPU-v1',
            'full-native-token-suffix-readback-CPU-v1', 'full-corrected-multimodal-batch-readback-CPU-v1']:
        source = ROOT / 'reports' / name
        observed = next(source / n for n in ['failure.json', 'complete.json', 'progress.json'] if (source / n).exists())
        sources[name] = dict(actual_owner=observe(read(source / 'plan.json')['owner']), observed_file=observed.name, value=read(observed))
    snapshot = dict(at=at, actual_original_human_progress=human, actual_original_GPUowners=GPUowners,
        actual_new_data_only_admission_owner=worker, actual_new_data_only_progress=read(folder / 'progress.json'),
        actual_source_observations=sources,
        actual_GPU_rows=subprocess.check_output(['nvidia-smi', '--query-gpu=index,name,utilization.gpu,memory.used,memory.total',
            '--format=csv,noheader'], text=True), actual_disk_rows=subprocess.check_output(['df', '-B1', str(ROOT)], text=True),
        source_data_admission_exists=(ROOT / 'data/native-human-code-suffix-v1/admission.json').exists(),
        no_existing_neural_worker_changed_and_no_new_neural_or_GPUjob=True,
        mutable_progress_queues_registries_and_live_source_journals_observed_not_bound=True,
        shared_cache_cleanup_actual_user_reply_pending=not (ROOT / 'reports/shared-pretrained-cache-reclamation-v1/user-approval.json').exists())
    requirements = deepcopy(read(PREVIOUS)['requirements'])
    requirements['public_data']['evidence'] += (
        ' An indexed per-record gzip source transport and strict data adapter are implemented. Eight actual '
        'transport/source checks pass: a complete real public control proof and byte-LF Unicode survive out-of-order '
        'reads; corrupt members, index gaps, false raw bounds, raw quota overflow and incomplete declared populations '
        'reject. Source-only schedules enumerate all actual14884original descriptors for1and3epochs with exact '
        'original context/CE/horizon totals and ordered4/16/64/unhinted levels. A separate data-only admission owner '
        'waits for the exact full independent native reader, then packs and freshly reads every complete original '
        'producer proof/reader result before publishing admission. Every failed/trivial/abstaining/disagreeing view '
        'remains CE-only. The production data adapter requires the complete terminal admission owner, actual receipt, '
        'all source bindings and complete pack. Full production construction remains pending, not proven by the '
        'explicit CPU fixture. That fixture has9presentations of1actual human source, not9independent human sources.')
    requirements['gpu_runtime']['evidence'] += (
        ' The future coding source schedule has3724global groups/14884presentations per epoch, with4real one-source/'
        'three-empty-rank tails at the level boundaries; these cannot be replaced with duplicate examples. Three '
        'epochs retain11172updates/44652presentations/7010616original CE-EOS targets. The indexed transport reads '
        'one bounded compressed member per source and retains original per-record SHA/offset/length metadata, '
        'avoiding repeated whole-corpus decompression. The new idle source admission worker cannot admit or launch '
        'GPUruntime. Full four-rank code stage/capacity/recovery/production/export/quality remain pending.')
    requirements['human_feedback']['evidence'] = (
        f'The unchanged original four-L40chosen-response arm is{human["actual_updates"]}/5362updates. All85786pairs per '
        'original chosen_CE/human_DPOarm remain. Public human-ranked other-model texts are not fresh own-output '
        'judgments. New indexed source controls are not human feedback. Full original human completion and quality '
        'remain pending.')
    requirements['serious_full_model_training']['evidence'] += (
        f' Latest original human first-arm observation:{human["actual_updates"]}/5362. The new eight data/source '
        'controls construct0neural models, perform0neural forwards and0optimizer updates. Proposed full coding '
        'curriculum counts are source schedule evidence only, not completed model training or admitted runtime.')
    for value in requirements.values():
        if value['status'] != 'incomplete_for_requested_final_model': raise ValueError('All ten final requirements remain incomplete')
        value['previous_evidence'] = str(PREVIOUS)
    if len(requirements) != 10: raise ValueError('Preserve all ten final requirements')
    names = ['src/slm_lab/indexed_gzip_records_v1.py', 'src/slm_lab/native_suffix_train_data_v1.py',
        'tests/test_native_suffix_indexed_data_v1.py', 'scripts/run_native_suffix_indexed_data_v1_tests.py',
        'scripts/admit_native_human_suffix_data_CPU_v1.py', 'reports/native-human-code-suffix-data-admission-CPU-v1-launch.json']
    files = {str(p): sha(p) for p in [Path(__file__), PREVIOUS, tests, *schedules, transport_path, adapter_path,
        folder / 'plan.json', *[ROOT / n for n in names]]}
    classification = ('progress:8actual indexed-source transport/data/full-schedule controls pass; strict source adapter '
        'retains whole original CE targets and ordered complete source views; exact full data-only admission owner '
        'queued;0new neural training or quality gain')
    OUT.mkdir()
    atomic(OUT / 'runtime-snapshot.json', snapshot)
    atomic(OUT / 'complete.json', dict(complete=True, at=at, source_files_sha256=files, actual_source_data_controls=8,
        actual_original_sources=3721, actual_original_views_per_epoch=14884, actual_global_groups_per_epoch=3724,
        actual_complete_context_tokens_per_epoch=9150088, actual_original_CE_EOS_targets_per_epoch=2336872,
        actual_full_native_horizon_sum_per_epoch=7695421, actual_single_source_three_empty_rank_steps_per_epoch=4,
        actual_three_epoch_groups=11172, actual_three_epoch_presentations=44652, actual_three_epoch_CE_EOS_targets=7010616,
        actual_CPU_fixture_presentations=9, actual_CPU_fixture_distinct_human_sources=1,
        indexed_random_source_transport_retains_complete_original_proofs=True,
        actual_data_only_admission_owner=plan['owner'], full_source_data_admission_pending=True,
        full_size_GPUruntime_capacity_training_and_quality_not_admitted=True,
        actual_original_human_updates=human['actual_updates'], new_neural_models=0, new_neural_forwards=0,
        new_optimizer_updates=0, new_model_quality_gain=False, all_ten_final_requirements_incomplete=True, goal_achieved=False))
    metrics = (
        '**Eight data/source controls pass.** A new indexed gzip transport retains a complete actual public native-control '
        'proof and Unicode byte-LF records exactly during random reads. Per-record offsets, compressed/raw lengths and '
        'SHA256 allow one bounded record to be decoded independently. Corrupt gzip members, gaps, false raw lengths, '
        'raw quota overflow and incomplete declared populations reject. This is source transport evidence, not GPU speed.\n\n'
        '**The complete ordered source schedule preserves every original view.** Each epoch uses all 3,721 human sources '
        'at all 4/16/64/unhinted levels: 14,884 presentations, 9,150,088 original context tokens, 2,336,872 complete '
        'response/EOS targets and 7,695,421 native horizon positions. It requires 3,724 global groups, including four '
        'real one-source/three-empty-rank tails at the level boundaries. Three epochs retain 11,172 groups, 44,652 '
        'presentations and 7,010,616 targets. These counts come from the complete actual closed descriptor population; '
        'they do not establish native-profile admission, production updates or quality.\n\n'
        '**The source API keeps failed execution views usable for complete reference CE.** Qualified views yield the '
        'exact existing source-bound native episode; ineligible views yield no execution episode and retain their '
        'original full CE batch. The explicit CPU fixture repeats one actual source nine times: 612 original targets '
        'and the unchanged 512-action source budget. It claims one distinct human source and cannot admit production. '
        'No neural model, forward or optimizer update occurs in these checks. Full production adapter construction '
        'still requires the genuine complete terminal admission and is not substituted by this fixture.\n\n'
        '**A data-only admission worker is actually queued.** It waits for the exact independent native reader, aligns '
        'every original producer proof with its fresh result, packs all 14,884 records, then freshly decompresses and '
        'compares every record against both original streams. Failed/trivial/abstaining/disagreeing cases remain. '
        'Only after that does it publish a separate receipt for data use. Complete four-rank coding runtime, actual '
        'GPU capacity/recovery/training/export and final evaluation remain separate unfinished requirements.\n\n'
        f'The original four-L40 human first arm is **{human["actual_updates"]}/5,362 updates**. Existing neural workers '
        'remain unchanged. The new worker is CUDA-hidden, nice19/SCHED_IDLE, one thread on CPU22; it starts no model, '
        'GPU context or lease. All original 17 families, human pairs, complete reasoning/visual scopes and nine final '
        'endpoints remain. The 2 GiB raw indexed-pack quota and 40 GiB other-writer floor remain explicit.\n\n'
        '**No new model-quality gain is established. Latest completed coding correctness remains 1/48 at every '
        'original study endpoint. All ten final requirements remain incomplete.** The earlier shared-cache cleanup '
        'question awaits an actual user reply; no shared pretrained weights were used or deleted.\n')
    (OUT / 'metrics.md').write_text(metrics)
    for p in [OUT / 'complete.json', OUT / 'runtime-snapshot.json', OUT / 'metrics.md']: files[str(p)] = sha(p)
    atomic(OUT / 'review.json', dict(at=at, requirements=requirements, source_files_sha256=files,
        previous_full_scope_review=str(PREVIOUS), current_goal_turn_classification=classification, goal_achieved=False))
    audit_path = ROOT / 'reports/completion-audit.json'
    audit = read(audit_path)
    audit.update(at=at, updated_at=at, requirements=requirements, latest_full_scope_review=str(OUT / 'review.json'),
        latest_native_suffix_indexed_data_readiness=str(OUT / 'complete.json'),
        next_native_suffix_train_data=str(ROOT / 'src/slm_lab/native_suffix_train_data_v1.py'),
        live_native_suffix_data_admission_directory=str(folder), current_goal_turn_classification=classification, goal_achieved=False)
    atomic(audit_path, audit)
    entry = (f'Indexed complete native-code curriculum, **{at}**: eight source/data checks pass; all 14,884 views '
        'retain 2,336,872 complete original targets per epoch and four real level-boundary tails. Indexed source '
        'transport avoids whole-corpus scans. The data-only admission worker is queued; full-source admission and '
        f'GPU runtime remain pending. Original human first arm: {human["actual_updates"]}/5,362 updates. **No new model '
        'training or quality gain; all ten final requirements remain incomplete.** '
        '[Evidence](reports/native-suffix-indexed-data-readiness-v1-review/metrics.md), '
        '[full review](reports/native-suffix-indexed-data-readiness-v1-review/review.json). Earlier entries are historical snapshots.\n\n')
    for name in ['README.md', 'RESEARCH.md']:
        p = ROOT / name
        p.write_text(entry + p.read_text())
    print(str(OUT / 'complete.json'))


if __name__ == '__main__': main()
