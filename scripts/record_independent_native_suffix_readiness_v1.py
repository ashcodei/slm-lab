"""Record actual independent source controls and the complete queued reader."""
from copy import deepcopy
from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess

from record_visual_stage_and_first_repair_arm_v1 import atomic, read, sha

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/independent-native-suffix-readiness-v1-review'
PREVIOUS = ROOT / 'reports/reasoning-aligned-GPU-readiness-v1-review/review.json'


def observe(owner):
    try: fields = Path('/proc', str(owner['pid']), 'stat').read_text().split(') ', 1)[1].split()
    except FileNotFoundError: return dict(owner=owner, live=False)
    live = fields[19] == owner['ticks'] and fields[0] not in ['Z', 'X']
    return dict(owner=owner, live=live, actual_ticks=fields[19], state=fields[0],
        nice=os.getpriority(os.PRIO_PROCESS, owner['pid']) if live else None,
        scheduler=os.sched_getscheduler(owner['pid']) if live else None)


def main():
    if OUT.exists(): raise ValueError('Preserve every earlier review')
    tests = ROOT / 'reports/independent-native-suffix-controls-v1-tests.log'
    if '6 passed in 9.64s' not in tests.read_text(): raise ValueError('All six actual source/control tests must pass')
    proofs = sorted((ROOT / 'reports/independent-native-suffix-controls-v1-pytest').rglob('actual-independent-*-native-source-controls.json'))
    if len(proofs) != 2 or {read(p)['kind'] for p in proofs} != {'apps', 'documented'}:
        raise ValueError('Both actual original public source formats must be freshly checked')
    for path in proofs:
        p = read(path)
        if (p['actual_fresh_native_gold']['reward'] != 1 or p['actual_fresh_native_prefix']['reward'] != 0
                or not p['all_actual_program_tests_outputs_and_verdict_semantics_agree']
                or p['models_constructed'] != 0 or p['neural_forwards'] != 0 or p['production_updates'] != 0):
            raise ValueError('Actual bounded gold/prefix controls and no-neural source scope must remain explicit')
    at = datetime.now(timezone.utc).isoformat()
    folder = ROOT / 'reports/full-native-token-suffix-readback-CPU-v1'
    plan = read(folder / 'plan.json')
    worker = observe(plan['owner'])
    if not worker['live'] or worker['nice'] != 19 or worker['scheduler'] != 5 or (folder / 'failure.json').exists():
        raise ValueError('The exact new idle source reader must actually be live')
    human = read(ROOT / 'reports/phase-human-preference-v8/chosen_CE-progress.json')
    GPUowners = [observe(dict(pid=pid, ticks=birth)) for pid, birth in [(151931, '55253729'), (151932, '55253730'),
        (151933, '55253731'), (151934, '55253731')]]
    if not all(v['live'] for v in GPUowners): raise ValueError('Inspect the exact original neural-worker transition')
    source_names = ['full-public-token-suffix-readback-gzip-recovery-CPU-v2', 'full-native-token-suffix-qualification-CPU-v1',
        'full-corrected-multimodal-batch-readback-CPU-v1', 'full-corrected-multimodal-source-admission-CPU-v1',
        'batched-reasoning-aligned-preparation-v1']
    observations = {}
    for name in source_names:
        source = ROOT / 'reports' / name
        file = next(source / n for n in ['failure.json', 'complete.json', 'progress.json'] if (source / n).exists())
        observations[name] = dict(actual_owner=observe(read(source / 'plan.json')['owner']),
            observed_file=file.name, value=read(file))
    snapshot = dict(at=at, actual_original_human_progress=human, actual_original_GPUowners=GPUowners,
        actual_new_independent_native_reader=worker, actual_new_native_reader_progress=read(folder / 'progress.json'),
        actual_other_source_observations=observations,
        actual_GPU_rows=subprocess.check_output(['nvidia-smi', '--query-gpu=index,name,utilization.gpu,memory.used,memory.total',
            '--format=csv,noheader'], text=True), actual_disk_rows=subprocess.check_output(['df', '-B1', str(ROOT)], text=True),
        no_existing_neural_worker_signalled_reprioritized_restarted_or_changed=True,
        no_new_neural_model_training_or_GPUcontext_job_or_lease=True,
        mutable_progress_registries_queues_and_live_journals_observed_not_bound=True,
        shared_cache_cleanup_actual_user_reply_pending=not (ROOT / 'reports/shared-pretrained-cache-reclamation-v1/user-approval.json').exists())
    requirements = deepcopy(read(PREVIOUS)['requirements'])
    requirements['public_data']['evidence'] += (
        ' Six independent native source/control tests now pass. The reader rebuilds original Unicode-exact native '
        'prefix/suffix IDs, whole CE/EOS masks and full horizons without importing the producer view helper. Actual '
        'APPSplain-code and documented-fence gold/prefix controls are freshly rerun and agree with both retained '
        'control repetitions on actual worker program/test/output/verdict semantics. Shortened native budgets, '
        'changed actual worker program hashes and dropped CEtargets reject; preserved adapter failure remains '
        'rewardNone and never becomes source eligibility. A new complete independent native-profile reader waits '
        'for the exact full producer, retains all3721sources/14884views, rechecks both original legacy controls '
        'and freshly runs gold/prefix controls for every attempted native view. Every failure/triviality/abstention/'
        'disagreement remains CE-only; full source admission and GPUcapacity are still separate incomplete gates.')
    requirements['gpu_runtime']['evidence'] += (
        ' The new source-only reader is actually live at nice19/SCHED_IDLE onCPU22, CUDAhidden and one thread. '
        'It constructs0neural models and cannot grant itself production RLor GPUownership. Complete source '
        'observations use lossless byte-LF gzip with2GiBraw quota/40GiBother-writer floor/16MiBnext-record headroom. '
        'Only closed parent journals will be bound after the exact producer becomes terminal; its live journal '
        'is not frozen or patched. Existing full-size neural workers and all original scopes remain unchanged.')
    requirements['human_feedback']['evidence'] = (
        f'The original four-L40chosen-response arm is{human["actual_updates"]}/5362updates. Both original arms retain '
        '85786public human-ranked pairs. Those other-model texts ranked by humans are distinct from fresh human '
        'judgments on this student. The new source controls are machine evidence, not human RLHF. Full human '
        'completion and held quality remain pending.')
    requirements['serious_full_model_training']['evidence'] += (
        f' Latest original human first-arm observation:{human["actual_updates"]}/5362. The new six source tests '
        'construct0models, run0neural forwards and perform0optimizer updates. Full complete native source '
        'qualification/independent readback/admission, actual full-size code/reasoning/visual RLtraining and final '
        'quality remain pending. Source controls do not establish student reward density or unassisted quality.')
    for value in requirements.values():
        if value['status'] != 'incomplete_for_requested_final_model': raise ValueError('All ten final requirements remain incomplete')
        value['previous_evidence'] = str(PREVIOUS)
    if len(requirements) != 10: raise ValueError('Preserve all final requirements')
    names = ['src/slm_lab/independent_native_suffix_controls_v1.py', 'tests/test_independent_native_suffix_controls_v1.py',
        'scripts/run_independent_native_suffix_controls_v1_tests.py', 'scripts/read_full_native_token_suffixes_CPU_v1.py',
        'reports/full-native-token-suffix-readback-CPU-v1-launch.json']
    files = {str(p): sha(p) for p in [Path(__file__), PREVIOUS, tests, *proofs, folder / 'plan.json', *[ROOT / n for n in names]]}
    classification = ('progress:6actual independent native gold/prefix/source-mask/full-budget/worker-program and '
        'missing-verdict controls pass; complete3721source14884view independent native-profile reader queued '
        'without changing neural workers;0new model training or quality gain')
    OUT.mkdir()
    atomic(OUT / 'runtime-snapshot.json', snapshot)
    atomic(OUT / 'complete.json', dict(complete=True, at=at, source_files_sha256=files,
        actual_independent_native_source_controls=6, actual_plain_and_documented_gold_reward1_prefix_reward0=True,
        actual_full_native_horizons_whole_CE_masks_worker_bytes_and_verdict_semantics_checked=True,
        missing_adapter_verdict_preserved_as_None_without_source_eligibility=True,
        actual_new_independent_native_reader=plan['owner'], entire_original_sources=3721, entire_original_views=14884,
        every_level_retains_original_context_tokens=2287522, every_level_retains_original_CE_EOS_targets=584218,
        complete_native_profile_source_readback=False, source_training_admitted=False, production_RL_admitted=False,
        actual_original_human_updates=human['actual_updates'], new_neural_models=0, new_neural_forwards=0,
        new_optimizer_updates=0, new_model_quality_gain=False, all_ten_final_requirements_incomplete=True, goal_achieved=False))
    metrics = (
        '**Six independent native source/control tests pass.** Actual public APPSplain programs and documented '
        'Python-fence functions retain original complete gold reward1and prefix-only reward0. A new reader independently '
        'rebuilds original source IDs, Unicode-exact hints, missing suffix/EOS IDs, whole unconditional CElabels and '
        'uncropped native horizons using the separate source reconstruction implementation. Real fresh worker calls '
        'agree with both retained native controls on program/test/returned-output/verdict semantics; walltime identity '
        'is not asserted. Documented incomplete outer fences receive their observed structural zero, with no invented '
        'worker execution.\n\n'
        'Changing an actual worker program SHA, substituting a5action CPUmechanics budget for the original512budget, '
        'or dropping an original CEtarget rejects. A controlled unavailable adapter observation retains rewardNone '
        'and cannot qualify a source. These source checks construct0neural models/run0neural forwards/perform0updates. '
        'They are not observed student rewards or an unassisted coding result.\n\n'
        '**A complete independent native-profile reader is actually queued.** After its exact producer finishes, it '
        'will reconstruct every3721human source/all14884views, reconcile both retained legacy observations, validate '
        'every native control request/program/mask/horizon and freshly run native gold/prefix controls for every '
        'previously attempted view. Every failed/trivial/abstaining/disagreeing view remains CE-only. Each level '
        'retains2287522original context tokens/584218whole response-EOS targets. This reader does not grant production '
        'source/runtime admission or complete the future four-rank coding RLstage.\n\n'
        'The exact new reader is CUDAhidden, nice19/SCHED_IDLE, one CPUthread onCPU22. Its lossless byte-LF gzip '
        'journal retains a2GiBraw bound and40GiBother-writer floor, with16MiBfor the next complete bounded record. '
        'No live parent journal or mutable queue/progress/child registry is source-bound. Existing neural workers '
        'are not signalled, moved, patched or restarted.\n\n'
        f'Original four-L40human first arm: **{human["actual_updates"]}/5362updates**. The original full code-source '
        'producer already finished all3721references/14884prefix controls; its full independent legacy reader and '
        'native-profile producer remain separate live gates. All original17families,85786human pairs/arm, complete '
        'reasoning/visual scopes and9final endpoints remain.\n\n'
        '**No new model-quality gain is established. Latest completed coding correctness remains1/48at every '
        'original study endpoint. All ten final requirements remain incomplete.** The earlier shared-cache cleanup '
        'question still awaits an actual user reply; no shared pretrained weights were used or deleted.\n')
    (OUT / 'metrics.md').write_text(metrics)
    for p in [OUT / 'complete.json', OUT / 'runtime-snapshot.json', OUT / 'metrics.md']: files[str(p)] = sha(p)
    atomic(OUT / 'review.json', dict(at=at, requirements=requirements, source_files_sha256=files,
        previous_full_scope_review=str(PREVIOUS), current_goal_turn_classification=classification, goal_achieved=False))
    audit_path = ROOT / 'reports/completion-audit.json'
    audit = read(audit_path)
    audit.update(at=at, updated_at=at, requirements=requirements, latest_full_scope_review=str(OUT / 'review.json'),
        latest_independent_native_code_suffix_readiness=str(OUT / 'complete.json'),
        live_independent_native_code_suffix_reader_directory=str(folder), current_goal_turn_classification=classification, goal_achieved=False)
    atomic(audit_path, audit)
    entry = (f'Independent native-code source readback, **{at}**: six actual plain-code/documented-fence source controls '
        'pass with fresh worker outcomes/full native budgets/whole CEtargets and strict missing-verdict handling. '
        'The complete3721source/14884view independent native-profile reader is queued. '
        f'Original human first arm: {human["actual_updates"]}/5362updates. **No new model training or quality gain; '
        'all ten final requirements remain incomplete.** [Evidence](reports/independent-native-suffix-readiness-v1-review/metrics.md), '
        '[full review](reports/independent-native-suffix-readiness-v1-review/review.json). Earlier entries are historical snapshots.\n\n')
    for name in ['README.md', 'RESEARCH.md']:
        p = ROOT / name
        p.write_text(entry + p.read_text())
    print(str(OUT / 'complete.json'))


if __name__ == '__main__': main()
