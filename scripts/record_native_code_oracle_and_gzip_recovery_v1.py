"""Record native code-reward controls, exact source recovery and fresh pixels."""
from copy import deepcopy
from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess

from record_visual_stage_and_first_repair_arm_v1 import atomic, read, sha
from slm_lab.exact_waiting_process_retirement_v1 import terminal

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/native-code-oracle-and-gzip-recovery-v1-review'
PREVIOUS = ROOT / 'reports/batched-reasoning-capacity-controller-v1-review/review.json'


def main():
    if OUT.exists(): raise ValueError('Preserve every earlier complete or failed review')
    logs = {'native-code-execution-oracle-v2-tests.log': '7 passed',
        'bounded-gzip-observation-stream-v1-tests.log': '5 passed', 'full-token-suffix-gzip-recovery-v2-tests.log': '3 passed'}
    for name, expected in logs.items():
        if expected not in (ROOT / 'reports' / name).read_text(): raise ValueError('All15actual source/oracle/stream/recovery controls must pass')
    proofs_root = ROOT / 'reports/native-code-execution-oracle-v2-pytest'
    format_path = next(proofs_root.rglob('actual-all3721-source-format-readback.json'))
    formats = read(format_path)
    if (formats['actual_source_format_counts'] != {'documented:one_exact_Python_fence': 173, 'apps:plain_Python': 3548}
            or formats['original_context_tokens'] != 2287522 or formats['original_positive_response_EOS_targets'] != 584218
            or not formats['every_original_human_program_and_native_mask_reconstructed'] or formats['models_constructed'] != 0):
        raise ValueError('Every original human source byte/format and full CE/EOS mask must reconcile')
    program_proofs = []
    for kind in ['documented', 'apps']:
        p = next(proofs_root.rglob('actual-' + kind + '-native-program-controls.json'))
        value = read(p)
        if (value['actual_original_gold_control']['reward'] != 1 or value['actual_original_extracting_scorer']['reward'] != 1
                or value['actual_native_scorer_rejection']['reward'] != 0 or value['actual_gold_without_EOS']['reward'] != .9
                or value['actual_legacy_style_decode_then_score']['reward'] != 1
                or value['actual_injected_native_control_delivery']['reward'] != 0
                or not value['actual_original_gold_control']['all_observed_worker_program_bytes_equal_complete_native_program']
                or not value['test_responses_are_not_generated_student_samples']):
            raise ValueError('All actual response-format, raw-worker, native-control and true-EOS comparisons must remain explicit')
        program_proofs.append(p)
    launch_root = ROOT / 'reports/full-token-suffix-gzip-recovery-v2-launch'
    launch_path = launch_root / 'complete.json'
    launch = read(launch_path)
    retained_path = launch_root / 'actual-live-gzip-prefix-readback.json'
    retained = read(retained_path)
    if (not launch['complete'] or launch['actual_original_closed_source_rows_reconstructed'] != 2222
            or launch['actual_remaining_sources'] != 1499 or not launch['old_owners_terminal_before_recovery']
            or not retained['equal_to_every_closed_original_control_byte'] or retained['actual_retained_original_sources'] != 2222
            or retained['actual_retained_reference_and_prefix_controls'] != 11110
            or not retained['live_compressed_journal_observed_not_source_bound']):
        raise ValueError('All2222prior source rows/11110controls must survive actual gzip transport byte exactly')
    source = ROOT / 'reports/full-public-token-suffix-qualification-gzip-recovery-CPU-v2'
    reader = ROOT / 'reports/full-public-token-suffix-readback-gzip-recovery-CPU-v2'
    if (source / 'failure.json').exists() or (reader / 'failure.json').exists():
        raise ValueError('Inspect the exact new failed source owner before claiming live recovery')
    producer_live, reader_live = not terminal(launch['actual_source_producer']), not terminal(launch['actual_independent_fresh_reader'])
    if not producer_live and not (source / 'complete.json').exists() or not reader_live and not (reader / 'complete.json').exists():
        raise ValueError('Each exact new owner must be live or have a complete authoritative result')
    source_observation = read(source / ('complete.json' if (source / 'complete.json').exists() else 'progress.json'))
    if source_observation['actual_counts']['complete_original_sources'] <= 2222:
        raise ValueError('The actual source worker must have progressed beyond its complete retained prefix')
    old_producer, old_reader = ROOT / 'reports/full-public-token-suffix-qualification-CPU-v1', ROOT / 'reports/full-public-token-suffix-readback-CPU-v1'
    if (not terminal(dict(pid=146084, ticks='55111340')) or not terminal(dict(pid=146633, ticks='55128888'))
            or 'Original disk reserve remains unavailable after the bounded wait' not in read(old_producer / 'failure.json')['error']):
        raise ValueError('Original failed source owners must stay terminal and their exact failed gates preserved')
    pixel_root = ROOT / 'reports/full-multimodal-pixel-readback-CPU-v1'
    pixel_path = pixel_root / 'complete.json'
    pixels = read(pixel_path)
    if (not pixels['complete'] or not pixels['passed'] or not terminal(pixels['owner'])
            or pixels['actual_original_physical_paths'] != 195121 or pixels['actual_original_native_views'] != 302627
            or pixels['actual_original_whole_word_views'] != 91585 or pixels['actual_counts']['fresh_integrity_passes'] != 195121
            or pixels['actual_end_file_state_differences'] != 0 or not pixels['all195121physical_sources_freshly_observed']
            or pixels['models_constructed'] != 0 or pixels['training_admitted'] is not False):
        raise ValueError('The full actual independent image/word-source population must finish without state differences')
    pixel_journal = pixel_root / 'all195121-independent-fresh-pixel-observations.jsonl'
    if sha(pixel_journal) != pixels['fresh_observations_sha256']:
        raise ValueError('Every closed independent physical-source observation byte must match')
    human = ROOT / 'reports/phase-human-preference-v8'
    human_plan, human_progress = read(human / 'plan.json'), read(human / 'chosen_CE-progress.json')
    children = [p for p in read(human / 'children.json') if not terminal(p)]
    if (terminal(human_plan['owner']) or (human / 'failure.json').exists()
            or len([p for p in children if p['subject'].startswith('train-chosen_CE-rank-')]) != 4):
        raise ValueError('The original whole human run and all4actual training ranks must remain live')
    at = datetime.now(timezone.utc).isoformat()
    batch_root = ROOT / 'reports/full-corrected-multimodal-batch-readback-CPU-v1'
    admission_root = ROOT / 'reports/full-corrected-multimodal-source-admission-CPU-v1'
    snapshot = dict(at=at, actual_human_progress=human_progress, original_human_owner=human_plan['owner'], actual_live_human_children=children,
        actual_GPU_rows=subprocess.check_output(['nvidia-smi', '--query-gpu=index,name,utilization.gpu,memory.used,memory.total',
            '--format=csv,noheader'], text=True), actual_free_bytes=os.statvfs(ROOT).f_bavail * os.statvfs(ROOT).f_frsize,
        old_source_owner_terminal=True, old_source_reader_terminal=True, exact_new_source_owner=launch['actual_source_producer'],
        exact_new_fresh_reader=launch['actual_independent_fresh_reader'], new_source_owner_live=producer_live,
        new_fresh_reader_live=reader_live, new_source_observation=source_observation,
        new_reader_observation=read(reader / ('complete.json' if (reader / 'complete.json').exists() else 'progress.json')),
        full_visual_pixel_readback_complete=True,
        full_visual_batch_reader_live=not terminal(read(batch_root / 'plan.json')['owner']),
        full_visual_batch_progress=read(batch_root / 'progress.json'),
        full_visual_source_admission_owner_live=not terminal(read(admission_root / 'plan.json')['owner']),
        shared_cache_user_approval_pending=not (ROOT / 'reports/shared-pretrained-cache-reclamation-v1/user-approval.json').exists(),
        shared_pretrained_cache_removal_applied=(ROOT / 'reports/shared-pretrained-cache-reclamation-v1/complete.json').exists(),
        no_existing_model_training_or_neural_evaluation_worker_signalled_reprioritized_restarted_or_patched=True,
        only_own_obsolete_non_neural_metadata_test_interrupted=True, no_new_GPU_context_job_or_lease=True,
        mutable_progress_registries_and_current_compressed_journal_observed_not_source_bound=True)
    requirements = deepcopy(read(PREVIOUS)['requirements'])
    requirements['public_data']['evidence'] += (
        ' A new native-code oracle preserves every3721original human source/2287522contexts/584218CE-EOS targets. '
        'All173documented responses use one exact Python fence; all3548APPS responses are plain Python. Those declared '
        'envelopes preserve original program bytes; arbitrary prose/code search is disabled. Seven source/worker checks '
        'pass, including both actual documented/APPS gold controls, strict EOS and native control-token handling. These '
        'are source/control observations, not sampled student gains. New private identity-parser function bindings leave '
        'all parent scorer globals and existing jobs unchanged. The source profile requires separate complete qualification '
        'before production RL; old extracting-profile results are not retroactively relabelled or waived.')
    requirements['gpu_runtime']['evidence'] += (
        ' The old full-source qualifier and its reader become authoritatively terminal after the preserved40GiB+1GiB '
        'reserve failure. Five lossless-stream and3actual-prefix controls pass. A new source-only recovery preserves all2222 '
        'closed source records/11110raw controls and finishes the remaining1499under the unchanged original oracle. '
        'Independent live-gzip-prefix readback matches every retained byte to the closed original journal. The full raw1GiB '
        'quota and40GiBother-writer floor remain; lossless gzip reserves16MiBfor the next bounded record instead of reserving '
        'the entire raw quota on every append. The old failed gate remains failed. The new producer has made actual progress '
        'past2222; its exact independent reader still requires all18605fresh controls/all14884view dispositions. '
        'Both new jobs are CUDA-hidden nice19single-thread idle CPUsource jobs; no neural worker or GPUlease changed.')
    requirements['text_and_images']['evidence'] += (
        ' The independent physical-pixel reader completes all195121paths/302627native views/91585whole-word views, with0end '
        'file-state differences. Its complete closed observation journal matches its recorded SHA. It constructs0models and '
        'performs0neural forwards; this proves physical/native preprocessing integrity, not labels or visual capability. '
        'The complete corrected4733step/18932rank-call batch reader and separate source admission remain separate gates. '
        'Full visual GPUcapacity/training/export/held-quality remain pending.')
    requirements['serious_full_model_training']['evidence'] += (
        f" Latest original chosen_CE progress:{human_progress['actual_updates']}/5362updates. All85786pairs per arm, complete "
        'code/reasoning/visual source scopes and9final endpoints remain. No new on-policy student group or production RLupdate '
        'is created by these15source/data controls or source recovery. Current coding correctness remains1/48at every completed '
        'coding-study endpoint; no new overall quality improvement or frontier capability is established.')
    for value in requirements.values(): value['previous_evidence'] = str(PREVIOUS)
    if len(requirements) != 10 or any(v['status'] != 'incomplete_for_requested_final_model' for v in requirements.values()):
        raise ValueError('All ten final requirements remain incomplete')
    names = ['src/slm_lab/native_code_execution_oracle_v1.py', 'src/slm_lab/native_code_execution_oracle_v2.py',
        'tests/test_native_code_execution_oracle_v1.py', 'tests/test_native_code_execution_oracle_v2.py',
        'scripts/run_native_code_execution_oracle_v1_tests.py', 'scripts/run_native_code_execution_oracle_v2_tests.py',
        'reports/native-code-source-identity-probe-v1.json', 'reports/native-code-source-identity-probe-v1.stderr',
        'reports/native-code-execution-oracle-v1-tests.log', 'reports/native-code-oracle-v1-source-test-interruption.json',
        'src/slm_lab/bounded_gzip_observation_stream_v1.py', 'tests/test_bounded_gzip_observation_stream_v1.py',
        'scripts/run_bounded_gzip_observation_stream_v1_tests.py', 'scripts/qualify_full_public_token_suffixes_gzip_recovery_CPU_v2.py',
        'scripts/read_full_public_token_suffixes_gzip_recovery_CPU_v2.py', 'scripts/launch_full_token_suffix_gzip_recovery_CPU_v2.py',
        'tests/test_full_token_suffix_gzip_recovery_v2.py', 'scripts/run_full_token_suffix_gzip_recovery_v2_tests.py',
        'reports/shared-pretrained-cache-reclamation-v1/plan.json']
    closed = [Path(__file__), PREVIOUS, format_path, *program_proofs, launch_path, retained_path,
        launch_root / 'closed-original-prefix-readback.json', source / 'retained-original-prefix.json',
        old_producer / 'failure.json', old_reader / 'failure.json', pixel_path, pixel_journal,
        pixel_root / 'fresh-end-file-state-differences.jsonl',
        *[ROOT / n for n in names], *[ROOT / 'reports' / n for n in logs],
        *sorted(proofs_root.rglob('actual*.json')),
        *sorted((ROOT / 'reports/full-token-suffix-gzip-recovery-v2-pytest').rglob('actual*.json'))]
    files = {str(p): sha(p) for p in closed}
    classification = ('progress:7native source/oracle controls preserve all3721human formats and reject extraction/control-token '
        'credit;8lossless source/recovery controls pass; exact2222source prefix survives actual gzip and recovery progresses '
        'under full unchanged scope; independent195121path physical reader completes;0new production RLor accuracy gain')
    OUT.mkdir()
    atomic(OUT / 'runtime-snapshot.json', snapshot)
    atomic(OUT / 'complete.json', dict(complete=True, at=at, source_files_sha256=files,
        actual_native_oracle_controls=7, actual_lossless_stream_controls=5, actual_retained_prefix_controls=3,
        all3721original_source_formats_program_bytes_and_native_masks_reconstructed=True,
        actual_original_context_tokens=2287522, actual_original_CE_EOS_targets=584218,
        both_actual_gold_worker_controls_reward1=True, actual_prose_and_native_control_contrasts_preserved=True,
        no_neural_models_constructed_by_these_tests=True, no_production_code_RLprofile_admission=True,
        exact_retained_original_source_rows=2222, exact_retained_original_reference_prefix_controls=11110,
        actual_remaining_original_source_rows=1499, all_prior_gzip_observation_bytes_equal_original=True,
        original_failed_source_and_reader_preserved=True, new_source_progress_past_retained_prefix=True,
        entire18605fresh_control_independent_readback_still_required=True,
        full195121physical_path302627view91585word_independent_reader_complete=True,
        actual_fresh_pixel_end_file_state_differences=0, fresh_pixel_reader_no_models_or_neural_forwards=True,
        actual_human_first_arm_updates=human_progress['actual_updates'], actual_human_first_arm_planned_updates=5362,
        actual_original_broad_and9final_scopes_preserved=True, shared_cache_removal_user_approval_still_required=True,
        new_student_generated_groups=0, new_production_RL_updates=0, new_model_quality_gain=False,
        all_ten_final_requirements_incomplete=True, goal_achieved=False))
    metrics = (
        f"The original four-L40human first arm is **{human_progress['actual_updates']}/5362updates** at this snapshot. "
        'No existing model-training or neural-evaluation worker was restarted, reprioritized or patched.\n\n'
        '**Seven native code-source/oracle checks pass.** Every3721human source format and native input/label array is checked: '
        '173documented responses have a declared single Python fence;3548APPS responses are plain source. All2287522contexts '
        'and584218positive/EOS targets remain. Declared fence markers are decoded once; program bytes inside remain exact, '
        'without searching elsewhere in the response. Actual documented/APPS worker controls keep gold reward1, give.9without '
        'trueEOS and give0to added prose or generated control tokens. The old extracting profile gives1to those constructed '
        'prose/control inputs. **This is an explicit protocol difference on test inputs, not observed student behavior or a '
        'retroactive correction to the original benchmarks.** Parent scorer globals stay unchanged. Actual nonfinite candidate '
        'observations stay JSON-safe and uncredited. Full new-profile qualification and on-policy gradients remain pending.\n\n'
        'Source checks run before/after a checked gradient-only batch. Per-call role/code bindings still validate; changed '
        'inputs reject at the closing boundary before the caller may use Adam. The obsolete own metadata-only predecessor '
        'test was explicitly interrupted after the successor passed; it built0models. Its source,310.96second interrupted '
        'log and exact-PID explanation remain. The new7checks finish in15.42seconds. **This is no GPUthroughput measurement.**\n\n'
        '**The full code-source chain recovers from an actual disk-reserve failure.** Both original source owners became '
        'terminal and both failure receipts remain. The closed journal contained2222complete rows, beyond its last2208row '
        'progress snapshot. All2222rows/11110reference-prefix controls are reconstructed and copied byte exactly into gzip; '
        'an independent read of the actual live gzip prefix matches the entire closed original journal SHA. The new source '
        f"owner has progressed to{source_observation['actual_counts']['complete_original_sources']}/3721. Its independent reader "
        'will still freshly execute every18605control and rebuild every14884view.\n\n'
        '**Five stream and three retained-prefix tests pass.** Compression preserves byte-LF Unicode, failed/trivial/abstaining '
        'records and the aggregate original1GiBraw quota. The40GiBother-writer floor remains; the new version reserves16MiB '
        'for the next bounded record. The original failed full-quota preflight is preserved, not relabelled passed. All17public '
        'families,3721whole program anchors,85786human pairs, reasoning/visual budgets and9final endpoints remain. Both new '
        'source jobs use CUDA-hidden nice19single-thread idle CPUexecution and own no GPUlease.\n\n'
        '**The independent image/word pixel reader is complete:**195121physical sources,302627native views and91585whole-word '
        'views,0end file-state differences,0models/forwards. The closed complete journal matches its SHA. This establishes '
        'physical/preprocessing integrity; source-label semantics, the separate complete rank-batch/source-admission gates '
        'and visual GPUcapacity/training/quality remain pending.\n\n'
        '**No new student accuracy gain is established.** Coding correctness remains1/48across all four completed coding '
        'endpoints. New production RLupdates:0. The additional32.1GiBshared Qwen-cache removal remains awaiting actual user '
        'approval because the caches sit outside this project. All ten requested final requirements remain incomplete.\n')
    (OUT / 'metrics.md').write_text(metrics)
    for p in [OUT / 'complete.json', OUT / 'runtime-snapshot.json', OUT / 'metrics.md']: files[str(p)] = sha(p)
    atomic(OUT / 'review.json', dict(at=at, requirements=requirements, source_files_sha256=files,
        previous_full_scope_review=str(PREVIOUS), previous_goal_turn_classification=read(PREVIOUS)['current_goal_turn_classification'],
        current_goal_turn_classification=classification, goal_achieved=False))
    path = ROOT / 'reports/completion-audit.json'
    audit = read(path)
    audit.update(at=at, updated_at=at, requirements=requirements, latest_full_scope_review=str(OUT / 'review.json'),
        latest_native_code_oracle_and_gzip_recovery=str(OUT / 'complete.json'),
        next_native_code_execution_oracle=str(ROOT / 'src/slm_lab/native_code_execution_oracle_v2.py'),
        live_full_public_suffix_qualification_directory=str(source), live_full_public_suffix_fresh_reader_directory=str(reader),
        current_goal_turn_classification=classification, goal_achieved=False)
    atomic(path, audit)
    entry = (f'Native code oracle and lossless source recovery, **{at}**:15source/data controls pass. '
        'All3721human source formats/whole CEtargets reconstruct; actual worker controls preserve exact declared programs '
        'and reject extracted-prose/native-control credit. This is not a student quality result. After the original disk '
        'failure, all2222closed source rows/11110controls survive gzip byte exactly; the full source/re-execution chain '
        'continues. Independent195121physical/302627view/91585word pixel readback completes with0end-state differences. '
        f"Original human first arm:{human_progress['actual_updates']}/5362updates. **All ten final requirements remain incomplete.** "
        '[Evidence](reports/native-code-oracle-and-gzip-recovery-v1-review/metrics.md), '
        '[full review](reports/native-code-oracle-and-gzip-recovery-v1-review/review.json). Earlier entries are historical snapshots.\n\n')
    for name in ['README.md', 'RESEARCH.md']:
        p = ROOT / name
        p.write_text(entry + p.read_text())
    print(str(OUT / 'complete.json'))


if __name__ == '__main__': main()
