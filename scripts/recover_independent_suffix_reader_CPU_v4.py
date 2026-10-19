"""Finish a terminal disk-failed reader without repeating its closed controls.

Write a separate recovered catalogue. The original failure and every original
journal stay unchanged; existing failed successors are neither resumed nor
made successful. The recovery does not admit native-profile data or GPU work.
"""
from collections import Counter
from datetime import datetime, timezone
import os
from pathlib import Path

from record_visual_stage_and_first_repair_arm_v1 import atomic, read, sha

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'reports/full-public-token-suffix-qualification-gzip-recovery-CPU-v2'
OLD = ROOT / 'reports/full-public-token-suffix-readback-gzip-recovery-CPU-v2'
OUT = ROOT / 'reports/full-public-token-suffix-readback-gzip-recovery-CPU-v4'
REPLACED = ROOT / 'reports/full-public-token-suffix-readback-gzip-recovery-CPU-v3'
PARENT = dict(pid=158779, ticks='55770044')
RESERVE = 40 * 1024**3
MAX_RAW = 1024**3
FRESH = 'all3721-fresh-reference-and14884-fresh-prefix-controls.jsonl.gz'
CATALOGUE = 'all14884-view-dispositions.jsonl.gz'


def main():
    if (OUT.exists() or os.environ.get('CUDA_VISIBLE_DEVICES') != '' or os.nice(0) < 19
            or any(os.environ.get(k) != '1' for k in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'])):
        raise ValueError('A fresh CUDA-hidden nice19 one-thread recovery owner is required')
    os.sched_setscheduler(0, os.SCHED_IDLE, os.sched_param(0))
    from slm_lab.exact_waiting_process_retirement_v1 import terminal
    if not terminal(PARENT) or read(OLD / 'plan.json')['owner'] != PARENT:
        raise ValueError('The exact failed original reader must already be terminal')
    failure = read(OLD / 'failure.json')
    if (failure['owner'] != PARENT or (OLD / 'complete.json').exists()
            or 'Original disk reserve remains unavailable after the bounded wait' not in failure['error']):
        raise ValueError('Only the preserved actual terminal disk failure may be recovered')
    completed = read(SOURCE / 'complete.json')
    if (not terminal(completed['owner']) or not completed['complete'] or (SOURCE / 'failure.json').exists()
            or completed['owner'] != read(SOURCE / 'plan.json')['owner']
            or completed['actual_counts']['complete_original_sources'] != 3721
            or completed['actual_counts']['full_reference_runs'] != 3721
            or completed['actual_counts']['prefix_control_runs'] != 14884
            or completed['source_training_admitted'] is not False
            or completed['all_original3721sources_and_every_unconditional_CE_target_preserved'] is not True):
        raise ValueError('The complete original unadmitted producer is required')
    tests = ROOT / 'reports/independent-suffix-reader-recovery-v1-tests.log'
    if '5 passed' not in tests.read_text():
        raise ValueError('Actual retained-byte, reconstruction, rejection and new-worker controls must pass')
    guard_tests = ROOT / 'reports/closed-source-recovery-guard-v1-tests.log'
    if '3 passed' not in guard_tests.read_text():
        raise ValueError('Actual complete boundary hashing, byte corruption and resource/thread controls must pass')
    replaced = read(REPLACED / 'plan.json')['owner']
    if (not terminal(replaced) or read(REPLACED / 'failure.json')['owner'] != replaced
            or read(REPLACED / 'failure.json')['error'] != 'KeyboardInterrupt()'):
        raise ValueError('The exact inefficient owned recovery trial must be terminal and its interruption preserved')
    import torch
    from slm_lab.active_execution_input_bindings_v1 import input_bindings, merge_bindings
    from slm_lab.bounded_disk_reserve_wait_v1 import wait_for_reserve
    from slm_lab.bounded_gzip_observation_stream_v1 import Writer, RawBudget, NEXT_RECORD, lines
    from slm_lab.closed_source_recovery_guard_v1 import SourceOnlyRecoveryGuard
    from slm_lab.expanded_execution_corpus_v1 import ExpandedExecutionCorpus
    from slm_lab.independent_suffix_reader_recovery_v1 import validate_retained, fresh, LEVELS
    from slm_lab.local_source_closure_v2 import python_sources
    torch.set_num_threads(1)
    if torch.cuda.is_initialized():
        raise ValueError('Source recovery cannot initialize CUDA')
    corpus = ExpandedExecutionCorpus(ROOT)
    files = {}
    for mapping in [read(OLD / 'closed-source-bindings.json')['source_files_sha256'],
            completed['source_files_sha256'], input_bindings(ROOT, corpus)]:
        merge_bindings(files, mapping)
    metadata_path, controls_path = Path(completed['actual_metadata_path']), Path(completed['complete_control_file'])
    if (sha(metadata_path) != completed['actual_all14884_native_descriptors_sha256']
            or sha(controls_path) != completed['actual_all18605_runtime_controls_sha256']):
        raise ValueError('All original closed producer observations must retain their bytes')
    seeds = [Path(__file__), ROOT / 'tests/test_independent_suffix_reader_recovery_v1.py']
    paths = [*python_sources(ROOT, seeds), SOURCE / 'complete.json', SOURCE / 'plan.json',
        OLD / 'plan.json', OLD / 'failure.json', OLD / 'progress.json', OLD / 'closed-source-bindings.json',
        OLD / FRESH, OLD / CATALOGUE, metadata_path, controls_path, tests,
        ROOT / 'scripts/run_independent_suffix_reader_recovery_v1_tests.py', guard_tests,
        ROOT / 'tests/test_closed_source_recovery_guard_v1.py', ROOT / 'scripts/run_closed_source_recovery_guard_v1_tests.py',
        ROOT / 'scripts/recover_independent_suffix_reader_CPU_v3.py',
        REPLACED / 'plan.json', REPLACED / 'failure.json', REPLACED / 'closed-source-bindings.json',
        REPLACED / FRESH, REPLACED / CATALOGUE,
        ROOT / 'reports/independent-suffix-reader-recovery-v3-io-regression/operator-stop.json']
    merge_bindings(files, {str(p): sha(p) for p in paths})
    def resource_only():
        if os.sched_getscheduler(0) != os.SCHED_IDLE or torch.cuda.is_initialized():
            raise ValueError('Recovery must keep idle scheduling and its CUDA-free context')

    # FileBindings hashes every payload during initialization. Validate all
    # bytes again before publication; between these boundaries only immutable
    # observations are reconstructed, with no model or optimizer operations.
    guard = SourceOnlyRecoveryGuard(files, resource_only)
    owner = dict(pid=os.getpid(), ticks=Path('/proc/self/stat').read_text().split(') ', 1)[1].split()[19])
    OUT.mkdir()
    atomic(OUT / 'plan.json', dict(owner=owner, at=datetime.now(timezone.utc).isoformat(),
        exact_failed_original_reader=PARENT, exact_complete_original_producer=completed['owner'],
        source_files_sha256=files, original_failure_preserved=failure, original_sources=3721, native_views=14884,
        original_complete_reference_prefix_controls=18605, complete_context_tokens_per_level=2287522,
        complete_response_EOS_targets_per_level=584218, retain_exact_closed_byte_LF_observations=True,
        independently_reconstruct_every_retained_descriptor_and_disposition=True,
        new_executions_only_for_unfinished_original_sources=True, original_raw_quota=MAX_RAW,
        original_other_writer_reserve_bytes=RESERVE, source_training_admitted=False,
        complete_all_input_SHA_checks_before_reading_and_before_publication=True,
        source_metadata_checks_every_at_most_64_resource_calls=True,
        exact_earlier_owned_recovery_interruption_preserved=replaced,
        production_RL_admitted=False, models_constructed=0, neural_forwards=0, GPUcontexts=0, goal_achieved=False))
    files[str(OUT / 'plan.json')] = sha(OUT / 'plan.json')
    guard = SourceOnlyRecoveryGuard(files, resource_only)
    atomic(OUT / 'closed-source-bindings.json', dict(owner=owner, source_files_sha256=files,
        exact_original_reader_terminal=True, exact_original_reader_failure_preserved=True))

    def resources():
        guard.check()

    def reserve():
        wait_for_reserve(ROOT, minimum_bytes=RESERVE + NEXT_RECORD, guard=resources)

    counts, retained_counts, new_counts = Counter(), Counter(), Counter()
    totals = {str(level): Counter() for level in LEVELS}
    discrepancies = []
    controls, metadata = iter(lines(controls_path)), iter(lines(metadata_path))
    old_fresh, old_catalogue = iter(lines(OLD / FRESH)), iter(lines(OLD / CATALOGUE))
    budget = RawBudget(MAX_RAW)
    old_prefix_finished = False
    try:
        reserve()
        with Writer(OUT / FRESH, root=ROOT, budget=budget, reserve_bytes=RESERVE, guard=resources) as journal, Writer(
                OUT / CATALOGUE, root=ROOT, budget=budget, reserve_bytes=RESERVE, guard=resources) as catalogue:
            for position in range(len(corpus)):
                reserve()
                original = next(controls)[1]
                md = [next(metadata)[1] for _ in LEVELS]
                retained = next(old_fresh, None) if not old_prefix_finished else None
                if retained is not None:
                    raw_catalogue = [next(old_catalogue) for _ in LEVELS]
                    result = validate_retained(corpus, position, original, md, retained[1], [r for _, r in raw_catalogue])
                    for raw, _ in raw_catalogue: catalogue.append_raw(raw)
                    journal.append_raw(retained[0])
                    retained_counts.update(result['counts'])
                else:
                    if not old_prefix_finished:
                        if next(old_catalogue, None) is not None:
                            raise ValueError('Preserve and inspect orphaned partial catalogue observations; never discard them')
                        old_prefix_finished = True
                        if not 0 < retained_counts['complete_original_sources'] < 3721:
                            raise ValueError('A genuine partial closed prefix is required')
                        atomic(OUT / 'retained-prefix.json', dict(owner=owner,
                            actual_retained_counts=dict(retained_counts), closed_original_prefix_bytes_preserved=True,
                            all_closed_masks_and_dispositions_reconstructed=True, old_progress_not_used_as_actual_row_count=True,
                            original_failure_preserved=True, new_execution_runs=0, neural_forwards=0))
                    result = fresh(corpus, position, original, md)
                    for record in result['catalogue']: catalogue.append(record)
                    journal.append(result['record'])
                    new_counts.update(result['counts'])
                counts.update(result['counts'])
                for level in LEVELS: totals[str(level)].update(result['level_totals'][str(level)])
                discrepancies.extend(result['discrepancies'])
                if (position + 1) % 64 == 0 or old_prefix_finished:
                    journal.fsync(); catalogue.fsync()
                    atomic(OUT / 'progress.json', dict(owner=owner,
                        phase='retaining exact independently reconstructed observations' if not old_prefix_finished else 'actual remaining-source worker executions',
                        actual_counts=dict(counts), actual_retained_counts=dict(retained_counts),
                        actual_new_counts=dict(new_counts), all_fresh_discrepancies_preserved=len(discrepancies),
                        source_training_admitted=False, neural_forwards=0))
            if (next(controls, None) is not None or next(metadata, None) is not None
                    or next(old_fresh, None) is not None or next(old_catalogue, None) is not None):
                raise ValueError('No hidden or excess source/control/metadata observations are allowed')
        if (not old_prefix_finished or counts['complete_original_sources'] != 3721
                or counts['fresh_full_reference_runs'] != 3721 or counts['fresh_prefix_control_runs'] != 14884
                or counts['all_native_views_rebuilt'] != 14884
                or any(v != dict(views=3721, original_context_tokens=2287522, original_response_EOS_targets=584218)
                    for v in totals.values())):
            raise ValueError('All original sources, full contexts, complete CE targets and every disposition must remain')
        reserve(); guard.verify_final()
        atomic(OUT / 'complete.json', dict(complete=True, integrity_readback_passed=True,
            at=datetime.now(timezone.utc).isoformat(), owner=owner, source_files_sha256=files,
            actual_counts=dict(counts), actual_retained_counts=dict(retained_counts), actual_new_counts=dict(new_counts),
            original_disk_failed_reader=PARENT, original_failure_preserved=failure,
            retained_controls_not_relabelled_as_new_executions=True,
            all_four_level_metadata_totals={k: dict(v) for k, v in totals.items()},
            actual_fresh_execution_discrepancies=discrepancies, all_fresh_discrepancies_preserved=True,
            actual_all3721source14884view_dispositions_sha256=sha(OUT / CATALOGUE),
            actual_all18605fresh_execution_controls_sha256=sha(OUT / FRESH),
            actual_fresh_execution_control_file=str(OUT / FRESH), actual_view_disposition_file=str(OUT / CATALOGUE),
            lossless_observation_streams=dict(fresh=journal.receipt(), catalog=catalogue.receipt()),
            every_original_unconditional_CE_context_and_target_retained=True,
            no_source_or_level_dropped_for_failure_triviality_abstention_or_disagreement=True,
            source_training_admitted=False, production_RL_admitted=False, student_generations=0,
            neural_forwards=0, optimizer_updates=0, GPUcontexts=0,
            recovered_catalogue_consumers_require_explicit_new_source_bindings=True,
            native_profile_controls_data_admission_and_full_GPUcapacity_still_required=True, goal_achieved=False))
        print(dict(complete=True, actual_counts=dict(counts), retained_sources=retained_counts['complete_original_sources'],
            genuinely_new_sources=new_counts['complete_original_sources'], new_worker_runs=5 * new_counts['complete_original_sources']), flush=True)
    except BaseException as exc:
        atomic(OUT / 'failure.json', dict(owner=owner, error=repr(exc), automatic_retry=False,
            actual_counts=dict(counts), actual_retained_counts=dict(retained_counts), actual_new_counts=dict(new_counts),
            all_original_and_new_partial_observations_preserved=True, source_training_admitted=False,
            production_RL_admitted=False, goal_achieved=False))
        raise


if __name__ == '__main__':
    main()
