"""Independent complete native-profile source/control readback and fresh runs.

Wait for the exact native source producer. Retain all original sources, every
conditional view and complete unconditional CE masks. This never starts a
model, grants a GPUlease or admits production RL.
"""
from collections import Counter
from contextlib import ExitStack
from datetime import datetime, timezone
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'reports/full-native-token-suffix-qualification-gzip-recovery-CPU-v3'
OUT = ROOT / 'reports/full-native-token-suffix-readback-gzip-recovery-CPU-v3'
PARENT = None
LEGACY_SOURCE = ROOT / 'reports/full-public-token-suffix-qualification-gzip-recovery-CPU-v2'
LEGACY_READER = ROOT / 'reports/full-public-token-suffix-readback-gzip-recovery-CPU-v4'
RESERVE = 40 * 1024**3
MAX_RAW = 2 * 1024**3


def main():
    if (OUT.exists() or os.environ.get('CUDA_VISIBLE_DEVICES') != '' or os.nice(0) < 19
            or any(os.environ.get(k) != '1' for k in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'])):
        raise ValueError('A fresh CUDA-hidden nice19 single-thread idle source reader is required')
    os.sched_setscheduler(0, os.SCHED_IDLE, os.sched_param(0))
    from record_visual_stage_and_first_repair_arm_v1 import atomic, read, sha
    from slm_lab.exact_waiting_process_retirement_v1 import terminal
    previous = ROOT / 'reports/full-native-token-suffix-readback-CPU-v2'
    previous_failure = read(previous / 'failure.json')
    if (previous_failure['owner'] != dict(pid=183192, ticks='56951924')
            or not terminal(previous_failure['owner']) or (previous / 'complete.json').exists()):
        raise ValueError('The exact previous failed native reader must remain failed and terminal')
    retained_folder = ROOT / 'reports/full-native-token-suffix-qualification-CPU-v2'
    retained_path = retained_folder / 'all14884-native-profile-view-controls.jsonl.gz'
    retained_manifest = retained_folder / 'closed-source-bindings.json'
    retained_count = read(SOURCE / 'plan.json')['actual_retained_closed_native_views']
    retained_tests = ROOT / 'reports/recovered-native-source-controls-v1-tests.log'
    if '3 passed' not in retained_tests.read_text(): raise ValueError('Actual retained native source/semantic tests required')
    original = ROOT / 'reports/full-native-token-suffix-readback-CPU-v1'
    old_failure = read(original / 'failure.json')
    if (not terminal(old_failure['owner']) or old_failure['owner'] != dict(pid=166298, ticks='56079802')):
        raise ValueError('The original failed native reader must remain terminal and preserved')
    for name in ['closed-source-recovery-guard-v1-tests.log', 'recovered-legacy-suffix-catalogue-v1-tests.log']:
        if '3 passed' not in (ROOT / 'reports' / name).read_text():
            raise ValueError('Actual recovered catalogue and complete source-boundary controls must pass')
    if read(SOURCE / 'plan.json')['owner'] != PARENT:
        raise ValueError('The exact original native-profile source producer birth is required')
    tests = ROOT / 'reports/independent-native-suffix-controls-v1-tests.log'
    if '6 passed' not in tests.read_text(): raise ValueError('All actual independent native source/control checks must pass')
    names = ['src/slm_lab/independent_native_suffix_controls_v1.py', 'src/slm_lab/independent_token_suffix_readback_v1.py',
        'tests/test_independent_native_suffix_controls_v1.py', 'scripts/run_independent_native_suffix_controls_v1_tests.py']
    extra = [previous / 'failure.json', previous / 'plan.json', retained_path, retained_manifest, retained_tests, ROOT / 'src/slm_lab/recovered_legacy_suffix_catalogue_v1.py',
        ROOT / 'src/slm_lab/closed_source_recovery_guard_v1.py', original / 'failure.json',
        ROOT / 'reports/closed-source-recovery-guard-v1-tests.log',
        ROOT / 'reports/recovered-legacy-suffix-catalogue-v1-tests.log']
    files = {str(p): sha(p) for p in [Path(__file__), SOURCE / 'plan.json', tests, *[ROOT / n for n in names], *extra]}
    from slm_lab.local_source_closure_v2 import python_sources
    for path in python_sources(ROOT, [Path(__file__)]): files[str(path)] = sha(path)
    states = {p: (Path(p).stat().st_ino, Path(p).stat().st_size, Path(p).stat().st_mtime_ns) for p in files}
    def waiting_guard():
        if os.sched_getscheduler(0) != os.SCHED_IDLE or any(
                (Path(p).stat().st_ino, Path(p).stat().st_size, Path(p).stat().st_mtime_ns) != s for p, s in states.items()):
            raise ValueError('Every exact immutable reader/source input and idle context must remain')
    owner = dict(pid=os.getpid(), ticks=Path('/proc/self/stat').read_text().split(') ', 1)[1].split()[19])
    OUT.mkdir()
    atomic(OUT / 'plan.json', dict(owner=owner, exact_original_native_source_producer=PARENT,
        at=datetime.now(timezone.utc).isoformat(), source_files_sha256=files,
        all_original_sources=3721, all_native_views=14884, original_context_tokens=2287522,
        original_unconditional_response_EOS_targets=584218, independent_native_descriptor_builder=True,
        all_original_native_controls_and_both_legacy_source_observations_rechecked=True,
        new_fresh_gold_and_prefix_run_for_every_previously_attempted_view=True,
        every_failed_trivial_abstaining_or_disagreeing_view_retained=True,
        failed_original_reader_preserved=old_failure, previous_failed_native_reader_preserved=previous_failure,
        actual_retained_source_prefix_views_to_compare_byte_exact=retained_count, recovered_legacy_reader_used_explicitly=True,
        complete_input_hash_validation_before_execution_and_before_publication=True,
        original_sources_per_checked_native_oracle_batch=16,
        maximum_additional_raw_output_bytes=MAX_RAW, minimum_other_writer_reserve_bytes=RESERVE,
        source_training_admitted=False, production_RL_admitted=False, models_constructed=0,
        neural_forwards=0, GPUleases=0, optimizer_updates=0))
    try:
        deadline = time.monotonic() + 7 * 86400
        while not terminal(PARENT):
            waiting_guard()
            if time.monotonic() > deadline: raise ValueError('The source-only readback waiting deadline expired')
            atomic(OUT / 'progress.json', dict(owner=owner, phase='waiting for exact complete native-profile source producer',
                exact_original_producer_live=True, models_constructed=0, fresh_native_runs=0, source_training_admitted=False))
            time.sleep(15)
        if (SOURCE / 'failure.json').exists() or not (SOURCE / 'complete.json').exists():
            raise ValueError('The exact native source producer did not finish; preserve every failed or partial observation')
        complete = read(SOURCE / 'complete.json')
        manifest = SOURCE / 'closed-source-bindings.json'
        stream_path = Path(complete['complete_control_file'])
        if (not complete['complete'] or complete['owner'] != PARENT or complete['source_training_admitted'] is not False
                or complete['actual_counts']['all_original_sources'] != 3721 or complete['actual_counts']['all_original_views'] != 14884
                or not complete['every_source_view_and_CE_target_retained']
                or sha(manifest) != complete['closed_source_manifest_sha256']
                or read(manifest)['source_files_sha256'] != complete['source_files_sha256']
                or sha(stream_path) != complete['complete_control_file_sha256']):
            raise ValueError('The entire original closed native-profile source population and every actual control byte must match')
        import torch
        from slm_lab.active_execution_input_bindings_v1 import input_bindings, merge_bindings
        from slm_lab.bounded_disk_reserve_wait_v1 import wait_for_reserve
        from slm_lab.bounded_gzip_observation_stream_v1 import Writer, RawBudget, NEXT_RECORD, lines
        from slm_lab.closed_source_recovery_guard_v1 import SourceOnlyRecoveryGuard
        from slm_lab.expanded_execution_corpus_v1 import ExpandedExecutionCorpus
        from slm_lab.independent_native_suffix_controls_v1 import check, delivery, agrees
        from slm_lab.independent_token_suffix_readback_v1 import rebuild
        from slm_lab.native_code_execution_oracle_v2 import NativeCodeExecutionOracle
        from slm_lab.parallel_function_repair_stage_v1 import value_digest
        from slm_lab.qualified_native_token_suffix_episode_v1 import PROOF
        from slm_lab.recovered_legacy_suffix_catalogue_v1 import load_recovered_legacy_catalogue
        torch.set_num_threads(1)
        corpus = ExpandedExecutionCorpus(ROOT)
        oracle = NativeCodeExecutionOracle(corpus)
        catalogue = load_recovered_legacy_catalogue(oracle)
        for mapping in [complete['source_files_sha256'], input_bindings(ROOT, corpus), oracle.sources, catalogue.files]:
            merge_bindings(files, mapping)
        files.update({str(p): sha(p) for p in [SOURCE / 'complete.json', manifest, stream_path, OUT / 'plan.json']})
        def resources():
            if os.sched_getscheduler(0) != os.SCHED_IDLE or torch.cuda.is_initialized():
                raise ValueError('Only source-only idle CPU readback may produce native control evidence')
        guard = SourceOnlyRecoveryGuard(files, resources)
        guard.check()
        atomic(OUT / 'closed-source-bindings.json', dict(source_files_sha256=files, complete_legacy_catalogue_sha256=catalogue.digest))
        def reserve(): wait_for_reserve(ROOT, minimum_bytes=RESERVE + NEXT_RECORD, guard=resources)
        def deliver(position, native, actions, origin):
            try:
                return oracle.delivery(position, native['original_reference_prefix_ids'], actions,
                    delivered_action_budget=native['metadata']['full_native_action_budget'], origin=origin)
            except Exception as exc:
                return dict(reward=None, status='preserved_native_control_adapter_exception', error=repr(exc),
                    requested_source_position=position, requested_public_reference_ids=native['original_reference_prefix_ids'],
                    requested_native_actions=list(actions), requested_action_budget=native['metadata']['full_native_action_budget'], requested_origin=origin)
        native_rows = iter(lines(stream_path))
        legacy_original = iter(lines(Path(read(LEGACY_SOURCE / 'complete.json')['complete_control_file'])))
        legacy_fresh = iter(lines(Path(read(LEGACY_READER / 'complete.json')['actual_fresh_execution_control_file'])))
        expected_proof_files = {str(manifest): sha(manifest)}
        retained_proof_files = {str(retained_manifest): sha(retained_manifest)}
        if complete['actual_retained_closed_native_views'] != retained_count or not 0 < retained_count < 14884:
            raise ValueError('The complete actual retained native prefix count must remain')
        retained_rows = iter(lines(retained_path))
        counts, totals = Counter(), {str(level): Counter() for level in [4, 16, 64, 0]}
        journal_path = OUT / 'all14884-independent-native-view-controls.jsonl.gz'
        with Writer(journal_path, root=ROOT, budget=RawBudget(MAX_RAW), reserve_bytes=RESERVE, guard=resources) as journal, ExitStack() as checked_sources:
            for position in range(len(corpus)):
                reserve()
                guard.check()
                if position % 16 == 0:
                    checked_sources.close()
                    checked_sources.enter_context(oracle.checked_source_batch())
                row = corpus.row(position)
                ids, labels = corpus.sequence(position)
                a, b = next(legacy_original)[1], next(legacy_fresh)[1]
                if (a['source_position'] != position or b['source_position'] != position
                        or a['corpus_id'] != row['id'] or b['corpus_id'] != row['id']):
                    raise ValueError('Every original and fresh legacy source identity must remain ordered and exact')
                for j, level in enumerate([4, 16, 64, 0]):
                    raw_observed, observed = next(native_rows)
                    ordinal = position * 4 + j
                    if ordinal < retained_count:
                        raw_retained, _ = next(retained_rows)
                        if raw_observed != raw_retained:
                            raise ValueError('Every complete retained producer record must remain byte-LF exact')
                        actual_proof_files = retained_proof_files
                        counts['actual_original_retained_native_views_byte_compared'] += 1
                    else:
                        actual_proof_files = expected_proof_files
                    proof = observed['native_profile_control_proof']
                    original = catalogue.rows[(position, level)]
                    metadata = proof['native_view']
                    native = rebuild(corpus.tokenizer, row['original'], ids, labels, level, metadata)
                    expected_attempt = original['bounded_nontrivial_control_evidence'] and not any(v in [0, 1, 3] for v in ids[1:-1])
                    if (observed['source_position'] != position or observed['level'] != level or observed['corpus_id'] != row['id']
                            or observed['original_legacy_disposition'] != original or metadata != original['exact_native_descriptor']
                            or proof['format'] != PROOF or proof['scope'] != 'complete-source-catalogue'
                            or proof['source_position'] != position or proof['nominal_missing_non_EOS_tokens'] != level
                            or proof['source_information'] != oracle.source_information(position) or proof['oracle_binding'] != oracle.binding()
                            or proof['source_files_sha256'] != actual_proof_files or proof['complete_legacy_catalogue_sha256'] != catalogue.digest
                            or proof['source_training_admitted'] is not False or proof['production_RL_admitted'] is not False
                            or proof['model_generations'] != 0 or proof['model_updates'] != 0
                            or observed['all_original_unconditional_CE_targets_retained'] is not True
                            or bool(proof['actual_controls']) != bool(expected_attempt)):
                        raise ValueError('Every exact native source/proof/oracle/full mask/budget and legacy disposition must reconstruct')
                    controls = proof['actual_controls']
                    old_qualified, fresh_gold, fresh_prefix, same, qualified = False, None, None, False, False
                    if expected_attempt:
                        native, old_qualified = check(corpus, oracle, position, level, metadata, controls)
                        old = [a['views'][j]['counterfactual_prefix_then_EOS_execution'], b['all_four_fresh_views'][j]['fresh_prefix_execution']]
                        for repeat, control in enumerate(controls):
                            if (control['legacy_prefix_then_EOS'] != old[repeat]
                                    or control['retained_legacy_origin'] != ['original_producer', 'independent_fresh_reader'][repeat]):
                                raise ValueError('Both actually retained legacy controls must match their original closed observations')
                        fresh_gold = deliver(position, native, native['original_missing_suffix_and_EOS_ids'], 'original_reference_control')
                        fresh_prefix = deliver(position, native, [2], 'test_fixture')
                        delivery(corpus, oracle, position, native, fresh_gold, native['original_missing_suffix_and_EOS_ids'], 'original_reference_control')
                        delivery(corpus, oracle, position, native, fresh_prefix, [2], 'test_fixture')
                        same = agrees(controls, fresh_gold, fresh_prefix)
                        qualified = bool(old_qualified and same and fresh_gold.get('reward') == 1
                            and fresh_prefix.get('reward') is not None and 0 <= fresh_prefix['reward'] < 1)
                        counts['actual_fresh_native_gold_runs'] += 1
                        counts['actual_fresh_native_prefix_runs'] += 1
                        counts['native_execution_disagreements_preserved'] += int(not same)
                    if observed['eligible_for_later_native_profile'] is not old_qualified or proof['complete'] is not old_qualified:
                        raise ValueError('Every original qualification boolean must reconstruct from the actual retained controls')
                    disposition = ('native_controls_agree_and_nontrivial' if qualified else 'legacy_or_boundary_CE_only' if not expected_attempt
                        else 'native_execution_disagreement_CE_only' if not same else 'native_failed_trivial_or_abstaining_CE_only')
                    journal.append(dict(source_position=position, corpus_id=row['id'], level=level,
                        independent_native_descriptor=metadata, actual_closed_producer_proof_sha256=value_digest(proof),
                        original_native_profile_eligible=old_qualified, actual_fresh_native_gold=fresh_gold,
                        actual_fresh_native_prefix=fresh_prefix, all_original_actual_control_semantics_agree=same,
                        native_profile_eligible_in_both_readers=qualified, disposition=disposition,
                        all_original_unconditional_CE_targets_retained=True, source_training_admitted=False, production_RL_admitted=False))
                    counts['all_original_views_rebuilt'] += 1
                    counts[disposition] += 1
                    totals[str(level)].update(views=1, eligible=int(qualified), original_context_tokens=len(ids),
                        original_response_EOS_targets=sum(v != -100 for v in labels))
                counts['all_original_sources'] += 1
                counts['original_context_tokens'] += len(ids)
                counts['original_response_EOS_targets'] += sum(v != -100 for v in labels)
                if (position + 1) % 16 == 0 or position + 1 == len(corpus):
                    journal.fsync()
                    atomic(OUT / 'progress.json', dict(owner=owner, phase='all original native views rebuilt and controls freshly run',
                        actual_counts=dict(counts), actual_level_totals={k: dict(v) for k, v in totals.items()},
                        source_training_admitted=False, models_constructed=0, neural_forwards=0))
            if any(next(iterator, None) is not None for iterator in [native_rows, legacy_original, legacy_fresh, retained_rows]):
                raise ValueError('Every complete source stream must end at exactly the full original population')
        guard.verify_final()
        if (counts['all_original_sources'] != 3721 or counts['all_original_views_rebuilt'] != 14884
                or counts['original_context_tokens'] != 2287522 or counts['original_response_EOS_targets'] != 584218
                or counts['actual_original_retained_native_views_byte_compared'] != retained_count
                or 2 * counts['actual_fresh_native_gold_runs'] != (complete['actual_counts']['new_actual_native_gold_runs']
                    + complete['actual_counts']['actual_retained_native_gold_control_invocations'])
                or 2 * counts['actual_fresh_native_prefix_runs'] != (complete['actual_counts']['new_actual_native_prefix_runs']
                    + complete['actual_counts']['actual_retained_native_prefix_control_invocations'])
                or any(v['views'] != 3721 or v['original_context_tokens'] != 2287522
                    or v['original_response_EOS_targets'] != 584218 for v in totals.values())):
            raise ValueError('Every original source/view/full CEtarget and newly run control must reconcile')
        atomic(OUT / 'complete.json', dict(complete=True, integrity_readback_passed=True, owner=owner,
            at=datetime.now(timezone.utc).isoformat(), source_files_sha256=files, actual_counts=dict(counts),
            actual_level_totals={k: dict(v) for k, v in totals.items()}, complete_control_file=str(journal_path),
            complete_control_file_sha256=sha(journal_path), lossless_observation_stream=journal.receipt(),
            all_original_sources_views_full_CE_masks_and_failed_dispositions_retained=True,
            previous_failed_native_reader_preserved=True, actual_retained_prefix_records_compared_byte_exact=retained_count,
            actual_program_test_output_verdict_semantics_compared_without_walltime_equality=True,
            source_training_admitted=False, production_RL_admitted=False, models_constructed=0, neural_forwards=0,
            student_generations=0, GPUleases=0, optimizer_updates=0,
            separate_complete_source_admission_and_full_GPUruntime_still_required=True, goal_achieved=False))
        print(str(OUT / 'complete.json'), flush=True)
    except BaseException as exc:
        atomic(OUT / 'failure.json', dict(owner=owner, error=repr(exc), automatic_retry=False,
            all_actual_source_attempts_preserved=True, source_training_admitted=False, production_RL_admitted=False, goal_achieved=False))
        raise


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--parent-pid', type=int, required=True)
    parser.add_argument('--parent-ticks', required=True)
    args = parser.parse_args()
    PARENT = dict(pid=args.parent_pid, ticks=args.parent_ticks)
    main()
