"""Independent complete native-profile source/control readback and fresh runs.

Wait for the exact native source producer. Retain all original sources, every
conditional view and complete unconditional CE masks. This never starts a
model, grants a GPUlease or admits production RL.
"""
from collections import Counter
from datetime import datetime, timezone
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'reports/full-native-token-suffix-qualification-CPU-v1'
OUT = ROOT / 'reports/full-native-token-suffix-readback-CPU-v1'
PARENT = dict(pid=162287, ticks='55947633')
LEGACY_SOURCE = ROOT / 'reports/full-public-token-suffix-qualification-gzip-recovery-CPU-v2'
LEGACY_READER = ROOT / 'reports/full-public-token-suffix-readback-gzip-recovery-CPU-v2'
RESERVE = 40 * 1024**3
MAX_RAW = 2 * 1024**3


def main():
    if (OUT.exists() or os.environ.get('CUDA_VISIBLE_DEVICES') != '' or os.nice(0) < 19
            or any(os.environ.get(k) != '1' for k in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'])):
        raise ValueError('A fresh CUDA-hidden nice19 single-thread idle source reader is required')
    os.sched_setscheduler(0, os.SCHED_IDLE, os.sched_param(0))
    from record_visual_stage_and_first_repair_arm_v1 import atomic, read, sha
    from slm_lab.exact_waiting_process_retirement_v1 import terminal
    if read(SOURCE / 'plan.json')['owner'] != PARENT:
        raise ValueError('The exact original native-profile source producer birth is required')
    tests = ROOT / 'reports/independent-native-suffix-controls-v1-tests.log'
    if '6 passed' not in tests.read_text(): raise ValueError('All actual independent native source/control checks must pass')
    names = ['src/slm_lab/independent_native_suffix_controls_v1.py', 'src/slm_lab/independent_token_suffix_readback_v1.py',
        'tests/test_independent_native_suffix_controls_v1.py', 'scripts/run_independent_native_suffix_controls_v1_tests.py']
    files = {str(p): sha(p) for p in [Path(__file__), SOURCE / 'plan.json', tests, *[ROOT / n for n in names]]}
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
        from slm_lab.codenet_checkpoint_v1 import FileBindings
        from slm_lab.expanded_execution_corpus_v1 import ExpandedExecutionCorpus
        from slm_lab.independent_native_suffix_controls_v1 import check, delivery, agrees
        from slm_lab.independent_token_suffix_readback_v1 import rebuild
        from slm_lab.native_code_execution_oracle_v2 import NativeCodeExecutionOracle
        from slm_lab.parallel_function_repair_stage_v1 import value_digest
        from slm_lab.qualified_native_token_suffix_episode_v1 import CompleteLegacySuffixCatalogue, PROOF
        torch.set_num_threads(1)
        corpus = ExpandedExecutionCorpus(ROOT)
        oracle = NativeCodeExecutionOracle(corpus)
        catalogue = CompleteLegacySuffixCatalogue(oracle)
        for mapping in [complete['source_files_sha256'], input_bindings(ROOT, corpus), oracle.sources, catalogue.files]:
            merge_bindings(files, mapping)
        files.update({str(p): sha(p) for p in [SOURCE / 'complete.json', manifest, stream_path, OUT / 'plan.json']})
        guard = FileBindings(files)
        guard.check(full=True)
        atomic(OUT / 'closed-source-bindings.json', dict(source_files_sha256=files, complete_legacy_catalogue_sha256=catalogue.digest))
        def resources():
            if os.sched_getscheduler(0) != os.SCHED_IDLE or torch.cuda.is_initialized():
                raise ValueError('Only source-only idle CPU readback may produce native control evidence')
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
        counts, totals = Counter(), {str(level): Counter() for level in [4, 16, 64, 0]}
        journal_path = OUT / 'all14884-independent-native-view-controls.jsonl.gz'
        with Writer(journal_path, root=ROOT, budget=RawBudget(MAX_RAW), reserve_bytes=RESERVE, guard=resources) as journal:
            for position in range(len(corpus)):
                reserve()
                guard.check()
                row = corpus.row(position)
                ids, labels = corpus.sequence(position)
                a, b = next(legacy_original)[1], next(legacy_fresh)[1]
                if (a['source_position'] != position or b['source_position'] != position
                        or a['corpus_id'] != row['id'] or b['corpus_id'] != row['id']):
                    raise ValueError('Every original and fresh legacy source identity must remain ordered and exact')
                with oracle.checked_source_batch():
                    for j, level in enumerate([4, 16, 64, 0]):
                        observed = next(native_rows)[1]
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
                                or proof['source_files_sha256'] != expected_proof_files or proof['complete_legacy_catalogue_sha256'] != catalogue.digest
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
            if any(next(iterator, None) is not None for iterator in [native_rows, legacy_original, legacy_fresh]):
                raise ValueError('Every complete source stream must end at exactly the full original population')
        guard.check(full=True)
        if (counts['all_original_sources'] != 3721 or counts['all_original_views_rebuilt'] != 14884
                or counts['original_context_tokens'] != 2287522 or counts['original_response_EOS_targets'] != 584218
                or 2 * counts['actual_fresh_native_gold_runs'] != complete['actual_counts']['new_actual_native_gold_runs']
                or 2 * counts['actual_fresh_native_prefix_runs'] != complete['actual_counts']['new_actual_native_prefix_runs']
                or any(v['views'] != 3721 or v['original_context_tokens'] != 2287522
                    or v['original_response_EOS_targets'] != 584218 for v in totals.values())):
            raise ValueError('Every original source/view/full CEtarget and newly run control must reconcile')
        atomic(OUT / 'complete.json', dict(complete=True, integrity_readback_passed=True, owner=owner,
            at=datetime.now(timezone.utc).isoformat(), source_files_sha256=files, actual_counts=dict(counts),
            actual_level_totals={k: dict(v) for k, v in totals.items()}, complete_control_file=str(journal_path),
            complete_control_file_sha256=sha(journal_path), lossless_observation_stream=journal.receipt(),
            all_original_sources_views_full_CE_masks_and_failed_dispositions_retained=True,
            actual_program_test_output_verdict_semantics_compared_without_walltime_equality=True,
            source_training_admitted=False, production_RL_admitted=False, models_constructed=0, neural_forwards=0,
            student_generations=0, GPUleases=0, optimizer_updates=0,
            separate_complete_source_admission_and_full_GPUruntime_still_required=True, goal_achieved=False))
        print(str(OUT / 'complete.json'), flush=True)
    except BaseException as exc:
        atomic(OUT / 'failure.json', dict(owner=owner, error=repr(exc), automatic_retry=False,
            all_actual_source_attempts_preserved=True, source_training_admitted=False, production_RL_admitted=False, goal_achieved=False))
        raise


if __name__ == '__main__': main()
