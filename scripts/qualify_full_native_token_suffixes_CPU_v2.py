"""Qualify the exact native execution profile after the complete legacy reader.

All original sources/views remain in the journal, including CE-only cases.
This source-only successor cannot launch models, GPU jobs or optimizer updates.
Two new native gold/prefix runs accompany the two retained legacy controls.
"""
from collections import Counter
from contextlib import ExitStack
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/full-native-token-suffix-qualification-CPU-v2'
READER = ROOT / 'reports/full-public-token-suffix-readback-gzip-recovery-CPU-v4'
PRODUCER = ROOT / 'reports/full-public-token-suffix-qualification-gzip-recovery-CPU-v2'
PARENT = dict(pid=182750, ticks='56898334')
RESERVE = 40 * 1024**3
MAX_RAW = 2 * 1024**3


def main():
    if (OUT.exists() or os.environ.get('CUDA_VISIBLE_DEVICES') != '' or os.nice(0) < 19
            or any(os.environ.get(k) != '1' for k in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'])):
        raise ValueError('A fresh CUDA-hidden nice19 one-thread source-only successor is required')
    os.sched_setscheduler(0, os.SCHED_IDLE, os.sched_param(0))
    from record_visual_stage_and_first_repair_arm_v1 import atomic, read, sha
    from slm_lab.exact_waiting_process_retirement_v1 import terminal
    from slm_lab.codenet_checkpoint_v1 import FileBindings
    from slm_lab.closed_source_recovery_guard_v1 import SourceOnlyRecoveryGuard

    original = ROOT / 'reports/full-native-token-suffix-qualification-CPU-v1'
    old_failure = read(original / 'failure.json')
    if (not terminal(old_failure['owner']) or old_failure['owner'] != dict(pid=162287, ticks='55947633')):
        raise ValueError('The exact failed original native qualifier must remain terminal and preserved')
    for name in ['closed-source-recovery-guard-v1-tests.log', 'recovered-legacy-suffix-catalogue-v1-tests.log']:
        if '3 passed' not in (ROOT / 'reports' / name).read_text():
            raise ValueError('Both actual complete recovery catalogue and source-boundary controls must pass')

    parent_plan = READER / 'plan.json'
    if read(parent_plan)['owner'] != PARENT:
        raise ValueError('The exact original independent reader birth is required')
    names = ['src/slm_lab/qualified_native_token_suffix_episode_v1.py',
        'src/slm_lab/native_code_execution_oracle_v2.py', 'src/slm_lab/public_token_suffix_curriculum_v1.py',
        'src/slm_lab/bounded_gzip_observation_stream_v1.py',
        'reports/native-code-execution-oracle-v2-tests.log',
        'reports/batched-native-suffix-group-gradients-v1-tests.log']
    extra = [ROOT / 'src/slm_lab/recovered_legacy_suffix_catalogue_v1.py',
        ROOT / 'src/slm_lab/closed_source_recovery_guard_v1.py', original / 'failure.json',
        ROOT / 'reports/closed-source-recovery-guard-v1-tests.log',
        ROOT / 'reports/recovered-legacy-suffix-catalogue-v1-tests.log']
    files = {str(p): sha(p) for p in [Path(__file__), parent_plan, *[ROOT / n for n in names], *extra]}
    if ('7 passed' not in (ROOT / names[-1]).read_text()
            or '7 passed' not in (ROOT / names[-2]).read_text()):
        raise ValueError('Both actual native oracle and connected gradient controls must pass')
    guard = FileBindings(files)
    OUT.mkdir()
    owner = dict(pid=os.getpid(), ticks=Path('/proc/self/stat').read_text().split(') ', 1)[1].split()[19])
    atomic(OUT / 'plan.json', dict(owner=owner, exact_original_fresh_reader=PARENT, source_files_sha256=files,
        all_original_sources=3721, all_original_views=14884, all_original_context_tokens=2287522,
        all_original_response_EOS_targets=584218, every_legacy_ineligible_view_retained_as_CE_only=True,
        new_native_repeats_per_legacy_eligible_view=2, new_native_runs_per_eligible_view=4,
        retained_legacy_controls_not_relabelled_as_new_executions=True,
        recovered_legacy_reader_used_explicitly=True, failed_original_qualifier_preserved=old_failure,
        complete_input_hash_validation_before_execution_and_before_publication=True,
        original_sources_per_checked_native_oracle_batch=16,
        maximum_additional_raw_output_bytes=MAX_RAW, minimum_other_writer_reserve_bytes=RESERVE,
        source_training_admitted=False, production_RL_admitted=False, models_constructed=0,
        neural_forwards=0, GPUleases=0, optimizer_updates=0))
    files[str(OUT / 'plan.json')] = sha(OUT / 'plan.json')
    guard = FileBindings(files)
    try:
        deadline = time.monotonic() + 7 * 86400
        while not terminal(PARENT):
            guard.check()
            if time.monotonic() > deadline:
                raise ValueError('The source-only waiting deadline expired')
            atomic(OUT / 'progress.json', dict(owner=owner, phase='waiting for the entire original independent reader',
                original_reader_live=True, new_native_executions=0, models_constructed=0, source_training_admitted=False))
            time.sleep(15)
        if (READER / 'failure.json').exists() or not (READER / 'complete.json').exists():
            raise ValueError('The exact original independent reader did not complete; preserve its failure')

        import torch
        from slm_lab.active_execution_input_bindings_v1 import input_bindings
        from slm_lab.bounded_disk_reserve_wait_v1 import wait_for_reserve
        from slm_lab.bounded_gzip_observation_stream_v1 import Writer, RawBudget, NEXT_RECORD, lines
        from slm_lab.expanded_execution_corpus_v1 import ExpandedExecutionCorpus
        from slm_lab.native_code_execution_oracle_v2 import NativeCodeExecutionOracle
        from slm_lab.public_token_suffix_curriculum_v1 import LEVELS, view, descriptor
        from slm_lab.qualified_native_token_suffix_episode_v1 import QualifiedNativeTokenSuffixEpisode, PROOF
        from slm_lab.recovered_legacy_suffix_catalogue_v1 import load_recovered_legacy_catalogue
        torch.set_num_threads(1)
        corpus = ExpandedExecutionCorpus(ROOT)
        oracle = NativeCodeExecutionOracle(corpus)
        catalogue = load_recovered_legacy_catalogue(oracle)
        files.update(input_bindings(ROOT, corpus))
        files.update(oracle.sources)
        files.update(catalogue.files)
        def resource_only():
            if os.sched_getscheduler(0) != os.SCHED_IDLE or torch.cuda.is_initialized():
                raise ValueError('Native source qualification must remain idle and CUDA-free')
        guard = SourceOnlyRecoveryGuard(files, resource_only)
        guard.check()
        manifest = OUT / 'closed-source-bindings.json'
        atomic(manifest, dict(source_files_sha256=files, complete_legacy_catalogue_sha256=catalogue.digest,
            exact_original_fresh_reader_terminal=True))
        proof_files = {str(manifest): sha(manifest)}
        compact_guard = FileBindings(proof_files)

        def resources():
            if os.sched_getscheduler(0) != os.SCHED_IDLE or torch.cuda.is_initialized():
                raise ValueError('Only idle source execution without a CUDA context is admitted')
            compact_guard.check()

        def reserve():
            resources()
            wait_for_reserve(ROOT, minimum_bytes=RESERVE + NEXT_RECORD, guard=resources)

        def deliver(position, prefix_ids, actions, horizon, origin):
            try:
                return oracle.delivery(position, prefix_ids, actions, delivered_action_budget=horizon, origin=origin)
            except Exception as exc:
                # An adapter exception is missing evidence, never an invented loss.
                # Binding failures still fail the enclosing checked source batch.
                return dict(reward=None, status='preserved_native_control_adapter_exception', error=repr(exc),
                    requested_source_position=position, requested_public_reference_ids=list(prefix_ids),
                    requested_native_actions=list(actions), requested_action_budget=horizon, requested_origin=origin)

        old = iter(lines(Path(read(PRODUCER / 'complete.json')['complete_control_file'])))
        fresh = iter(lines(Path(read(READER / 'complete.json')['actual_fresh_execution_control_file'])))
        counts, levels = Counter(), {str(level): Counter() for level in LEVELS}
        journal_path = OUT / 'all14884-native-profile-view-controls.jsonl.gz'
        budget = RawBudget(MAX_RAW)
        with Writer(journal_path, root=ROOT, budget=budget, reserve_bytes=RESERVE, guard=resources) as journal, ExitStack() as checked_sources:
            for position in range(len(corpus)):
                reserve()
                guard.check()
                if position % 16 == 0:
                    checked_sources.close()
                    checked_sources.enter_context(oracle.checked_source_batch())
                row = corpus.row(position)
                ids, labels = corpus.sequence(position)
                a, b = next(old)[1], next(fresh)[1]
                if (a['source_position'] != position or b['source_position'] != position
                        or a['corpus_id'] != row['id'] or b['corpus_id'] != row['id']
                        or len(a['views']) != 4 or len(b['all_four_fresh_views']) != 4):
                    raise ValueError('Every original and fresh control must retain its ordered human source')
                for level, original, independent in zip(LEVELS, a['views'], b['all_four_fresh_views']):
                    native = view(corpus.tokenizer, row['original'], ids, labels, level)
                    desc = descriptor(native)
                    previous = catalogue.rows[(position, level)]
                    if (previous['exact_native_descriptor'] != desc or original['native_view'] != desc
                            or original['nominal_missing_non_EOS_tokens'] != level
                            or independent['nominal_missing_non_EOS_tokens'] != level
                            or independent['disposition'] != previous['disposition']):
                        raise ValueError('All original native masks, whole budgets and both legacy views must match')
                    legacy_eligible = previous['bounded_nontrivial_control_evidence']
                    extra_boundary = any(t in [0, 1, 3] for t in ids[1:-1])
                    controls = []
                    if legacy_eligible and not extra_boundary:
                        catalogue.check_view(position, level, desc)
                        for repeat, legacy in enumerate([
                                original['counterfactual_prefix_then_EOS_execution'], independent['fresh_prefix_execution']]):
                            reserve()
                            gold = deliver(position, native['public_reference_prefix_ids'],
                                native['original_missing_suffix_and_EOS_ids'], native['full_native_action_budget'], 'original_reference_control')
                            prefix = deliver(position, native['public_reference_prefix_ids'], [2],
                                native['full_native_action_budget'], 'test_fixture')
                            controls.append(dict(repeat=repeat, native_gold=gold, native_prefix_then_EOS=prefix,
                                legacy_prefix_then_EOS=legacy,
                                retained_legacy_origin='original_producer' if repeat == 0 else 'independent_fresh_reader'))
                            counts['new_actual_native_gold_runs'] += 1
                            counts['new_actual_native_prefix_runs'] += 1
                    qualified = bool(controls) and all(c['native_gold']['reward'] == 1.
                        and c['native_prefix_then_EOS']['reward'] is not None
                        and c['native_prefix_then_EOS']['reward'] < 1. for c in controls)
                    proof = dict(format=PROOF, complete=qualified, scope='complete-source-catalogue',
                        source_position=position, nominal_missing_non_EOS_tokens=level,
                        source_information=oracle.source_information(position), oracle_binding=oracle.binding(),
                        native_view=desc, actual_controls=controls, source_files_sha256=proof_files,
                        complete_legacy_catalogue_sha256=catalogue.digest,
                        source_training_admitted=False, production_RL_admitted=False,
                        model_generations=0, model_updates=0)
                    if qualified:
                        # Use the exact downstream type check before recording eligibility.
                        # The temporary file is solely this source worker's validation input.
                        with tempfile.TemporaryDirectory(prefix='native-source-proof-', dir=OUT) as directory:
                            proof_path = Path(directory) / 'proof.json'
                            atomic(proof_path, proof)
                            episode = QualifiedNativeTokenSuffixEpisode(oracle, proof_path, catalogue=catalogue)
                            episode.check()
                            del episode
                    disposition = ('native_nontrivial_controls_qualified' if qualified else
                        'legacy_ineligible_CE_only' if not legacy_eligible else
                        'embedded_original_boundary_CE_only' if extra_boundary else 'native_control_failed_or_abstained_CE_only')
                    journal.append(dict(source_position=position, corpus_id=row['id'], level=level,
                        disposition=disposition, eligible_for_later_native_profile=qualified,
                        native_profile_control_proof=proof, original_legacy_disposition=previous,
                        all_original_unconditional_CE_targets_retained=True,
                        source_training_admitted=False, production_RL_admitted=False))
                    counts['all_original_views'] += 1
                    counts[disposition] += 1
                    levels[str(level)].update(views=1, qualified=int(qualified),
                        original_context_tokens=len(ids), original_response_EOS_targets=len(ids) - row['prefix_positions'])
                counts['all_original_sources'] += 1
                counts['original_context_tokens'] += len(ids)
                counts['original_response_EOS_targets'] += len(ids) - row['prefix_positions']
                if (position + 1) % 16 == 0 or position + 1 == len(corpus):
                    atomic(OUT / 'progress.json', dict(owner=owner, phase='actual native gold and prefix controls',
                        actual_counts=dict(counts), actual_level_counts={k: dict(v) for k, v in levels.items()},
                        actual_stream=journal.receipt(), source_training_admitted=False, neural_forwards=0))
                    journal.fsync()
            if next(old, None) is not None or next(fresh, None) is not None:
                raise ValueError('The complete source journals cannot contain extra hidden records')
        guard.verify_final()
        resources()
        if (len(corpus) != 3721 or counts['all_original_sources'] != 3721 or counts['all_original_views'] != 14884
                or counts['original_context_tokens'] != 2287522 or counts['original_response_EOS_targets'] != 584218
                or any(v['views'] != 3721 or v['original_context_tokens'] != 2287522
                    or v['original_response_EOS_targets'] != 584218 for v in levels.values())):
            raise ValueError('Every original source, level, context and complete CE/EOS target must remain')
        atomic(OUT / 'complete.json', dict(complete=True, at=datetime.now(timezone.utc).isoformat(), owner=owner,
            source_files_sha256=files, closed_source_manifest_sha256=sha(manifest),
            actual_counts=dict(counts), actual_level_counts={k: dict(v) for k, v in levels.items()},
            complete_control_file=str(journal_path), complete_control_file_sha256=sha(journal_path),
            lossless_observation_stream=journal.receipt(), every_source_view_and_CE_target_retained=True,
            each_qualified_episode_validated_by_exact_downstream_type=True,
            independent_full_native_profile_readback_still_required=True,
            source_training_admitted=False, production_RL_admitted=False, models_constructed=0,
            student_generations=0, neural_forwards=0, optimizer_updates=0, GPUleases=0, goal_achieved=False))
        print(json.dumps(dict(complete=True, actual_counts=dict(counts), production_RL_admitted=False)), flush=True)
    except BaseException as exc:
        atomic(OUT / 'failure.json', dict(owner=owner, error=repr(exc), automatic_retry=False,
            all_attempts_preserved=True, source_training_admitted=False, production_RL_admitted=False, goal_achieved=False))
        raise


if __name__ == '__main__': main()
