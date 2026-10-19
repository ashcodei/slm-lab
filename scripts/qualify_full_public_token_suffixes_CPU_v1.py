"""Reconstruct every3721human source at all4levels and execute all controls."""
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/full-public-token-suffix-qualification-CPU-v1'
MAX_OUTPUT = 1024**3
RESERVE = 40 * 1024**3


def main():
    if (OUT.exists() or os.environ.get('CUDA_VISIBLE_DEVICES') != '' or os.nice(0) < 19
            or any(os.environ.get(k) != '1' for k in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'])):
        raise ValueError('Fresh CUDA-hidden nice19 single-thread source-only qualification required')
    os.sched_setscheduler(0, os.SCHED_IDLE, os.sched_param(0))
    from slm_lab.diverse_images import atomic_json
    owner = dict(pid=os.getpid(), ticks=Path('/proc/self/stat').read_text().split(') ', 1)[1].split()[19])
    OUT.mkdir()
    atomic_json(OUT / 'owner.json', dict(owner=owner, model_generations=0, optimizer_updates=0, GPUleases=0))
    try:
        import torch
        from slm_lab.active_execution_input_bindings_v1 import input_bindings, merge_bindings
        from slm_lab.bounded_disk_reserve_wait_v1 import wait_for_reserve
        from slm_lab.codenet_checkpoint_v1 import FileBindings
        from slm_lab.corpus import file_hash
        from slm_lab.expanded_execution_corpus_v1 import ExpandedExecutionCorpus
        from slm_lab.local_source_closure_v2 import python_sources
        from slm_lab.public_token_suffix_curriculum_v1 import LEVELS, view, descriptor, nontrivial
        torch.set_num_threads(1)
        tests = ROOT / 'reports/public-token-suffix-curriculum-v1-tests.log'
        if '10 passed' not in tests.read_text():
            raise ValueError('All exact native-mask/role/Unicode/maximum-context controls must pass')
        corpus = ExpandedExecutionCorpus(ROOT)
        files = input_bindings(ROOT, corpus)
        merge_bindings(files, {str(p): file_hash(p) for p in python_sources(ROOT, [Path(__file__),
            ROOT / 'tests/test_public_token_suffix_curriculum_v1.py'])})
        merge_bindings(files, {str(tests): file_hash(tests)})
        guard = FileBindings(files)
        guard.check(full=True)
        atomic_json(OUT / 'plan.json', dict(owner=owner, at=datetime.now(timezone.utc).isoformat(),
            source_files_sha256=files, actual_original_human_train_sources=3721,
            actual_documented_sources=173, actual_APPS_sources=3548,
            nominal_missing_non_EOS_token_levels=list(LEVELS), zero_level_means_original_unhinted_prompt=True,
            all_original_CE_contexts_per_level=2287522, all_original_response_EOS_targets_per_level=584218,
            full_original_reference_executions=3721, counterfactual_prefix_then_EOS_executions=14884,
            counterfactuals_are_not_student_samples=True, every_source_and_disposition_preserved=True,
            original_unhinted_CE_targets_always_retained=True, maximum_total_output_bytes=MAX_OUTPUT,
            minimum_other_writer_reserve_bytes=RESERVE, source_training_admitted=False,
            production_RL_admitted=False, no_existing_neural_worker_changed=True, goal_achieved=False))
        def intact():
            if os.sched_getscheduler(0) != os.SCHED_IDLE or torch.cuda.is_initialized():
                raise ValueError('Source-only idle execution must never acquire a CUDAcontext')
            guard.check()
        wait_totals = dict(waited_seconds=0., failed_space_observations=0)
        def reserve():
            result = wait_for_reserve(ROOT, minimum_bytes=RESERVE + MAX_OUTPUT, guard=intact)
            for k in wait_totals:
                wait_totals[k] += result[k]
        def score(position, code):
            try:
                return corpus.score(position, code, stopped_on_eos=True)
            except Exception as exc:
                return dict(reward=None, status='preserved_execution_adapter_abstention', error=repr(exc))
        totals = {str(n): Counter() for n in LEVELS}
        used = 0
        def append(stream, value):
            nonlocal used
            raw = (json.dumps(value, ensure_ascii=True, allow_nan=False, separators=(',', ':')) + '\n').encode()
            if len(raw) > 8 * 1024**2 or used + len(raw) > MAX_OUTPUT:
                raise ValueError('Preserve all raw outcomes; never truncate or drop cases to fit the output budget')
            stream.write(raw)
            stream.flush()
            used += len(raw)
        counts = Counter()
        reserve()
        with (OUT / 'all14884-native-view-descriptors.jsonl').open('xb') as metadata:
            for position in range(len(corpus)):
                if position % 32 == 0:
                    reserve()
                row = corpus.row(position)
                ids, labels = corpus.sequence(position)
                for level in LEVELS:
                    value = view(corpus.tokenizer, row['original'], ids, labels, level)
                    summary = descriptor(value)
                    append(metadata, dict(source_position=position, corpus_id=row['id'], kind=row['kind'], **summary))
                    totals[str(level)].update(views=1, original_context_tokens=len(ids),
                        original_response_EOS_targets=value['original_response_EOS_targets'],
                        retained_reference_tokens=value['retained_reference_tokens'],
                        missing_original_response_EOS_targets=value['actual_missing_response_EOS_targets'],
                        views_with_actual_hint=int(value['retained_reference_tokens'] > 0))
            os.fsync(metadata.fileno())
        if any(v['views'] != 3721 or v['original_context_tokens'] != 2287522
                or v['original_response_EOS_targets'] != 584218 for v in totals.values()):
            raise ValueError('All four levels must preserve every complete original context and CE/EOS target')
        with (OUT / 'all3721-reference-and14884-prefix-controls.jsonl').open('xb') as journal:
            for position in range(len(corpus)):
                reserve()
                row = corpus.row(position)
                ids, labels = corpus.sequence(position)
                reference = score(position, row['original']['response'])
                counts['full_reference_runs'] += 1
                counts['full_reference_reward1'] += int(reference.get('reward') == 1.)
                counts['full_reference_abstentions'] += int(reference.get('reward') is None)
                controls = []
                for level in LEVELS:
                    value = view(corpus.tokenizer, row['original'], ids, labels, level)
                    actual = score(position, value['public_reference_prefix_text'])
                    eligible = (nontrivial(value, reference, actual) if reference.get('reward') is not None
                        and actual.get('reward') is not None else False)
                    controls.append(dict(nominal_missing_non_EOS_tokens=level, native_view=descriptor(value),
                        original_prefix_text=value['public_reference_prefix_text'], counterfactual_prefix_then_EOS_execution=actual,
                        proposed_nontrivial_bounded_execution_view=eligible, source_training_admitted=False))
                    counts['prefix_control_runs'] += 1
                    counts['prefix_only_reward1_rejected'] += int(actual.get('reward') == 1.)
                    counts['prefix_control_abstentions'] += int(actual.get('reward') is None)
                    counts['proposed_nontrivial_views'] += int(eligible)
                append(journal, dict(source_position=position, corpus_id=row['id'], kind=row['kind'],
                    difficulty=row['original'].get('difficulty'), full_original_reference_execution=reference,
                    views=controls, original_CE_context_tokens=len(ids), original_CE_response_EOS_targets=len(ids) - row['prefix_positions'],
                    every_original_source_and_complete_CE_anchor_retained=True,
                    counterfactuals_not_student_samples=True, source_role='train', source_training_admitted=False,
                    production_RL_admitted=False))
                counts['complete_original_sources'] += 1
                if (position + 1) % 16 == 0 or position + 1 == len(corpus):
                    atomic_json(OUT / 'progress.json', dict(owner=owner, phase='full original human references and all4counterfactual controls',
                        actual_counts=dict(counts), original_sources=3721, actual_output_bytes=used,
                        bounded_disk_waits=wait_totals, source_training_admitted=False, neural_forwards=0))
                if (position + 1) % 64 == 0:
                    os.fsync(journal.fileno())
            os.fsync(journal.fileno())
        if counts['complete_original_sources'] != 3721 or counts['prefix_control_runs'] != 14884:
            raise ValueError('Every original source and all4explicit curriculum levels must have an actual disposition')
        guard.check(full=True)
        atomic_json(OUT / 'complete.json', dict(complete=True, at=datetime.now(timezone.utc).isoformat(), owner=owner,
            source_files_sha256=files, actual_counts=dict(counts), actual_all_four_level_metadata_totals={k: dict(v) for k, v in totals.items()},
            actual_all14884_native_descriptors_sha256=file_hash(OUT / 'all14884-native-view-descriptors.jsonl'),
            actual_all18605_runtime_controls_sha256=file_hash(OUT / 'all3721-reference-and14884-prefix-controls.jsonl'),
            all_original3721sources_and_every_unconditional_CE_target_preserved=True,
            original_full_CE_contexts=2287522, original_full_CE_response_EOS_targets=584218,
            all_failures_abstentions_and_trivial_prefixes_preserved=True,
            independent_full_metadata_and_fresh_execution_readback_still_required=True,
            actual_student_reward_density_unassisted_code_quality_and_GPUcapacity_not_measured=True,
            source_training_admitted=False, production_RL_admitted=False, student_generations=0,
            model_updates=0, GPUcontexts=0, existing_neural_workers_unchanged=True, goal_achieved=False))
        print(json.dumps(dict(complete=True, actual_counts=dict(counts), source_training_admitted=False)), flush=True)
    except BaseException as exc:
        atomic_json(OUT / 'failure.json', dict(owner=owner, error=repr(exc), automatic_retry=False,
            source_training_admitted=False, optimizer_updates=0, goal_achieved=False))
        raise


if __name__ == '__main__':
    main()
