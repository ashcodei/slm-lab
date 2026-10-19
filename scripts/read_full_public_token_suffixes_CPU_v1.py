"""Independently rebuild all14884views and freshly execute all18605controls.

Wait for the exact source-only producer. Preserve every failed, trivial,
abstaining or disagreeing view. This creates a complete evidence catalog,
not production RL admission or a claim of student reward/capability gains.
"""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'reports/full-public-token-suffix-qualification-CPU-v1'
OUT = ROOT / 'reports/full-public-token-suffix-readback-CPU-v1'
PARENT = dict(pid=146084, ticks='55111340')
MAX_OUTPUT = 1024**3
RESERVE = 40 * 1024**3


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def lines(path):
    with Path(path).open('rb') as stream:
        for raw in stream:
            if not raw.endswith(b'\n'):
                raise ValueError('All original closed LF source observations must be complete')
            yield json.loads(raw)


def main():
    if (OUT.exists() or os.environ.get('CUDA_VISIBLE_DEVICES') != '' or os.nice(0) < 19
            or any(os.environ.get(k) != '1' for k in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'])):
        raise ValueError('Fresh CUDA-hidden nice19 one-thread source-only independent reader required')
    os.sched_setscheduler(0, os.SCHED_IDLE, os.sched_param(0))
    from record_visual_stage_and_first_repair_arm_v1 import atomic
    from slm_lab.exact_waiting_process_retirement_v1 import terminal
    plan = read(SOURCE / 'plan.json')
    if plan['owner'] != PARENT:
        raise ValueError('The exact original full-source qualification birth is required')
    tests = ROOT / 'reports/independent-token-suffix-readback-v2-tests.log'
    if '8 passed' not in tests.read_text():
        raise ValueError('All independent source reconstruction and changed-target/control rejection tests must pass')
    files = dict(plan['source_files_sha256'])
    extra = [Path(__file__), SOURCE / 'plan.json', tests, ROOT / 'src/slm_lab/independent_token_suffix_readback_v1.py',
        ROOT / 'tests/test_independent_token_suffix_readback_v2.py', ROOT / 'scripts/run_independent_token_suffix_readback_tests_v2.py']
    files.update({str(p): sha(p) for p in extra})
    states = {p: (Path(p).stat().st_dev, Path(p).stat().st_ino, Path(p).stat().st_size, Path(p).stat().st_mtime_ns) for p in files}
    def guard():
        if os.sched_getscheduler(0) != os.SCHED_IDLE or any(
                (Path(p).stat().st_dev, Path(p).stat().st_ino, Path(p).stat().st_size, Path(p).stat().st_mtime_ns) != value
                for p, value in states.items()):
            raise ValueError('All immutable original source inputs, reader code and own idle scheduling must remain exact')
    OUT.mkdir()
    owner = dict(pid=os.getpid(), ticks=Path('/proc/self/stat').read_text().split(') ', 1)[1].split()[19])
    atomic(OUT / 'plan.json', dict(owner=owner, exact_original_source_producer=PARENT, source_files_sha256=files,
        all_original_sources=3721, all_native_views=14884, all_fresh_runtime_controls=18605,
        all_original_unconditional_CE_targets=584218, all_original_unconditional_context_tokens=2287522,
        independently_rebuild_without_importing_producer_view_helper=True,
        every_fresh_failed_trivial_abstaining_or_disagreeing_view_preserved=True,
        maximum_total_output_bytes=MAX_OUTPUT, minimum_other_writer_reserve_bytes=RESERVE,
        source_training_admitted=False, production_RL_admitted=False, neural_forwards=0, GPUleases=0))
    try:
        expires = time.monotonic() + 7 * 86400
        while not terminal(PARENT):
            guard()
            if time.monotonic() > expires:
                raise ValueError('The declared independent source-only waiting deadline expired')
            atomic(OUT / 'progress.json', dict(owner=owner, phase='waiting for exact complete full human-code control producer',
                exact_original_producer_live=True, neural_forwards=0, source_training_admitted=False))
            time.sleep(15)
        if (SOURCE / 'failure.json').exists():
            raise ValueError('The original producer failed; preserve its failure without relabelling a partial population complete')
        complete = read(SOURCE / 'complete.json')
        if (not complete['complete'] or complete['owner'] != PARENT
                or complete['actual_counts']['complete_original_sources'] != 3721
                or complete['actual_counts']['full_reference_runs'] != 3721
                or complete['actual_counts']['prefix_control_runs'] != 14884
                or not complete['all_original3721sources_and_every_unconditional_CE_target_preserved']
                or complete['source_training_admitted']):
            raise ValueError('The entire unadmitted original producer and every source disposition are required')
        for name, expected in [('all14884-native-view-descriptors.jsonl', complete['actual_all14884_native_descriptors_sha256']),
                ('all3721-reference-and14884-prefix-controls.jsonl', complete['actual_all18605_runtime_controls_sha256'])]:
            path = SOURCE / name
            if sha(path) != expected:
                raise ValueError('Every closed producer observation byte must match its original completion receipt')
            files[str(path)] = expected
        files[str(SOURCE / 'complete.json')] = sha(SOURCE / 'complete.json')
        import torch
        from slm_lab.bounded_disk_reserve_wait_v1 import wait_for_reserve
        from slm_lab.expanded_execution_corpus_v1 import ExpandedExecutionCorpus
        from slm_lab.independent_token_suffix_readback_v1 import rebuild, eligible
        from slm_lab.local_source_closure_v2 import python_sources
        torch.set_num_threads(1)
        if torch.cuda.is_initialized():
            raise ValueError('Independent public source execution cannot create a CUDAcontext')
        files.update({str(p): sha(p) for p in python_sources(ROOT, [Path(__file__), ROOT / 'tests/test_independent_token_suffix_readback_v2.py'])})
        states.update({p: (Path(p).stat().st_dev, Path(p).stat().st_ino, Path(p).stat().st_size, Path(p).stat().st_mtime_ns) for p in files})
        atomic(OUT / 'closed-source-bindings.json', dict(owner=owner, source_files_sha256=files, original_producer_terminal=True))
        corpus = ExpandedExecutionCorpus(ROOT)
        def reserve():
            wait_for_reserve(ROOT, minimum_bytes=RESERVE + MAX_OUTPUT, guard=guard)
            if torch.cuda.is_initialized():
                raise ValueError('Source reader cannot acquire a GPUcontext')
        def score(position, text):
            try:
                return corpus.score(position, text, stopped_on_eos=True)
            except Exception as exc:
                return dict(reward=None, status='preserved_execution_adapter_abstention', error=repr(exc))
        used = 0
        def append(stream, value):
            nonlocal used
            raw = (json.dumps(value, ensure_ascii=True, allow_nan=False, separators=(',', ':')) + '\n').encode()
            if len(raw) > 8 * 1024**2 or used + len(raw) > MAX_OUTPUT:
                raise ValueError('No original/fresh case can be truncated or omitted to fit the reserved output budget')
            stream.write(raw)
            stream.flush()
            used += len(raw)
        counts = Counter()
        totals = {str(n): Counter() for n in [4, 16, 64, 0]}
        discrepancies = []
        metadata = iter(lines(SOURCE / 'all14884-native-view-descriptors.jsonl'))
        controls = iter(lines(SOURCE / 'all3721-reference-and14884-prefix-controls.jsonl'))
        with (OUT / 'all3721-fresh-reference-and14884-fresh-prefix-controls.jsonl').open('xb') as journal, (
                OUT / 'all14884-view-dispositions.jsonl').open('xb') as catalog:
            for position in range(len(corpus)):
                reserve()
                row = corpus.row(position)
                ids, labels = corpus.sequence(position)
                observed = next(controls)
                if (observed['source_position'] != position or observed['corpus_id'] != row['id']
                        or observed['kind'] != row['kind'] or len(observed['views']) != 4
                        or observed['original_CE_context_tokens'] != len(ids)
                        or observed['original_CE_response_EOS_targets'] != len(ids) - row['prefix_positions']):
                    raise ValueError('Every original source/control position and full CE budget must reconstruct')
                fresh_ref = score(position, row['original']['response'])
                old_ref = observed['full_original_reference_execution']
                changed_ref = any(old_ref.get(k) != fresh_ref.get(k) for k in ['reward', 'status', 'all_finite_original_cases_passed'])
                if changed_ref:
                    discrepancies.append(dict(source_position=position, phase='full_reference', original_reward=old_ref.get('reward'), fresh_reward=fresh_ref.get('reward')))
                counts['fresh_full_reference_runs'] += 1
                counts['fresh_reference_reward1'] += int(fresh_ref.get('reward') == 1.)
                actual_views = []
                for level, original_control in zip([4, 16, 64, 0], observed['views']):
                    original = next(metadata)
                    identity = {k: original.pop(k) for k in ['source_position', 'corpus_id', 'kind']}
                    if identity != dict(source_position=position, corpus_id=row['id'], kind=row['kind']):
                        raise ValueError('Every exact original native descriptor identity must match')
                    fresh = rebuild(corpus.tokenizer, row['original'], ids, labels, level, original)
                    info = fresh['metadata']
                    if (original_control['nominal_missing_non_EOS_tokens'] != level
                            or original_control['native_view'] != info
                            or original_control['original_prefix_text'] != fresh['original_prefix_text']
                            or original_control['proposed_nontrivial_bounded_execution_view'] != eligible(info, old_ref,
                                original_control['counterfactual_prefix_then_EOS_execution'])):
                        raise ValueError('Every original whole prefix/control/qualification disposition must independently reconstruct')
                    actual = score(position, fresh['original_prefix_text'])
                    old = original_control['counterfactual_prefix_then_EOS_execution']
                    changed = changed_ref or any(old.get(k) != actual.get(k) for k in ['reward', 'status', 'all_finite_original_cases_passed'])
                    if changed:
                        discrepancies.append(dict(source_position=position, level=level, phase='prefix', original_reward=old.get('reward'), fresh_reward=actual.get('reward')))
                    both = eligible(info, old_ref, old) and eligible(info, fresh_ref, actual) and not changed
                    disposition = ('nontrivial_in_both_complete_control_runs' if both else
                        'fresh_execution_disagreement' if changed else 'trivial_prefix_rejected' if actual.get('reward') == 1 else
                        'execution_abstention' if fresh_ref.get('reward') is None or actual.get('reward') is None else
                        'full_reference_not_complete_credit' if fresh_ref.get('reward') != 1 else
                        'embedded_original_boundary_token' if info['original_has_embedded_boundary_tokens'] else
                        'no_actual_hint_at_this_nominal_level')
                    append(catalog, dict(source_position=position, corpus_id=row['id'], nominal_missing_non_EOS_tokens=level,
                        exact_native_descriptor=info, disposition=disposition, bounded_nontrivial_control_evidence=both,
                        all_original_unconditional_CE_targets_retained=True, source_training_admitted=False, production_RL_admitted=False))
                    actual_views.append(dict(nominal_missing_non_EOS_tokens=level,
                        fresh_prefix_execution=actual, execution_disagreed=changed, disposition=disposition))
                    counts['fresh_prefix_control_runs'] += 1
                    counts['all_native_views_rebuilt'] += 1
                    counts['bounded_nontrivial_views_in_both_runs'] += int(both)
                    counts['fresh_prefix_reward1_rejected'] += int(actual.get('reward') == 1.)
                    totals[str(level)].update(views=1, original_context_tokens=len(ids),
                        original_response_EOS_targets=len(ids) - row['prefix_positions'])
                append(journal, dict(source_position=position, corpus_id=row['id'], fresh_full_reference_execution=fresh_ref,
                    full_reference_execution_disagreed=changed_ref, all_four_fresh_views=actual_views,
                    no_student_generation_or_training=True))
                counts['complete_original_sources'] += 1
                if (position + 1) % 16 == 0 or position + 1 == len(corpus):
                    atomic(OUT / 'progress.json', dict(owner=owner, phase='all native views independently rebuilt and all controls freshly executed',
                        actual_counts=dict(counts), original_sources=3721, preserved_discrepancies=len(discrepancies),
                        source_training_admitted=False, neural_forwards=0))
                if (position + 1) % 64 == 0:
                    os.fsync(journal.fileno())
                    os.fsync(catalog.fileno())
            if next(metadata, None) is not None or next(controls, None) is not None:
                raise ValueError('Original source observations cannot contain extra hidden rows')
            os.fsync(journal.fileno())
            os.fsync(catalog.fileno())
        if (counts['complete_original_sources'] != 3721 or counts['all_native_views_rebuilt'] != 14884
                or counts['fresh_full_reference_runs'] != 3721 or counts['fresh_prefix_control_runs'] != 14884
                or any(v != dict(views=3721, original_context_tokens=2287522, original_response_EOS_targets=584218) for v in totals.values())):
            raise ValueError('The complete source, all four levels and every original whole-CE target must be retained')
        reserve()
        if any(sha(p) != h for p, h in files.items()):
            raise ValueError('Every original input and closed source observation byte must remain unchanged')
        atomic(OUT / 'complete.json', dict(complete=True, integrity_readback_passed=True, at=datetime.now(timezone.utc).isoformat(),
            owner=owner, source_files_sha256=files, actual_counts=dict(counts), all_four_level_metadata_totals={k: dict(v) for k, v in totals.items()},
            actual_fresh_execution_discrepancies=discrepancies, all_fresh_discrepancies_preserved=True,
            actual_all3721source14884view_dispositions_sha256=sha(OUT / 'all14884-view-dispositions.jsonl'),
            actual_all18605fresh_execution_controls_sha256=sha(OUT / 'all3721-fresh-reference-and14884-fresh-prefix-controls.jsonl'),
            every_original_unconditional_CE_context_and_target_retained=True,
            no_source_or_level_dropped_for_failure_triviality_abstention_or_disagreement=True,
            counterfactual_controls_not_student_reward_density_or_unassisted_capability=True,
            source_training_admitted=False, production_RL_admitted=False, student_generations=0, neural_forwards=0,
            full_size_GPUcapacity_private_recovery_actual_RLtraining_and_quality_still_required=True, goal_achieved=False))
        print(json.dumps(dict(complete=True, actual_counts=dict(counts), actual_discrepancies=len(discrepancies), production_RL_admitted=False)), flush=True)
    except BaseException as exc:
        atomic(OUT / 'failure.json', dict(owner=owner, error=repr(exc), automatic_retry=False,
            source_training_admitted=False, production_RL_admitted=False, goal_achieved=False))
        raise


if __name__ == '__main__':
    main()
