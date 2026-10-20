"""Screen whole 16K reasoning views for the fixed original 4K problem subset.

Every original shard row is retained. All 6,116 original candidate roles are
inherited, never reassigned from fresh execution or model scores. This does
not admit data, extend a model, or change the running 4K experiments.
"""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
JOIN = Path('/tmp/slm-public-reasoned-code-question-join-v2')
SCREEN = Path('/tmp/slm-public-reasoned-code-screen-v2')
HELD = Path('/tmp/slm-public-reasoned-code-held-registry-v1')
CANDIDATE = Path('/tmp/slm-public-reasoned-code-candidate-v1/original-python-00000.parquet')
OUT = Path('/tmp/slm-whole-reasoning-16K-screen-v1')
MAXIMUM_CONTEXT = 16384


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(name, value):
    if shutil.disk_usage(OUT).free < 40 * 1024**3:
        raise ValueError('Preserve the original scratch floor and all partial evidence')
    with (OUT / name).open('x') as stream:
        stream.write(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def main():
    if (OUT.exists() or os.environ.get('CUDA_VISIBLE_DEVICES') != '' or os.nice(0) != 19
            or 'torch' in sys.modules or any(os.environ.get(k) != '1' for k in
                ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'))):
        raise ValueError('Fresh CUDA-hidden nice19 single-thread source-only screen required')
    os.sched_setscheduler(0, os.SCHED_IDLE, os.sched_param(0))
    memory = dict(line.split()[:2] for line in Path('/sys/fs/cgroup/memory.stat').read_text().splitlines())
    headroom = int(Path('/sys/fs/cgroup/memory.max').read_text()) - sum(int(memory[k]) for k in ('anon', 'shmem', 'kernel'))
    if headroom < 56 * 1024**3 or shutil.disk_usage(OUT.parent).free < 42 * 1024**3:
        raise ValueError('8GiB own/48GiB other-job memory and2GiB own/40GiB scratch reserves required')
    joined = json.loads((JOIN / 'complete.json').read_text())
    original = json.loads((SCREEN / 'complete.json').read_text())
    held = json.loads((HELD / 'complete.json').read_text())
    assert joined['complete'] and original['complete'] and held['complete']
    assert sha(CANDIDATE) == joined['source_files_sha256'][str(CANDIDATE)]
    assert sha(JOIN / 'every-candidate-disposition.jsonl') == joined['ledger_sha256']
    assert sha(SCREEN / 'every-original-row-screen.jsonl') == original['ledger_sha256']
    assert sha(HELD / 'held-question-registry.json') == held['held_registry_sha256']
    assert held['total_held_question_stems'] == 38791
    tests = ROOT / 'reports/whole-reasoning-view-screen-v1-tests.log'
    assert '3 passed' in tests.read_text()
    paths = [Path(__file__).resolve(), CANDIDATE, JOIN / 'complete.json', JOIN / 'every-candidate-disposition.jsonl',
        SCREEN / 'complete.json', SCREEN / 'every-original-row-screen.jsonl', HELD / 'complete.json',
        HELD / 'held-question-registry.json', ROOT / 'src/slm_lab/held_text_overlap_v1.py',
        ROOT / 'src/slm_lab/whole_reasoning_view_screen_v1.py',
        ROOT / 'tests/test_whole_reasoning_view_screen_v1.py', tests]
    files = {str(p): sha(p) for p in paths}
    OUT.mkdir()
    owner = dict(pid=os.getpid(), birth=Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19])
    write('owner-plan.json', dict(owner=owner, at=datetime.now(timezone.utc).isoformat(), source_files_sha256=files,
        maximum_context=MAXIMUM_CONTEXT, fixed_original_candidate_subset=6116,
        all_original_shard_rows_retained=20000, source_roles_never_reassigned=True,
        exact_original_execution_results_not_read_or_used_for_selection=True,
        original_4K_workers_and_sources_unchanged=True, transitive_source_history_not_rehashed=True,
        existing_literal_stem_policy_retained=True, semantic_or_historical_independence_not_certified=True,
        model_updates=0, model_context_changed=False, training_admitted=False, goal_achieved=False))
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
        from slm_lab.held_text_overlap_v1 import HeldTextOverlap
        from slm_lab.whole_reasoning_view_screen_v1 import apply_component_veto, screen
        pa.set_cpu_count(1); pa.set_io_thread_count(1)
        overlap = HeldTextOverlap(json.loads((HELD / 'held-question-registry.json').read_text()))
        registry = []
        with (SCREEN / 'every-original-row-screen.jsonl').open() as stream:
            for position, line in enumerate(stream):
                row = json.loads(line)
                assert row['original_shard_position'] == position and row['training_admitted'] is False
                registry.append(row)
        assert len(registry) == 20000
        candidates = []; rows = []; role_counts = Counter(); geometrical_counts = Counter()
        with (JOIN / 'every-candidate-disposition.jsonl').open() as source_stream:
            for batch in pq.ParquetFile(CANDIDATE).iter_batches(columns=['id', 'question_id', 'solution', 'r1_generation'],
                    batch_size=16, use_threads=False):
                for raw in batch.to_pylist():
                    position = len(rows); source = json.loads(source_stream.readline()); old = registry[position]
                    assert source['position'] == position
                    assert source['id'] == old['id'] == raw['id']
                    assert source['question_id'] == old['question_id'] == raw['question_id']
                    result = dict(original_shard_position=position, id=raw['id'], question_id=raw['question_id'],
                        source_role=old['source_role'], original_4K_disposition=old['status'],
                        status='outside_fixed_original_candidate_subset', fitting_screened_views=[], training_admitted=False)
                    rows.append(result)
                    if not old['fresh_execution_pending']:
                        continue
                    assert old['status'] == 'screened_metadata_positive_pending_fresh_execution'
                    assert old['source_role'] in ('train', 'reserved') and source['split'] == 'train'
                    assert old['publisher_solution_sha256'] == hashlib.sha256(raw['solution'].encode()).hexdigest()
                    assert old['publisher_solution_sha256'] == source['views']['code_only']['entire_target_utf8_sha256']
                    assert 'code_only' in old['fitting_views'] and source['views']['code_only']['complete_context_fits_4096']
                    observation = screen(source['views']['reasoning'], raw['r1_generation'], raw['solution'],
                        maximum_context=MAXIMUM_CONTEXT, overlap=overlap)
                    view = source['views']['reasoning']
                    if (view['entire_response_tokens_including_EOS'] > 1
                            and not view['contains_reserved_native_source_token']
                            and view['entire_context_tokens'] <= MAXIMUM_CONTEXT):
                        geometrical_counts[old['source_role']] += 1
                    result.update(fixed_pre_execution_role=old['fixed_pre_execution_role'],
                        dataset=old['dataset'], source_index=old['source_index'],
                        publisher_solution_sha256=old['publisher_solution_sha256'],
                        original_4K_fitting_views=old['fitting_views'], reasoning_screen=observation)
                    candidates.append(result); role_counts[old['source_role']] += 1
                if len(rows) % 256 == 0:
                    print(json.dumps(dict(phase='whole_16K_target_literal_and_code_identity_screen',
                        original_rows=len(rows), fixed_candidates=len(candidates))), flush=True)
            assert source_stream.readline() == ''
        assert len(rows) == 20000 and len(candidates) == 6116
        assert role_counts == {'train': 5509, 'reserved': 607}
        assert geometrical_counts == {'train': 4930, 'reserved': 540}
        screened = {r['original_shard_position']: r for r in apply_component_veto(candidates)}
        counts = Counter(); roles_after = {role: Counter() for role in role_counts}; reason_status = Counter()
        for position, result in enumerate(rows):
            if position in screened:
                result = rows[position] = screened[position]
                role = result['source_role']; reason_status[result['reasoning_screen']['status']] += 1
                roles_after[role]['positions_retained'] += 1
                roles_after[role]['component_quarantined'] += result['component_quarantined']
                for name in result['fitting_screened_views']:
                    roles_after[role][name + '_metadata_positive_views'] += 1
                if 'reasoning' in result['fitting_screened_views']:
                    roles_after[role]['reasoning_response_EOS_targets'] += result['reasoning_screen']['response_EOS_tokens']
            counts[result['status']] += 1
        assert all(sha(p) == h for p, h in files.items())
        ledger = OUT / 'every-original-row-screen.jsonl'; written = 0
        with ledger.open('x') as stream:
            for result in rows:
                text = json.dumps(result, ensure_ascii=False, separators=(',', ':'), allow_nan=False) + '\n'
                written += len(text.encode())
                if written > 512 * 1024**2:
                    raise ValueError('Whole-screen metadata allowance exhausted; preserve partial output')
                stream.write(text)
        write('complete.json', dict(complete=True, owner=owner, at=datetime.now(timezone.utc).isoformat(),
            source_files_sha256=files, original_rows=20000, fixed_original_candidate_positions=6116,
            maximum_context=MAXIMUM_CONTEXT, unchanged_original_roles=dict(role_counts),
            geometric_reasoning_fit_counts=dict(geometrical_counts), counts=dict(counts),
            by_original_role=roles_after, whole_reasoning_status_counts=dict(reason_status),
            held_literal_screen_configuration=overlap.configuration, ledger_sha256=sha(ledger),
            paired_original_4K_candidate_subset_not_exhaustive_16K_source_selection=True,
            no_new_role_assignment_or_execution_result_selection=True,
            whole_responses_and_original_code_hashes_checked=True,
            independent_target_and_component_readback_still_required=True,
            original_full_execution_and_returned_value_readback_still_required=True,
            native_16K_packing_and_independent_admission_still_required=True,
            actual_model_context=4096, model_context_changed=False,
            human_feedback=False, RL_oracle_admission_forbidden=True, training_admitted=False,
            model_updates=0, model_quality_gain_established=False, goal_achieved=False))
        print(json.dumps(dict(complete=True, counts=counts, by_original_role=roles_after,
            whole_reasoning_status_counts=reason_status)), flush=True)
    except BaseException as error:
        write('failure.json', dict(error=repr(error), traceback=traceback.format_exc(),
            partial_evidence_retained=True, training_admitted=False, model_updates=0, goal_achieved=False))
        raise


if __name__ == '__main__':
    main()
