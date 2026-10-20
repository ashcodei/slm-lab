"""Reconstruct every longer-view decision from complete original text.

Does not import either producer helper. Uses the original shared held-stem
matcher, so it is not a second independent literal-matching algorithm. This
is source/decision readback, not native token packing or training admission.
"""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path('/tmp/slm-whole-reasoning-16K-screen-v1')
JOIN = Path('/tmp/slm-public-reasoned-code-question-join-v2')
OLD = Path('/tmp/slm-public-reasoned-code-screen-v2')
HELD = Path('/tmp/slm-public-reasoned-code-held-registry-v1/held-question-registry.json')
RAW = Path('/tmp/slm-public-reasoned-code-candidate-v1/original-python-00000.parquet')
OUT = ROOT / 'reports/whole-reasoning-16K-readback-v1'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(name, value):
    with (OUT / name).open('x') as stream:
        stream.write(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def main():
    if (OUT.exists() or os.environ.get('CUDA_VISIBLE_DEVICES') != '' or os.nice(0) != 19
            or 'torch' in sys.modules):
        raise ValueError('Fresh CUDA-hidden nice19 source-only readback required')
    os.sched_setscheduler(0, os.SCHED_IDLE, os.sched_param(0))
    done = json.loads((SOURCE / 'complete.json').read_text())
    if (done.get('complete') is not True or (SOURCE / 'failure.json').exists()
            or done['original_rows'] != 20000 or done['maximum_context'] != 16384
            or done['fixed_original_candidate_positions'] != 6116 or done['training_admitted'] is not False):
        raise ValueError('The successful complete original whole-response screen is required')
    owner = done['owner']; process = Path('/proc') / str(owner['pid']) / 'stat'
    if process.exists():
        stat = process.read_text().rsplit(')', 1)[1].split()
        if stat[19] == owner['birth'] and stat[0] not in ('Z', 'X'):
            raise ValueError('Never bind still-running producer outputs')
    files = dict(done['source_files_sha256'])
    for path in [Path(__file__).resolve(), SOURCE / 'complete.json', SOURCE / 'every-original-row-screen.jsonl']:
        files[str(path)] = sha(path)
    if (sha(SOURCE / 'every-original-row-screen.jsonl') != done['ledger_sha256']
            or any(sha(p) != digest for p, digest in files.items())):
        raise ValueError('Every exact original source and completed decision byte must remain unchanged')
    OUT.mkdir()
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
        from slm_lab.held_text_overlap_v1 import HeldTextOverlap
        pa.set_cpu_count(1); pa.set_io_thread_count(1)
        matcher = HeldTextOverlap(json.loads(HELD.read_text()))
        old = [json.loads(line) for line in (OLD / 'every-original-row-screen.jsonl').open()]
        actual = [json.loads(line) for line in (SOURCE / 'every-original-row-screen.jsonl').open()]
        assert len(old) == len(actual) == 20000
        expected = []; bad_components = set(); component_roles = {}; role_counts = Counter(); geometry = Counter()
        with (JOIN / 'every-candidate-disposition.jsonl').open() as joined:
            for batch in pq.ParquetFile(RAW).iter_batches(columns=['id', 'question_id', 'solution', 'r1_generation'],
                    batch_size=16, use_threads=False):
                for original in batch.to_pylist():
                    position = len(expected); previous = old[position]; source = json.loads(joined.readline())
                    assert previous['original_shard_position'] == source['position'] == position
                    assert original['id'] == previous['id'] == source['id']
                    assert original['question_id'] == previous['question_id'] == source['question_id']
                    result = dict(original_shard_position=position, id=original['id'], question_id=original['question_id'],
                        source_role=previous['source_role'], original_4K_disposition=previous['status'],
                        status='outside_fixed_original_candidate_subset', fitting_screened_views=[], training_admitted=False)
                    expected.append(result)
                    if not previous['fresh_execution_pending']:
                        continue
                    role = previous['source_role']; fixed = previous['fixed_pre_execution_role']; role_counts[role] += 1
                    assert role == fixed['source_role'] and fixed['role_fixed_before_fresh_replay'] is True
                    assert fixed['optimization_forbidden'] is (role == 'reserved')
                    component = fixed['declared_duplicate_component_sha256']
                    assert component not in component_roles or component_roles[component] == role
                    component_roles[component] = role
                    code_hash = hashlib.sha256(original['solution'].encode()).hexdigest()
                    assert code_hash == previous['publisher_solution_sha256'] == source['views']['code_only']['entire_target_utf8_sha256']
                    assert 'code_only' in previous['fitting_views']
                    view = source['views']['reasoning']; text = original['r1_generation']
                    digest = hashlib.sha256(text.encode()).hexdigest(); assert digest == view['entire_target_utf8_sha256']
                    p = view['prompt_tokens_including_BOS']; r = view['entire_response_tokens_including_EOS']; n = view['entire_context_tokens']
                    assert all(type(v) is int and v > 0 for v in (p, r, n)) and n == p + r
                    assert type(view['contains_reserved_native_source_token']) is bool
                    valid = r > 1 and not view['contains_reserved_native_source_token']
                    assert view['complete_context_fits_4096'] is (valid and n <= 4096)
                    observation = dict(status='whole_context_or_native_source_token_qualification',
                        whole_response_sha256=digest, prompt_tokens=p, response_EOS_tokens=r,
                        whole_context_tokens=n, maximum_context=16384, original_4K_fit=view['complete_context_fits_4096'],
                        component_overlap_veto=False, training_admitted=False)
                    if valid and n <= 16384:
                        geometry[role] += 1
                        if any('\0' in value for value in (view['prompt'], text, original['solution'])):
                            observation['status'] = 'source_NUL_qualification'
                        elif not original['solution'].strip() or original['solution'].strip() not in text:
                            observation['status'] = 'publisher_solution_not_present_in_entire_reasoning'
                        else:
                            matches = dict(prompt_literal=matcher.literal(view['prompt']), target_literal=matcher.literal(text))
                            veto = any(matches.values())
                            observation.update(status='held_literal_overlap' if veto else 'whole_reasoning_metadata_screen_positive',
                                entire_target_literal_overlap=matches, component_overlap_veto=veto,
                                complete_publisher_solution_occurs_in_response=True, publisher_solution_utf8_sha256=code_hash)
                            if veto: bad_components.add(component)
                    result.update(fixed_pre_execution_role=fixed, dataset=previous['dataset'], source_index=previous['source_index'],
                        publisher_solution_sha256=code_hash, original_4K_fitting_views=previous['fitting_views'], reasoning_screen=observation)
                if len(expected) % 2048 == 0:
                    print(json.dumps(dict(phase='independent_full_original_target_readback', original_rows=len(expected))), flush=True)
            assert joined.readline() == ''
        assert len(expected) == 20000 and role_counts == {'train': 5509, 'reserved': 607}
        counts = Counter(); roles = {role: Counter() for role in role_counts}; statuses = Counter()
        for row in expected:
            if 'reasoning_screen' in row:
                blocked = row['fixed_pre_execution_role']['declared_duplicate_component_sha256'] in bad_components
                views = [] if blocked else ['code_only']
                if not blocked and row['reasoning_screen']['status'] == 'whole_reasoning_metadata_screen_positive': views.append('reasoning')
                row.update(status='long_context_overlap_component_quarantined' if blocked
                    else 'metadata_positive_pending_original_execution_readback', fitting_screened_views=views,
                    component_quarantined=blocked, optimization_forbidden=row['source_role'] == 'reserved')
                role = row['source_role']; roles[role]['positions_retained'] += 1; roles[role]['component_quarantined'] += blocked
                statuses[row['reasoning_screen']['status']] += 1
                for view_name in views: roles[role][view_name + '_metadata_positive_views'] += 1
                if 'reasoning' in views: roles[role]['reasoning_response_EOS_targets'] += row['reasoning_screen']['response_EOS_tokens']
            assert row == actual[row['original_shard_position']]
            counts[row['status']] += 1
        assert dict(counts) == done['counts'] and roles == done['by_original_role']
        assert statuses == done['whole_reasoning_status_counts'] and geometry == done['geometric_reasoning_fit_counts']
        assert all(sha(p) == digest for p, digest in files.items())
        write('complete.json', dict(complete=True, at=datetime.now(timezone.utc).isoformat(), source_files_sha256=files,
            all_original_rows_reconstructed=20000, original_fixed_candidates=6116, unchanged_roles=dict(role_counts),
            every_whole_response_and_program_hash_and_code_inclusion_reconstructed=True,
            every_literal_match_and_component_veto_and_view_count_reconstructed=True,
            by_original_role=roles, shared_original_literal_matcher_not_independent_matching_algorithm=True,
            producer_helper_or_script_not_imported=True, native_tokenization_not_replayed_by_reader=True,
            full_original_execution_readback_and_native_16K_packing_still_required=True,
            actual_model_context=4096, larger_context_GPU_capacity_or_neural_quality_proved=False,
            training_admitted=False, RL_oracle_admission_forbidden=True, model_updates=0, goal_achieved=False))
        print(json.dumps(dict(complete=True, original_rows=20000, by_original_role=roles)), flush=True)
    except BaseException as error:
        write('failure.json', dict(error=repr(error), traceback=traceback.format_exc(), partial_outputs_retained=True,
            training_admitted=False, model_updates=0, goal_achieved=False))
        raise


if __name__ == '__main__':
    main()
