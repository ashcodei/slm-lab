"""Reconcile closed whole-context counts and correct string-score counters.

The original join and zero-candidate screen stay unchanged. This reader does
not rerun tokenization or certify source independence/execution correctness.
"""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
JOIN = Path('/tmp/slm-public-reasoned-code-question-join-v2')
CANDIDATE = Path('/tmp/slm-public-reasoned-code-candidate-v1')
OUT = ROOT / 'reports/public-reasoned-code-full-fit-readback-v1'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    if OUT.exists() or os.environ.get('CUDA_VISIBLE_DEVICES') != '' or os.nice(0) != 19:
        raise ValueError('Fresh CPU-hidden nice19 source-only reconciliation required')
    os.sched_setscheduler(0, os.SCHED_IDLE, os.sched_param(0))
    import pyarrow as pa
    import pyarrow.parquet as pq
    from slm_lab.public_reasoned_code_metadata_v1 import normalized_pass_rate, reported_all_pass
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    source = json.loads((JOIN / 'complete.json').read_text())
    survey = json.loads((CANDIDATE / 'survey-complete.json').read_text())
    assert source['complete'] and survey['complete']
    assert sha(JOIN / 'every-candidate-disposition.jsonl') == source['ledger_sha256']
    files = dict(source['source_files_sha256'])
    files.update({str(path): sha(path) for path in [Path(__file__).resolve(), JOIN / 'complete.json',
        JOIN / 'every-candidate-disposition.jsonl', CANDIDATE / 'survey-complete.json',
        ROOT / 'src/slm_lab/public_reasoned_code_metadata_v1.py',
        ROOT / 'tests/test_public_reasoned_code_metadata_v1.py',
        ROOT / 'reports/public-reasoned-code-metadata-v1-tests.log']})
    assert all(sha(path) == value for path, value in files.items())
    parquet = pq.ParquetFile(CANDIDATE / 'original-python-00000.parquet')
    assert str(parquet.schema_arrow.field('pass_rate').type) == 'string'
    counts = Counter()
    types = Counter()
    with (JOIN / 'every-candidate-disposition.jsonl').open() as ledger:
        for batch in parquet.iter_batches(columns=['id', 'question_id', 'dataset', 'split', 'index', 'pass_rate'],
                batch_size=512, use_threads=False):
            for row in batch.to_pylist():
                saved = json.loads(next(ledger))
                assert saved['position'] == counts['original_rows'] and saved['id'] == row['id']
                assert saved['question_id'] == row['question_id']
                assert (saved['dataset'], saved['split'], saved['index']) == (row['dataset'], row['split'], int(row['index']))
                counts['original_rows'] += 1
                types[type(row['pass_rate']).__name__] += 1
                positive = reported_all_pass(row['pass_rate'])
                counts['reported_pass_rate_one_normalized'] += positive
                counts['reported_unknown_or_invalid_pass_rate'] += normalized_pass_rate(row['pass_rate']) in (None, -1.)
                if row['split'] != 'train':
                    counts['upstream_non_train_quarantined'] += 1
                    continue
                counts['whole_train_question_joined'] += 1
                assert saved['publisher_reported_pass_rate'] == row['pass_rate']
                veto = bool(saved['known_held_source_component'] or saved['APPS_internal_validation']
                    or saved['known_excluded_CodeContests_normalized_question_matches'])
                counts['known_held_identity_or_exact_normalized_question_flagged'] += veto
                for name, view in saved['views'].items():
                    context = view['prompt_tokens_including_BOS'] + view['entire_response_tokens_including_EOS']
                    assert context == view['entire_context_tokens']
                    expected = (view['entire_response_tokens_including_EOS'] > 1
                        and not view['contains_reserved_native_source_token'] and context <= 4096)
                    assert view['complete_context_fits_4096'] is expected
                    counts[name + '_complete_context_fits_4096'] += expected
                    counts[name + '_maximum_whole_context_tokens'] = max(counts[name + '_maximum_whole_context_tokens'], context)
                    counts[name + '_fits_no_current_identity_flag_reported_pass_one_normalized'] += expected and not veto and positive
        assert ledger.readline() == ''
    assert counts['original_rows'] == 20000 and counts['whole_train_question_joined'] == 17424
    assert counts['reported_pass_rate_one_normalized'] == survey['counts']['reported_pass_rate_one'] == 11806
    assert counts['upstream_non_train_quarantined'] == 2576
    assert counts['known_held_identity_or_exact_normalized_question_flagged'] == 5323
    assert counts['code_only_complete_context_fits_4096'] == 17307
    assert counts['reasoning_complete_context_fits_4096'] == 3940
    assert all(sha(path) == value for path, value in files.items())
    OUT.mkdir()
    result = dict(complete=True, at=datetime.now(timezone.utc).isoformat(), source_files_sha256=files,
        original_rows=20000, counts=dict(counts), original_pass_rate_Python_types=dict(types),
        original_join_and_zero_candidate_screen_unchanged=True,
        supersedes_original_join_two_strict_numeric_comparison_counters_only=True,
        original_whole_context_fit_counts_and_all_roles_preserved=True,
        token_lengths_not_recomputed_by_this_reader=True,
        all_reported_native_context_length_arithmetic_reconciled=True,
        publisher_numeric_strings_parsed_before_metadata_positive_counting=True,
        full_overlap_screen_and_fresh_execution_and_independent_reader_still_required=True,
        data_admitted_to_training=False, model_updates=0, model_quality_gain_established=False, goal_achieved=False)
    (OUT / 'complete.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(complete=True, counts=dict(counts), pass_rate_types=dict(types))), flush=True)


if __name__ == '__main__':
    main()
