"""Explain one frozen fresh-test disagreement without rerunning a program.

The original public prompt explicitly permits any email-list order, whereas
its synthetic assertions require one order. Keep both original verdicts and
all actual returned values. Do not change the live reader, source, roles,
tests, reward protocol or training admission.
"""
import ast
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

import pyarrow.parquet as pq

from slm_lab.public_code_disposition_readback_v1 import row_sha, verify_observation
from slm_lab.public_code_instruction_screen_v1 import parse_whole

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/public-code-unordered-contract-diagnosis-CPU-v1'
POSITION = 5114


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def record(path, position):
    # This may read a complete prefix of a live journal; it never source-binds
    # its changing inode. The one exact complete LF record is copied below.
    with path.open('rb') as stream:
        for line in stream:
            if not line.endswith(b'\n'):
                break
            value = json.loads(line)
            if value['original_shard_position'] == position:
                return value, line
    raise ValueError('The exact original complete LF record is not yet present')


def main():
    if (OUT.exists() or os.environ.get('CUDA_VISIBLE_DEVICES') != '' or os.nice(0) < 19
            or any(os.environ.get(k) != '1' for k in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'])):
        raise ValueError('Fresh one-thread CUDA-hidden nice19 source-only diagnosis required')
    os.sched_setscheduler(0, os.SCHED_IDLE, os.sched_param(0))
    parquet = ROOT / 'data/public-code-instruction-candidate-v1/train-00000-of-00050.parquet'
    if sha(parquet) != '342757e0c6b706c8f68cf0a867ead5c417d0273726453dbcfc934f5c3c6ca891':
        raise ValueError('Exact original pinned public shard required')
    offset = 0
    row = None
    for batch in pq.ParquetFile(parquet).iter_batches(batch_size=256, use_threads=False):
        if offset <= POSITION < offset + batch.num_rows:
            row = batch.to_pylist()[POSITION - offset]
            break
        offset += batch.num_rows
    if row is None or row_sha(row) != '90ce2d85479a493b5b019689af77a1e0661fd08f95ffced5266db7a646a5641d':
        raise ValueError('Exact original public instruction, program and assertions required')
    old, old_bytes = record(ROOT / 'reports/full-public-code-shard-qualification-CPU-v1/all100000-original-dispositions.jsonl', POSITION)
    fresh, fresh_bytes = record(ROOT / 'reports/full-public-code-shard-readback-CPU-v1/all100000-fresh-readback-observations.jsonl', POSITION)
    parsed = parse_whole(row)
    a = old['actual_whole_execution']
    b = fresh['fresh_whole_program_execution']
    verify_observation(parsed, a)
    verify_observation(parsed, b)
    if (not a['all_published_tests_passed'] or b['status'] != 'published_test_mismatch'
            or fresh['original_entry_sha256'] != row_sha(row)
            or fresh['training_admitted'] is not False
            or 'The order of the emails in the list does not matter.' not in row['input']):
        raise ValueError('Exact completed fresh disagreement and explicit public unordered contract required')
    cases = []
    for i, (case, actual_text) in enumerate(zip(parsed['cases'], b['actual_value_literals'])):
        expected, actual = ast.literal_eval(case['expected_literal']), ast.literal_eval(actual_text)
        if (type(expected) is not list or type(actual) is not list
                or any(type(value) is not str for value in expected + actual)):
            raise ValueError('This diagnostic is restricted to the actual finite string-list outputs')
        contract_equal = len(actual) == len(set(actual)) and set(actual) == set(expected)
        cases.append(dict(original_test_position=i, original_expected=expected, actual_fresh_value=actual,
            original_list_equality=actual == expected,
            declared_unique_unordered_set_equality=contract_equal))
    if len(cases) != 10 or sum(c['original_list_equality'] for c in cases) != 9 or not all(
            c['declared_unique_unordered_set_equality'] for c in cases):
        raise ValueError('Every original finite value must support the limited order-only diagnosis')
    OUT.mkdir()
    (OUT / 'unchanged-public-source-row.json').write_text(json.dumps(row, indent=2) + '\n')
    (OUT / 'exact-original-qualification-record.jsonl').write_bytes(old_bytes)
    (OUT / 'exact-fresh-reader-record.jsonl').write_bytes(fresh_bytes)
    paths = [Path(__file__), parquet, OUT / 'unchanged-public-source-row.json',
        OUT / 'exact-original-qualification-record.jsonl', OUT / 'exact-fresh-reader-record.jsonl',
        ROOT / 'src/slm_lab/public_code_disposition_readback_v1.py',
        ROOT / 'src/slm_lab/public_code_instruction_screen_v1.py']
    proof = dict(complete=True, at=datetime.now(timezone.utc).isoformat(),
        source_files_sha256={str(p): sha(p) for p in paths}, original_shard_position=POSITION,
        source_id=row['id'], original_entry_sha256=row_sha(row), all_original_finite_cases=cases,
        original_qualification_published_assertions_passed=10, fresh_published_assertions_passed=9,
        actual_fresh_values_satisfying_declared_unique_unordered_result=10,
        original_published_test_mismatch_not_relabelled=True,
        diagnosis='Synthetic assertions require one list order that the original public question explicitly leaves unrestricted.',
        whole_program_or_arbitrary_email_semantic_correctness_not_certified=True,
        no_code_execution_or_program_repair_or_reroll=True, no_existing_reader_or_source_or_role_changed=True,
        no_published_benchmark_or_RL_reward_protocol_changed=True,
        current_public_synthetic_tests_remain_forbidden_as_independent_RL_truth=True,
        training_admitted=False, model_updates=0, goal_achieved=False)
    (OUT / 'complete.json').write_text(json.dumps(proof, indent=2) + '\n')
    print(json.dumps(dict(report=str(OUT / 'complete.json'), finite_order_only_mismatches=1,
        new_program_executions=0, training_admitted=False)), flush=True)


if __name__ == '__main__':
    main()
