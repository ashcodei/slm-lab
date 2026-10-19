"""Inspect every Python file in one pinned public Parquet and repeat doctests.

This creates an unadmitted function-contract candidate corpus. Public expected
answers remain visible inside unchanged original docstrings. Passing every
finite example twice is not hidden-test accuracy or general correctness.
"""
from collections import Counter,deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from datetime import datetime, timezone
import ast
import gzip
import json
import os
from pathlib import Path
import shutil
import time
import traceback

from slm_lab.codenet_checkpoint_v1 import FileBindings
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'reports/public-function-contracts-pilot-CPU-v1'
DATA = ROOT/'data/public-function-contracts-pilot-v1'
RESERVE = 40*1024**3
MAXIMUM = 512*1024**2


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or OUT.exists() or DATA.exists():
        raise ValueError('Fresh GPU-hidden actual original-source function pilot required')
    import pyarrow.parquet as pq
    from tokenizers import Tokenizer
    from slm_lab.held_text_overlap_v1 import HeldTextOverlap
    from slm_lab.local_source_closure_v2 import python_sources
    from slm_lab.public_pretraining_intake_v2 import candidate as source_candidate
    from slm_lab.public_function_contracts_v1 import from_function,execute_candidate,verified_training_row
    tests = ROOT/'reports/public-function-contracts-and-termination-v1-final-tests.log'
    if '7 passed' not in tests.read_text():raise ValueError('Actual original-source/native/EOS/finite-example controls must pass')
    base = ROOT/'reports/public-pretraining-source-preflight-CPU-v1'
    spec = next(s for s in json.loads((base/'plan.json').read_text())['pinned_sources'] if s['family']=='code')
    path = ROOT/'data/public-pretraining-source-preflight-v1/code/original.parquet'
    if file_hash(path) != spec['sha256'] or path.stat().st_size != spec['size']:
        raise ValueError('Actual complete original public source must match its pinned publisher LFS bytes')
    tokenizer_path = ROOT/'data/tokenizer-corpus/tokenizer.json'
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    registry_path = ROOT/'reports/CodeContests-native-held-text-v2/unchanged-held-text-registry.json'
    registry = json.loads(registry_path.read_text())
    if len(registry) != 32953:raise ValueError('Every original unchanged declared held stem required')
    overlap = HeldTextOverlap(registry)
    inputs = [tests,base/'plan.json',path,tokenizer_path,registry_path,*python_sources(ROOT,[Path(__file__),
        ROOT/'tests/test_public_function_contracts_v1.py',ROOT/'tests/test_function_termination_objective_v1.py'])]
    files = {str(p):file_hash(p) for p in inputs};guard = FileBindings(files)
    if shutil.disk_usage(ROOT).free < RESERVE+MAXIMUM:raise ValueError('The bounded candidate and all original persistent writers must fit')
    OUT.mkdir();DATA.mkdir();started = time.monotonic();last = started
    owner = dict(pid=os.getpid(),ticks=Path('/proc/self/stat').read_text().split(') ',1)[1].split()[19])
    parquet = pq.ParquetFile(path)
    atomic_json(OUT/'plan.json',dict(at=datetime.now(timezone.utc).isoformat(),owner=owner,source_files_sha256=files,
        source=spec,original_pinned_file_rows=parquet.metadata.num_rows,
        scope='Every Python original row and every top-level function of qualifying Python files in the complete first pinned source file',
        source_file_filter='Existing full-document own-BPE, per-file license and all32953held-text screens',
        candidate_filter='Undecorated ordinary top-level functions with literal docstrings containing2to128unskipped published examples and no unresolved declared-stdlib context',
        execution_workers=2,maximum_inflight_whole_candidates=4,each_candidate_fresh_whole_example_runs=2,per_run_wall_seconds=3,
        original_example_answers_visible_inside_original_function_docstring=True,
        separate_expected_output_assertions_or_comparison_artifacts_not_supplied=True,
        complete_original_function_response_and_EOS_required=True,maximum_complete_native_context=4096,
        original_raw_document_train_valid_role_unchanged=True,maximum_own_candidate_bytes=MAXIMUM,
        full_independent_source_native_and_execution_readback_still_required=True,
        optimization_admitted=False,neural_updates=0,goal_achieved=False))
    counts = Counter();roles = Counter();observed = 0;sources = set();seen_programs = set();inflight = deque()
    def work(candidate):
        actual = [execute_candidate(candidate,seconds=3),execute_candidate(candidate,seconds=3)]
        passed = all(r.get('status')=='all_original_examples_passed' for r in actual)
        error = None
        if passed:
            try:verified_training_row(candidate,actual,tokenizer,overlap)
            except ValueError as issue:passed=False;error=str(issue)
        return actual,passed,error
    try:
        with ExitStack() as stack:
            source_stream = stack.enter_context(gzip.open(DATA/'original-python-source-documents.jsonl.gz','xt',encoding='utf-8'))
            observations = stack.enter_context(gzip.open(DATA/'complete-candidate-observations.jsonl.gz','xt',encoding='utf-8'))
            ledger = stack.enter_context(gzip.open(DATA/'all-original-source-and-function-dispositions.jsonl.gz','xt',encoding='utf-8'))
            streams = [source_stream,observations,ledger]
            executor = stack.enter_context(ThreadPoolExecutor(max_workers=2))
            def progress():
                for stream in streams:stream.flush();os.fsync(stream.fileno())
                size = sum(p.stat().st_size for p in DATA.iterdir() if p.is_file())
                if size > MAXIMUM or shutil.disk_usage(ROOT).free < RESERVE:raise ValueError('Actual function-source storage exceeded its bound or writer reserve')
                atomic_json(OUT/'progress.json',dict(at=datetime.now(timezone.utc).isoformat(),original_rows_observed=observed,
                    actual_counts=dict(counts),actual_verified_roles=dict(roles),inflight_whole_candidates=len(inflight),
                    own_artifact_bytes=size,seconds=time.monotonic()-started,optimization_admitted=False,neural_updates=0,goal_achieved=False))
            def finish_one():
                candidate,future = inflight.popleft();actual,passed,error = future.result()
                original = candidate['source_original_file']
                compact = {k:v for k,v in candidate.items() if k != 'source_original_file'}
                compact['original_source_document_id'] = original['id']
                compact['original_source_row_ordinal'] = original['original_parquet_row_ordinal']
                reason = 'all_original_published_examples_pass_twice_with_exact_native_function' if passed else 'published_example_or_declared_context_or_repeatability_qualification'
                observations.write(json.dumps(dict(candidate=compact,actual_fresh_observations=actual,
                    verification_passed=passed,qualification_error=error,optimization_admitted=False),ensure_ascii=False,default=str)+'\n')
                ledger.write(json.dumps(dict(kind='function',original_row_ordinal=original['original_parquet_row_ordinal'],
                    original_lineno=candidate['original_function_span']['lineno'],candidate_id=candidate['id'],disposition=reason))+'\n')
                counts[reason] += 1
                if passed:
                    roles[candidate['role']+'_functions'] += 1
                    roles[candidate['role']+'_response_EOS_targets'] += candidate['response_EOS_targets']
                    roles[candidate['role']+'_native_context_tokens'] += candidate['complete_native_context_tokens']
                    roles[candidate['role']+'_original_published_examples'] += candidate['whole_original_example_count']
            ordinal = 0
            for batch in parquet.iter_batches(batch_size=128,use_threads=False):
                for row in batch.to_pylist():
                    current = ordinal;ordinal += 1;observed += 1
                    if row.get('language') != 'Python':reason='outside_this_original_Python_function_scope';original=None
                    else:
                        counts['original_Python_rows'] += 1
                        original,reason = source_candidate(row,spec,tokenizer,overlap)
                    ledger.write(json.dumps(dict(kind='file',original_row_ordinal=current,disposition=reason))+'\n')
                    counts[reason] += 1
                    if original is not None:
                        original['original_parquet_row_ordinal'] = current
                        original['original_source_file_sha256'] = spec['sha256']
                        for node in ast.parse(row['code']).body:
                            if not isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)):continue
                            candidate,rejection = from_function(original,node,tokenizer,overlap)
                            if candidate is not None and candidate['original_function_program_sha256'] in seen_programs:
                                candidate=None;rejection='exact_original_function_program_duplicate_in_this_pilot'
                            if candidate is None:
                                counts[rejection] += 1
                                ledger.write(json.dumps(dict(kind='function',original_row_ordinal=current,original_lineno=node.lineno,
                                    candidate_id=None,disposition=rejection))+'\n')
                                continue
                            seen_programs.add(candidate['original_function_program_sha256'])
                            if original['id'] not in sources:
                                source_stream.write(json.dumps(original,ensure_ascii=False,default=str)+'\n');sources.add(original['id'])
                            inflight.append((candidate,executor.submit(work,candidate)));counts['actual_static_whole_function_candidates'] += 1
                            if len(inflight) >= 4:finish_one()
                    if time.monotonic()-last >= 10:guard.check();progress();last=time.monotonic()
            while inflight:finish_one();progress()
            if observed != parquet.metadata.num_rows:raise ValueError('The complete original public file must be inspected without source-prefix sampling')
            progress()
        guard.check(full=True)
        artifacts = [p for p in DATA.iterdir() if p.is_file()]
        atomic_json(DATA/'manifest.json',dict(format='own-public-function-literal-doc-contract-candidate-v1',
            original_source=spec,source_files_sha256={str(p):file_hash(p) for p in artifacts},
            actual_source_rows_observed=observed,actual_original_source_documents_retained=len(sources),
            actual_counts=dict(counts),actual_verified_roles=dict(roles),
            complete_original_function_targets_and_all_public_examples_preserved=True,
            original_example_answers_visible_inside_original_program_docstring=True,
            finite_public_examples_not_hidden_tests_or_general_correctness=True,
            independent_source_native_and_fresh_runtime_readback_required_before_admission=True,
            optimization_admitted=False,neural_updates=0,goal_achieved=False))
        atomic_json(OUT/'complete.json',dict(at=datetime.now(timezone.utc).isoformat(),complete=True,
            actual_original_source_rows=observed,actual_counts=dict(counts),actual_verified_roles=dict(roles),
            all_original_Python_rows_and_top_level_functions_in_declared_file_observed=True,
            all_accepted_functions_pass_every_original_published_example_in_two_fresh_confined_interpreters=True,
            complete_original_function_native_inputs_masks_and_EOS_rebuilt=True,
            source_files_sha256={**files,str(OUT/'plan.json'):file_hash(OUT/'plan.json'),str(DATA/'manifest.json'):file_hash(DATA/'manifest.json')},
            seconds=time.monotonic()-started,optimization_admitted=False,new_neural_updates=0,
            fresh_independent_readback_training_and_generated_quality_assessment_still_required=True,goal_achieved=False))
        print(json.dumps(dict(complete=str(OUT/'complete.json'),actual_verified_roles=dict(roles),optimization_admitted=False)),flush=True)
    except BaseException as error:
        atomic_json(OUT/'failure.json',dict(error=repr(error),traceback=traceback.format_exc(),
            every_original_job_and_partial_new_source_artifact_preserved=True,optimization_admitted=False,goal_achieved=False))
        raise


if __name__ == '__main__':main()
