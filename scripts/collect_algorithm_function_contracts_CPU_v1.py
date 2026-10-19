"""Pinned direct-GitHub function/example corpus, not a teacher or final model.

Downloads a complete source archive, checks every ordinary file against its
pinned Git blob, and repeats every qualified function's public examples twice
in confined interpreters. No archive path is extracted or source imported in
the parent. The resulting candidate remains unadmitted until full readback.
"""
from collections import Counter,deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from datetime import datetime, timezone
import ast
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import traceback
from urllib.request import Request,urlopen

from slm_lab.codenet_checkpoint_v1 import FileBindings
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'reports/direct-algorithm-function-contracts-CPU-v1'
DATA = ROOT/'data/direct-algorithm-function-contracts-v1'
REPOSITORY = 'TheAlgorithms/Python'
REVISION = '25c712f2692449e91a57748d328ef34fccf25654'
TREE = '85d4f12cb691afddb21e3a54e4a69583570e41eb'
RESERVE = 40*1024**3
MAXIMUM = 512*1024**2


def get(url):
    return urlopen(Request(url,headers={'User-Agent':'slm-lab-public-function-contract-sources/1.0'}),timeout=60)


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or OUT.exists() or DATA.exists():
        raise ValueError('Fresh GPU-hidden pinned direct-repository source collection required')
    from tokenizers import Tokenizer
    from slm_lab.held_text_overlap_v1 import HeldTextOverlap
    from slm_lab.local_source_closure_v2 import python_sources
    from slm_lab.public_function_contracts_v1 import execute_candidate
    from slm_lab.public_git_function_sources_v1 import archive_files,source_entry,repository_function,verified_repository_row
    tests = ROOT/'reports/public-git-function-sources-v1-final-tests.log'
    function_tests = ROOT/'reports/public-function-contracts-and-termination-v1-final-tests.log'
    if '4 passed' not in tests.read_text() or '7 passed' not in function_tests.read_text():
        raise ValueError('Actual source/blob/path/partition/whole-function/runtime/EOS controls must pass')
    protected = ROOT/'data/code-search-expanded-v1/protected-repositories.txt'
    if REPOSITORY.casefold() in {p.casefold().strip() for p in protected.read_text().splitlines()}:
        raise ValueError('An original protected held repository cannot supply this source')
    tokenizer_path = ROOT/'data/tokenizer-corpus/tokenizer.json'
    registry_path = ROOT/'reports/CodeContests-native-held-text-v2/unchanged-held-text-registry.json'
    registry = json.loads(registry_path.read_text())
    if len(registry) != 32953:raise ValueError('Every original unchanged declared held stem required')
    overlap = HeldTextOverlap(registry);tokenizer = Tokenizer.from_file(str(tokenizer_path))
    paths = [tests,function_tests,protected,tokenizer_path,registry_path,*python_sources(ROOT,[Path(__file__),
        ROOT/'tests/test_public_git_function_sources_v1.py'])]
    files = {str(p):file_hash(p) for p in paths};guard = FileBindings(files)
    if shutil.disk_usage(ROOT).free < RESERVE+MAXIMUM:raise ValueError('The bounded source corpus and persistent reserve must fit')
    OUT.mkdir();DATA.mkdir();started = time.monotonic()
    owner = dict(pid=os.getpid(),ticks=Path('/proc/self/stat').read_text().split(') ',1)[1].split()[19])
    atomic_json(OUT/'plan.json',dict(at=datetime.now(timezone.utc).isoformat(),owner=owner,source_files_sha256=files,
        repository=REPOSITORY,revision=REVISION,git_tree_sha1=TREE,
        scope='Every regular Python source file and its top-level functions in the complete pinned repository',
        ordinary_archive_files_must_match_every_pinned_Git_blob=True,archive_members_never_extracted=True,
        original_repository_MIT_notice_required_and_preserved=True,no_publisher_train_or_test_partition_exists=True,
        our_document_roles_use_same_original_UTF8_hash_rule_as_new_raw_corpus=True,
        all32953original_held_stems_screen_whole_source_and_complete_composed_native_text=True,
        execution_workers=2,maximum_inflight_whole_candidates=4,per_run_wall_seconds=3,fresh_complete_example_runs=2,
        public_example_answers_visible_in_original_program_docstring=True,
        not_hidden_tests_or_general_correctness_or_efficiency_certification=True,
        source_archive_not_imported_or_used_as_model_weights=True,
        maximum_own_artifact_bytes=MAXIMUM,full_independent_source_native_and_execution_readback_required=True,
        optimization_admitted=False,neural_updates=0,goal_achieved=False))
    try:
        tree_url='https://api.github.com/repos/'+REPOSITORY+'/git/trees/'+TREE+'?recursive=1'
        with get(tree_url) as response:tree=json.load(response)
        if tree.get('sha') != TREE or tree.get('truncated') is not False:
            raise ValueError('The exact complete immutable original Git tree is required')
        atomic_json(DATA/'original-git-tree.json',tree)
        archive = DATA/'original-source.tar.gz';written = 0
        with get('https://codeload.github.com/'+REPOSITORY+'/tar.gz/'+REVISION) as response,archive.open('xb') as output:
            while True:
                chunk = response.read(1024**2)
                if not chunk:break
                written += len(chunk)
                if written > 128*1024**2 or shutil.disk_usage(ROOT).free-len(chunk) < RESERVE:
                    raise ValueError('The pinned complete archive exceeded its declared bounded source size')
                output.write(chunk)
            output.flush();os.fsync(output.fileno())
        git_proofs = [];python_files = [];license_raw = None
        for path,raw,entry in archive_files(archive,tree,'Python-'+REVISION):
            git_proofs.append(dict(path=path,original_git_blob_sha1=entry['sha'],bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()))
            if path == 'LICENSE.md':license_raw=raw;(DATA/'original-LICENSE.md').write_bytes(raw)
            if path == 'README.md':(DATA/'original-README.md').write_bytes(raw)
            if path.endswith('.py'):python_files.append((path,raw,entry))
        if (license_raw is None or b'Permission is hereby granted, free of charge' not in license_raw
                or b'THE SOFTWARE IS PROVIDED' not in license_raw):
            raise ValueError('The actual original repository MIT license text must remain present and byte verified')
        license_sha = hashlib.sha256(license_raw).hexdigest()
        atomic_json(DATA/'complete-original-Git-blob-proofs.json',git_proofs)
        python_files.sort(key=lambda item:item[0])
        files.update({str(p):file_hash(p) for p in [archive,DATA/'original-git-tree.json',DATA/'original-LICENSE.md',
            DATA/'original-README.md',DATA/'complete-original-Git-blob-proofs.json']})
        guard = FileBindings(files)
        counts = Counter();roles = Counter();inflight = deque();seen = set();observed = 0;last = time.monotonic()
        def work(candidate):
            observations = [execute_candidate(candidate,seconds=3),execute_candidate(candidate,seconds=3)]
            passed = all(r.get('status') == 'all_original_examples_passed' for r in observations);qualification = None
            if passed:
                try:verified_repository_row(candidate,observations,tokenizer,overlap)
                except ValueError as error:passed=False;qualification=str(error)
            return observations,passed,qualification
        with ExitStack() as stack:
            source_stream = stack.enter_context(gzip.open(DATA/'original-qualified-Python-sources.jsonl.gz','xt',encoding='utf-8'))
            candidates = stack.enter_context(gzip.open(DATA/'complete-function-candidate-observations.jsonl.gz','xt',encoding='utf-8'))
            ledger = stack.enter_context(gzip.open(DATA/'every-original-Python-file-and-function-disposition.jsonl.gz','xt',encoding='utf-8'))
            executor = stack.enter_context(ThreadPoolExecutor(max_workers=2))
            def progress():
                for stream in [source_stream,candidates,ledger]:stream.flush();os.fsync(stream.fileno())
                size = sum(p.stat().st_size for p in DATA.iterdir() if p.is_file())
                if size > MAXIMUM or shutil.disk_usage(ROOT).free < RESERVE:raise ValueError('The declared source bound or original writers reserve was exceeded')
                atomic_json(OUT/'progress.json',dict(at=datetime.now(timezone.utc).isoformat(),original_Python_files_observed=observed,
                    original_Python_file_population=len(python_files),actual_counts=dict(counts),actual_verified_roles=dict(roles),
                    inflight_whole_candidates=len(inflight),own_artifact_bytes=size,seconds=time.monotonic()-started,
                    optimization_admitted=False,neural_updates=0,goal_achieved=False))
            def finish_one():
                candidate,future=inflight.popleft();actual,passed,qualification=future.result()
                original=candidate['source_original_file']
                compact={k:v for k,v in candidate.items() if k != 'source_original_file'}
                compact['original_source_document_id']=original['id'];compact['original_source_git_path']=original['source_git_path']
                status='every_original_public_example_passed_twice' if passed else 'public_example_or_context_or_repeatability_qualification'
                candidates.write(json.dumps(dict(candidate=compact,actual_fresh_observations=actual,verification_passed=passed,
                    qualification_error=qualification,optimization_admitted=False),ensure_ascii=False,default=str)+'\n')
                ledger.write(json.dumps(dict(kind='function',original_git_path=original['source_git_path'],
                    original_lineno=candidate['original_function_span']['lineno'],candidate_id=candidate['id'],disposition=status))+'\n')
                counts[status]+=1
                if passed:
                    role=candidate['role'];roles[role+'_functions']+=1
                    roles[role+'_response_EOS_targets']+=candidate['response_EOS_targets']
                    roles[role+'_native_context_tokens']+=candidate['complete_native_context_tokens']
                    roles[role+'_original_public_examples']+=candidate['whole_original_example_count']
            for path,raw,entry in python_files:
                observed+=1
                original,reason=source_entry(path,raw,entry,repository=REPOSITORY,revision=REVISION,
                    repository_license_sha256=license_sha,tokenizer=tokenizer,overlap=overlap)
                status=reason or 'whole_original_git_Python_source_ready';counts[status]+=1
                ledger.write(json.dumps(dict(kind='file',original_git_path=path,disposition=status))+'\n')
                if original is not None:
                    source_stream.write(json.dumps(original,ensure_ascii=False)+'\n')
                    for node in ast.parse(original['original_public_text']).body:
                        if not isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)):continue
                        candidate,rejection=repository_function(original,node,tokenizer,overlap)
                        if candidate is not None and candidate['original_function_program_sha256'] in seen:
                            candidate=None;rejection='exact_whole_function_duplicate_in_this_repository_candidate'
                        if candidate is None:
                            counts[rejection]+=1
                            ledger.write(json.dumps(dict(kind='function',original_git_path=path,original_lineno=node.lineno,
                                candidate_id=None,disposition=rejection))+'\n');continue
                        seen.add(candidate['original_function_program_sha256']);counts['actual_complete_static_function_candidates']+=1
                        inflight.append((candidate,executor.submit(work,candidate)))
                        if len(inflight)>=4:finish_one()
                if time.monotonic()-last>=10:guard.check();progress();last=time.monotonic()
            while inflight:finish_one();progress()
            progress()
        if observed!=len(python_files):raise ValueError('Every original Python source file in the complete pinned repository must remain observed')
        guard.check(full=True)
        artifacts=[p for p in DATA.iterdir() if p.is_file()]
        atomic_json(DATA/'manifest.json',dict(format='own-pinned-Git-function-public-example-candidate-v1',
            repository=REPOSITORY,revision=REVISION,git_tree_sha1=TREE,original_license_sha256=license_sha,
            source_files_sha256={str(p):file_hash(p) for p in artifacts},actual_original_regular_files=len(git_proofs),
            actual_original_Python_files=observed,actual_counts=dict(counts),actual_verified_roles=dict(roles),
            actual_complete_original_Git_blobs_verified=True,original_public_example_answers_visible_in_docstrings=True,
            no_publisher_train_or_test_partition_exists=True,original_file_notices_and_full_MIT_license_preserved=True,
            independent_full_source_native_and_fresh_execution_readback_required=True,
            finite_examples_not_general_correctness_or_empirical_efficiency=True,optimization_admitted=False,neural_updates=0,goal_achieved=False))
        atomic_json(OUT/'complete.json',dict(at=datetime.now(timezone.utc).isoformat(),complete=True,
            actual_original_regular_files=len(git_proofs),actual_original_Python_files=observed,
            actual_counts=dict(counts),actual_verified_roles=dict(roles),
            source_files_sha256={**files,str(OUT/'plan.json'):file_hash(OUT/'plan.json'),str(DATA/'manifest.json'):file_hash(DATA/'manifest.json')},
            every_accepted_original_function_passes_all_public_examples_twice=True,
            independent_source_and_runtime_admission_training_and_generated_quality_still_required=True,
            seconds=time.monotonic()-started,optimization_admitted=False,neural_updates=0,goal_achieved=False))
        print(json.dumps(dict(complete=str(OUT/'complete.json'),actual_verified_roles=dict(roles),optimization_admitted=False)),flush=True)
    except BaseException as error:
        atomic_json(OUT/'failure.json',dict(error=repr(error),traceback=traceback.format_exc(),all_original_jobs_and_partial_source_data_preserved=True,
            optimization_admitted=False,goal_achieved=False))
        raise


if __name__ == '__main__':main()
