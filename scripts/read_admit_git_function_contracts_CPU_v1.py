"""Independently rebuild every Git/source/native disposition and repeat execution.

Only train-role original whole functions are admitted for CE. Public source
docstrings contain the example answers; this is not hidden-test accuracy or
human preference data. The producer selection/building modules are not imported.
"""
from collections import Counter,deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from datetime import datetime,timezone
import ast
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

from slm_lab.codenet_checkpoint_v1 import FileBindings
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'reports/direct-algorithm-function-contracts-CPU-v1'
DATA=ROOT/'data/direct-algorithm-function-contracts-v1'
OUT=ROOT/'reports/direct-algorithm-function-contracts-v1-readback-admission'
PACK=ROOT/'data/direct-algorithm-function-contracts-admitted-v1'


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or OUT.exists() or PACK.exists():
        raise ValueError('Fresh GPU-hidden complete independent Git-function source/admission reader required')
    import numpy as np
    import torch
    from tokenizers import Tokenizer
    from slm_lab.codenet_token_pack_v1 import CodePackWriter,CompleteCodePack
    from slm_lab.exact_waiting_process_retirement_v1 import terminal
    from slm_lab.git_function_readback_v1 import original_files,qualify_file,qualify_function,verify_candidate,observe,observations_pass
    from slm_lab.held_text_overlap_v1 import HeldTextOverlap
    from slm_lab.local_source_closure_v2 import python_sources
    torch.set_num_threads(1)
    plan_path=SOURCE/'plan.json';producer= json.loads(plan_path.read_text());done=json.loads((SOURCE/'complete.json').read_text())
    if not terminal(producer['owner']) or not done['complete'] or (SOURCE/'failure.json').exists():
        raise ValueError('The exact original whole repository collector must finish before admission')
    if producer['repository']!='TheAlgorithms/Python' or producer['revision']!='25c712f2692449e91a57748d328ef34fccf25654':
        raise ValueError('Only the complete declared pinned original repository is eligible')
    tests=ROOT/'reports/git-function-readback-v1-final2-tests.log'
    if '4 passed' not in tests.read_text():raise ValueError('Actual independent source/native/partition/raw-outcome controls required')
    manifest_path=DATA/'manifest.json';manifest=json.loads(manifest_path.read_text())
    files={**producer['source_files_sha256'],**done['source_files_sha256'],**manifest['source_files_sha256']}
    header=ROOT/'checkpoints/phase-joint-v3/trained/model.pt'
    files.update({str(p):file_hash(p) for p in [tests,plan_path,SOURCE/'complete.json',manifest_path,header,
        *python_sources(ROOT,[Path(__file__),ROOT/'tests/test_git_function_readback_v1.py'])]})
    guard=FileBindings(files);tokenizer_path=ROOT/'data/tokenizer-corpus/tokenizer.json'
    tokenizer=Tokenizer.from_file(str(tokenizer_path))
    registry=json.loads((ROOT/'reports/CodeContests-native-held-text-v2/unchanged-held-text-registry.json').read_text())
    if len(registry)!=32953:raise ValueError('All original unchanged declared held stems must remain screened')
    overlap=HeldTextOverlap(registry);tree=json.loads((DATA/'original-git-tree.json').read_text())
    if tree['sha']!=producer['git_tree_sha1']:raise ValueError('The complete original pinned Git tree identity must remain exact')
    originals=original_files(DATA/'original-source.tar.gz',tree,producer['revision'])
    if len(originals)!=done['actual_original_regular_files']:raise ValueError('Every original ordinary Git file must reconstruct')
    git_proofs=json.loads((DATA/'complete-original-Git-blob-proofs.json').read_text())
    expected_proofs={p:dict(path=p,original_git_blob_sha1=e['sha'],bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
        for p,(raw,e) in originals.items()}
    if len(git_proofs)!=len(expected_proofs) or {p['path']:p for p in git_proofs}!=expected_proofs:
        raise ValueError('Every original full Git-blob observation must independently rederive')
    license_raw=originals['LICENSE.md'][0];license_sha=hashlib.sha256(license_raw).hexdigest()
    if (DATA/'original-LICENSE.md').read_bytes()!=license_raw or (DATA/'original-README.md').read_bytes()!=originals['README.md'][0]:
        raise ValueError('Original whole repository notice and README bytes must remain exact')
    source_rows={}
    with gzip.open(DATA/'original-qualified-Python-sources.jsonl.gz','rt') as stream:
        for line in stream:
            row=json.loads(line);path=row['source_git_path']
            if path in source_rows:raise ValueError('Every original qualified source path must occur once')
            source_rows[path]=row
    ledger={}
    with gzip.open(DATA/'every-original-Python-file-and-function-disposition.jsonl.gz','rt') as stream:
        for line in stream:
            row=json.loads(line);key=(row['kind'],row['original_git_path'],row.get('original_lineno'))
            if key in ledger:raise ValueError('Every original file/function disposition must occur once')
            ledger[key]=row
    observations=[]
    with gzip.open(DATA/'complete-function-candidate-observations.jsonl.gz','rt') as stream:
        observations=[json.loads(line) for line in stream]
    if len(observations)!=done['actual_counts']['actual_complete_static_function_candidates']:
        raise ValueError('Every static candidate, including qualification failures, must remain present')
    observed=iter(observations);counts=Counter();seen=set();work=[];source_count=0
    def check_ledger(row):
        key=(row['kind'],row['original_git_path'],row.get('original_lineno'))
        if ledger.pop(key,None)!=row:raise ValueError('Every original source/function status and native candidate identity must independently reconstruct')
    for path,(raw,entry) in sorted(originals.items()):
        if not path.endswith('.py'):continue
        source_count+=1
        original,reason=qualify_file(path,raw,entry,producer['repository'],producer['revision'],license_sha,tokenizer,overlap)
        status=reason or 'whole_original_git_Python_source_ready';counts[status]+=1
        check_ledger(dict(kind='file',original_git_path=path,disposition=status))
        if original is None:continue
        if source_rows.pop(path,None)!=original:raise ValueError('Every original source byte/metadata/notice/derived role must rebuild')
        for node in ast.parse(original['original_public_text']).body:
            if not isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)):continue
            candidate,rejection=qualify_function(original,node,tokenizer,overlap)
            if candidate is not None and candidate['original_function_program_sha256'] in seen:
                candidate=None;rejection='exact_whole_function_duplicate_in_this_repository_candidate'
            if candidate is None:
                counts[rejection]+=1
                check_ledger(dict(kind='function',original_git_path=path,original_lineno=node.lineno,candidate_id=None,disposition=rejection));continue
            seen.add(candidate['original_function_program_sha256']);counts['actual_complete_static_function_candidates']+=1
            retained=next(observed,None)
            if retained is None:raise ValueError('Every original qualifying whole function needs its actual retained observations')
            verify_candidate(retained['candidate'],candidate)
            passed=observations_pass(candidate,retained['actual_fresh_observations'])
            if retained['verification_passed']!=passed or retained['optimization_admitted'] is not False:
                raise ValueError('Original complete public-case outcome and candidate-only scope must reconstruct')
            status='every_original_public_example_passed_twice' if passed else 'public_example_or_context_or_repeatability_qualification'
            counts[status]+=1
            check_ledger(dict(kind='function',original_git_path=path,original_lineno=node.lineno,candidate_id=candidate['id'],disposition=status))
            work.append((candidate,retained,original))
    if (source_count!=done['actual_original_Python_files'] or ledger or source_rows or next(observed,None) is not None
            or dict(counts)!=done['actual_counts']):
        raise ValueError('Every original file/function/candidate population and exclusion count must independently reconcile')
    OUT.mkdir();PACK.mkdir();started=time.monotonic();owner=dict(pid=os.getpid(),ticks=Path('/proc/self/stat').read_text().split(') ',1)[1].split()[19])
    atomic_json(OUT/'plan.json',dict(at=datetime.now(timezone.utc).isoformat(),owner=owner,source_files_sha256=files,
        producer_owner=producer['owner'],actual_original_Git_files=len(originals),actual_original_Python_files=source_count,
        every_static_candidate_and_source_exclusion_rebuilt=True,all906candidates_require_two_new_whole_example_runs=True,
        producer_selection_and_building_modules_not_imported=True,expected_answers_visible_in_original_docstrings=True,
        optimization_admitted=False,new_neural_updates=0,goal_achieved=False))
    shutil.copyfile(tokenizer_path,PACK/'tokenizer.json')
    (PACK/'DATASET-LICENSE.txt').write_bytes(license_raw);(PACK/'publisher-README.md').write_bytes(originals['README.md'][0])
    raw_model=torch.load(header,map_location='cpu',weights_only=True,mmap=True)
    atomic_json(PACK/'model-config.json',raw_model['model']['config']);del raw_model
    writer=CodePackWriter(PACK,maximum_bytes=256*1024**2,minimum_free_gib=40)
    expected_native=[];roles=Counter();fresh_count=0
    try:
        with ThreadPoolExecutor(max_workers=2) as pool,(OUT/'all906-independent-fresh-example-observations.jsonl').open('x') as output:
            pending=deque()
            def fresh(candidate):return [observe(candidate),observe(candidate)]
            def finish():
                nonlocal fresh_count
                candidate,retained,original,future=pending.popleft();actual=future.result()
                passed=observations_pass(candidate,actual)
                if passed!=retained['verification_passed']:raise ValueError('Fresh independent public-example qualification differs from the completed producer')
                if passed and actual[0]['raw_results']!=retained['actual_fresh_observations'][0]['raw_results']:
                    raise ValueError('Every originally accepted function must reproduce its original raw outputs and entry calls')
                output.write(json.dumps(dict(id=candidate['id'],verification_passed=passed,actual_fresh_observations=actual),ensure_ascii=False)+'\n')
                fresh_count+=1
                if passed:
                    role='train' if candidate['role']=='train' else 'validation'
                    slim={k:v for k,v in candidate.items() if k not in ['whole_native_ids','whole_native_labels']}
                    row=dict(id=candidate['id'],group_id='public-file:'+original['original_UTF8_sha256'],split=role,
                        publisher_split=None,original_publisher_partition=None,language='Python',prompt=candidate['prompt'],response=candidate['response'],
                        source_candidate=slim,original_public_URL=original['original_public_URL'],rights=original['rights'],
                        complete_original_function_response_unchanged=True,original_published_example_count=candidate['whole_original_example_count'],
                        actual_fresh_example_repetitions_including_producer=4,public_example_answers_visible_in_original_docstring=True,
                        no_publisher_train_or_test_partition_exists=True,finite_examples_not_general_correctness_or_efficiency=True,
                        human_preferences=False,external_neural_teacher_used=False,execution_reward_training_admitted=False,
                        dataset_admitted_to_training=False)
                    writer.add(row,candidate['whole_native_ids'],candidate['whole_native_labels'])
                    expected_native.append((role,candidate['whole_native_ids'],candidate['whole_native_labels']))
                    roles[candidate['role']+'_functions']+=1;roles[candidate['role']+'_response_EOS_targets']+=candidate['response_EOS_targets']
                    roles[candidate['role']+'_native_context_tokens']+=candidate['complete_native_context_tokens']
                    roles[candidate['role']+'_original_public_examples']+=candidate['whole_original_example_count']
                if fresh_count%25==0:
                    output.flush();os.fsync(output.fileno());guard.check()
                    atomic_json(OUT/'progress.json',dict(actual_static_candidates_freshly_reexecuted=fresh_count,planned_static_candidates=len(work),
                        actual_independently_verified_roles=dict(roles),seconds=time.monotonic()-started,neural_updates=0,goal_achieved=False))
            for candidate,retained,original in work:
                pending.append((candidate,retained,original,pool.submit(fresh,candidate)))
                if len(pending)>=4:finish()
            while pending:finish()
            output.flush();os.fsync(output.fileno())
        if dict(roles)!=done['actual_verified_roles'] or fresh_count!=len(work):raise ValueError('All original candidate and admitted-role counts must reconcile after fresh execution')
        if any(n in sys.modules for n in ['slm_lab.public_function_contracts_v1','slm_lab.public_git_function_sources_v1']):
            raise ValueError('The independent reader must not execute producer selection/building code')
        guard.check(full=True)
        packed=writer.finish(dict(original_repository=producer['repository'],original_revision=producer['revision'],
            source_collector_complete_sha256=file_hash(SOURCE/'complete.json'),all_original_files_and_source_exclusions_rebuilt=True,
            all906static_candidates_independently_reexecuted_twice=True,all_accepted_public_examples_pass_four_fresh_runs=True,
            original_example_answers_visible_in_original_docstrings=True,no_neural_teacher_or_generated_target=True,
            no_publisher_train_partition_exists=True,declared_held_text_screens_not_semantic_or_all_history_independence=True))
        for role in ['train','validation']:
            pack=CompleteCodePack(PACK,role,'audit')
            expected=[(ids,labels) for r,ids,labels in expected_native if r==role]
            if len(pack)!=len(expected):raise ValueError('Exact sealed native role population required')
            for i,(ids,labels) in enumerate(expected):
                if pack.sequence(i)!=(ids,labels):raise ValueError('Every admitted complete source native byte, mask and EOS must survive durable pack readback')
        readback=OUT/'readback-complete.json'
        atomic_json(readback,dict(at=datetime.now(timezone.utc).isoformat(),complete=True,actual_original_regular_Git_files=len(originals),
            actual_original_Python_files=source_count,actual_static_candidates_independently_reexecuted=fresh_count,
            actual_verified_roles=dict(roles),all_original_source_native_roles_and_dispositions_rebuilt=True,
            all_accepted_original_examples_reproduced_in_two_additional_fresh_interpreters=True,
            every_durable_native_sequence_and_mask_rebuilt=True,producer_selection_and_building_modules_not_imported=True,
            source_files_sha256={**files,str(PACK/'manifest.json'):file_hash(PACK/'manifest.json'),
                str(OUT/'all906-independent-fresh-example-observations.jsonl'):file_hash(OUT/'all906-independent-fresh-example-observations.jsonl')},
            optimization_admission_scope='Source-train whole-function CE only; no held-role, human preference or execution-RL admission',
            new_neural_updates=0,goal_achieved=False))
        atomic_json(PACK/'admission.json',dict(admitted_to_training=True,dataset_manifest_sha256=file_hash(PACK/'manifest.json'),
            independent_readback_sha256=file_hash(readback),independent_readback=str(readback),actual_train_functions=826,actual_valid_functions=15,
            only_train_role_eligible_for_optimization=True,human_preference_training_admitted=False,execution_reward_training_admitted=False,
            source_files_sha256={str(readback):file_hash(readback)},new_neural_updates=0,goal_achieved=False))
        atomic_json(OUT/'complete.json',dict(at=datetime.now(timezone.utc).isoformat(),complete=True,dataset_admitted_to_function_CE_training=True,
            actual_verified_roles=dict(roles),source_files_sha256={str(readback):file_hash(readback),str(PACK/'admission.json'):file_hash(PACK/'admission.json'),
                str(PACK/'manifest.json'):file_hash(PACK/'manifest.json'),str(OUT/'plan.json'):file_hash(OUT/'plan.json')},
            source_example_verification_not_generated_model_accuracy=True,neural_training_updates=0,goal_achieved=False))
        print(json.dumps(dict(complete=str(OUT/'complete.json'),actual_verified_roles=dict(roles),data_admitted=True,neural_updates=0)),flush=True)
    except BaseException as error:
        atomic_json(OUT/'failure.json',dict(error=repr(error),traceback=traceback.format_exc(),partial_new_data_and_all_original_jobs_preserved=True,
            optimization_admitted=False,goal_achieved=False));raise


if __name__=='__main__':main()
