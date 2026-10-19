"""Rebuild all600 reserved source records and run all1816original controls."""
from collections import Counter
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

import numpy as np
from tokenizers import Tokenizer

from slm_lab.MBPP_original_population_v1 import BUDGET,FORMAT,prompt,role
from slm_lab.codenet_checkpoint_v1 import FileBindings
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.local_source_closure_v2 import python_sources
from slm_lab.mbpp_original_function_cases_v1 import cases,execute,implementation_sources
from slm_lab.source_jsonl_records_v1 import records

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/MBPP-all-reserved-original-v1-preparation'
DATA=ROOT/'data/MBPP-all-reserved-original-evaluation-v1'


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or OUT.exists() or DATA.exists():raise ValueError('Fresh CPU-only whole-original MBPP preparation required')
    tests=ROOT/'reports/mbpp-original-function-cases-v1-tests.log'
    if '3 passed' not in tests.read_text():raise ValueError('Actual class/global/container/complex/gold-separation/challenge/resource controls required')
    cache=ROOT/'data/verified-code-source-cache-v1';meta=json.loads((cache/'sources.json').read_text())
    source=next(r for r in meta['files'] if r['path']=='mbpp.jsonl')
    if file_hash(cache/'mbpp.jsonl')!=source['sha256']:raise ValueError('Pinned complete original publisher source must remain exact')
    reserved_path=ROOT/'data/verified-code-expanded-v1/reserved-mbpp.jsonl'
    expanded_manifest=reserved_path.with_name('manifest.json');expanded=json.loads(expanded_manifest.read_text())
    if file_hash(reserved_path)!=expanded['files_sha256'][reserved_path.name]:raise ValueError('Entire original held reservation must remain exact')
    original={r['task_id']:r for r in records(cache/'mbpp.jsonl')};rows=list(records(reserved_path))
    if [r['task_id'] for r in rows]!=list(range(1,601)) or any(r!=original[r['task_id']] for r in rows):
        raise ValueError('All600unmodified original publisher problems must reconstruct including all code/setup/tests/challenges')
    if sum(len(cases(r)) for r in rows)!=1816:raise ValueError('Every original test/challenge equality must be supported without filtering')
    token_path=ROOT/'data/tokenizer-corpus/tokenizer.json';tokenizer=Tokenizer.from_file(str(token_path))
    phase=json.loads((ROOT/'reports/phase-joint-v3/plan.json').read_text())
    if file_hash(token_path)!=phase['tokenizer_sha256']:raise ValueError('Exact own main/future-model tokenizer required')
    registry_path=ROOT/'data/full-held-question-pool-v2/question-registry.jsonl'
    reserved_registry={r['id']:r for r in records(registry_path) if r['benchmark']=='reserved-MBPP'}
    if len(reserved_registry)!=600 or any(reserved_registry[str(r['task_id'])]['question']!=r['text'] for r in rows):
        raise ValueError('The same whole600held question population must already be reserved and audited')
    source_files={str(p):file_hash(p) for p in [tests,cache/'sources.json',cache/'mbpp.jsonl',cache/'README.md',cache/'repository-README.md',
        expanded_manifest,reserved_path,token_path,registry_path,
        ROOT/'reports/full-held-question-audit-v2/complete.json',ROOT/'tests/test_mbpp_original_function_cases_v1.py',
        *python_sources(ROOT,[Path(__file__),ROOT/'tests/test_mbpp_original_function_cases_v1.py'])]}
    source_files.update(implementation_sources());guard=FileBindings(source_files);OUT.mkdir();DATA.mkdir();started=time.monotonic()
    owner=dict(pid=os.getpid(),ticks=Path('/proc/self/stat').read_text().split(') ',1)[1].split()[19])
    atomic_json(OUT/'plan.json',dict(at=datetime.now(timezone.utc).isoformat(),owner=owner,source_files_sha256=source_files,
        original_reserved_problem_ids=[1,600],original_problems=600,original_test_challenge_cases=1816,
        publisher_test_problems=500,publisher_validation_problems=90,publisher_prompting_problems=10,
        every_original_reference_control_runs_without_rewriting_or_eligibility_exclusion=True,
        every_original_expected_RHS_stays_outside_candidate_process_and_filesystem=True,
        models_receive_only_original_task_and_public_signatures_no_test_values_or_code_body=True,
        model_protocol='Zero-shot native instruction; greedy cached512actions; standard-library CPython3.11 OS confinement',
        source_control_runtime_seconds=4.,reference_qualifications_retained_and_not_used_to_drop_model_questions=True,
        CPU_workers=1,automatic_retry=False,models_constructed=0,model_updates=0,goal_achieved=False))
    shutil.copy2(reserved_path,DATA/'original-reserved-rows.jsonl')
    shutil.copy2(cache/'README.md',DATA/'publisher-README.md');shutil.copy2(cache/'repository-README.md',DATA/'publisher-repository-README.md')
    counts=Counter();controls=[];input_rows=[]
    try:
        for row in rows:
            guard.check();native_prompt=prompt(row);ids=[1]+tokenizer.encode(native_prompt).ids
            if (len(ids)+BUDGET>4096 or any(v<4 or v>=32768 for v in ids[1:])):
                raise ValueError('Every full original native prompt plus unchanged512generation must fit without clipping')
            original_cases=cases(row);bucket=role(row['task_id']);control=execute(row['code'],row,seconds=4.)
            if control['original_cases']!=len(original_cases):raise ValueError('Every original case denominator must survive actual execution')
            counts[bucket+':problems']+=1;counts[bucket+':reference_full_case_passes']+=control['functional_passed']
            counts['original_cases']+=len(original_cases);counts['actual_observed_cases']+=control['actual_observed_cases']
            counts['actual_passed_cases']+=control['passed_cases'];counts['reference_fully_qualified_problems']+=control['functional_passed']
            counts['reference_syntax_valid_problems']+=control['syntax_valid'];counts['complete_native_prompt_positions']+=len(ids)
            control['publisher_role']=bucket;controls.append(control)
            input_rows.append(dict(original_task_id=row['task_id'],publisher_role=bucket,original_prompt=native_prompt,
                native_input_ids=ids,native_input_ids_sha256=hashlib.sha256(np.asarray(ids,dtype='<u2').tobytes()).hexdigest(),
                prompt_tokens=len(ids),generation_budget=BUDGET,original_cases=len(original_cases),
                original_reference_full_case_pass=control['functional_passed'],optimization_forbidden=True))
            atomic_json(OUT/f'original-source-control-{row["task_id"]:04d}.json',control)
            if row['task_id']%25==0:
                atomic_json(OUT/'progress.json',dict(owner=owner,actual_original_problems=row['task_id'],original_problems=600,
                    actual_counts=dict(counts),elapsed_seconds=time.monotonic()-started,models_constructed=0,model_updates=0,goal_achieved=False))
        if (counts['publisher_test:problems'],counts['publisher_validation:problems'],counts['publisher_prompting:problems'],counts['original_cases'])!=(500,90,10,1816):
            raise ValueError('Every original publisher split and all1816original case positions required')
        guard.check(full=True)
        atomic_json(DATA/'all-original-reference-controls.json',controls)
        with (DATA/'original-native-inputs.jsonl').open('x') as stream:
            for row in input_rows:stream.write(json.dumps(row,ensure_ascii=False)+'\n')
        atomic_json(DATA/'manifest.json',dict(format=FORMAT,at=datetime.now(timezone.utc).isoformat(),original_problems=600,
            original_population_sha256=hashlib.sha256(json.dumps(rows,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest(),
            original_reserved_source_sha256=file_hash(reserved_path),original_publisher_source=source,publisher_revision=meta['revision'],
            tokenizer_sha256=file_hash(token_path),original_native_generation_budget=BUDGET,actual_counts=dict(counts),
            all_original_fields_reference_source_tests_challenges_and_setup_retained=True,
            no_reference_eligibility_exclusions_or_source_repairs=True,optimization_forbidden=True,
            not_official_few_shot_or_sanitized_MBPP_protocol=True,
            source_ancestry_and_semantic_independence_not_certified=True,
            files_sha256={name:file_hash(DATA/name) for name in ['original-reserved-rows.jsonl','original-native-inputs.jsonl',
                'all-original-reference-controls.json','publisher-README.md','publisher-repository-README.md']}))
        atomic_json(OUT/'complete.json',dict(at=datetime.now(timezone.utc).isoformat(),complete=True,
            every600original_source_and_every1816case_reconstructed_without_exclusion=True,
            all600actual_original_reference_programs_executed=True,all600native_prompts_and_full512budgets_fit=True,
            actual_counts=dict(counts),actual_CPython_version=sys.version,
            model_questions_never_filtered_by_reference_qualification=True,
            every_expected_RHS_kept_in_trusted_parent_and_every_actual_LHS_return_compared=True,
            source_files_sha256={str(p):file_hash(p) for p in [OUT/'plan.json',DATA/'manifest.json',DATA/'all-original-reference-controls.json',DATA/'original-native-inputs.jsonl']},
            models_constructed=0,model_updates=0,goal_achieved=False))
        print(json.dumps(dict(complete=str(OUT/'complete.json'),actual_counts=dict(counts),models_constructed=0)),flush=True)
    except BaseException as error:
        atomic_json(OUT/'failure.json',dict(error=repr(error),traceback=traceback.format_exc(),models_constructed=0,model_updates=0,goal_achieved=False));raise


if __name__=='__main__':main()
