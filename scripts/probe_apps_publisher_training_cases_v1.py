"""Finite original-training-reference checks using the existing restricted worker.

One source-selected original per admitted APPS train problem. No publisher-test
or internal-validation programs, model generations, gradients or new labels.
This conservative function subset is not the official APPS benchmark runtime.
"""
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from urllib.request import Request, urlopen

from slm_lab.code_quality import execute
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'data/public-apps-original-inventory-v1/original-train'
VIEW = ROOT / 'data/public-apps-human-solution-view-v1'
OUTPUT = ROOT / 'reports/APPS-original-training-cases-v1'
PUBLISHER = 'https://raw.githubusercontent.com/hendrycks/apps/362aedc3c71cd7d9bd2fc96a6c80e11dbc38c7a5/eval/testing_util.py'


def contains_dict(value):
    pending=[value]
    while pending:
        item=pending.pop()
        if isinstance(item,dict):return True
        if isinstance(item,list):pending.extend(item)
    return False


def check(row):
    i=row['source_problem_id'];folder=SOURCE/i;path=folder/'input_output.json'
    base=dict(id=row['id'],problem=i,publisher_role='train',candidate_role='train',
        original_solution_index=row['original_solution_index'],method='minimum_target_tokens_then_original_ID_per_admitted_train_problem')
    if not path.exists():return dict(**base,status='source_missing_tests')
    raw=path.read_bytes();base['original_tests_sha256']=hashlib.sha256(raw).hexdigest()
    if len(raw)>1024**2:return dict(**base,status='full_case_payload_exceeds_bounded_probe')
    try:tests=json.loads(raw)
    except (ValueError,RecursionError):return dict(**base,status='original_numeric_or_JSON_parser_review')
    name=tests.get('fn_name');inputs=tests.get('inputs');outputs=tests.get('outputs')
    if not isinstance(name,str) or not name:return dict(**base,status='original_stdin_or_unspecified_interface')
    if (not isinstance(inputs,list) or not isinstance(outputs,list) or len(inputs)!=len(outputs)
            or not 2<=len(inputs)<=1000 or not all(isinstance(args,list) for args in inputs)):
        return dict(**base,status='original_complete_case_schema_or_size_review')
    if contains_dict(inputs) or contains_dict(outputs):return dict(**base,status='publisher_dictionary_key_conversion_requires_separate_backend')
    programs=json.loads((folder/'solutions.json').read_bytes());program=programs[row['original_solution_index']]
    base.update(original_program_sha256=hashlib.sha256(program.encode()).hexdigest(),original_function=name,complete_publisher_cases=len(inputs))
    cases=[dict(args=args,expected=expected) for args,expected in zip(inputs,outputs)]
    direct=execute(dict(language='Python',code=program,function=name,cases=cases,stress=[]))
    if direct.get('functional_passed') and direct.get('passed_cases')==len(cases):
        return dict(**base,status='complete_strict_original_cases_pass',comparison='original_literal_expected',worker=direct)
    # The publisher's function comparator accepts either the full expected
    # value or its first element when nonempty. A whole-suite uniform first-
    # element match is sufficient but less permissive than that per-case OR.
    # Never label other outcomes incorrect or synthesize expected values.
    alternate=None
    if all(isinstance(value,list) and value for value in outputs):
        alternate=execute(dict(language='Python',code=program,function=name,
            cases=[dict(args=args,expected=expected[0]) for args,expected in zip(inputs,outputs)],stress=[]))
        if alternate.get('functional_passed') and alternate.get('passed_cases')==len(cases):
            return dict(**base,status='complete_strict_original_cases_pass',comparison='publisher_first_expected_element_uniform_whole_suite',worker=alternate)
    return dict(**base,status='not_verified_under_conservative_restricted_backend',direct=direct,alternate=alternate,
        actual_original_program_incorrect_proven=False)


def main():
    if OUTPUT.exists():raise ValueError('Preserve earlier source probe; no implicit restart')
    OUTPUT.mkdir()
    with urlopen(Request(PUBLISHER,headers={'User-Agent':'slm-lab-public-research/1.0'}),timeout=45) as response:
        declaration=response.read(1024**2)
    (OUTPUT/'publisher-testing_util.py').write_bytes(declaration)
    groups=defaultdict(list)
    for row in map(json.loads,(VIEW/'train-records.jsonl').read_text().splitlines()):
        if row['split']!='train':raise ValueError('Original admitted train-only source required')
        groups[row['source_problem_id']].append(row)
    chosen=[min(rows,key=lambda r:(r['target_tokens'],r['id'])) for _,rows in sorted(groups.items())]
    atomic_json(OUTPUT/'plan.json',dict(created_at=datetime.now(timezone.utc).isoformat(),source_view_admission_sha256=file_hash(VIEW/'admission.json'),
        original_train_problems=len(chosen),source_selection=[r['id'] for r in chosen],maximum_concurrent_restricted_workers=8,
        publisher_comparator_url=PUBLISHER,publisher_comparator_sha256=hashlib.sha256(declaration).hexdigest(),
        held_sources_used=False,model_updates=0,labels_admitted_to_training=False,goal_achieved=False))
    counts=Counter();results=[]
    with ThreadPoolExecutor(max_workers=8) as pool,(OUTPUT/'observations.jsonl').open('x') as stream:
        futures=[pool.submit(check,row) for row in chosen]
        for future in as_completed(futures):
            record=future.result();results.append(record);counts[record['status']]+=1
            stream.write(json.dumps(record,separators=(',',':'))+'\n');stream.flush()
            if len(results)%100==0:print('original_APPS_train_problems_checked',len(results),dict(counts),flush=True)
    passing=[r for r in results if r['status']=='complete_strict_original_cases_pass']
    report=dict(completed_at=datetime.now(timezone.utc).isoformat(),source_selected_original_train_programs=len(chosen),
        outcomes=dict(counts),complete_supported_original_programs_passing=len(passing),
        original_publisher_case_exposures_passing=sum(r['complete_publisher_cases'] for r in passing),
        full_original_cases_not_clipped=True,publisher_function_comparison_sufficient_but_conservative=True,
        supported_subset_not_official_APPS_score=True,no_incorrect_labels_from_unsupported_or_unverified_outcomes=True,
        observations_sha256=file_hash(OUTPUT/'observations.jsonl'),plan_sha256=file_hash(OUTPUT/'plan.json'),
        model_generations=0,model_updates=0,new_preference_or_RLHF_labels_admitted=False,goal_achieved=False)
    atomic_json(OUTPUT/'summary.json',report);print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
