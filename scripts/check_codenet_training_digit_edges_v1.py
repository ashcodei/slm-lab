"""Targeted original-training-source counterexamples; never benchmark labels.

The source's p00002 statement explicitly permits both inputs to be zero and
requires the decimal digit count of their sum. Historical Accepted annotations
alone do not certify this boundary. Only a completed wrong output can produce a
counterexample label; compiler/runtime failures remain separately qualified.
"""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

from slm_lab.codenet_checkpoint_v1 import FileBindings
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.native_code import toolchain
from slm_lab.standalone_program_v1 import compile_cpp_program, prepare_python, run_bounded

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/codenet-training-digit-edges-v1'


def main():
    if OUT.exists():raise ValueError('Preserve original targeted execution evidence; no implicit restart')
    if shutil.disk_usage(ROOT).free<26*1024**3+64*1024**2 or shutil.disk_usage(tempfile.gettempdir()).free<12*1024**3:
        raise ValueError('Fund targeted native work above persistent26GiB/temporary12GiB reserves')
    pack=ROOT/'data/codenet-durable-pack-v2'
    desc_path=ROOT/'reports/codenet-description-format-v1/records.jsonl'
    desc_summary=ROOT/'reports/codenet-description-format-v1/summary.json'
    if file_hash(desc_path)!=json.loads(desc_summary.read_text())['records_sha256']:
        raise ValueError('Original full publisher descriptions required')
    desc=next(r for r in map(json.loads,desc_path.open()) if r['problem_id']=='p00002')
    rows=[]
    with gzip.open(pack/'original-rows.jsonl.gz','rt') as stream:
        for line in stream:
            row=json.loads(line)
            if row['source_problem_id']=='p00002' and row['language'] in {'C++','Python'}:
                if row['split']!='train' or row['group_id']!='p00002' or row['original_html_sha256']!=desc['original_html_sha256']:
                    raise ValueError('Only the exact original sealed training problem is eligible')
                if hashlib.sha256(row['response'].encode()).hexdigest()!=row['original_source_body_sha256']:
                    raise ValueError('Every original full public programme byte must match')
                rows.append(row)
    if len(rows)!=8 or len({r['id'] for r in rows})!=8:
        raise ValueError('Every original4C++/4Python source variant must remain accounted for')
    # Directly reviewed original bounds/input/output contract; no external model.
    if ('0 ≤ a, b ≤ 1,000,000' not in desc['text'] or 'Print the number of digits of a + b for each data set.' not in desc['text']):
        raise ValueError('Original unambiguous digit-count contract changed')
    values=[(0,0),(0,1),(1,0),(4,5),(5,5),(49,50),(50,50),
        (499,500),(500,500),(4999,5000),(5000,5000),(49999,50000),
        (50000,50000),(499999,500000),(500000,500000),(1000000,1000000)]
    generated_input=''.join(f'{a} {b}\n' for a,b in values)
    generated_output=''.join(str(len(str(a+b)))+'\n' for a,b in values)
    # Check the derived oracle by an independent threshold calculation, including
    # original published samples. This is one problem's finite training cases.
    def threshold_digits(n):
        return 1+sum(n>=10**power for power in range(1,7))
    assert all(len(str(a+b))==threshold_digits(a+b) and 0<=a<=1000000 and 0<=b<=1000000 for a,b in values)
    if not desc['recognized_samples_usable'] or desc['sample_parse_issues'] or not desc['samples']:
        raise ValueError('Complete original recognized publisher sample required')
    for pair in desc['samples']:
        ints=list(map(int,pair['input'].split()))
        assert len(ints)%2==0
        assert [str(threshold_digits(a+b)) for a,b in zip(ints[::2],ints[1::2])]==pair['output'].split()
    cases=[dict(kind='original_publisher_sample',**pair) for pair in desc['samples']]
    cases.append(dict(kind='derived_source_boundaries',input=generated_input,output=generated_output))
    tools=toolchain()
    paths=[Path(__file__),pack/'manifest.json',pack/'admission.json',pack/'original-rows.jsonl.gz',
        desc_path,desc_summary,ROOT/'src/slm_lab/standalone_program_v1.py',ROOT/'src/slm_lab/native_code.py',
        ROOT/'scripts/native_sandbox.py',ROOT/'scripts/python_program_sandbox_v1.py',ROOT/'.tools/native-toolchain.json',
        Path(tools['compiler_path'])]
    files={str(p):file_hash(p) for p in paths};guard=FileBindings(files)
    OUT.mkdir()
    atomic_json(OUT/'plan.json',dict(created_at=datetime.now(timezone.utc).isoformat(),source_files_sha256=files,
        original_problem_id='p00002',original_source_role='train',source_rows=8,maximum_CPU_workers=2,
        source_contract=desc['text'],original_html_sha256=desc['original_html_sha256'],cases=cases,
        targeted_after_observing_suspected_zero_case_bug=True,representative_population_claim=False,
        compile_or_runtime_failures_are_not_wrong_code_labels=True,neural_updates=0,
        dataset_admitted_to_training=False,human_preference_labels=False,goal_achieved=False))

    def execute(row):
        guard.check()
        record={k:row[k] for k in ['id','source_problem_id','group_id','split','language','original_language',
            'original_archive_member','original_source_body_sha256','complete_input_ids_sha256','target_tokens']}
        record.update(original_judge_status=row['original_judge_status'],original_code_modified=False)
        if row['language']=='Python' and not row['original_language'].startswith('Python3'):
            return dict(**record,qualification='original_Python_runtime_ambiguous; no wrong-code label',cases=[])
        with tempfile.TemporaryDirectory(prefix='slm-training-digit-edge-') as work:
            if row['language']=='Python':constraints=prepare_python(row['response'].encode(),work)
            else:
                constraints,compilation=compile_cpp_program(row['response'].encode(),work,tools)
                record['compilation']={k:(v.decode(errors='replace') if isinstance(v,bytes) else v) for k,v in compilation.items()}
                if constraints is None:return dict(**record,qualification='compiler_environment_review; no wrong-code label',cases=[])
            observed=[]
            for case in cases:
                result=run_bounded(constraints,case['input'].encode(),python=row['language']=='Python')
                actual=result['stdout'].decode(errors='replace')
                passed=result['status']=='completed' and actual.split()==case['output'].split()
                result={k:(v.decode(errors='replace') if isinstance(v,bytes) else v) for k,v in result.items()}
                observed.append(dict(kind=case['kind'],input=case['input'],expected=case['output'],
                    passed=passed,finished_output_counterexample=result['status']=='completed' and not passed,**result))
        qualified=all(c['status']=='completed' for c in observed)
        original_passed=all(c['passed'] for c in observed if c['kind']=='original_publisher_sample')
        has_counterexample=any(c['finished_output_counterexample'] for c in observed)
        return dict(**record,cases=observed,all_original_publisher_samples_passed=original_passed,
            qualification=('completed_source_counterexample' if qualified and has_counterexample else
                'all_targeted_training_cases_pass' if qualified else 'runtime_environment_review; no wrong-code label'),
            verified_targeted_wrong_output=qualified and has_counterexample,
            unbounded_or_all_inputs_correctness_claim=False)
    with ThreadPoolExecutor(max_workers=2) as pool:records=list(pool.map(execute,rows))
    guard.check(full=True)
    atomic_json(OUT/'results.json',dict(completed_at=datetime.now(timezone.utc).isoformat(),
        plan_sha256=file_hash(OUT/'plan.json'),source_records=8,records=records,
        completed_source_counterexamples=sum(r.get('verified_targeted_wrong_output',False) for r in records),
        all_targeted_cases_pass=sum(r['qualification']=='all_targeted_training_cases_pass' for r in records),
        original_train_only=True,source_programmes_edited=False,neural_updates=0,
        dataset_admitted_to_training=False,human_preferences=False,goal_achieved=False,
        scope='Targeted original-program execution against one original training problem and16derived boundary lines. Completed exact-output failures establish these finite counterexamples only. Compiler/runtime qualifications never become wrong-code labels; finite passing cases do not certify whole programs. No held benchmark labels or model-generated outputs were used.'))
    print(json.dumps({r['id']:r['qualification'] for r in records},indent=2))


if __name__=='__main__':main()
