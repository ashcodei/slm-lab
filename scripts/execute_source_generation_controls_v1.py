"""Execute sealed raw C++/Python source controls on original published samples."""
import argparse
import base64
from collections import Counter
from datetime import datetime,timezone
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

from check_codenet_standalone_references_v1 import original_python3
from slm_lab.codenet_checkpoint_v1 import FileBindings
from slm_lab.codenet_sample_oracles_v1 import SUPPORTED,check_sample
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.native_code import toolchain
from slm_lab.standalone_numpy_v1 import numpy_runtime_paths,prepare_numpy
from slm_lab.standalone_program_v1 import compile_cpp_program,run_bounded

ROOT=Path(__file__).resolve().parents[1]
SOURCES={'four_source':'four-source-code-learning-probe-v1',
         'repeated_source':'code-residual-repeated-source-pilot-v1'}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--experiment',choices=SOURCES,required=True)
    args=parser.parse_args();source=ROOT/'reports'/SOURCES[args.experiment]
    reports=ROOT/'reports'/f'{SOURCES[args.experiment]}-published-sample-controls'
    if reports.exists():raise ValueError('Preserve earlier control executions')
    if shutil.disk_usage(ROOT).free < 16*1024**3+32*1024**2:
        raise ValueError('Bound32MiB diagnostic report above16GiB remaining space required')
    result_path=source/'results.json';result=json.loads(result_path.read_text())
    generation_path=source/'generations.json'
    if file_hash(generation_path)!=result['generation_report_sha256']:
        raise ValueError('All raw outputs must be sealed before execution')
    generations=json.loads(generation_path.read_text())['records']
    if args.experiment=='four_source':generations=[g for g in generations if g['step']==256]
    assert len(generations)==len({g['id'] for g in generations})==8
    selected=[g for g in generations if g['language'] in {'C++','Python'}]
    assert len(selected)==4 and Counter(g['role'] for g in selected)=={'train':2,'validation':2}
    rows={}
    with gzip.open(ROOT/'data/codenet-durable-pack-v2/original-rows.jsonl.gz','rt') as stream:
        for line in stream:
            row=json.loads(line)
            if row['id'] in {g['id'] for g in selected}:rows[row['id']]=row
    assert len(rows)==4
    descriptions_path=ROOT/'reports/codenet-description-format-v1/records.jsonl'
    descriptions={r['problem_id']:r for r in map(json.loads,descriptions_path.open())}
    tools=toolchain()
    files=[Path(__file__),result_path,generation_path,descriptions_path,
        ROOT/'data/codenet-durable-pack-v2/original-rows.jsonl.gz',
        ROOT/'src/slm_lab/standalone_program_v1.py',ROOT/'src/slm_lab/standalone_numpy_v1.py',
        ROOT/'src/slm_lab/codenet_sample_oracles_v1.py',ROOT/'src/slm_lab/native_code.py',
        ROOT/'scripts/native_sandbox.py',ROOT/'scripts/python_program_sandbox_v1.py',
        ROOT/'scripts/check_codenet_standalone_references_v1.py',Path(tools['compiler_path']),
        ROOT/'.tools/native-toolchain.json']
    files+=sorted({p.resolve() for d in numpy_runtime_paths() for p in d.rglob('*') if p.is_file()})
    bound={str(p):file_hash(p) for p in files};guard=FileBindings(bound)
    reports.mkdir()
    atomic_json(reports/'plan.json',dict(created_at=datetime.now(timezone.utc).isoformat(),
        source_files_sha256=bound,selected_model_source_ids=[g['id'] for g in selected],
        selection='All four original C++/Python controls; final predeclared state only, no result selection',
        source_generation_sealed_before_execution=True,raw_code_repaired_or_extracted=False,
        unconfigured_original_D_and_Java_controls=4,run_seconds_per_sample=3,combined_output_bytes_per_sample=1024**2,
        original_primary_benchmark_or_scorer_modified=False,optimizer_updates=0,goal_achieved=False,
        scope='Published samples on source-only training/internal-validation controls. Not hidden cases, efficiency, primary benchmark scores or independent final generalization. Environment/compile failures remain qualified.'))
    records=[]
    for generated in selected:
        guard.check()
        if shutil.disk_usage(tempfile.gettempdir()).free < 12*1024**3+300*1024**2:
            raise ValueError('Preserve original temporary reserve/compiler workspace')
        row=rows[generated['id']];problem=descriptions[row['source_problem_id']]
        assert generated['role']==row['split'] and generated['language']==row['language']
        code=generated['raw_program_text'].encode()
        assert hashlib.sha256(code).hexdigest()==generated['raw_program_sha256']
        assert problem['recognized_samples_usable'] and not problem['sample_parse_issues'] and problem['samples']
        record=dict(id=row['id'],role=row['split'],language=row['language'],source_problem_id=row['source_problem_id'],
            kind='raw_model_generated_source_control',original_source_body_sha256=row['original_source_body_sha256'],
            generated_source_sha256=generated['raw_program_sha256'],samples=[])
        if not code or len(code)>32768 or b'\0' in code:
            record['qualification']='empty_or_unsupported_source_encoding'
        elif row['language']=='Python' and not original_python3(row['original_language']):
            record['qualification']='original_runtime_not_configured'
        else:
            with tempfile.TemporaryDirectory(prefix='slm-source-generated-control-') as work:
                if row['language']=='Python':constraints=prepare_numpy(code,work)
                else:
                    constraints,compilation=compile_cpp_program(code,work,tools)
                    record['compilation']={k:v for k,v in compilation.items() if k not in {'stdout','stderr'}}
                    record['compilation']['diagnostic_tail']=compilation['stderr'][-4000:].decode(errors='replace')
                if constraints is None:record['qualification']='compiler_or_environment_review_needed'
                else:
                    for number,pair in enumerate(problem['samples'],1):
                        observed=run_bounded(constraints,pair['input'].encode(),python=row['language']=='Python')
                        oracle=None
                        if observed['status']=='completed' and row['source_problem_id'] in SUPPORTED:
                            oracle=check_sample(row['source_problem_id'],pair['input'],pair['output'],observed['stdout'].decode(errors='replace'))
                        agrees=bool(observed['status']=='completed' and (oracle['passed'] if oracle is not None else observed['stdout'].split()==pair['output'].encode().split()))
                        record['samples'].append(dict(number=number,
                            **{k:v for k,v in observed.items() if k not in {'stdout','stderr'}},
                            stdout_base64=base64.b64encode(observed['stdout']).decode(),
                            stdout_sha256=hashlib.sha256(observed['stdout']).hexdigest(),
                            stderr_tail=observed['stderr'][-2000:].decode(errors='replace'),
                            source_specific_oracle=oracle,sample_agrees=agrees))
                    record['qualification']=('all_original_published_samples_agree' if all(s['sample_agrees'] for s in record['samples'])
                        else 'one_or_more_original_published_samples_disagree_or_runtime_failure')
        records.append(record);atomic_json(reports/'records.json',dict(records=records))
        print('generated_source_control',record['id'],record['qualification'],flush=True)
    guard.check(full=True)
    assert sum(p.stat().st_size for p in reports.iterdir() if p.is_file())<=32*1024**2
    summary={role:dict(programs=2,all_published_samples_agree=sum(r['qualification']=='all_original_published_samples_agree' for r in records if r['role']==role)) for role in ['train','validation']}
    atomic_json(reports/'results.json',dict(completed_at=datetime.now(timezone.utc).isoformat(),
        plan_sha256=file_hash(reports/'plan.json'),records_sha256=file_hash(reports/'records.json'),
        source_model_checkpoint_sha256=result['checkpoint_sha256'],records=4,summary=summary,
        unsupported_D_Java_controls_not_counted=4,primary_benchmark_scores_changed=False,
        model_updates=0,candidate_promoted=False,goal_achieved=False,
        scope='Four raw generated C++/Python source controls on eleven original published samples, not hidden tests or independent generalization. Original controls include training sources.'))
    print(json.dumps(summary))


if __name__=='__main__':main()
