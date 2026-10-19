"""A bounded exact-integer counterexample to synthetic-test-only admission.

This diagnoses one unchanged public teacher answer, not our neural model.
No corpus-wide error rate or training improvement is inferred.
"""
import ast
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

from slm_lab.public_code_instruction_screen_v1 import HARNESS, parse_whole, observe
from slm_lab.standalone_program_v1 import python_policy, run_bounded

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data/public-code-instruction-candidate-v1'
OUT=ROOT/'reports/public-code-instruction-precision-counterexample-CPU-v1'
POSITION=30


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,value):
    with path.open('x') as stream:stream.write(json.dumps(value,indent=2,ensure_ascii=True)+'\n')


def main():
    if (os.environ.get('CUDA_VISIBLE_DEVICES')!='' or os.nice(0)<19 or OUT.exists()
            or any(os.environ.get(k)!='1' for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'])
            or shutil.disk_usage(ROOT).free<40*1024**3):raise ValueError('Fresh bounded low-priority CPU observation required')
    collection=json.loads((ROOT/'reports/public-code-instruction-collection-CPU-v1/complete.json').read_text())
    original_path=DATA/'first1024-original-entries.jsonl'
    if sha(original_path)!=collection['artifact_sha256'][str(original_path.relative_to(ROOT))]:
        raise ValueError('Pinned whole original candidate rows required')
    with original_path.open() as stream:entries=list(map(json.loads,stream))
    entry=entries[POSITION];row=entry['original_entry'];parsed=parse_whole(row)
    if (entry['original_shard_position']!=POSITION or parsed['function']!='find_closest_pair_of_points'
            or '-10^9' not in row['input'] or '1000' not in row['input']):
        raise ValueError('The unchanged explicitly bounded closest-pair source is required')
    paths=[Path(__file__),original_path,HARNESS,ROOT/'src/slm_lab/public_code_instruction_screen_v1.py',
        ROOT/'src/slm_lab/standalone_program_v1.py',ROOT/'src/slm_lab/native_code.py',
        ROOT/'scripts/python_program_sandbox_v1.py',ROOT/'scripts/native_sandbox.py']
    files={str(p):sha(p) for p in paths}
    OUT.mkdir()
    write(OUT/'plan.json',dict(source_files_sha256=files,source_id=row['id'],original_shard_position=POSITION,
        exact_original_question=row['input'],exact_original_response=row['output'],
        original_published_tests=json.loads(row['unit_tests']),new_model_forwards=0,new_neural_updates=0,
        original_teacher_data_synthetic=True,training_admitted=False,goal_achieved=False))
    original=observe(parsed,seconds=3)
    if not original['all_published_tests_passed']:raise ValueError('Fresh actual original synthetic tests must first pass')
    # All three points satisfy the original integer-coordinate and size bounds.
    # Squared distance preserves Euclidean ordering using exact integer arithmetic.
    points=[(-10**9,0),(0,1),(10**9,1)]
    distances=[dict(indices=[i,j],squared_distance=(points[i][0]-points[j][0])**2+(points[i][1]-points[j][1])**2)
        for i in range(len(points)) for j in range(i+1,len(points))]
    best=min(distances,key=lambda r:r['squared_distance'])
    expected=tuple(points[i] for i in best['indices'])
    if (not 2<=len(points)<=1000 or not all(type(v) is int and -10**9<=v<=10**9 for p in points for v in p)
            or sum(r['squared_distance']==best['squared_distance'] for r in distances)!=1):
        raise ValueError('The fresh case must satisfy every original bound and have one strict optimum')
    with tempfile.TemporaryDirectory(prefix='slm-public-precision-v1-') as temporary:
        work=Path(temporary);(work/'program.py').write_text(parsed['program']);(work/'harness.py').write_bytes(HARNESS.read_bytes())
        settings=dict(program=str(work/'program.py'),function=parsed['function'],
            inputs=[dict(args_literal=repr([points]),kwargs_literal='{}')])
        (work/'settings.json').write_text(json.dumps(settings))
        policy=python_policy(work);policy['command'][-1]=str(work/'harness.py');policy['command'].append(str(work/'settings.json'))
        policy['memory_bytes']=512*1024**2;policy['cpu_seconds']=3
        actual=run_bounded(policy,b'',seconds=3,output_bytes=1024**2,python=True)
    if actual['status']!='completed':raise ValueError('A full actual confined return value is required')
    raw=json.loads(actual['stdout'])
    if raw['format']!='public-code-literal-values-v1' or len(raw['actual_value_literals'])!=1:
        raise ValueError('One complete actual value required')
    value=ast.literal_eval(raw['actual_value_literals'][0])
    if value==expected:raise ValueError('Expected precision counterexample was not observed; do not fabricate a failure')
    if any(sha(p)!=digest for p,digest in files.items()):raise ValueError('Every original byte must remain unchanged')
    proof=dict(complete=True,at=datetime.now(timezone.utc).isoformat(),source_files_sha256=files,
        source_id=row['id'],original_shard_position=POSITION,actual_all_original_synthetic_tests=original,
        independent_exact_integer_counterexample=dict(input_literal=repr(points),original_bounds_satisfied=True,
            all_pair_squared_distances=distances,unique_exact_expected_pair_literal=repr(expected),
            actual_unchanged_teacher_pair_literal=repr(value),actual_output_differs_from_unique_mathematical_optimum=True),
        explanation='Distinct legal squared distances 1000000000000000001 and 1000000000000000000 round to the same float square root; strict less-than retains the earlier, longer pair.',
        expected_answer_not_sent_to_confined_worker=True,original_program_not_repaired=True,
        independent_semantic_veto_for_this_source=True,all_other_sources_not_semantically_certified=True,
        dataset_wide_error_rate_not_estimated=True,synthetic_tests_not_sufficient_RL_or_admission_oracle=True,
        our_model_not_evaluated_or_updated=True,training_admitted=False,new_neural_updates=0,goal_achieved=False)
    write(OUT/'complete.json',proof)
    write(OUT/'semantic-veto.json',dict(source_id=row['id'],original_shard_position=POSITION,
        original_entry_sha256=hashlib.sha256(json.dumps(row,sort_keys=True,ensure_ascii=True).encode()).hexdigest(),
        evidence_path=str(OUT/'complete.json'),evidence_sha256=sha(OUT/'complete.json'),
        whole_original_response_forbidden_as_positive_without_independent_correction=True))
    print(json.dumps(dict(complete=True,original_tests_all_pass=True,exact_legal_counterexample_failed=True,training_admitted=False)))


if __name__=='__main__':main()
