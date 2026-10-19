"""Re-execute all retained joint-model public Python and local C++ outputs on CPU."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path

from slm_lab.code_quality import execute,instruction_code
from slm_lab.code_tasks import TASKS,request
from slm_lab.corpus import file_hash
from slm_lab.cpp_evaluation import extract
from slm_lab.diverse_images import atomic_json
from slm_lab.evaluation_provenance import capture
from slm_lab.native_code import execute as native_execute

ROOT=Path(__file__).resolve().parents[1]


def main():
    paths={name:ROOT/'reports/extended-curriculum-v1'/(name+'.json') for name in ['public-code','cpp-code']}
    reports={name:json.loads(path.read_text()) for name,path in paths.items()}
    source=ROOT/'data/public-code-assessment-v1/questions.jsonl'
    manifest_path=source.parent/'manifest.json';manifest=json.loads(manifest_path.read_text())
    base=ROOT/'checkpoints/extended-curriculum-v1/joint/model.pt';tokenizer=base.parent/'tokenizer.json'
    base_sha=file_hash(base);token_sha=file_hash(tokenizer)
    for report in reports.values():
        if report['checkpoint_sha256']!=base_sha or report['tokenizer_sha256']!=token_sha or report['evaluation_provenance']!=capture():
            raise ValueError('Actual completed own base/tokenizer/evaluator provenance differs')
    rows=list(map(json.loads,source.read_text().splitlines()));by_id={r['id']:r for r in rows}
    public=reports['public-code'];cpp=reports['cpp-code']
    if (file_hash(source)!=manifest['questions_sha256'] or public['data_sha256']!=file_hash(source)
            or len(by_id)!=175 or len(public['results'])!=175 or {r['id'] for r in public['results']}!=set(by_id)
            or manifest['official_sanitized_test_problems']!=257 or manifest['excluded_problems']!=82):
        raise ValueError('Complete original supported public benchmark/questions required')
    def python_result(result):
        row=by_id[result['id']]
        if result['task_id']!=row['task_id'] or instruction_code(result['raw_response'])!=result['code']:
            raise ValueError('Every retained original raw response/extracted Python program must match')
        actual=execute(dict(language='Python',code=result['code'],function=row['function'],cases=row['cases'],stress=[]))
        for key in ['syntax_valid','functional_passed','passed_cases','total_cases']:
            if actual.get(key)!=result['execution'].get(key):
                raise ValueError('Independent original candidate execution differs: '+result['id']+' '+key)
        return dict(id=result['id'],task_id=row['task_id'],execution=actual)
    with ThreadPoolExecutor(max_workers=2) as executor:python=list(executor.map(python_result,public['results']))
    tasks={t[0]:t for t in TASKS}
    if len(cpp['results'])!=12 or {r['task'] for r in cpp['results']}!=set(tasks):
        raise ValueError('All original twelve local C++ tasks required')
    native=[]
    for row in cpp['results']:
        if extract(row['raw_response'])!=row['code']:raise ValueError('Retained raw C++ extraction differs')
        actual=native_execute(request(tasks[row['task']],'C++',row['code']))
        for key in ['syntax_valid','functional_passed','passed_cases','total_cases']:
            if actual.get(key)!=row['candidate'].get(key):raise ValueError('Original C++ candidate result differs: '+row['task']+' '+key)
        native.append(dict(task=row['task'],execution=actual))
    counts=dict(public_Python=dict(tasks=175,syntax=sum(r['execution']['syntax_valid'] for r in python),
        published_case_correct=sum(r['execution']['functional_passed'] for r in python),
        excluded_original_problems=82,official_source_test_problems=257),
        local_CPP=dict(tasks=12,syntax=sum(r['execution']['syntax_valid'] for r in native),
            functional=sum(r['execution']['functional_passed'] for r in native)))
    if (public['syntax_rate_on_supported_subset']!=counts['public_Python']['syntax']/175
            or public['pass_at_1_on_supported_subset']!=counts['public_Python']['published_case_correct']/175
            or cpp['syntax_pass_rate']!=counts['local_CPP']['syntax']/12
            or cpp['functional_pass_at_1']!=counts['local_CPP']['functional']/12):
        raise ValueError('Re-executed counts do not match complete original report aggregates')
    if counts['local_CPP']['syntax']!=0 or cpp['efficiency_stress_pass_rate']!=0:
        raise ValueError('Changed C++ outcome requires actual renewed reference-relative stress measurement')
    bound=list(paths.values())+[source,manifest_path,base,tokenizer,Path(__file__).resolve()]
    target=ROOT/'reports/completed-joint-public-code-readback-v1.json'
    atomic_json(target,dict(observed_at=datetime.now(timezone.utc).isoformat(),
        base_checkpoint_sha256=base_sha,tokenizer_sha256=token_sha,counts=counts,
        actual_CPU_reexecution_of_every_retained_candidate=True,Python=python,CPP=native,
        original_report_aggregates_match=True,model_regenerated=False,training_updates=0,
        limitations=public['limitations']+cpp['limitations']+[
            'C++ candidates all fail syntax, so their efficiency score is zero; original reference stress timings were not rerun.',
            'This replays finite existing development/public cases, not independent final held-out capability or universal correctness.'],
        high_end_capability_established=False,goal_achieved=False,evaluation_provenance=capture(),
        files_sha256={str(p):file_hash(p) for p in bound}))
    print(json.dumps(dict(report=str(target),counts=counts),indent=2))


if __name__=='__main__':main()
