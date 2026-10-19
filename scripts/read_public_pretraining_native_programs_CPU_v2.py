"""Fresh all96native recovered program executions bound to actual actions.

Wait for the exact public-pretraining parent and every registered child to
finish. Preserve all raw reports, both frames/languages/endpoints and every
fresh verdict difference. This creates no model, generation or training data.
"""
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import time

from tokenizers import Tokenizer

from slm_lab.bounded_disk_reserve_wait_v1 import wait_for_reserve
from slm_lab.code_quality import execute,efficiency
from slm_lab.code_tasks import TASKS,request
from slm_lab.decoded_native_code_reexecution_v1 import reconstruct
from slm_lab.local_source_closure_v2 import python_sources
from slm_lab.native_code_action_readback_v1 import validate
from slm_lab.native_code_shards_v1 import reconcile,positions
from slm_lab.ordinary_python_function_cases_v1 import execute as ordinary_execute

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'reports/public-pretraining-v1'
RUNTIME=ROOT/'reports/public-pretraining-assessment-recovery-v1'
ASSESSMENT=RUNTIME
EPOCH=ROOT/'reports/public-pretraining-complete-epoch-value-readback-CPU-v2/complete.json'
OUT=ROOT/'reports/public-pretraining-native-program-action-readback-CPU-v2'
TESTS=ROOT/'reports/native-code-action-readback-v1-tests.log'
PARENT=dict(pid=100343,ticks='54121978')
RESERVE=40*1024**3


def read(path):return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def atomic(path,value):
    pending=path.with_suffix(path.suffix+'.pending')
    with pending.open('x') as stream:
        stream.write(json.dumps(value,indent=2)+'\n');stream.flush();os.fsync(stream.fileno())
    pending.replace(path)


def live(identity):
    try:parts=Path('/proc',str(identity['pid']),'stat').read_text().split(') ',1)[1].split()
    except FileNotFoundError:return False
    return parts[19]==identity['ticks'] and parts[0]!='Z'


def lines(path):
    with path.open('rb') as stream:
        result=[]
        for line in stream:
            if not line.endswith(b'\n'):raise ValueError('A closed native action journal cannot contain a partial LF record')
            result.append(json.loads(line))
    return result


def main():
    if (OUT.exists() or os.environ.get('CUDA_VISIBLE_DEVICES')!='' or os.nice(0)<19
            or any(os.environ.get(k)!='1' for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'])
            or '13 passed' not in TESTS.read_text()):
        raise ValueError('Fresh tested CUDA-hidden one-thread nice19 code/action readback required')
    os.sched_setscheduler(0,os.SCHED_IDLE,os.sched_param(0))
    epoch=read(EPOCH);plan=read(SOURCE/'plan.json');runtime=read(RUNTIME/'plan.json')
    if (not epoch['passed'] or epoch['actual_complete_updates']!=4447 or epoch['actual_unique_neural_values']!=984399690
            or {k:runtime['owner'].get(k) for k in ['pid','ticks']}!=PARENT):
        raise ValueError('The complete original source/order/value readback and exact original parent required')
    files={str(p):sha(p) for p in [EPOCH,SOURCE/'plan.json',RUNTIME/'plan.json',TESTS,
        ROOT/'reports/decoded-native-code-reexecution-v1-tests.log',
        *python_sources(ROOT,[Path(__file__),ROOT/'tests/test_native_code_action_readback_v1.py'])]}
    states={p:(Path(p).stat().st_dev,Path(p).stat().st_ino,Path(p).stat().st_size,Path(p).stat().st_mtime_ns) for p in files}
    me=dict(pid=os.getpid(),ticks=Path('/proc/self/stat').read_text().split(') ',1)[1].split()[19])
    OUT.mkdir();atomic(OUT/'plan.json',dict(owner=me,source_files_sha256=files,exact_original_parent=PARENT,
        all_original_native_programs=96,additional_ordinary_Python_programs=48,
        actual_original_generation_actions_and_EOS_must_reconstruct=True,
        original_frames=['completion','instruction'],original_languages=['Python','JavaScript'],original_action_budget=512,
        every_fresh_verdict_disagreement_preserved=True,reference_and_candidate_stress_timings_reobserved_separately=True,
        no_program_repair_or_new_neural_generation=True,source_role='evaluation',optimization_forbidden=True,
        all_current_training_and_evaluation_jobs_unchanged=True,training_admitted=False,goal_achieved=False))
    def guard():
        if os.sched_getscheduler(0)!=os.SCHED_IDLE or any(
                (Path(p).stat().st_dev,Path(p).stat().st_ino,Path(p).stat().st_size,Path(p).stat().st_mtime_ns)!=v for p,v in states.items()):
            raise ValueError('Every immutable original source/runtime byte and idle scheduling must remain exact')
    def checked():wait_for_reserve(ROOT,minimum_bytes=RESERVE,guard=guard)
    try:
        while live(PARENT):
            guard();atomic(OUT/'progress.json',dict(owner=me,phase='waiting for exact original GPU parent complete and restored',
                parent_live=True,models_constructed=0,new_neural_forwards=0));time.sleep(15)
        if ((RUNTIME/'failure.json').exists() or (SOURCE/'failure.json').exists()
                or any(not (folder/name).is_file() for folder,name in [(ASSESSMENT,'complete.json'),(ASSESSMENT,'restored.json'),
                    (SOURCE,'training-complete.json'),(SOURCE,'native-assessment-inputs.json'),(RUNTIME,'complete.json'),(RUNTIME,'restored.json')])
                or any(live(child) for child in read(RUNTIME/'children.json'))):
            raise ValueError('The full original epoch, all native suites and every exact registered GPU child must finish and restore')
        done=read(ASSESSMENT/'complete.json')
        if not done['all_original_native_populations_complete'] or done['original_complete_training_updates']!=4447:
            raise ValueError('No partial assessment or partial epoch may substitute for the complete source producer')
        inputs=read(SOURCE/'native-assessment-inputs.json');jobs=[];profiles=[]
        for arm in ['initial','trained']:
            dest=OUT/arm;dest.mkdir()
            token_path=ROOT/'checkpoints/public-pretraining-v1'/arm/'tokenizer.json'
            if (sha(token_path)!=plan['tokenizer_sha256'] or inputs[arm]['checkpoint_sha256']!=epoch['source_files_sha256'][str(token_path.with_name('model.pt'))]):
                raise ValueError('Both actual neural checkpoints and packaged own tokenizers must match the complete value readback')
            files[str(token_path)]=sha(token_path);tokenizer=Tokenizer.from_file(str(token_path))
            for frame in ['completion','instruction']:
                envelopes=[ASSESSMENT/arm/f'code-{frame}-shard-{rank}.json' for rank in range(4)]
                report=reconcile(envelopes,dest/f'original-code-{frame}.json')
                if (report['task_count']!=24 or report['checkpoint_sha256']!=inputs[arm]['checkpoint_sha256']
                        or report['suite_sha256']!=sha(ROOT/'src/slm_lab/code_tasks.py') or report['tokenizer_sha256']!=plan['tokenizer_sha256']):
                    raise ValueError('Every full original task/frame/language/weight/tokenizer identity required')
                profiles.append({k:report[k] for k in ['generation_budget','decoding','suite_sha256','tokenizer_sha256','inference_device_type','inference_precision']})
                files.update(report['source_files_sha256'])
            for rank in range(4):
                action_path=ASSESSMENT/arm/f'code-actual-native-actions-{rank}.jsonl';events=lines(action_path)
                expected=2*len(positions(rank,4))
                if len(events)!=2*expected:raise ValueError('Every actual original two-frame source action event is required')
                files[str(action_path)]=sha(action_path)
                for frame_number,frame in enumerate(['completion','instruction']):
                    report=read(ASSESSMENT/arm/f'code-{frame}-shard-{rank}.raw.json')
                    if len(report['results'])!=expected:raise ValueError('Every original assigned frame case must remain present')
                    for i,case in enumerate(report['results']):
                        proof=validate(case,frame,events[frame_number*expected+i],tokenizer,expected_case_position=frame_number*expected+i)
                        jobs.append(dict(endpoint=arm,frame=frame,case=case,action_proof=proof))
        if len(jobs)!=96 or any(p!=profiles[0] for p in profiles):
            raise ValueError('All96original code deliveries must share the full matched native runtime profile')
        for folder,name in [(ASSESSMENT,'complete.json'),(ASSESSMENT,'restored.json'),(SOURCE,'training-complete.json'),
                (SOURCE,'native-assessment-inputs.json'),(RUNTIME,'complete.json'),(RUNTIME,'restored.json'),(RUNTIME,'children.json')]:
            path=folder/name;files[str(path)]=sha(path)
        states.update({p:(Path(p).stat().st_dev,Path(p).stat().st_ino,Path(p).stat().st_size,Path(p).stat().st_mtime_ns) for p in files})
        atomic(OUT/'closed-source-bindings.json',dict(owner=me,source_files_sha256=files,all96actual_original_action_sequences_verified=True,
            actual_matched_native_profile=profiles[0],all_source_workers_terminal=True,source_role='evaluation',optimization_forbidden=True))
        checked();controls={}
        for task in TASKS:
            for language,code in [('Python',task[4]),('JavaScript',task[5])]:
                actual=execute(request(task,language,code,True))
                controls[(language,task[0])]=actual
        atomic(OUT/'all24-fresh-native-reference-controls.json',dict(results=[dict(language=k[0],task=k[1],execution=v) for k,v in controls.items()]))
        controls_passed=all(v.get('functional_passed') and v.get('efficiency_completed') for v in controls.values())
        rows=[];differences=[]
        with (OUT/'all96-action-bound-fresh-program-executions.jsonl').open('x') as stream:
            for position,job in enumerate(jobs):
                if position%8==0:checked()
                arm,frame,case=job['endpoint'],job['frame'],job['case'];task,code=reconstruct(case,frame)
                payload=request(task,case['language'],code);actual=execute(payload)
                difference=any(bool(actual.get(k))!=bool(case['candidate'].get(k)) for k in ['syntax_valid','functional_passed'])
                if difference:differences.append(dict(endpoint=arm,frame=frame,language=case['language'],task=case['task'],producer=case['candidate'],fresh=actual))
                reference=controls[(case['language'],case['task'])]
                efficient,measure=efficiency(actual,reference,task[3]) if reference.get('efficiency_completed') else (False,{})
                ordinary=ordinary_execute(code,task[0],payload['cases']) if case['language']=='Python' else None
                row=dict(endpoint=arm,frame=frame,language=case['language'],task=case['task'],actual_unchanged_code=code,
                    actual_code_sha256=hashlib.sha256(code.encode()).hexdigest(),actual_original_finite_cases=len(payload['cases']),
                    original_action_readback=job['action_proof'],fresh_native_execution=actual,fresh_secondary_ordinary_Python=ordinary,
                    fresh_reference_efficiency_pass=bool(efficient),fresh_efficiency_measurement=measure,
                    original_efficiency_pass=case['efficiency_stress_passed'],native_verdict_difference=difference,
                    source_role='evaluation',optimization_forbidden=True,no_new_neural_generation_or_program_repair=True)
                rows.append(row);stream.write(json.dumps(row,ensure_ascii=True)+'\n');stream.flush()
                atomic(OUT/'progress.json',dict(owner=me,phase='fresh whole-program native and ordinary Python execution',
                    original_programs_complete=len(rows),all_original_programs=96,actual_native_verdict_differences=len(differences),
                    reference_controls_passed=controls_passed,new_neural_forwards=0))
            os.fsync(stream.fileno())
        totals={}
        for arm in ['initial','trained']:
            group=[v for v in rows if v['endpoint']==arm]
            totals[arm]=dict(tasks=len(group),syntax=sum(bool(v['fresh_native_execution'].get('syntax_valid')) for v in group),
                functional=sum(bool(v['fresh_native_execution'].get('functional_passed')) for v in group),
                fresh_empirical_efficiency=sum(v['fresh_reference_efficiency_pass'] for v in group),
                original_empirical_efficiency=sum(v['original_efficiency_pass'] for v in group),
                ordinary_Python_tasks=sum(v['language']=='Python' for v in group),
                ordinary_Python_functional=sum(bool(v['fresh_secondary_ordinary_Python']['functional_passed']) for v in group if v['language']=='Python'),
                actual_EOS_stops=sum(v['original_action_readback']['actual_stopped_on_EOS'] for v in group),
                actual_generated_actions=sum(v['original_action_readback']['actual_native_action_count'] for v in group))
        if any(v['tasks']!=48 or v['ordinary_Python_tasks']!=24 for v in totals.values()):raise ValueError('Every full endpoint and secondary Python population required')
        checked()
        if any(sha(path)!=digest for path,digest in files.items()):raise ValueError('Every closed original source/program/action/runtime byte must remain immutable')
        atomic(OUT/'complete.json',dict(complete=True,passed=controls_passed and not differences,at=datetime.now(timezone.utc).isoformat(),owner=me,
            source_files_sha256=files,actual_endpoint_totals=totals,all24_fresh_reference_controls_passed=controls_passed,
            all96actual_original_prompts_actions_raw_decoded_responses_and_programs_reconstructed=True,
            all96_original_native_programs_freshly_executed=True,all48_Python_programs_additionally_executed_under_ordinary_protocol=True,
            actual_native_verdict_differences=differences,all_fresh_differences_preserved_without_case_omission=True,
            actual_fresh_program_observations_sha256=sha(OUT/'all96-action-bound-fresh-program-executions.jsonl'),
            empirical_stress_timings_not_asymptotic_or_arbitrary_code_efficiency_proof=True,
            actual_action_ids_and_EOS_certified_but_log_probabilities_not_retained=True,
            source_role='evaluation',optimization_forbidden=True,models_constructed=0,neural_forwards=0,
            training_admitted=False,new_production_optimizer_updates=0,goal_achieved=False))
        print(json.dumps(dict(complete=True,passed=controls_passed and not differences,actual_endpoint_totals=totals)),flush=True)
    except BaseException as error:
        atomic(OUT/'failure.json',dict(owner=me,error=repr(error),automatic_retry=False,goal_achieved=False));raise


if __name__=='__main__':main()
