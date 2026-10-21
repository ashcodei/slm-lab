"""Reexecute every unchanged distributed-code delivery as each profile closes.

Checks all original prompts, actual actions/EOS and complete program bytes.
Native and ordinary-Python results remain distinct. No model is loaded or
queried; these reused local tasks do not replace the full final benchmarks.
"""
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'reports/durable-distributed-code-v1'
MODELS=ROOT/'checkpoints/durable-distributed-code-v1'
OUT=ROOT/'reports/distributed-dense-program-readback-CPU-v1'


def read(path):return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def write(name,value):
    path=OUT/name;pending=path.with_suffix(path.suffix+'.pending')
    with pending.open('x') as stream:
        json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n');stream.flush();os.fsync(stream.fileno())
    os.replace(pending,path)


def live(actor):
    try:fields=Path('/proc',str(actor['pid']),'stat').read_text().rsplit(')',1)[1].split()
    except FileNotFoundError:return False
    return fields[0] not in ['Z','X'] and fields[19]==actor['birth']


def main():
    if OUT.exists() or os.environ.get('CUDA_VISIBLE_DEVICES')!='':
        raise ValueError('Fresh explicit CPU-only program readback required')
    os.nice(10);os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
    from tokenizers import Tokenizer
    from slm_lab.codenet_checkpoint_v1 import FileBindings
    from slm_lab.code_tasks import TASKS,request
    from slm_lab.code_quality import execute as native_execute,efficiency
    from slm_lab.decoded_native_code_reexecution_v1 import reconstruct
    from slm_lab.native_code_action_readback_v1 import validate
    from slm_lab.native_code_shards_v1 import positions
    from slm_lab.ordinary_python_function_cases_v1 import execute as ordinary_execute,implementation_sources
    from slm_lab.local_source_closure_v2 import python_sources
    study=read(RUN/'study.json');profiles=list(study['rates'])+['initial'];attempt=RUN/'attempts/1'
    owner=read(attempt/'plan.json')['owner']
    if (study['updates_per_arm']!=230 or study['views_per_arm']!=3675 or study['world_size']!=4 or len(TASKS)!=12):
        raise ValueError('Every original full-source/48-program/four-shard comparison is required')
    if shutil.disk_usage(ROOT).free<8*1024**3+64*1024**2:
        raise ValueError('All actual program receipts must fit above the current shared writer reserve')
    OUT.mkdir();files={str(p):sha(p) for p in [RUN/'study.json',attempt/'plan.json',*python_sources(ROOT,[Path(__file__).resolve()])]}
    files.update(implementation_sources());guard=FileBindings(files)
    write('plan.json',dict(at=datetime.now(timezone.utc).isoformat(),owner=dict(pid=os.getpid(),birth=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19]),
        exact_original_GPU_owner=owner,profiles=profiles,source_files_sha256=files,programs_per_profile=48,
        action_budget=512,actual_native_action_journals_required=True,original_four_shards_required=True,
        no_GPU_or_model_load_or_new_neural_generation=True,full_final_benchmarks_still_required=True,goal_achieved=False))
    try:
        references={};ordinary_references={}
        for task in TASKS:
            for language,index in [('Python',4),('JavaScript',5)]:
                result=native_execute(request(task,language,task[index],True))
                if not result['functional_passed'] or not result['efficiency_completed']:
                    raise ValueError('Every actual fresh native reference/efficiency control must pass')
                references[language+':'+task[0]]=result
            result=ordinary_execute(task[4],task[0],request(task,'Python',task[4])['cases'])
            if not result['functional_passed'] or not result['all_cases_observed']:
                raise ValueError('Every complete ordinary-Python reference control must pass')
            ordinary_references[task[0]]=result
        write('reference-controls.json',dict(native=references,ordinary_Python=ordinary_references,
            actual_native_positive_controls=24,actual_ordinary_Python_positive_controls=12))
        summaries={}
        for profile in profiles:
            folder=RUN/profile/'assessment-attempt-1'
            while not all((folder/f'code-{frame}.json').exists() for frame in ['completion','instruction']):
                if not live(owner):raise ValueError('The exact original GPU owner ended before all original code reports closed')
                guard.check();write('progress.json',dict(phase='wait_exact_original_code_deliveries',profile=profile,
                    exact_GPU_owner_confirmed_live=True,completed_profiles=summaries));time.sleep(10)
            tokenizer_path=(MODELS/profile/'tokenizer.json') if profile!='initial' else ROOT/'data/public-reasoned-code-SFT-v1/tokenizer.json'
            if sha(tokenizer_path)!='0072bb6de61da87bd9ba5f4e6739b594cc3e60c62c75a59d25d499629a70088f':
                raise ValueError('The exact original own vocabulary is required for every profile')
            tokenizer=Tokenizer.from_file(str(tokenizer_path));files[str(tokenizer_path)]=sha(tokenizer_path)
            checkpoint=MODELS/profile/'model.pt' if profile!='initial' else Path('/tmp/slm-durable-distributed-code-v1/initial/model.pt')
            if not checkpoint.exists():raise ValueError('The original actual checkpoint must remain available during delivery readback')
            checkpoint_sha=sha(checkpoint);files[str(checkpoint)]=checkpoint_sha
            if profile!='initial':
                training=read(RUN/profile/'training-complete.json')
                if not training['complete'] or training['updates']!=230 or training['checkpoint_sha256']!=checkpoint_sha:
                    raise ValueError('Every complete original training epoch and its final model are required')
                files[str(RUN/profile/'training-complete.json')]=sha(RUN/profile/'training-complete.json')
            records=[]
            for frame in ['completion','instruction']:
                merged=read(folder/f'code-{frame}.json');files[str(folder/f'code-{frame}.json')]=sha(folder/f'code-{frame}.json')
                if merged['task_count']!=24 or merged['generation_budget']!=512 or merged['checkpoint_sha256']!=checkpoint_sha:
                    raise ValueError('The complete original code report and actual final checkpoint must agree')
                by_case={(row['language'],row['task']):row for row in merged['results']}
                if len(by_case)!=24:raise ValueError('Every original case must occur exactly once')
                frame_records=[]
                for rank in range(4):
                    envelope=folder/f'code-{frame}-shard-{rank}.json';raw=folder/f'code-{frame}-shard-{rank}.raw.json'
                    journal=folder/f'code-{frame}-rank-{rank}-actual-actions.jsonl'
                    packet=read(envelope);report=read(raw);actions=[json.loads(line) for line in journal.read_text().splitlines()]
                    for path in [envelope,raw,journal]:files[str(path)]=sha(path)
                    if (packet['rank']!=rank or packet['world']!=4 or packet['raw_report_sha256']!=sha(raw)
                            or packet['original_task_positions']!=positions(rank,4) or report['task_count']!=6
                            or len(actions)!=6 or report['generation_budget']!=512 or report['prompt_mode']!=frame
                            or report['inference_device_type']!='cuda' or report['inference_precision']!='bf16'
                            or report['checkpoint_sha256']!=checkpoint_sha or report['tokenizer_sha256']!=sha(tokenizer_path)):
                        raise ValueError('Every complete native shard, actual six deliveries and original protocol must agree')
                    for rel,digest in report['evaluation_provenance']['files_sha256'].items():
                        path=ROOT/rel
                        if sha(path)!=digest:raise ValueError('An actual original producer source changed')
                        files[str(path)]=digest
                    expected=[(language,TASKS[i]) for language in ['Python','JavaScript'] for i in positions(rank,4)]
                    for local,(case,event,(language,task)) in enumerate(zip(report['results'],actions,expected)):
                        if case['language']!=language or case['task']!=task[0] or by_case[(language,task[0])]!=case:
                            raise ValueError('Every original shard/language/task and merged source must agree')
                        delivery=validate(case,frame,event,tokenizer,expected_case_position=local)
                        actual_task,code=reconstruct(case,frame)
                        if actual_task!=task:raise ValueError('The full original specification and unchanged code bytes are required')
                        actual=native_execute(request(task,language,code))
                        if any(actual[key]!=case['candidate'][key] for key in ['syntax_valid','functional_passed']):
                            write(f'{profile}-{frame}-{language}-{task[0]}-discrepancy.json',dict(original=case,actual=actual))
                            raise ValueError('A fresh native verdict differs; retain the discrepancy without repair')
                        efficient,measurement=efficiency(actual,references[language+':'+task[0]],task[3])
                        ordinary=ordinary_execute(code,task[0],request(task,language,code)['cases']) if language=='Python' else None
                        record=dict(profile=profile,frame=frame,rank=rank,language=language,task=task[0],delivery_proof=delivery,
                            actual_actions=event,actual_unchanged_code=code,fresh_native=actual,fresh_ordinary_Python=ordinary,
                            producer_empirical_efficiency=case['efficiency_stress_passed'],fresh_empirical_efficiency=efficient,
                            fresh_efficiency_measurement=measurement,shared_host_timing_not_asymptotic_proof=True)
                        write(f'{profile}-{frame}-{language}-{task[0]}.json',record);frame_records.append(record);records.append(record)
                if (len(frame_records)!=24 or merged['syntax_pass_rate']!=sum(r['fresh_native']['syntax_valid'] for r in frame_records)/24
                        or merged['functional_pass_at_1']!=sum(r['fresh_native']['functional_passed'] for r in frame_records)/24
                        or merged['efficiency_stress_pass_rate']!=sum(r['producer_empirical_efficiency'] for r in frame_records)/24):
                    raise ValueError('Every original full-frame aggregate must reconstruct from every actual case')
            if len(records)!=48:raise ValueError('All original48programs per profile must be freshly executed')
            summary=dict(programs=48,native_syntax=sum(r['fresh_native']['syntax_valid'] for r in records),
                native_functional=sum(r['fresh_native']['functional_passed'] for r in records),
                ordinary_Python_syntax=sum(r['fresh_ordinary_Python']['syntax_valid'] for r in records if r['fresh_ordinary_Python'] is not None),
                ordinary_Python_functional=sum(r['fresh_ordinary_Python']['functional_passed'] for r in records if r['fresh_ordinary_Python'] is not None),
                producer_efficiency=sum(r['producer_empirical_efficiency'] for r in records),
                fresh_empirical_efficiency=sum(r['fresh_empirical_efficiency'] for r in records),
                EOS=sum(r['actual_actions']['actual_stopped_on_EOS'] for r in records))
            guard=FileBindings(files);guard.check(full=True);summaries[profile]=summary
            write(profile+'-complete.json',dict(complete=True,profile=profile,summary=summary,source_files_sha256=files,
                every_original_prompt_action_EOS_and_program_rebuilt=True,all48native_and24ordinary_Python_deliveries_reexecuted=True,
                all_original_syntax_and_functional_verdicts_agree=True,no_neural_generation_or_likelihood_replayed=True,
                reused_local_development_suite_not_full_independent_benchmarks=True,candidate_promoted=False,goal_achieved=False))
            print(json.dumps(dict(profile=profile,summary=summary)),flush=True)
        write('complete.json',dict(complete=True,at=datetime.now(timezone.utc).isoformat(),source_files_sha256=files,
            summaries=summaries,actual_native_program_reexecutions=240,actual_ordinary_Python_reexecutions=120,
            actual_native_reference_controls=24,actual_ordinary_Python_reference_controls=12,
            all_five_original_profiles_and_actions_preserved=True,no_model_or_GPU_or_new_generation=True,
            all_original_full_final_benchmarks_and_human_multimodal_scope_still_required=True,candidate_promoted=False,goal_achieved=False))
    except BaseException as error:
        write('failure.json',dict(error=repr(error),traceback=traceback.format_exc(),partial_evidence_preserved=True,
            no_original_source_program_or_GPU_process_changed=True,goal_achieved=False));raise


if __name__=='__main__':main()
