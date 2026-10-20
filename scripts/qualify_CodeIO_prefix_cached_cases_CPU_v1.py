"""Qualify the first32 fixed original function rows after all roles are fixed.

Every original cached case is retained. Reference calls use two seed contexts
and a fresh confined stdlib process per case; a complete no-op control follows
only full agreement. This produces source observations, not model metrics,
training admission, human preferences or an independent RL reward oracle.
"""
from collections import Counter
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
INDEX=Path('/tmp/slm-CodeIO-whole-source-index-v1')
RAW=Path('/tmp/slm-CodeIO-raw-candidate-v1')
OUT=Path('/tmp/slm-CodeIO-prefix-cached-case-qualification-v1')
COUNT=32
GIB=1024**3


def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def live(actor):
    try:f=Path('/proc',str(actor['pid']),'stat').read_text().rsplit(')',1)[1].split()
    except FileNotFoundError:return False
    return f[19]==actor['birth'] and f[0] not in ['Z','X']


def write(name,value):
    p=OUT/name;q=p.with_suffix(p.suffix+'.pending')
    with q.open('x') as f:f.write(json.dumps(value,indent=2,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
    os.replace(q,p)


def main():
    if OUT.exists() or os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('Fresh CPU-only source qualification required')
    os.nice(10);os.sched_setaffinity(0,{22})
    from slm_lab.codeio_published_cases_v1 import contract,execute,strict_loads,implementation_sources
    from slm_lab.local_source_closure_v2 import python_sources
    predecessor=json.loads((INDEX/'plan.json').read_text())['owner']
    tests=ROOT/'reports/codeio-published-cases-v1-tests.log'
    if '3 passed' not in tests.read_text():raise ValueError('Actual confined keyword/value/schema/isolation checks must pass')
    source=json.loads((RAW/'complete.json').read_text());path=Path(source['original_source_file'])
    if not source['complete'] or sha(path)!=source['original_source_sha256']:raise ValueError('The exact complete original source must remain unchanged')
    if shutil.disk_usage(OUT.parent).free<44*GIB:raise ValueError('All observations must fit above the unchanged source writer reserve')
    files={str(p):sha(p) for p in [*python_sources(ROOT,[Path(__file__).resolve()]),tests,ROOT/'tests/test_codeio_published_cases_v1.py',INDEX/'plan.json',RAW/'complete.json',path]}
    files.update(implementation_sources());OUT.mkdir()
    write('plan.json',dict(owner=dict(pid=os.getpid(),birth=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19]),
        at=datetime.now(timezone.utc).isoformat(),source_files_sha256=files,exact_index_predecessor=predecessor,
        original_positions=list(range(COUNT)),selection_fixed_before_fresh_source_outcomes=True,
        every_original_cached_case_required=True,declared_seeds=[0,1],fresh_process_per_case=True,
        source_module_name='__main__',keyword_call_and_original_JSON_conversion=True,
        original_source_code_never_edited=True,no_OP_control_required=True,
        maximum_whole_source_seconds_per_seed=20,CPU_threads=1,own_nice=10,CPU_affinity=[22],
        no_model_or_GPU_or_live_teacher_used=True,data_admitted_to_training=False,RL_oracle_admitted=False,goal_achieved=False))
    counts=Counter()
    try:
        while not (INDEX/'complete.json').exists():
            if not live(predecessor):raise ValueError('The exact source index ended without complete fixed roles')
            write('progress.json',dict(phase='wait_all_original_roles_and_screens',exact_predecessor_live=True));time.sleep(10)
        if (INDEX/'failure.json').exists():raise ValueError('An incomplete/failed index cannot supply source roles')
        index=json.loads((INDEX/'complete.json').read_text())
        ledger=INDEX/'every-original-source-with-fixed-role.jsonl'
        if not index['complete'] or sha(ledger)!=index['final_role_ledger_sha256']:
            raise ValueError('The complete original role/screen ledger must remain exact before source execution')
        for p in [INDEX/'complete.json',ledger]:files[str(p)]=sha(p)
        selected=[]
        with ledger.open() as f:
            for i in range(COUNT):
                r=json.loads(next(f));assert r['original_position']==i;selected.append(r)
        write('execution-plan.json',dict(source_files_sha256=files,index_complete_sha256=sha(INDEX/'complete.json'),
            exact_selected_original_sources=selected,role_change_after_outcomes_forbidden=True,source_execution_only=True))
        results=[]
        with path.open('rb') as stream:
            for position in range(COUNT):
                line=next(stream);record=selected[position]
                if hashlib.sha256(line).hexdigest()!=record['original_row_sha256']:
                    raise ValueError('The exact original complete selected source row differs')
                result=dict(original_position=position,original_row_sha256=record['original_row_sha256'],
                    fixed_role=record['fixed_role'],whole_component_quarantined=record['entire_component_quarantined_for_protected_signal'],
                    source_optimization_forbidden=record['optimization_forbidden'],all_original_cases_and_seeds_match=False,
                    source_only_no_training_admission=True,independent_semantic_correctness_certified=False,RL_reward_admitted=False)
                try:row=strict_loads(line);cases=contract(row)
                except (ValueError,TypeError,KeyError,RecursionError) as error:
                    result.update(status='original_complete_schema_qualification',diagnostic=str(error))
                else:
                    observations=[]
                    for seed in [0,1]:
                        observation=execute(cases,seed=seed,seconds=20);observations.append(observation)
                        counts['reference_program_contexts']+=1
                        counts['actual_reference_case_processes']+=len(observation['records'])
                        if not observation['all_original_cases_observed']:break
                    whole=len(observations)==2 and all(o['all_original_cases_observed'] for o in observations)
                    agree=whole and all(o['all_original_cached_values_match'] for o in observations)
                    result.update(total_original_cases=cases['total_cases'],original_program_sha256=cases['original_program_sha256'],
                        original_cases_sha256=cases['original_cases_sha256'],reference_observations=observations)
                    if not whole:result['status']='reference_runtime_or_resource_population_qualification'
                    elif not agree:result['status']='complete_local_context_cached_value_mismatch'
                    else:
                        noop='def '+cases['function']+'(**kwargs):\n    return None\n'
                        control=execute(cases,seed=0,seconds=20,program=noop)
                        counts['no_OP_case_processes']+=len(control['records'])
                        result['no_OP_control']=control
                        if not control['all_original_cases_observed'] or control['all_original_cached_values_match']:
                            result['status']='no_OP_runtime_or_trivial_population_qualification'
                        else:
                            result.update(status='all_original_cached_values_match_two_local_seed_contexts',all_original_cases_and_seeds_match=True)
                    result['fresh_context_agreement_does_not_certify_general_determinism']=True
                counts[result['status']]+=1
                output=OUT/(f'source-{position:05d}.json');write(output.name,result);results.append(dict(position=position,path=str(output),sha256=sha(output)))
                write('progress.json',dict(phase='complete_source_IO_qualification',actual_original_sources=position+1,planned_original_sources=COUNT,counts=dict(counts)))
        if any(sha(p)!=h for p,h in files.items()):raise ValueError('An original/index/runtime source byte changed')
        write('complete.json',dict(complete=True,source_files_sha256=files,exact_original_positions=list(range(COUNT)),
            all_fixed_sources_accounted_for=True,source_results=results,counts=dict(counts),
            every_original_case_preserved_or_whole_source_qualification=True,
            source_roles_fixed_before_fresh_outcomes=True,reference_contexts_are_local_stdlib_not_full_publisher_environment=True,
            actual_model_generations_or_training_updates=0,human_feedback=False,data_admitted_to_training=False,
            RL_reward_oracle_admitted=False,model_quality_improvement_proved=False,goal_achieved=False))
    except BaseException as error:
        write('failure.json',dict(error=repr(error),traceback=traceback.format_exc(),partial_outputs_retained=True,data_admitted_to_training=False));raise


if __name__=='__main__':main()
