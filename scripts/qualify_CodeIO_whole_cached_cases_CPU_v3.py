"""Qualify whole cached-case populations without editing original functions.

The prefix mode compares all32 retained V1 sources before the full mode may
run across every original row. Whole-context observations use fresh module
namespaces, not a fresh interpreter per case. No source is trained here.
"""
import argparse
from collections import Counter
from datetime import datetime,timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
RAW=Path('/tmp/slm-CodeIO-raw-candidate-v1')
INDEX=Path('/tmp/slm-CodeIO-whole-source-index-v1')
ORIGINAL=Path('/tmp/slm-CodeIO-prefix-cached-case-qualification-v1')
PREFIX=Path('/tmp/slm-CodeIO-whole-cached-case-prefix-v2')
FULL=Path('/tmp/slm-CodeIO-whole-cached-case-qualification-v2')
GIB=1024**3


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def write(folder,name,value):
    p=folder/name;q=p.with_suffix(p.suffix+'.pending')
    with q.open('x') as f:
        f.write(json.dumps(value,indent=2,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
    os.replace(q,p)


def main(mode):
    folder=PREFIX if mode=='prefix' else FULL;count=32 if mode=='prefix' else 227669
    if folder.exists() or os.environ.get('CUDA_VISIBLE_DEVICES')!='':
        raise ValueError('A fresh CPU-only whole-source observation directory is required')
    os.nice(10);os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
    from slm_lab.codeio_published_cases_v1 import contract,equal_json,strict_loads
    from slm_lab.codeio_whole_cases_v2 import execute,implementation_sources
    from slm_lab.local_source_closure_v2 import python_sources
    tests=ROOT/'reports/codeio-whole-cases-v2-tests.log'
    if '2 passed' not in tests.read_text():raise ValueError('The actual fresh/whole-context and confinement controls must pass')
    source=json.loads((RAW/'complete.json').read_text());index=json.loads((INDEX/'complete.json').read_text())
    path=Path(source['original_source_file']);roles=INDEX/'every-original-source-with-fixed-role.jsonl'
    if (not source['complete'] or not index['complete'] or sha(path)!=source['original_source_sha256']
            or sha(roles)!=index['final_role_ledger_sha256'] or index['complete_original_rows']!=227669):
        raise ValueError('The complete original raw file and all fixed pre-execution roles are required')
    files={str(p):sha(p) for p in [*python_sources(ROOT,[Path(__file__).resolve()]),tests,
        ROOT/'tests/test_codeio_whole_cases_v2.py',RAW/'complete.json',INDEX/'complete.json',path,roles]}
    files.update(implementation_sources());originals={}
    if mode=='prefix':
        receipt=json.loads((ORIGINAL/'complete.json').read_text())
        if not receipt['complete'] or receipt['exact_original_positions']!=list(range(32)):
            raise ValueError('Every original V1 prefix source is required')
        files[str(ORIGINAL/'complete.json')]=sha(ORIGINAL/'complete.json')
        for entry in receipt['source_results']:
            p=Path(entry['path'])
            if sha(p)!=entry['sha256']:raise ValueError('The original V1 observations changed')
            originals[entry['position']]=json.loads(p.read_text());files[str(p)]=entry['sha256']
    else:
        preflight=json.loads((PREFIX/'complete.json').read_text())
        if (not preflight['complete'] or preflight['actual_original_sources']!=32
                or not preflight['fresh_context_controls_match_original_v1']
                or any(sha(p)!=h for p,h in preflight['source_files_sha256'].items())
                or sha(PREFIX/'every-original-source-observation.jsonl.gz')!=preflight['observations_sha256']):
            raise ValueError('Complete unchanged V1/V2 prefix agreement must precede full-source execution')
        for p in [PREFIX/'complete.json',PREFIX/'every-original-source-observation.jsonl.gz']:files[str(p)]=sha(p)
    if shutil.disk_usage(folder.parent).free<48*GIB:
        raise ValueError('The complete observations must fit above the40GiB source-writer floor')
    folder.mkdir();started=time.monotonic();counts=Counter();role_counts={r:Counter() for r in ['train','reserved']}
    write(folder,'plan.json',dict(at=datetime.now(timezone.utc).isoformat(),
        owner=dict(pid=os.getpid(),birth=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19]),
        mode=mode,original_positions='all original positions in ascending order' if mode=='full' else list(range(32)),
        actual_declared_source_population=count,source_files_sha256=files,
        source_roles_fixed_before_fresh_outcomes=True,whole_original_cached_cases_required=True,
        seeds=[0,1],fresh_module_namespace_per_case=True,fresh_interpreter_per_case=False,
        maximum_seconds_per_whole_source_and_seed=20,memory_bytes_per_confined_process=768*1024**2,
        no_OP_control_required=True,no_model_or_GPU_or_live_teacher_used=True,
        CPU_affinity=sorted(os.sched_getaffinity(0)),CPU_threads=1,own_nice=10,data_admitted_to_training=False,RL_reward_admitted=False))
    ledger=folder/'every-original-source-observation.jsonl.gz'
    try:
        with path.open('rb') as raw,roles.open() as role_stream,ledger.open('xb') as binary:
            with gzip.GzipFile(fileobj=binary,mode='wb',mtime=0) as observations:
                for position in range(count):
                    line=next(raw);record=json.loads(next(role_stream))
                    if record['original_position']!=position or hashlib.sha256(line).hexdigest()!=record['original_row_sha256']:
                        raise ValueError('The exact original source position and row bytes must match the fixed roles')
                    role=record['fixed_role']['source_role']
                    result=dict(original_position=position,original_row_sha256=record['original_row_sha256'],
                        fixed_role=record['fixed_role'],source_optimization_forbidden=record['optimization_forbidden'],
                        whole_component_quarantined=record['entire_component_quarantined_for_protected_signal'],
                        all_original_cases_and_seeds_match=False,source_only_no_training_admission=True,
                        independent_semantic_correctness_certified=False,RL_reward_admitted=False)
                    try:cases=contract(strict_loads(line))
                    except (ValueError,TypeError,KeyError,RecursionError) as error:
                        result.update(status='original_complete_schema_qualification',diagnostic=str(error))
                    else:
                        contexts=[]
                        for seed in [0,1]:
                            observed=execute(cases,seed=seed,seconds=20);contexts.append(observed)
                            counts['reference_context_processes']+=1
                            counts['complete_reference_cases_observed']+=len(observed['records'])
                            if not observed['all_original_cases_observed']:break
                        whole=len(contexts)==2 and all(o['all_original_cases_observed'] for o in contexts)
                        agree=whole and all(o['all_original_cached_values_match'] for o in contexts)
                        result.update(total_original_cases=cases['total_cases'],original_program_sha256=cases['original_program_sha256'],
                            original_cases_sha256=cases['original_cases_sha256'],reference_observations=contexts)
                        if not whole:result['status']='reference_runtime_or_resource_population_qualification'
                        elif not agree:result['status']='complete_local_context_cached_value_mismatch'
                        else:
                            noop='def '+cases['function']+'(**kwargs):\n    return None\n'
                            control=execute(cases,seed=0,seconds=20,program=noop)
                            result['no_OP_control']=control;counts['no_OP_context_processes']+=1
                            counts['complete_no_OP_cases_observed']+=len(control['records'])
                            if not control['all_original_cases_observed'] or control['all_original_cached_values_match']:
                                result['status']='no_OP_runtime_or_trivial_population_qualification'
                            else:
                                result.update(status='all_original_cached_values_match_two_local_seed_contexts',all_original_cases_and_seeds_match=True)
                    if mode=='prefix':
                        old=originals[position]
                        if (old['original_row_sha256']!=result['original_row_sha256'] or old['fixed_role']!=result['fixed_role']
                                or old['status']!=result['status'] or old['all_original_cases_and_seeds_match']!=result['all_original_cases_and_seeds_match']):
                            raise ValueError('A complete original V1 source disposition differs in the separately named context')
                        if result['all_original_cases_and_seeds_match']:
                            for a,b in zip(old['reference_observations'],result['reference_observations']):
                                if not equal_json([r['returned']['value'] for r in a['records']],[r['value'] for r in b['records']]):
                                    raise ValueError('An original V1 returned value differs')
                        result['original_V1_complete_disposition_agrees']=True
                    counts[result['status']]+=1;role_counts[role][result['status']]+=1;counts['original_sources']+=1
                    observations.write((json.dumps(result,separators=(',',':'),allow_nan=False)+'\n').encode())
                    if (position+1)%10==0 or position+1==count:
                        observations.flush();binary.flush();os.fsync(binary.fileno())
                        if shutil.disk_usage(folder.parent).free<40*GIB:raise ValueError('Original source writer reserve must remain funded')
                        write(folder,'progress.json',dict(phase='whole_original_source_case_observations',
                            actual_original_sources=position+1,planned_original_sources=count,counts=dict(counts),
                            fixed_role_counts=role_counts,elapsed_seconds=time.monotonic()-started))
            binary.flush();os.fsync(binary.fileno())
            if mode=='full' and (raw.read(1) or role_stream.read(1)):raise ValueError('Every original raw and fixed-role row must be accounted for')
        if any(sha(p)!=h for p,h in files.items()):raise ValueError('An original raw/index/runtime/source byte changed')
        write(folder,'complete.json',dict(complete=True,at=datetime.now(timezone.utc).isoformat(),mode=mode,
            source_files_sha256=files,actual_original_sources=count,counts=dict(counts),fixed_role_counts=role_counts,
            observations_sha256=sha(ledger),fresh_context_controls_match_original_v1=True if mode=='prefix' else None,
            every_original_case_preserved_or_whole_source_qualification=True,
            fresh_module_namespaces_not_fresh_case_interpreters=True,publisher_full_runtime_reproduced=False,
            source_roles_unchanged=True,elapsed_seconds=time.monotonic()-started,
            data_admitted_to_training=False,actual_model_updates=0,human_feedback=False,
            independent_RL_reward_admitted=False,model_quality_gain_proved=False,goal_achieved=False))
    except BaseException as error:
        write(folder,'failure.json',dict(error=repr(error),traceback=traceback.format_exc(),partial_observations_preserved=True,
            original_inputs_unchanged=True,data_admitted_to_training=False));raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--mode',required=True,choices=['prefix','full'])
    main(parser.parse_args().mode)
