"""Extend fixed-role exposure screening to whole code and all cached values.

All original rows are read before any component can pass this new screen.
No source role is reassigned, no source code is executed, and neither model
scores nor fresh execution outcomes participate in this exposure screen.
"""
from collections import Counter
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
RAW=Path('/tmp/slm-CodeIO-raw-candidate-v1')
INDEX=Path('/tmp/slm-CodeIO-whole-source-index-v1')
HELD=Path('/tmp/slm-public-reasoned-code-held-registry-v1')
OUT=Path('/tmp/slm-CodeIO-whole-target-screen-v2')
GIB=1024**3


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def write(name,value):
    p=OUT/name;q=p.with_suffix(p.suffix+'.pending')
    with q.open('x') as f:
        f.write(json.dumps(value,indent=2,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
    os.replace(q,p)


def main():
    if OUT.exists() or os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('Fresh CPU-only complete source screen required')
    os.sched_setaffinity(0,{min(os.sched_getaffinity(0))});os.nice(10)
    from slm_lab.codeio_whole_text_screen_v2 import inspect,blocked_components
    from slm_lab.held_text_overlap_v1 import HeldTextOverlap
    from slm_lab.local_source_closure_v2 import python_sources
    source=json.loads((RAW/'complete.json').read_text());index=json.loads((INDEX/'complete.json').read_text())
    held=json.loads((HELD/'complete.json').read_text());path=Path(source['original_source_file'])
    roles=INDEX/'every-original-source-with-fixed-role.jsonl';registry=HELD/'held-question-registry.json'
    tests=ROOT/'reports/codeio-whole-text-screen-v2-tests.log'
    if ('2 passed' not in tests.read_text() or not source['complete'] or not index['complete'] or not held['complete']
            or sha(path)!=source['original_source_sha256'] or sha(roles)!=index['final_role_ledger_sha256']
            or sha(registry)!=held['held_registry_sha256'] or index['complete_original_rows']!=227669):
        raise ValueError('Actual late-target/component controls and complete unchanged source/index/held inputs required')
    files={str(p):sha(p) for p in [*python_sources(ROOT,[Path(__file__).resolve()]),tests,
        ROOT/'tests/test_codeio_whole_text_screen_v2.py',RAW/'complete.json',INDEX/'complete.json',
        HELD/'complete.json',path,roles,registry]}
    memory=dict(line.split()[:2] for line in Path('/sys/fs/cgroup/memory.stat').read_text().splitlines())
    nonfile=sum(int(memory[k]) for k in ['anon','shmem','kernel'])
    if (int(Path('/sys/fs/cgroup/memory.max').read_text())-nonfile<56*GIB
            or shutil.disk_usage(OUT.parent).free<44*GIB):
        raise ValueError('8GiB own/48GiB other-job host funding and40GiB scratch reserve required')
    resource.setrlimit(resource.RLIMIT_AS,(8*GIB,8*GIB));OUT.mkdir();started=time.monotonic();counts=Counter()
    write('plan.json',dict(at=datetime.now(timezone.utc).isoformat(),
        owner=dict(pid=os.getpid(),birth=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19]),
        source_files_sha256=files,original_rows=227669,all_original_code_and_cached_JSON_keys_values_screened=True,
        source_roles_unchanged=True,entire_fixed_components_quarantined_on_any_hit_or_screen_qualification=True,
        no_model_or_execution_outcomes_used=True,source_code_execution=False,maximum_own_virtual_bytes=8*GIB,
        CPU_affinity=sorted(os.sched_getaffinity(0)),CPU_threads=1,own_nice=10,data_admitted_to_training=False,RL_reward_admitted=False))
    try:
        matcher=HeldTextOverlap(json.loads(registry.read_text()));records=[]
        first=OUT/'every-original-source-whole-text-screen.jsonl'
        with path.open('rb') as raw,roles.open() as role_stream,first.open('x') as ledger:
            for position,line in enumerate(raw):
                prior=json.loads(next(role_stream))
                if prior['original_position']!=position or hashlib.sha256(line).hexdigest()!=prior['original_row_sha256']:
                    raise ValueError('Every exact original row and fixed source component must remain aligned')
                record=dict(original_position=position,original_row_sha256=prior['original_row_sha256'],fixed_role=prior['fixed_role'],
                    prior_component_quarantined=prior['entire_component_quarantined_for_protected_signal'],
                    whole_source_screen_qualification=False,screen={'literal_held_stem_ids':[],'lexical_held_stem_ids':[]})
                try:record['screen']=inspect(json.loads(line),matcher)
                except (ValueError,TypeError,KeyError,RecursionError,UnicodeError) as error:
                    record.update(whole_source_screen_qualification=True,qualification=str(error));counts['whole_source_screen_qualification']+=1
                counts['original_rows']+=1
                counts['direct_literal_signals']+=bool(record['screen']['literal_held_stem_ids'])
                counts['direct_lexical_signals']+=bool(record['screen']['lexical_held_stem_ids'])
                records.append(record);ledger.write(json.dumps(record,separators=(',',':'),allow_nan=False)+'\n')
                if (position+1)%1000==0:
                    ledger.flush()
                    if shutil.disk_usage(OUT.parent).free<40*GIB:raise ValueError('Original scratch writer reserve must remain funded')
                    write('progress.json',dict(phase='complete_original_code_and_cached_value_screen',actual_original_rows=position+1,
                        planned_original_rows=227669,counts=dict(counts),elapsed_seconds=time.monotonic()-started))
            if len(records)!=227669 or role_stream.read(1):raise ValueError('Every original row and fixed component is required')
            ledger.flush();os.fsync(ledger.fileno())
        blocked=blocked_components(records);final=OUT/'every-original-source-with-complete-target-screen.jsonl'
        with final.open('x') as ledger:
            for r in records:
                veto=r['fixed_role']['component_sha256'] in blocked;counts['entire_component_quarantined_rows']+=veto
                ledger.write(json.dumps(dict(**r,entire_component_quarantined_after_full_target_screen=veto,
                    optimization_forbidden=veto or r['fixed_role']['source_role']=='reserved',
                    source_role_unchanged=True,data_admitted_to_training=False),separators=(',',':'),allow_nan=False)+'\n')
            ledger.flush();os.fsync(ledger.fileno())
        if any(sha(p)!=h for p,h in files.items()):raise ValueError('An original raw/index/held/runtime byte changed')
        write('complete.json',dict(complete=True,at=datetime.now(timezone.utc).isoformat(),source_files_sha256=files,
            actual_original_rows=len(records),counts=dict(counts),blocked_fixed_components=len(blocked),
            original_component_roles_preserved=True,existing_protected_AST_component_vetoes_retained=True,
            whole_text_screen_ledger_sha256=sha(first),complete_target_role_ledger_sha256=sha(final),
            held_configuration=matcher.configuration,elapsed_seconds=time.monotonic()-started,
            semantic_and_historical_independence_not_certified=True,actual_source_code_executions=0,actual_model_updates=0,
            candidate_pack_not_modified=True,complete_execution_native_readback_source_review_and_admission_still_required=True,
            data_admitted_to_training=False,independent_RL_reward_admitted=False,goal_achieved=False))
    except BaseException as error:
        write('failure.json',dict(error=repr(error),traceback=traceback.format_exc(),partial_outputs_retained=True,
            original_inputs_and_roles_unchanged=True,data_admitted_to_training=False));raise


if __name__=='__main__':main()
