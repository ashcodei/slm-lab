"""Index every original function and fix roles before any fresh source replay.

Full reference bodies, whole descriptions, origin repositories and protected
held text/body signals are screened without executing code or training.
"""
from collections import Counter,defaultdict
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import traceback

ROOT=Path(__file__).resolve().parents[1]
RAW=Path('/tmp/slm-CodeIO-raw-candidate-v1')
HELD=Path('/tmp/slm-public-reasoned-code-held-registry-v1')
OUT=Path('/tmp/slm-CodeIO-whole-source-index-v1')
GIB=1024**3


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def write(name,value):
    p=OUT/name;q=p.with_suffix(p.suffix+'.pending')
    with q.open('x') as f:f.write(json.dumps(value,indent=2,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
    os.replace(q,p)


def main():
    if OUT.exists() or os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('Fresh CPU-only full-source index required')
    os.nice(10);os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
    from slm_lab.codeio_source_roles_v1 import assign,verify,digest
    from slm_lab.held_text_overlap_v1 import HeldTextOverlap,normalize
    from slm_lab.public_synthetic_code_screen_policy_v1 import body_fingerprint
    from slm_lab.local_source_closure_v2 import python_sources
    from slm_lab.code_tasks import TASKS
    source=json.loads((RAW/'complete.json').read_text());path=Path(source['original_source_file'])
    held=json.loads((HELD/'complete.json').read_text())
    if not source['complete'] or source['original_rows']!=227669 or sha(path)!=source['original_source_sha256']:
        raise ValueError('The complete pinned publisher function file must remain exact')
    if not held['complete'] or sha(HELD/'held-question-registry.json')!=held['held_registry_sha256']:
        raise ValueError('The complete original held-question exclusion registry must remain exact')
    tests=ROOT/'reports/codeio-source-roles-v1-tests.log'
    if '2 passed' not in tests.read_text():raise ValueError('The actual transitive-component and invalid-source controls must pass')
    inputs={str(p):sha(p) for p in [*python_sources(ROOT,[Path(__file__).resolve()]),tests,
        ROOT/'tests/test_codeio_source_roles_v1.py',RAW/'complete.json',path,HELD/'complete.json',HELD/'held-question-registry.json',
        ROOT/'data/verified-code-source-cache-v1/sanitized-mbpp.json',ROOT/'data/verified-code-expanded-v1/reserved-mbpp.jsonl']}
    if shutil.disk_usage(OUT.parent).free<44*GIB:raise ValueError('Full source index must fit above the unchanged40GiB writer floor')
    OUT.mkdir();write('plan.json',dict(owner=dict(pid=os.getpid(),birth=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19]),
        at=datetime.now(timezone.utc).isoformat(),source_files_sha256=inputs,original_rows=227669,
        roles_fixed_before_execution=True,whole_repository_question_and_body_components=True,
        code_execution_permitted=False,GPU_usage_permitted=False,data_admitted_to_training=False,model_updates=0,goal_achieved=False))
    try:
        protected=defaultdict(list)
        def protect(code,identity):
            try:protected[body_fingerprint(code)].append(identity)
            except (SyntaxError,ValueError,RecursionError):pass
        for row in json.loads((ROOT/'data/verified-code-source-cache-v1/sanitized-mbpp.json').read_text()):
            if 11<=row['task_id']<=510:protect(row['code'],'sanitized-MBPP:'+str(row['task_id']))
        for line in (ROOT/'data/verified-code-expanded-v1/reserved-mbpp.jsonl').open():
            row=json.loads(line);protect(row['code'],'reserved-MBPP:'+str(row['task_id']))
        for name,args,spec,scaling,py,js in TASKS:protect(py,'local-Python:'+name)
        overlap=HeldTextOverlap(json.loads((HELD/'held-question-registry.json').read_text()))
        records=[];counts=Counter();veto=set()
        with path.open('rb') as stream,(OUT/'every-original-source-before-roles.jsonl').open('x') as ledger:
            for position,line in enumerate(stream):
                r=json.loads(line);raw_sha=hashlib.sha256(line).hexdigest();qual=[]
                code=r.get('refcode');description=r.get('problem_description');requirements=r.get('io_requirements');meta=r.get('meta')
                if not all(isinstance(v,str) for v in [code,description,requirements,r.get('funcname')]):
                    qual.append('original_nontext_source_field');code=code if isinstance(code,str) else '';description=description if isinstance(description,str) else '';requirements=requirements if isinstance(requirements,str) else ''
                text=description+'\n'+requirements
                try:body=body_fingerprint(code);kind='whole_docstring_free_AST'
                except (SyntaxError,ValueError,RecursionError):body=digest(['unparseable_original_bytes',code]);kind='raw_unparseable';qual.append('whole_AST_qualification')
                repo=meta.get('repo_name') if isinstance(meta,dict) else None
                repo_key=['original_repository',repo.strip().casefold().rstrip('/')] if isinstance(repo,str) and repo.strip() else ['missing_repository',raw_sha]
                if not isinstance(repo,str) or not repo.strip():qual.append('origin_repository_missing')
                literal=overlap.literal(text);lexical=overlap.lexical_copies(text);ast_matches=protected.get(body,[])
                if literal or lexical or ast_matches:veto.add(position)
                ios=r.get('ios')
                if not isinstance(ios,list) or not ios:qual.append('no_original_cached_IO')
                try:json.dumps(ios,allow_nan=False)
                except (ValueError,TypeError,RecursionError):qual.append('nonfinite_or_nonJSON_original_IO')
                record=dict(original_position=position,original_row_sha256=raw_sha,original_code_utf8_sha256=hashlib.sha256(code.encode()).hexdigest(),
                    repository_group_sha256=digest(repo_key),question_group_sha256=digest(normalize(text)),whole_program_group_sha256=body,
                    body_group_kind=kind,cached_case_count=len(ios) if isinstance(ios,list) else 0,source_metadata_qualifications=qual,
                    literal_held_stem_ids=sorted(literal),lexical_held_stem_ids=sorted(lexical),protected_exact_body_ids=ast_matches)
                records.append(record);ledger.write(json.dumps(record,separators=(',',':'))+'\n')
                counts.update(qual);counts['original_rows']+=1;counts['with_any_protected_signal']+=bool(literal or lexical or ast_matches)
                if (position+1)%2000==0:ledger.flush();write('progress.json',dict(phase='complete_source_and_held_screen',original_rows=position+1,planned=227669))
            ledger.flush();os.fsync(ledger.fileno())
        if len(records)!=227669:raise ValueError('Every original publisher source row is required')
        roles=assign(records);summary=verify(records,roles);blocked={roles[p]['component_sha256'] for p in veto}
        with (OUT/'every-original-source-with-fixed-role.jsonl').open('x') as ledger:
            for r in records:
                role=roles[r['original_position']];quarantine=role['component_sha256'] in blocked
                ledger.write(json.dumps(dict(**r,fixed_role=role,entire_component_quarantined_for_protected_signal=quarantine,
                    optimization_forbidden=role['optimization_forbidden'] or quarantine,data_admitted_to_training=False),separators=(',',':'))+'\n')
                counts['quarantined_original_rows']+=quarantine
            ledger.flush();os.fsync(ledger.fileno())
        if any(sha(p)!=h for p,h in inputs.items()):raise ValueError('An original input or runtime source byte changed')
        write('complete.json',dict(complete=True,source_files_sha256=inputs,original_source_file=str(path),original_source_sha256=source['original_source_sha256'],
            complete_original_rows=len(records),fixed_role_summary=summary,counts=dict(counts),held_configuration=overlap.configuration,
            all_original_source_decisions_retained=True,roles_fixed_before_fresh_replay=True,
            final_role_ledger_sha256=sha(OUT/'every-original-source-with-fixed-role.jsonl'),quarantined_components=len(blocked),
            program_and_description_semantics_and_historical_independence_not_certified=True,
            original_license_attribution_retained_not_reinterpreted=True,source_execution_or_native_pack_or_training_admitted=False,
            new_source_code_executions=0,new_model_updates=0,goal_achieved=False))
    except BaseException as error:
        write('failure.json',dict(error=repr(error),traceback=traceback.format_exc(),partial_outputs_retained=True,data_admitted_to_training=False));raise


if __name__=='__main__':main()
