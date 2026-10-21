"""Resume whole CodeI/O case qualification from persistent original-source blocks.

All227669 original rows retain fixed roles and complete-target quarantines.
Only complete8-source blocks are committed; a crash can repeat the current
uncommitted block but never silently discard sources or relabel outcomes.
"""
from datetime import datetime,timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/CodeIO-durable-case-qualification-v1'
ROLE_RECOVERY=ROOT/'reports/CodeIO-role-recovery-20261021'
RAW=Path('/tmp/slm-CodeIO-raw-candidate-v1/0_368500_filtered_v2_ds25.sced.jsonl')
ROLES=Path('/tmp/slm-CodeIO-whole-source-index-v1/every-original-source-with-fixed-role.jsonl')
SCREEN=Path('/tmp/slm-CodeIO-whole-target-screen-v2/every-original-source-with-complete-target-screen.jsonl')
EXPECTED_RAW='e6f0b1d1db181764a5d6ca6d2dcf8b641b63c632468672d1abfc24837700cca4'
EXPECTED_ROLES='f88045b7e99094ecf4f6722917ee7f3d2686f21d3ca24a4e441225269aa05cca'
EXPECTED_SCREEN='8116c1acd65eb296e1c70c871ef11771a0e0c373e84e848e17215e8cb789fef6'
GIB=1024**3


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def write(path,value):
    temporary=path.with_suffix(path.suffix+'.pending')
    with temporary.open('w') as stream:
        json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n');stream.flush();os.fsync(stream.fileno())
    os.replace(temporary,path)


def actor(pid):
    try:fields=Path('/proc',str(pid),'stat').read_text().rsplit(')',1)[1].split()
    except FileNotFoundError:return None
    return None if fields[0] in ['Z','X'] else dict(pid=pid,birth=fields[19])


def wait_ready(path,owner,attempt):
    while not path.exists():
        if actor(owner['pid'])!=owner:raise ValueError('The exact source-recovery owner ended without its complete original ledger')
        write(attempt/'progress.json',dict(phase='wait_complete_original_source_ledger',required=str(path),exact_predecessor=owner))
        time.sleep(10)


def prefix(attempt,number):
    work=Path('/tmp')/f'slm-CodeIO-durable-prefix-attempt-{number}'
    if work.exists():raise ValueError('Preserve existing prefix observations; use a fresh owned attempt')
    work.mkdir()
    fresh=work/'fresh';whole=work/'whole'
    commands=[('fresh-prefix',
        'import sys;from pathlib import Path;sys.path.insert(0,'+repr(str(ROOT/'scripts'))+');'
        'import qualify_CodeIO_prefix_cached_cases_CPU_v2 as p;p.OUT=Path('+repr(str(fresh))+');p.main()'),
        ('whole-prefix',
        'import sys;from pathlib import Path;sys.path.insert(0,'+repr(str(ROOT/'scripts'))+');'
        'import qualify_CodeIO_whole_cached_cases_CPU_v3 as p;'
        'p.ORIGINAL=Path('+repr(str(fresh))+');p.PREFIX=Path('+repr(str(whole))+');p.main("prefix")')]
    for label,code in commands:
        with (attempt/(label+'.log')).open('x') as log:
            process=subprocess.Popen([sys.executable,'-u','-c',code],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
                env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'})
            write(attempt/'prefix-child.json',dict(actor=actor(process.pid),phase=label))
            if process.wait()!=0:raise ValueError('Original complete prefix comparison failed; preserve observations')
    result=json.loads((whole/'complete.json').read_text())
    fresh_result=json.loads((fresh/'complete.json').read_text())
    if (not result['complete'] or result['actual_original_sources']!=32
            or result['fresh_context_controls_match_original_v1'] is not True
            or fresh_result['counts'].get('all_original_cached_values_match_two_local_seed_contexts')!=28
            or result['counts'].get('all_original_cached_values_match_two_local_seed_contexts')!=28):
        raise ValueError('All original32 fresh/whole dispositions and28complete positives must match before the complete corpus')
    saved={}
    for source,label in [(fresh,'fresh-prefix'),(whole,'whole-prefix')]:
        destination=attempt/label;destination.mkdir()
        for path in source.iterdir():
            if not path.is_file():raise ValueError('Only original complete prefix observation files expected')
            target=destination/path.name;shutil.copyfile(path,target)
            if sha(path)!=sha(target):raise ValueError('Every original complete prefix observation must be retained')
            saved[str(target)]=sha(target)
    write(attempt/'prefix-complete.json',dict(complete=True,actual_original_sources=32,
        whole_context_matches_every_original_fresh_context_disposition=True,positive_sources=28,
        all_original_cases_and_both_seed_contexts_preserved=True,retained_original_files_sha256=saved,
        source_code_execution_is_confined_stdlib_only=True,data_admitted_to_training=False))


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or 'torch' in sys.modules:
        raise ValueError('A GPU-hidden data-only qualification process is required')
    from slm_lab.codeio_case_segments_v1 import Segments
    from slm_lab.codeio_whole_qualification_v3 import qualify
    from slm_lab.local_source_closure_v2 import python_sources
    checks=[(ROOT/'reports/CodeIO-post-reset-case-checks-20261021.log','5 passed'),
        (ROOT/'reports/codeio-case-segments-v1-tests.log','2 passed'),
        (ROOT/'reports/codeio-whole-qualification-v3-tests.log','2 passed')]
    if any(expected not in path.read_text() for path,expected in checks):
        raise ValueError('Actual confined/fresh/whole/resume/veto controls must pass')
    scripts=[Path(__file__).resolve(),ROOT/'scripts/qualify_CodeIO_prefix_cached_cases_CPU_v2.py',
        ROOT/'scripts/qualify_CodeIO_whole_cached_cases_CPU_v3.py']
    implementations={str(p):sha(p) for p in python_sources(ROOT,scripts)}
    fingerprint=dict(Python=platform.python_version(),interpreter_sha256=sha(Path(sys.executable).resolve()),
        libseccomp_sha256=sha(Path('/lib/x86_64-linux-gnu/libseccomp.so.2').resolve()),
        machine=platform.machine(),kernel=platform.release())
    bindings=dict(raw_source_sha256=EXPECTED_RAW,fixed_role_ledger_sha256=EXPECTED_ROLES,
        complete_target_veto_ledger_sha256=EXPECTED_SCREEN,implementation_sources_sha256=implementations,
        execution_runtime=fingerprint,seeds=[0,1],whole_source_wall_seconds_per_seed=20,
        source_process_memory_bytes=768*1024**2,source_case_context='fresh module namespace per case; shared confined interpreter per source/seed',
        every_original_case_required=True,whole_no_OP_control_required=True,expected_outputs_never_sent_to_children=True,
        no_training_or_RL_oracle_admission=True)
    OUT.mkdir(exist_ok=True)
    with (OUT/'owner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        attempts=OUT/'attempts';attempts.mkdir(exist_ok=True);number=len(list(attempts.iterdir()))+1
        attempt=attempts/str(number);attempt.mkdir()
        source_plan=json.loads((ROLE_RECOVERY/'plan.json').read_text());predecessor=actor(source_plan['owner_pid'])
        write(attempt/'plan.json',dict(at=datetime.now(timezone.utc).isoformat(),owner=actor(os.getpid()),
            exact_role_recovery_predecessor=predecessor,bindings=bindings,full_original_population=227669,
            persistent_segment_width=8,persistent_segment_budget_bytes=512*1024**2,
            original_uncommitted_block_may_repeat_after_interruption=True,no_GPU_or_model_updates=True,
            data_admitted_to_training=False,goal_achieved=False))
        if not ROLES.exists() or not (ROLES.parent/'complete.json').exists():
            if predecessor is None:raise ValueError('A live original source-recovery owner is required while waiting')
            wait_ready(ROLES.parent/'complete.json',predecessor,attempt)
        prefix(attempt,number)
        if not (ROLE_RECOVERY/'complete.json').exists():
            if predecessor is None:raise ValueError('A live original full-target recovery owner is required while waiting')
            wait_ready(ROLE_RECOVERY/'complete.json',predecessor,attempt)
        recovered=json.loads((ROLE_RECOVERY/'complete.json').read_text())
        if (not recovered['complete'] or recovered['exact_original_role_ledger_sha256']!=EXPECTED_ROLES
                or recovered['exact_original_complete_target_ledger_sha256']!=EXPECTED_SCREEN
                or any(sha(p)!=h for p,h in [(RAW,EXPECTED_RAW),(ROLES,EXPECTED_ROLES),(SCREEN,EXPECTED_SCREEN)])):
            raise ValueError('All complete original raw, source roles and full-target veto bytes must be exact')
        os.nice(10);os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
        writer=Segments(OUT/'segments',bindings=bindings,total=227669,width=8,
            maximum_total_bytes=512*1024**2,maximum_segment_bytes=128*1024**2)
        prior=iter(writer.rows());resume=writer.next_position;buffer=[];started=time.monotonic()
        write(attempt/'resume.json',dict(verified_original_closed_sources=resume,
            every_closed_original_source_revalidated_against_current_exact_raw_role_and_veto_bytes=True))
        with RAW.open('rb') as raw,ROLES.open() as roles,SCREEN.open() as screened:
            for position in range(227669):
                line=next(raw);original=json.loads(next(roles));veto=json.loads(next(screened))
                if (original['original_position']!=position or veto['original_position']!=position
                        or hashlib.sha256(line).hexdigest()!=original['original_row_sha256']
                        or veto['original_row_sha256']!=original['original_row_sha256']
                        or veto['fixed_role']!=original['fixed_role']):
                    raise ValueError('Every exact original complete source and role/veto join must reconstruct')
                if position<resume:
                    row=next(prior)
                    if (row['original_position']!=position or row['original_row_sha256']!=original['original_row_sha256']
                            or row['fixed_role']!=original['fixed_role']
                            or row['source_optimization_forbidden']!=veto['optimization_forbidden']
                            or row['whole_component_quarantined']!=veto['entire_component_quarantined_after_full_target_screen']):
                        raise ValueError('Every previously closed source observation must retain its exact original input/role/veto')
                    continue
                buffer.append(qualify(line,original,veto))
                if len(buffer)==8 or position+1==227669:
                    writer.append(buffer);buffer=[]
                    write(attempt/'progress.json',dict(phase='complete_original_confined_case_qualification',
                        actual_durable_original_sources=writer.next_position,planned_original_sources=227669,
                        complete_dispositions=writer.counts,committed_allocated_bytes=writer.committed_bytes,
                        elapsed_seconds=time.monotonic()-started,new_model_updates=0,data_admitted_to_training=False))
            if raw.read(1) or roles.read(1) or screened.read(1):raise ValueError('No original source may be omitted from the full qualification population')
        if writer.next_position!=227669 or any(sha(p)!=h for p,h in implementations.items()):
            raise ValueError('The full unchanged original source/runtime population is required')
        write(OUT/'complete.json',dict(complete=True,at=datetime.now(timezone.utc).isoformat(),actual_original_sources=227669,
            last_segment_sha256=writer.tip,complete_dispositions=writer.counts,committed_allocated_bytes=writer.committed_bytes,
            every_original_source_and_case_preserved_or_whole_source_qualification=True,
            full_native_readback_source_review_and_training_admission_still_required=True,
            independent_semantic_correctness_not_certified=True,data_admitted_to_training=False,RL_reward_admitted=False,
            actual_model_updates=0,goal_achieved=False))


if __name__=='__main__':
    try:main()
    except BaseException:
        traceback.print_exc();raise
