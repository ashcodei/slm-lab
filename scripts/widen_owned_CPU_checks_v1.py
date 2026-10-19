"""Widen only the two exact owned CPU checks; retain all numerical settings.

Owner plans describe their original placement and remain byte-identical. A
separate receipt records the runtime change, including in-flight work. This
does not establish numerical/model quality or a matched throughput speedup.
"""
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
JOBS=[dict(pid=302803,birth='62163023',entrypoint=ROOT/'scripts/evaluate_broad_interim_code_CPU_v1.py',
           owner_plan=Path('/tmp/slm-broad-code-step512-code-evaluation-CPU-v1/owner-plan.json')),
      dict(pid=303394,birth='62237688',entrypoint=ROOT/'scripts/probe_compact_future_order_full_CPU_v1.py',
           owner_plan=Path('/tmp/slm-compact-future-order-full-CPU-v1/owner-plan.json'))]
PROTECTED={293164,293165,293166,293167,18545,101711}


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def identity(pid):
    raw=(Path('/proc')/str(pid)/'stat').read_text().rsplit(')',1)[1].split()
    if raw[0] in ('Z','X'):raise ValueError('A terminal process cannot be repositioned')
    return raw[19]


def inspect(spec,mask,*,protected=PROTECTED):
    pid=spec['pid'];process=Path('/proc')/str(pid);owner_path=Path(spec['owner_plan']);owner=json.loads(owner_path.read_text());entry=Path(spec['entrypoint']).resolve()
    who=owner['owner']
    if (pid in protected or pid==os.getpid() or who['pid']!=pid or str(who.get('birth',who.get('ticks')))!=spec['birth']
            or identity(pid)!=spec['birth'] or not mask or not set(mask)<=os.sched_getaffinity(0)
            or (process/'cgroup').read_text()!=Path('/proc/self/cgroup').read_text()
            or owner['source_files_sha256'].get(str(entry))!=sha(entry)):
        raise ValueError('Only an exact owned CPU job, immutable entrypoint and approved instance CPUs may be repositioned')
    cwd=(process/'cwd').resolve();arguments=(process/'cmdline').read_bytes().split(b'\0')
    if not any((cwd/os.fsdecode(arg)).resolve()==entry for arg in arguments if arg.endswith(b'.py')):
        raise ValueError('The actual running entrypoint must match the immutable owner plan')
    for fd in (process/'fd').iterdir():
        try:target=os.readlink(fd)
        except FileNotFoundError:continue
        if 'nvidia' in target:raise ValueError('A process with GPU device descriptors cannot be repositioned here')
    threads=[]
    for path in (process/'task').iterdir():
        tid=int(path.name)
        try:
            birth=identity(tid);policy=os.sched_getscheduler(tid);nice=os.getpriority(os.PRIO_PROCESS,tid);affinity=sorted(os.sched_getaffinity(tid))
        except FileNotFoundError:continue
        if policy!=os.SCHED_IDLE or nice!=19 or not set(affinity)<=set(mask):
            raise ValueError('Every eligible owned thread must already be nice19/SCHED_IDLE inside the approved CPU mask')
        threads.append(dict(tid=tid,birth=birth,policy=policy,nice=nice,affinity=affinity))
    if not threads or identity(pid)!=spec['birth']:raise ValueError('Exact owned process and threads must remain live')
    return dict(pid=pid,birth=spec['birth'],owner_plan=str(owner_path),owner_plan_sha256=sha(owner_path),
                entrypoint=str(entry),entrypoint_sha256=sha(entry),threads=threads)


def widen(spec,mask,*,protected=PROTECTED):
    before=inspect(spec,mask,protected=protected)
    for _ in range(4):
        current=inspect(spec,mask,protected=protected)
        for thread in current['threads']:
            if identity(spec['pid'])!=spec['birth'] or identity(thread['tid'])!=thread['birth']:
                raise ValueError('Owned process/thread birth changed before its affinity syscall')
            os.sched_setaffinity(thread['tid'],mask)
        after=inspect(spec,mask,protected=protected)
        if all(set(t['affinity'])==set(mask) for t in after['threads']):break
    else:raise ValueError('Every exact owned thread must receive the same approved CPU placement')
    if (before['owner_plan_sha256']!=after['owner_plan_sha256'] or before['entrypoint_sha256']!=after['entrypoint_sha256']):
        raise ValueError('A running owner plan or source changed; preserve the partial placement receipt')
    return dict(before=before,after=after,numerical_threads_precision_weights_RNG_data_and_budgets_not_modified_by_tool=True,
        priority_and_policy_unchanged=True,only_kernel_CPU_eligibility_widened=True)


def write(path,value):
    with Path(path).open('x') as stream:stream.write(json.dumps(value,indent=2)+'\n');stream.flush();os.fsync(stream.fileno())


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or 'torch' in __import__('sys').modules:
        raise ValueError('The placement manager is CUDA-hidden and non-neural')
    output=ROOT/'reports/owned-CPU-affinity-placement-v1';mask=sorted(os.sched_getaffinity(0))
    if output.exists() or len(mask)<2:raise ValueError('A fresh receipt and multiple approved CPUs are required')
    # Preflight both exact targets before touching either. All original GPU
    # and older repair workers are explicitly excluded, independent of argv.
    previews=[inspect(spec,mask) for spec in JOBS];output.mkdir()
    files={str(Path(__file__).resolve()):sha(__file__)}
    files.update({row['owner_plan']:row['owner_plan_sha256'] for row in previews})
    write(output/'plan.json',dict(at=datetime.now(timezone.utc).isoformat(),source_files_sha256=files,approved_CPUs=mask,targets=previews,
        protected_original_PIDs=sorted(PROTECTED),in_flight_first_generation_or_control_step_may_span_placement_change=True,
        shared_host_timing_not_a_matched_speedup_measurement=True,goal_achieved=False))
    actual=[]
    try:
        for i,spec in enumerate(JOBS):
            actual.append(widen(spec,mask));write(output/f'job-{i}.json',actual[-1])
        write(output/'complete.json',dict(complete=True,at=datetime.now(timezone.utc).isoformat(),source_files_sha256=files,jobs=actual,
            approved_CPUs=mask,original_workers_sources_and_complete_assessment_scope_unchanged=True,
            first_generation_or_control_timing_may_span_changed_placement=True,no_measured_speedup_or_quality_gain_claim=True,goal_achieved=False))
        print(json.dumps(dict(complete=True,approved_CPUs=mask,owned_PIDs=[r['after']['pid'] for r in actual])))
    except BaseException as error:
        write(output/'failure.json',dict(error=repr(error),completed_placement_records=actual,original_workers_not_targeted=True,
            no_automatic_restart_or_retry=True,goal_achieved=False));raise


if __name__=='__main__':main()
