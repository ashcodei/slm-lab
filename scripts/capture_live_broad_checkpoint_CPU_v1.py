"""Capture one predeclared complete checkpoint prefix while training continues."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import traceback

from slm_lab.committed_snapshot_copy_v1 import copy_committed
from slm_lab.local_source_closure_v2 import python_sources

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'reports/broad-anchored-code-v7'


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def write(path,value):
    with Path(path).open('x') as stream:stream.write(json.dumps(value,indent=2)+'\n');stream.flush();os.fsync(stream.fileno())


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=Path('/tmp/slm-broad-code-step512-snapshot-CPU-v1'))
    parser.add_argument('--completed',type=int,default=512);args=parser.parse_args();out=args.output.absolute()
    if (out.exists() or out.resolve()!=out or os.environ.get('CUDA_VISIBLE_DEVICES')!='' or os.nice(0)!=19
            or 'torch' in __import__('sys').modules or any(os.environ.get(k)!='1' for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'])):
        raise ValueError('Fresh CUDA-hidden idle one-thread non-neural snapshot capture required')
    os.sched_setscheduler(0,os.SCHED_IDLE,os.sched_param(0))
    plan_path=SOURCE/'plan.json';raw_plan=plan_path.read_bytes();plan=json.loads(raw_plan)
    pointer_path=SOURCE/'train-anchor_CE-resume-distributed-state.json';pointer_bytes=pointer_path.read_bytes();pointer=json.loads(pointer_bytes)
    states=pointer['all_four_private_states']
    if (plan['planned_steps']!=3721 or pointer.get('completed')!=args.completed or not 0<args.completed<3721
            or len(states)!=4 or [r['rank'] for r in states]!=list(range(4))
            or any(r['actual_completed']!=args.completed for r in states)
            or len({(r['external_journal_prefix_bytes'],r['external_journal_prefix_sha256']) for r in states})!=1):
        raise ValueError('The exact predeclared incomplete checkpoint needs all four committed private states and one full journal prefix')
    files={str(p):sha(p) for p in python_sources(ROOT,[Path(__file__)])}
    def checked():
        if shutil.disk_usage(out.parent).free<42*1024**3 or any(sha(p)!=h for p,h in files.items()):
            raise ValueError('The full2GiBcapture budget,40GiBreserve and every capture source byte must remain')
    checked();out.mkdir();owner=dict(pid=os.getpid(),ticks=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19])
    write(out/'owner-plan.json',dict(owner=owner,at=datetime.now(timezone.utc).isoformat(),source_files_sha256=files,
        original_training_owner=plan['owner'],predeclared_arm='anchor_CE',predeclared_completed=args.completed,
        full_original_epoch_updates=3721,source_experiment_is_live_and_incomplete=True,
        producer_files_not_bound_as_immutable_and_never_modified=True,models_constructed=0,GPU_contexts=0,goal_achieved=False))
    captured=[]
    try:
        (out/'plan.json').write_bytes(raw_plan);(out/'original-commit-receipt.json').write_bytes(pointer_bytes)
        for row in states:
            captured.append(copy_committed(row['path'],out/f'rank-{row["rank"]}.pt',expected_sha256=row['sha256'],
                maximum_bytes=384*1024**2,checked=checked))
        first=states[0]
        captured.append(copy_committed(SOURCE/'anchor_CE-metrics.jsonl',out/'committed-journal-prefix.jsonl',
            expected_sha256=first['external_journal_prefix_sha256'],prefix_bytes=first['external_journal_prefix_bytes'],
            maximum_bytes=512*1024**2,checked=checked))
        initial=plan['starting_policy_inputs']
        captured.append(copy_committed(initial['checkpoint'],out/'starting-human-DPO-actor.pt',
            expected_sha256=initial['checkpoint_sha256'],maximum_bytes=128*1024**2,checked=checked))
        package=plan['tokenizer_package']
        if not isinstance(package,dict) or package.get('sha256')!=plan['tokenizer_sha256']:
            raise ValueError('The original packaged tokenizer and its independently recorded SHA must agree')
        token=Path(package['path'])
        captured.append(copy_committed(token,out/'tokenizer.json',expected_sha256=plan['tokenizer_sha256'],maximum_bytes=16*1024**2,checked=checked))
        counts=0
        with (out/'committed-journal-prefix.jsonl').open('rb') as stream:
            for counts,line in enumerate(stream,1):
                if not line.endswith(b'\n') or json.loads(line).get('step')!=counts:raise ValueError('Every original complete committed update must be present in order')
        if counts!=args.completed:raise ValueError('The complete captured journal prefix must exactly match the private checkpoint')
        immutable={str(p):sha(p) for p in out.iterdir() if p.is_file()};immutable.update(files)
        write(out/'complete.json',dict(complete=True,owner=owner,source_files_sha256=immutable,completed=args.completed,
            original_planned_updates=3721,arm='anchor_CE',actual_captured_artifacts=captured,
            source_checkpoint_is_interim_not_completed_epoch=True,original_mutable_files_not_bound_or_changed=True,
            no_model_or_GPU_work=True,no_optimizer_or_RNG_restored=True,model_quality_proved=False,goal_achieved=False))
        print(json.dumps(dict(complete=True,completed=args.completed,captured_bytes=sum(r['captured_bytes'] for r in captured))),flush=True)
    except BaseException as exc:
        write(out/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc(),
            only_own_unadmitted_capture_files_retained=True,producer_files_unchanged=True,goal_achieved=False));raise


if __name__=='__main__':main()
