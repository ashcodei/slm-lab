"""Bind all original actual sources to the complete joint checkpoint plan."""
from collections import Counter
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import traceback

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/complete-joint-objective-bridge-CPU-v1'
RESERVE=40*1024**3


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def write(path,value):
    raw=(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
    if len(raw)>32*1024**2 or shutil.disk_usage(ROOT).free-len(raw)<RESERVE:raise ValueError('The full original checkpoint plan must fit above the shared40GiBreserve')
    pending=Path(path).with_name(Path(path).name+'.pending')
    with pending.open('xb') as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())
    pending.replace(path)


def main():
    if (OUT.exists() or os.environ.get('CUDA_VISIBLE_DEVICES')!='' or os.nice(0)<19
            or any(os.environ.get(k)!='1' for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'])
            or shutil.disk_usage(ROOT).free<RESERVE+64*1024**2):raise ValueError('A fresh funded one-thread CUDA-hidden idle source check is required')
    os.sched_setscheduler(0,os.SCHED_IDLE,os.sched_param(0))
    from slm_lab.local_source_closure_v2 import python_sources
    paths=python_sources(ROOT,[Path(__file__),ROOT/'src/slm_lab/complete_joint_objective_bridge_v1.py'])
    files={str(p):sha(p) for p in paths}
    owner=dict(pid=os.getpid(),ticks=Path('/proc/self/stat').read_text().split(') ',1)[1].split()[19])
    OUT.mkdir();write(OUT/'plan.json',dict(owner=owner,at=datetime.now(timezone.utc).isoformat(),source_files_sha256=files,goal_achieved=False))
    try:
        import torch
        from slm_lab.complete_joint_objective_bridge_v1 import checkpoint_entry
        from slm_lab.complete_joint_training_data_v1 import CompleteJointTrainingDataV1
        from slm_lab.model import ModelConfig
        from slm_lab.parallel_function_repair_stage_v1 import value_digest
        torch.set_num_threads(1)
        source=CompleteJointTrainingDataV1(ROOT,ModelConfig.load(ROOT/'configs/model.json'),word_expansion=5)
        entries=[];totals=Counter();families=Counter()
        for step in range(len(source)):
            metadata=source.metadata(step);entry=checkpoint_entry(metadata,source_identity_sha256=source.identity_sha256)
            entries.append(entry)
            for row in entry['objectives']:
                totals['complete_context_tokens']+=row['complete_context_tokens'];totals['response_EOS_targets']+=row['response_EOS_targets']
                totals['whole_CTC_UTF8_bytes']+=row['word_UTF8_bytes'];denominators=row['normalization_denominators']
                for field in ['human_pairs','whole_words','image_pairs','corrective_pairs','negative_EOS']:
                    totals[field]+=denominators.get(field,0)
                families[row['task']+':'+row['source_family']]+=1
            if step%500==0:write(OUT/'progress.json',dict(owner=owner,actual_complete_joint_steps=step+1,totals=dict(totals)))
        expected=dict(complete_context_tokens=91100477,response_EOS_targets=34500081,whole_CTC_UTF8_bytes=644063,
            human_pairs=85786,whole_words=91585,image_pairs=78255,corrective_pairs=4,negative_EOS=80)
        if dict(totals)!=expected or len(entries)!=5362:raise ValueError('Every complete original source/context/target and independent denominator is required')
        source.guard.check(full=True)
        if torch.cuda.is_initialized():raise ValueError('A source-only checkpoint mapping cannot acquire a GPU')
        write(OUT/'joint-checkpoint-entries.json',entries)
        write(OUT/'complete.json',dict(complete=True,owner=owner,at=datetime.now(timezone.utc).isoformat(),
            source_files_sha256={**source.files,**files},actual_original_data_identity_sha256=source.identity_sha256,
            complete_checkpoint_schedule_typed_sha256=value_digest(entries),entries_path=str(OUT/'joint-checkpoint-entries.json'),
            entries_file_sha256=sha(OUT/'joint-checkpoint-entries.json'),actual_complete_joint_steps=len(entries),
            actual_original_totals=dict(totals),actual_objective_family_counts=dict(families),models_constructed=0,
            neural_forwards=0,optimizer_updates=0,GPU_contexts=0,owned_joint_controller_or_bounded_history_or_GPU_training_admitted=False,goal_achieved=False))
    except BaseException as exc:
        write(OUT/'failure.json',dict(owner=owner,error=repr(exc),traceback=traceback.format_exc(),automatic_retry=False,goal_achieved=False));raise


if __name__=='__main__':main()
