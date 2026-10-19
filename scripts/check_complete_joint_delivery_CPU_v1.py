"""Check the full actual joint metadata and unchanged encoder delivery.

Every5362step reference/budget is rebuilt. Actual pixel delivery is rechecked
for all new corrective negatives, each family maximum and the true empty
word tail; the existing complete302627view source admission remains required.
This process constructs zero neural models and starts no GPU work.
"""
from collections import Counter
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import traceback

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/complete-joint-delivery-CPU-v1'
RESERVE=40*1024**3


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def write(path,value):
    raw=(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
    if len(raw)>16*1024**2 or shutil.disk_usage(ROOT).free-len(raw)<RESERVE:raise ValueError('Complete metadata must fit above the original40GiBreserve')
    pending=Path(path).with_name(Path(path).name+'.pending')
    with pending.open('xb') as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())
    pending.replace(path)


def main():
    if (OUT.exists() or os.environ.get('CUDA_VISIBLE_DEVICES')!='' or os.nice(0)<19
            or any(os.environ.get(k)!='1' for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'])
            or shutil.disk_usage(ROOT).free<RESERVE+32*1024**2):raise ValueError('Fresh one-thread CUDA-hidden nice19 data check required')
    os.sched_setscheduler(0,os.SCHED_IDLE,os.sched_param(0))
    from slm_lab.local_source_closure_v2 import python_sources
    from slm_lab.complete_joint_source_schedule_v1 import digest
    owner=dict(pid=os.getpid(),ticks=Path('/proc/self/stat').read_text().split(') ',1)[1].split()[19])
    paths=python_sources(ROOT,[Path(__file__),ROOT/'src/slm_lab/complete_joint_training_data_v1.py'])
    files={str(p):sha(p) for p in paths};OUT.mkdir()
    write(OUT/'plan.json',dict(owner=owner,at=datetime.now(timezone.utc).isoformat(),source_files_sha256=files,
        models_constructed=0,neural_forwards=0,GPU_contexts=0,goal_achieved=False))
    try:
        import torch
        from slm_lab.complete_joint_training_data_v1 import CompleteJointTrainingDataV1
        from slm_lab.model import ModelConfig
        from slm_lab.full_visual_source_admission_v1 import FAMILIES
        torch.set_num_threads(1)
        source=CompleteJointTrainingDataV1(ROOT,ModelConfig.load(ROOT/'configs/model.json'),word_expansion=5)
        counts=Counter();visual_steps={};observations=[]
        for step in range(len(source)):
            row=source.metadata(step)
            for name,part in row['components'].items():
                entry=part['entry'];counts[name+'_steps']+=1;counts[name+'_presentations']+=len(entry['original_positions'])
                counts[name+'_context_tokens']+=entry['complete_context_tokens'];counts[name+'_response_EOS_targets']+=entry['response_EOS_targets']
                if name=='visual':visual_steps[part['reference']['source_step']]=row['components']['visual']
            if step%500==0:write(OUT/'progress.json',dict(owner=owner,phase='all complete actual source headers',actual_joint_steps=step+1,counts=dict(counts)))
        selected=set()
        for family in FAMILIES:
            rows=[e for e in source.streams['visual']['entries'] if e['family']==family]
            selected.add(max(rows,key=lambda e:(e['complete_context_tokens'],e['source_step']))['source_step'])
            selected.add(max(rows,key=lambda e:(e['response_EOS_targets'],e['source_step']))['source_step'])
        for e in source.streams['visual']['entries']:
            if e['complete_original_correction_indices'] or 0 in e['rank_source_counts']:selected.add(e['source_step'])
        negatives=negative_targets=empty_ranks=delivered=0;seen_corrections=set()
        for index,step in enumerate(sorted(selected),1):
            component=visual_steps[step];entry=component['entry']
            for rank in range(4):
                packet=source.prepare_visual(component,rank=rank,world=4,device='cpu')
                actual=packet['source'];positions=actual['positions'][rank::4]
                if actual['positions']!=entry['original_positions']:raise ValueError('An original rank source list changed')
                if not positions:
                    if packet['native'] is not None or packet['word'] is not None or packet['matching'] is not None or packet['negatives']:
                        raise ValueError('A true empty task rank cannot fabricate a source')
                    empty_ranks+=1
                else:
                    native=packet['native']
                    if int(native['attention_mask'].sum())!=sum(actual['context_lengths'][rank::4]) or int((native['labels'][:,1:]!=-100).sum())!=sum(actual['response_EOS_targets'][rank::4]):
                        raise ValueError('Every original local context and EOS target must reach delivery')
                    delivered+=len(positions)
                    if packet['word'] is not None:
                        texts=packet['word']['texts']
                        if [len(t.encode()) for t in texts]!=actual['UTF8_bytes'][rank::4]:raise ValueError('Whole CTC bytes changed')
                    for slot,negative in packet['negatives'].items():
                        positive=native['labels'][slot];start=int((positive!=-100).nonzero()[0]);ids=negative['input_ids'][0];labels=negative['labels'][0]
                        if (not torch.equal(ids[:start],native['input_ids'][slot,:start]) or int(ids[-1])!=2
                                or not torch.equal(labels[start:],ids[start:]) or not bool((labels[:start]==-100).all())
                                or not torch.equal(negative['pixel_values'],native['pixel_values'][slot:slot+1])):raise ValueError('The full corrective pair prompt/pixels/negative/EOS changed')
                        negatives+=1;negative_targets+=int((labels[1:]!=-100).sum())
                seen_corrections.update(entry['complete_original_correction_indices'])
                observations.append(dict(original_visual_step=step,rank=rank,kind=packet['kind'],original_positions=positions,
                    native_response_EOS_targets=sum(actual['response_EOS_targets'][rank::4]),negative_response_EOS_targets=sum(actual['negative_response_EOS_targets'][rank::4])))
            write(OUT/'progress.json',dict(owner=owner,phase='actual complete encoder/negative/extreme/tail delivery',completed_extreme_batches=index,total_extreme_batches=len(selected)))
        if negatives!=4 or negative_targets!=80 or seen_corrections!=set(range(4)) or empty_ranks!=3:raise ValueError('All four complete negative pairs/80targets and real empty tails must survive')
        expected=dict(code_steps=5256,code_presentations=84084,code_context_tokens=42271575,code_response_EOS_targets=21316734,
            visual_steps=4733,visual_presentations=302627,visual_context_tokens=17681556,visual_response_EOS_targets=4669243,
            human_steps=5362,human_presentations=85786,human_context_tokens=31147346,human_response_EOS_targets=8514104)
        if dict(counts)!=expected:raise ValueError('The entire original joint source/target budgets are required')
        source.guard.check(full=True)
        if torch.cuda.is_initialized():raise ValueError('A source-only delivery check cannot acquire a CUDA context')
        write(OUT/'actual-delivery-observations.json',observations)
        write(OUT/'complete.json',dict(complete=True,owner=owner,at=datetime.now(timezone.utc).isoformat(),source_files_sha256={**source.files,**files},
            data_identity_sha256=source.identity_sha256,all5362complete_source_entries_rebuilt=True,actual_global_counts=dict(counts),
            actual_extreme_global_visual_batches=len(selected),actual_fresh_visual_presentations=delivered,actual_real_empty_task_ranks=empty_ranks,
            actual_complete_correction_pairs=negatives,actual_complete_negative_EOS_targets=negative_targets,
            observations_sha256=sha(OUT/'actual-delivery-observations.json'),original_full302627pixel_encoder_admission_reused=True,
            only_named_extremes_and_all_new_negatives_redecoded=True,models_constructed=0,neural_forwards=0,GPU_contexts=0,
            production_joint_training_admitted=False,goal_achieved=False))
    except BaseException as exc:write(OUT/'failure.json',dict(owner=owner,error=repr(exc),traceback=traceback.format_exc(),automatic_retry=False,goal_achieved=False));raise


if __name__=='__main__':main()
