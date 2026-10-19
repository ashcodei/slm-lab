"""Read the closed full public epoch and native checkpoint values, without NN.

Wait for the exact four training ranks and immutable native input receipt.
The parent may continue its existing GPU assessments. No model is created,
no neural forward is run, and no existing process is changed.
"""
from collections import Counter
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import resource
import time

import numpy as np
import torch

from slm_lab.bounded_disk_reserve_wait_v1 import wait_for_reserve
from slm_lab.codenet_token_pack_v1 import INDEX_DTYPE
from slm_lab.local_source_closure_v2 import python_sources
from slm_lab.own_dense_standalone_v1 import FORMAT
from slm_lab.public_pretraining_native_v1 import provenance,sources
from slm_lab.deterministic_positions_v1 import PROTOCOL

ROOT=Path(__file__).resolve().parents[1]
PUBLIC=ROOT/'reports/public-pretraining-v1'
RUNTIME=ROOT/'reports/public-pretraining-recovery-v1'
OUTPUT=ROOT/'checkpoints/public-pretraining-v1'
OUT=ROOT/'reports/public-pretraining-complete-epoch-value-readback-CPU-v2'
MATH=PUBLIC/'plan.json'
PARENT=dict(pid=1241,ticks='49351203')
TRAIN=[dict(pid=p,ticks=t) for p,t in [(2287,'49534423'),(2288,'49534423'),(2289,'49534423'),(2290,'49534424')]]
RAW='raw:public_code_math_web'
FUNCTIONS='new:verified_original_Git_function_contracts'
RESERVE=40*1024**3


def read(path):return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def atomic(path,value):
    pending=path.with_suffix(path.suffix+'.pending')
    with pending.open('x') as stream:
        stream.write(json.dumps(value,indent=2)+'\n');stream.flush();os.fsync(stream.fileno())
    pending.replace(path)


def live(owner):
    try:parts=Path('/proc',str(owner['pid']),'stat').read_text().split(') ',1)[1].split()
    except FileNotFoundError:return False
    return parts[19]==owner['ticks'] and parts[0]!='Z'


def main():
    if (OUT.exists() or os.environ.get('CUDA_VISIBLE_DEVICES')!='' or os.nice(0)<19
            or any(os.environ.get(k)!='1' for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'])):
        raise ValueError('Fresh one-thread CUDA-hidden nice19 source/value-only readback required')
    os.sched_setscheduler(0,os.SCHED_IDLE,os.sched_param(0));torch.set_num_threads(1)
    plan,runtime=read(MATH),read(RUNTIME/'plan.json')
    if ({key:runtime['owner'].get(key) for key in ['pid','ticks']}!=PARENT
            or runtime['original_neural_plan_sha256']!=sha(MATH) or plan['planned_steps']!=4447):
        raise ValueError('Exact unchanged original4447step source plan and recovery owner required')
    files={str(p):sha(p) for p in [MATH,RUNTIME/'plan.json',*python_sources(ROOT,[Path(__file__)])]}
    states={p:(Path(p).stat().st_dev,Path(p).stat().st_ino,Path(p).stat().st_size,Path(p).stat().st_mtime_ns) for p in files}
    me=dict(pid=os.getpid(),ticks=Path('/proc/self/stat').read_text().split(') ',1)[1].split()[19])
    OUT.mkdir();atomic(OUT/'plan.json',dict(owner=me,source_files_sha256=files,exact_parent=PARENT,exact_train_ranks=TRAIN,
        waits_for_all_exact_train_ranks_terminal_and_complete_native_inputs=True,
        live_metric_or_parent_child_registry_files_not_bound=True,
        exact_training_updates=4447,all17families_and_full_original_budgets_required=True,
        actual_checkpoint_tensor_values_read_without_model_construction=True,
        CPU_working_allowance=16*1024**3,other_process_headroom=64*1024**3,parent_address_space_ceiling=24*1024**3,
        minimum_free_bytes=RESERVE,neural_forwards=0,training_or_evaluation_workers_unchanged=True,
        generated_quality_not_proved=True,training_admitted=False,goal_achieved=False))
    def guard():
        if os.sched_getscheduler(0)!=os.SCHED_IDLE or any(
                (Path(p).stat().st_dev,Path(p).stat().st_ino,Path(p).stat().st_size,Path(p).stat().st_mtime_ns)!=v for p,v in states.items()):
            raise ValueError('Every bound original/runtime byte and idle scheduling must remain exact')
    def checked():
        wait_for_reserve(ROOT,minimum_bytes=RESERVE,guard=guard)
        values={k:int(v) for k,v in (line.split() for line in Path('/sys/fs/cgroup/memory.stat').read_text().splitlines())}
        resident=values['anon']+values['shmem']+values['kernel'];limit=int(Path('/sys/fs/cgroup/memory.max').read_text())
        rss=next(int(line.split()[1])*1024 for line in Path('/proc/self/status').read_text().splitlines() if line.startswith('VmRSS:'))
        if limit-resident<64*1024**3 or rss>16*1024**3:raise ValueError('Original other-worker headroom and declared CPU value-readback allowance required')
    try:
        while any(live(value) for value in TRAIN) or not all((PUBLIC/name).is_file() for name in ['training-complete.json','native-assessment-inputs.json']):
            guard()
            if (RUNTIME/'failure.json').exists() or (PUBLIC/'failure.json').exists():
                raise ValueError('Original source producer reports failure; no complete epoch may be inferred')
            if not live(PARENT) and not any(live(value) for value in TRAIN):
                raise ValueError('The original parent/ranks ended without the full immutable training/native receipts')
            atomic(OUT/'progress.json',dict(owner=me,phase='waiting for exact original train ranks and closed native output',
                exact_live_train_ranks=[value for value in TRAIN if live(value)],parent_live=live(PARENT),neural_forwards=0))
            time.sleep(15)
        checked();resource.setrlimit(resource.RLIMIT_AS,(24*1024**3,24*1024**3))
        complete=read(PUBLIC/'training-complete.json');native_inputs=read(PUBLIC/'native-assessment-inputs.json')
        if (complete['actual_updates']!=4447 or not complete['all_new_admitted_raw_train_windows_once']
                or not complete['all826verified_train_functions_once'] or complete['actual_counts']!=plan['actual_counts']):
            raise ValueError('Every original complete epoch/parameter-count receipt must match')
        journal=RUNTIME/'training-metrics.jsonl';functions=ROOT/'data/direct-algorithm-function-contracts-admitted-v1'
        prior_path=ROOT/'reports/code-first-continuation-v3/plan.json'
        paths=[PUBLIC/'training-complete.json',PUBLIC/'native-assessment-inputs.json',journal,prior_path,
            functions/'manifest.json',functions/'index.npy',OUTPUT/'initial/model.pt',OUTPUT/'initial/tokenizer.json',
            OUTPUT/'trained/model.pt',OUTPUT/'trained/tokenizer.json']
        for path in paths:files[str(path)]=sha(path);s=path.stat();states[str(path)]=(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns)
        if (files[str(OUTPUT/'initial/model.pt')]!=runtime['initial_checkpoint_sha256']
                or any(files[str(OUTPUT/arm/'model.pt')]!=native_inputs[arm]['checkpoint_sha256'] for arm in ['initial','trained'])
                or any(files[str(OUTPUT/arm/'tokenizer.json')]!=plan['tokenizer_sha256'] for arm in ['initial','trained'])):
            raise ValueError('Both complete immutable native checkpoints and packaged own tokenizers must match their source receipts')
        atomic(OUT/'closed-artifact-bindings.json',dict(owner=me,source_files_sha256=files,
            all_original_train_ranks_terminal=True,parent_assessments_may_still_be_live=True,neural_forwards=0))
        raw_index=np.load(functions/'index.npy',mmap_mode='r',allow_pickle=False)
        if raw_index.dtype!=INDEX_DTYPE:raise ValueError('Original sealed train function schema required')
        train_positions=np.flatnonzero(raw_index['role']==0).tolist()
        if len(train_positions)!=826:raise ValueError('All826originaltrain functions and no held function required')
        retention=hashlib.sha256(json.dumps(dict(previous=read(prior_path)['source_schedule_sha256'],
            function_manifest_sha256=sha(functions/'manifest.json'),original_train_positions=train_positions),sort_keys=True).encode()).hexdigest()
        entries=[];budget={};raw_seen=Counter();function_seen=Counter();target_total=0
        with journal.open('rb') as stream:
            for step,line in enumerate(stream,1):
                if not line.endswith(b'\n'):raise ValueError('A closed source journal cannot contain a partial final record')
                row=json.loads(line);name=row['family'];positions=row['global_original_positions']
                if (row['step']!=step or name not in plan['selected_source_budget'] or row['global_examples']!=len(positions)
                        or not positions or any(type(p) is not int or not 0<=p<plan['complete_source_budget'][name]['examples'] for p in positions)
                        or row['resident_training_only_reference_parameters']!=0 or row['reference_forward_queried'] is not False
                        or type(row['response_EOS_targets']) not in [int,float] or row['response_EOS_targets']!=int(row['response_EOS_targets'])):
                    raise ValueError('Every actual update must retain its exact original whole source/target identity')
                entries.append((name,positions));counts=budget.setdefault(name,dict(presentations=0,complete_context_tokens=0,response_EOS_targets=0,updates=0))
                counts['presentations']+=len(positions);counts['complete_context_tokens']+=row['complete_context_tokens']
                counts['response_EOS_targets']+=int(row['response_EOS_targets']);counts['updates']+=1;target_total+=int(row['response_EOS_targets'])
                if name==RAW:raw_seen.update(positions)
                if name==FUNCTIONS:function_seen.update(positions)
        identity=hashlib.sha256(json.dumps(dict(settings=plan['actual_schedule_settings'],entries=entries,
            retention_identity=retention),separators=(',',':'),sort_keys=True).encode()).hexdigest()
        if (len(entries)!=4447 or len(budget)!=17 or budget!=plan['selected_source_budget']
                or budget!=complete['actual_selected_source_budget'] or identity!=plan['source_schedule_sha256']
                or raw_seen!=Counter(range(plan['complete_source_budget'][RAW]['examples']))
                or function_seen!=Counter(range(826)) or target_total!=1105735574
                or budget[RAW]['response_EOS_targets']!=1000085544):
            raise ValueError('All4447original batches,17family exact budgets,all raw windows and826functions once must reconstruct')
        atomic(OUT/'closed-epoch-observations.json',dict(complete=True,actual_updates=4447,actual_family_budgets=budget,
            reconstructed_exact_full_schedule_sha256=identity,actual_response_EOS_targets=target_total,
            all_original_raw_windows_once=True,all826train_functions_once=True,all_prior_selected_repetitions_preserved=True))
        atomic(OUT/'progress.json',dict(owner=me,phase='closed complete epoch verified; reading actual native checkpoint tensor values',neural_forwards=0))
        initial=torch.load(OUTPUT/'initial/model.pt',map_location='cpu',weights_only=True,mmap=True)
        trained=torch.load(OUTPUT/'trained/model.pt',map_location='cpu',weights_only=True,mmap=True)
        for arm,raw in [('initial',initial),('trained',trained)]:
            if (raw['format']!=FORMAT or raw['position_interpolation']!=PROTOCOL or raw['actual_counts']!=plan['actual_counts']
                    or raw['completed_updates']!=(0 if arm=='initial' else 4447) or raw['tokenizer_sha256']!=plan['tokenizer_sha256']
                    or raw['provenance']!=provenance(plan,arm,0 if arm=='initial' else 4447)
                    or raw['complete_dense']['sources']!=sources(plan,sha(MATH))
                    or raw['old_checkpoint_optimizer_or_training_reference_needed_for_inference'] is not False):
                raise ValueError('Exact own complete native dense model, full cursor, source provenance and standalone recipe required')
        if any(initial['complete_dense'][key]!=trained['complete_dense'][key] for key in ['config','settings','actual_counts']):
            raise ValueError('Actual architecture/head settings and learned-value inventory must remain unchanged')
        def flatten(raw):
            payload=raw['complete_dense'];return {group+'.'+name:t for group in ['network','heads'] for name,t in payload[group].items()}
        first,last=flatten(initial),flatten(trained)
        if first.keys()!=last.keys() or last['network.embedding.weight'] is not last['network.lm_head.weight']:
            raise ValueError('Every original tensor and one tied vocabulary must remain in the trained package')
        seen=set();unique_values=0;counts=Counter();changed_values=Counter()
        with (OUT/'all-native-tensor-value-observations.jsonl').open('x') as stream:
            for index,(name,tensor) in enumerate(last.items()):
                if index%32==0:checked()
                original=first[name]
                if tensor.shape!=original.shape or tensor.dtype!=torch.float32 or original.dtype!=torch.float32 or not bool(torch.isfinite(tensor).all()):
                    raise ValueError('Every actual finite original-shape FP32 native value must be present')
                key=(tensor.data_ptr(),tensor.storage_offset(),tuple(tensor.shape),tuple(tensor.stride()))
                alias=key in seen
                if not alias:unique_values+=tensor.numel();seen.add(key)
                active='vision' if name.startswith('network.vision.') else 'word' if name.startswith('heads.word.') else 'language' if name.startswith('network.') and not name.startswith('network.value_head.') else None
                different=int(torch.count_nonzero(tensor!=original))
                if active is None and different:raise ValueError('Every unoptimized auxiliary/critic value must remain bitexact')
                if not alias:
                    group=active or 'fixed';counts[group+'_tensors']+=1;counts[group+'_changed_tensors']+=bool(different)
                    changed_values[group]+=different
                stream.write(json.dumps(dict(name=name,shape=list(tensor.shape),dtype=str(tensor.dtype),all_actual_values_finite=True,
                    declared_active_group=active,tied_alias_already_counted=alias,actual_changed_values=different))+'\n')
            stream.flush();os.fsync(stream.fileno())
        if unique_values!=984399690 or counts['language_tensors']!=317 or counts['vision_tensors']!=112 or counts['word_tensors']!=16:
            raise ValueError('Actual complete under1B native model and all445original active leaves required')
        checked()
        if any(sha(path)!=digest for path,digest in files.items()):raise ValueError('Every bound closed source and checkpoint byte must remain immutable')
        if torch.cuda.is_initialized():raise ValueError('A value-only source readback must not initialize CUDA')
        atomic(OUT/'complete.json',dict(complete=True,passed=True,at=datetime.now(timezone.utc).isoformat(),owner=me,
            source_files_sha256=files,actual_complete_updates=4447,actual_family_budgets=budget,
            actual_response_EOS_targets=target_total,exact_original_schedule_sha256_reconstructed=identity,
            actual_unique_neural_values=unique_values,actual_active_and_fixed_tensor_counts=dict(counts),actual_changed_values_by_group=dict(changed_values),
            actual_tensor_value_observations_sha256=sha(OUT/'all-native-tensor-value-observations.jsonl'),
            all_actual_native_values_finite=True,all_unoptimized_auxiliary_and_critic_values_bitexact=True,
            all_closed_original_source_positions_targets_and_repetitions_preserved=True,
            full_parent_native_GPU_assessments_and_human_RL_and_nine_final_endpoints_still_required=True,
            no_model_constructed=True,neural_forwards=0,existing_neural_workers_unchanged=True,
            generated_quality_not_proved=True,training_admitted=False,new_production_optimizer_updates=0,goal_achieved=False))
        print(json.dumps(dict(complete=True,passed=True,actual_updates=4447,actual_unique_values=unique_values,
            actual_response_EOS_targets=target_total,counts=dict(counts))),flush=True)
    except BaseException as error:
        atomic(OUT/'failure.json',dict(owner=me,error=repr(error),automatic_retry=False,goal_achieved=False));raise


if __name__=='__main__':main()
