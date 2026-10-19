"""Measure a failed distributed raw-gradient gate without waiving it.

Actual small CPU replicas compare full-batch and per-word serial references,
pre/post-clipping gradients and Adam values. No production candidate is adopted.
"""
import copy
from datetime import datetime,timedelta,timezone
import hashlib
import json
import os
from pathlib import Path
import sys

import torch
from torch import distributed as dist
from torch.multiprocessing import spawn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tests'))
from test_global_word_stage_kernel_v1 import fixture,subset,optimizer,expected_backward
from slm_lab.global_word_stage_kernel_v1 import accumulate_global_word,named_word_selected
from slm_lab.shared_word_stage_v1 import ctc_word_sum
from slm_lab.local_source_closure_v2 import python_sources

OUT = ROOT/'reports/global-word-reduction-diagnosis-CPU-v1'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic(path,value):
    pending = path.with_suffix(path.suffix+'.pending')
    with pending.open('x') as stream:
        stream.write(json.dumps(value,indent=2)+'\n');stream.flush();os.fsync(stream.fileno())
    pending.replace(path)


def differences(left,right):
    count = sign = nonexact = 0;norm = square = maximum = 0.;worst = None
    for name,a in left.items():
        b = right[name];delta = (a-b).double();count += a.numel();nonexact += int((a != b).sum());sign += int((a*b < 0).sum())
        square += float(delta.square().sum());norm += float(b.double().square().sum());error = float(delta.abs().max())
        if error > maximum:maximum,worst = error,name
    return dict(values=count,nonexact=nonexact,opposite_sign_values=sign,maximum_absolute_difference=maximum,
        worst_tensor=worst,relative_L2_difference=(square/norm)**.5 if norm else None,reference_L2_norm=norm**.5)


def worker(rank,folder):
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    dist.init_process_group('gloo',init_method='file://'+str(Path(folder)/'rendezvous'),rank=rank,world_size=4,timeout=timedelta(minutes=4))
    try:
        view,batch = fixture();opt = optimizer(view);observations = []
        for slots,positions in [([0,1,2],[17,23,29]),([2],[29])]:
            full = copy.deepcopy(view);micro = copy.deepcopy(view);whole = subset(batch,slots)
            local_slots = slots[rank::4];local = subset(batch,local_slots) if local_slots else None
            counts = [int((r[1:] != -100).sum()) for r in whole['labels']];byte_counts = [len(t.encode()) for t in whole['texts']]
            record = accumulate_global_word(view,local,global_positions=positions,local_positions=positions[rank::4],
                global_response_EOS_targets=counts,global_word_UTF8_bytes=byte_counts,CTC_weight=.4,native_CE_weight=.7)
            actual = named_word_selected(view)
            for _,p in actual:
                if p.grad is None:p.grad = torch.zeros_like(p)
                dist.all_reduce(p.grad,op=dist.ReduceOp.SUM)
            full_values = expected_backward(full,whole)
            micro_parameters = [p for _,p in named_word_selected(micro)];micro_CTC = micro_CE = 0.
            for source_slot in slots:
                single = subset(batch,[source_slot]);logits = micro.actor.base.word_logits(single['word_pixels'])
                ctc,_ = ctc_word_sum(logits,single['texts']);(ctc*.4/len(slots)).backward(inputs=micro_parameters)
                micro_CTC += float(ctc.detach())
                logits = micro.native(**{k:single[k] for k in ['input_ids','attention_mask','pixel_values']})['logits']
                targets = single['labels'][:,1:];mask = targets != -100
                ce = torch.nn.functional.cross_entropy(logits[:,:-1].float()[mask],targets[mask],reduction='sum')
                (ce*.7/sum(counts)).backward(inputs=micro_parameters);micro_CE += float(ce.detach())
            actual_grads = {n:p.grad.detach().clone() for n,p in actual}
            full_grads = {n:p.grad.detach().clone() for n,p in named_word_selected(full)}
            micro_grads = {n:p.grad.detach().clone() for n,p in named_word_selected(micro)}
            against_full = differences(actual_grads,full_grads);against_micro = differences(actual_grads,micro_grads)
            norm = float(torch.nn.utils.clip_grad_norm_([p for _,p in actual],1.,error_if_nonfinite=True))
            torch.nn.utils.clip_grad_norm_([p for _,p in named_word_selected(full)],1.,error_if_nonfinite=True)
            torch.nn.utils.clip_grad_norm_(micro_parameters,1.,error_if_nonfinite=True)
            clipped = {n:p.grad.detach().clone() for n,p in actual}
            clipped_full = differences(clipped,{n:p.grad for n,p in named_word_selected(full)})
            clipped_micro = differences(clipped,{n:p.grad for n,p in named_word_selected(micro)})
            refopt = optimizer(full);refopt.load_state_dict(copy.deepcopy(opt.state_dict()))
            microopt = optimizer(micro);microopt.load_state_dict(copy.deepcopy(opt.state_dict()))
            opt.step();refopt.step();microopt.step();opt.zero_grad(set_to_none=True)
            weights = {n:p.detach() for n,p in actual}
            weight_full = differences(weights,{n:p.detach() for n,p in named_word_selected(full)})
            weight_micro = differences(weights,{n:p.detach() for n,p in named_word_selected(micro)})
            records = [None]*4;dist.all_gather_object(records,record)
            observations.append(dict(global_words=len(slots),actual_unique_parameters=view.actor.counts()['actual_unique'],
                raw_against_full_batch=against_full,raw_against_per_word_serial=against_micro,
                clipped_against_full_batch=clipped_full,clipped_against_per_word_serial=clipped_micro,
                Adam_against_full_batch=weight_full,Adam_against_per_word_serial=weight_micro,
                actual_global_gradient_norm_before_clip=norm,original_raw_absolute_gate_passed=against_full['maximum_absolute_difference'] < 5e-6,
                full_batch_serial_CTC_sum=full_values[0],per_word_serial_CTC_sum=micro_CTC,
                actual_replica_CTC_sum=sum(r['local_byte_normalized_CTC_sum'] for r in records),
                full_batch_serial_native_CE_sum=full_values[1],per_word_serial_native_CE_sum=micro_CE,
                actual_replica_native_CE_sum=sum(r['local_native_CE_sum'] for r in records),
                actual_rank_records=records,production_candidate_adopted=False))
        (Path(folder)/f'{rank}.json').write_text(json.dumps(dict(complete=True,rank=rank,observations=observations),indent=2)+'\n')
    finally:dist.destroy_process_group()


def main():
    if (OUT.exists() or os.environ.get('CUDA_VISIBLE_DEVICES') != '' or os.nice(0) < 19
            or any(os.environ.get(k) != '1' for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'])
            or '1 failed, 3 passed' not in (ROOT/'reports/global-word-stage-kernel-v2-tests.log').read_text()):
        raise ValueError('Original failed gate and fresh bounded low-priority CPU diagnostic required')
    files = {str(p):sha(p) for p in [ROOT/'reports/global-word-stage-kernel-v2-tests.log',
        *python_sources(ROOT,[Path(__file__),ROOT/'tests/test_global_word_stage_kernel_v1.py'])]}
    OUT.mkdir();atomic(OUT/'plan.json',dict(source_files_sha256=files,world_size=4,actual_small_neural_fixture=True,
        original_raw_absolute_gradient_gate=5e-6,gate_not_waived=True,full_batch_and_per_word_references=True,
        production_model_or_recipe_not_changed=True,training_admitted=False,goal_achieved=False))
    try:
        spawn(worker,args=(str(OUT),),nprocs=4,join=True)
        rows = [json.loads((OUT/f'{rank}.json').read_text()) for rank in range(4)]
        if any(not r['complete'] or len(r['observations']) != 2 for r in rows):raise ValueError('Every actual four-rank observation required')
        if any(sha(p) != h for p,h in files.items()):raise ValueError('Every failed-test/source/diagnostic byte must remain exact')
        atomic(OUT/'complete.json',dict(complete=True,at=datetime.now(timezone.utc).isoformat(),source_files_sha256=files,
            actual_rank_observations=rows,original_raw_gradient_gate_not_waived=True,
            no_full_size_GPU_or_model_quality_claim=True,training_admitted=False,new_production_optimizer_updates=0,goal_achieved=False))
        print(json.dumps(dict(complete=True,actual_ranks=4,original_gate_not_waived=True,training_admitted=False)),flush=True)
    except BaseException as error:
        atomic(OUT/'failure.json',dict(error=repr(error),automatic_retry=False,training_admitted=False,goal_achieved=False));raise


if __name__ == '__main__':main()
