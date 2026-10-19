"""Diagnose the preserved failed BF16 outer-autocast reference comparison."""
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tests'))
from test_global_photo_gradient_cache_stage_v1 import photo_fixture, selected, direct_backward
from test_global_native_visual_stage_kernel_v1 import specification
from slm_lab.global_native_visual_stage_kernel_v1 import native_microbatches
from slm_lab.global_photo_gradient_cache_stage_v1 import accumulate_global_photo_cached
from slm_lab.own_actor_vision_alignment_v1 import pair_positives,symmetric_pair_loss
from slm_lab.own_actor_vision_prefix_reuse_v1 import native_from_visual
from slm_lab.local_source_closure_v2 import python_sources

OUT = ROOT/'reports/photo-cache-autocast-gradient-diagnosis-CPU-v1'
FAILED = ROOT/'reports/global-photo-gradient-cache-stage-v1-tests.log'


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def fixed_reference(view,batch,ids,mask,keys,captions,groups,amp):
    parts=[];images_by_source={};texts_by_source={}
    for slots in groups:
        length=int(batch['attention_mask'][slots].sum(1).max())
        native={k:v[slots] if k=='pixel_values' else v[slots,:length] for k,v in batch.items()}
        with torch.no_grad(),amp():
            hidden=view.actor.policy(ids[slots],attention_mask=mask[slots],return_hidden=True,return_logits=False)['hidden']
            content=mask[slots] & (ids[slots]!=1) & (ids[slots]!=2)
            text=(hidden.float()*content[:,:,None]).sum(1)/content.sum(1,keepdim=True)
        with amp():
            visual=view.native.vision(native['pixel_values'],True)
            images=view.actor.base.heads['pointing']['patch'](visual.mean(1))
            texts=view.actor.base.heads['pointing']['query'](text.to(visual.dtype))
            ce=native_from_visual(view.native,native['input_ids'],visual,
                attention_mask=native['attention_mask'],labels=native['labels'])['loss']
        count=int((native['labels'][:,1:]!=-100).sum())
        parts.append((images,texts,ce,count))
        for j,source in enumerate(slots):images_by_source[source]=images[j];texts_by_source[source]=texts[j]
    images=torch.stack([images_by_source[i] for i in range(len(ids))])
    texts=torch.stack([texts_by_source[i] for i in range(len(ids))])
    loss=symmetric_pair_loss(images,texts,pair_positives(keys,captions),temperature=.7)
    # This is the only correction relative to the original attached-graph
    # reference: retain the full-pair derivative computation in FP32.
    with torch.autocast('cpu',enabled=False):
        gradients=torch.autograd.grad(.1*loss,[v for p in parts for v in p[:2]],retain_graph=True)
    ce_sum=0.;total=sum(p[3] for p in parts)
    for i,(image,text,ce,count) in enumerate(parts):
        torch.autograd.backward([image,text],grad_tensors=gradients[2*i:2*i+2],inputs=selected(view),retain_graph=True)
        (ce*count/total).backward(inputs=selected(view));ce_sum+=float(ce.detach())*count
    return float(loss.detach()),ce_sum


def main():
    if OUT.exists() or os.environ.get('CUDA_VISIBLE_DEVICES')!='' or os.nice(0)<19:
        raise ValueError('Fresh CUDA-hidden low-priority tiny-own-model diagnostic required')
    assert all(os.environ.get(k)=='1' for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'])
    os.sched_setscheduler(0,os.SCHED_IDLE,os.sched_param(0));torch.set_num_threads(1)
    assert '1 failed, 7 passed' in FAILED.read_text()
    files={str(p):sha(p) for p in [FAILED,*python_sources(ROOT,[Path(__file__)])]}
    OUT.mkdir()
    (OUT/'plan.json').write_text(json.dumps(dict(source_files_sha256=files,actual_small_own_model_only=True,
        reference_feature_gradient_guard_is_only_proposed_change=True,failed_original_gate_unwaived=True,
        production_optimizer_updates=0),indent=2)+'\n')
    observations={};saved={}
    for arm in ['cached_no_outer','cached_outer','reference_no_outer','reference_outer','reference_outer_FP32_feature_grad']:
        view,batch,ids,mask,keys,captions=photo_fixture();positions=list(range(9));spec=specification(batch,positions)
        amp=lambda:torch.autocast('cpu',dtype=torch.bfloat16)
        outer=amp if arm not in ['cached_no_outer','reference_no_outer'] else nullcontext
        with outer():
            if arm.startswith('cached'):
                record=accumulate_global_photo_cached(view,batch,ids,mask,local_positions=positions,pixel_keys=keys,
                    captions=captions,maximum_padded_tokens=450,temperature=.7,feature_dtype=torch.bfloat16,
                    native_logits_dtype=torch.bfloat16,amp=amp,**spec)
                loss=record['global_matching']['unweighted_loss'];ce=record['local_caption_CE_sum']
            else:
                groups=native_microbatches(spec['global_context_lengths'],prefix_tokens=196,maximum_padded_tokens=450)
                reference=fixed_reference if arm.endswith('FP32_feature_grad') else direct_backward
                loss,ce=reference(view,batch,ids,mask,keys,captions,groups,amp)
        gradients={name:p.grad.detach().clone() for name,p in view.named_selected()}
        saved[arm]=gradients
        differences={name:float((g-saved['cached_no_outer'][name]).abs().max()) for name,g in gradients.items()}
        worst=max(differences,key=differences.get)
        observations[arm]=dict(matching_loss=loss,complete_caption_CE_sum=ce,
            actual_own_unique_parameters=view.actor.counts()['actual_unique'],selected_values=sum(p.numel() for p in selected(view)),
            raw_max_gradient_difference_vs_cached_no_outer=differences[worst],worst_parameter=worst)
    assert all(sha(p)==digest for p,digest in files.items())
    assert not torch.cuda.is_initialized()
    result=dict(complete=True,source_files_sha256=files,observations=observations,
        original_failed_reference_comparison_preserved=True,no_tolerances_changed=True,
        no_production_model_or_optimizer_modified=True,production_optimizer_updates=0,goal_achieved=False)
    (OUT/'complete.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result['observations']),flush=True)


if __name__=='__main__':main()
