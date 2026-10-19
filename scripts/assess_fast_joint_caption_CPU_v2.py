"""Matched caption controls with the complete own native evaluator interface.

Repairs only the missing view property in a new side assessment, leaving the
bound production driver and all failed/partial reports unchanged.
"""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]
REPORTS=ROOT/'reports/fast-joint-caption-CPU-v2'


def worker(arm):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('Every GPU must be hidden')
    import torch
    from slm_lab.corpus import file_hash
    from slm_lab.diverse_images import atomic_json
    from slm_lab.fast_joint_inference_v2 import load
    from slm_lab.dense_native_view_v2 import CompleteDenseNativeView
    from slm_lab import vision_evaluation
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    path=ROOT/'checkpoints/fast-joint-v3'/arm/'model.pt';plan=ROOT/'reports/fast-joint-v3/plan.json'
    model,metadata=load(path,plan,arm,expected_checkpoint_sha256=file_hash(path),expected_plan_sha256=file_hash(plan),device='cpu')
    view=CompleteDenseNativeView(model).eval()
    if view.vision is not model.network.vision or sum(p.numel() for p in view.parameters())!=model.counts()['total_unique']:
        raise ValueError('The exact same native vision and every unique head must be counted once')
    original=vision_evaluation.load_checkpoint;vision_evaluation.load_checkpoint=lambda *args,**kwargs:(view,metadata)
    try:vision_evaluation.evaluate(path,ROOT/'data/fast-joint-v3-pulses/vision.jsonl',REPORTS/(arm+'.json'),device='cpu',max_examples=32,image_size=448)
    finally:vision_evaluation.load_checkpoint=original
    atomic_json(REPORTS/(arm+'-complete.json'),dict(arm=arm,actual_count=model.counts()['total_unique'],
        original_native_module_and_weights_reused=True,precision='cpu/fp32',no_extra_pretrained_encoder=True,goal_achieved=False))


def parent():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('Every GPU must be hidden')
    if REPORTS.exists():raise ValueError('Preserve prior caption attempts')
    import assess_joint_java_comparison_v1 as resources
    from slm_lab.corpus import file_hash
    from slm_lab.diverse_images import atomic_json
    source=ROOT/'data/fast-joint-v3-pulses/vision.jsonl';rows=list(map(json.loads,source.open()))
    if len({r['group_id'] for r in rows})!=len(rows) or len({r['response'] for r in rows})<2:
        raise ValueError('Unique images and differing captions required')
    REPORTS.mkdir();files=[Path(__file__),source,ROOT/'src/slm_lab/dense_native_view_v2.py']
    atomic_json(REPORTS/'plan.json',dict(at=datetime.now(timezone.utc).isoformat(),source_files_sha256={str(p):file_hash(p) for p in files},
        precision='cpu/fp32',role='evaluation_only',full_GPU_endpoint_driver_still_needs_same_interface_fix=True,goal_achieved=False))
    children=[];actors=[]
    for arm in ['initial','step-1024']:
        with (REPORTS/(arm+'.log')).open('xb') as log:
            p=subprocess.Popen(['.venv/bin/python','-u',str(Path(__file__).resolve()),'--arm',arm],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
                env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2'},start_new_session=True)
        children.append(p);actors.append(dict(**resources.identity(p.pid),arm=arm))
    atomic_json(REPORTS/'children.json',actors)
    if [p.wait() for p in children]!=[0,0]:raise ValueError('Both actual native caption controls must finish; preserve partial results')
    atomic_json(REPORTS/'complete.json',dict(both_matched_CPU_caption_arms_complete=True,full_GPU_endpoints_unchanged=True,goal_achieved=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--arm',choices=['initial','step-1024']);a=p.parse_args()
    if a.arm:worker(a.arm)
    else:parent()
