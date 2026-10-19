"""Corrected matched CPU/FP32 object/caption development checks.

The original GPU pulse's leading-row object subset cannot supply its wrong-
image controls. Preserve that failure; select answer-balanced independent
photos from the original evaluation partition into a separate named protocol.
"""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]
REPORTS=ROOT/'reports/fast-joint-corrected-visual-CPU-v1'


def worker(arm):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('No GPU may be used by this side comparison')
    import torch
    from slm_lab.corpus import file_hash
    from slm_lab.codenet_checkpoint_v1 import FileBindings
    from slm_lab.diverse_images import atomic_json
    from slm_lab.fast_joint_inference_v2 import load,DenseNativeView
    from slm_lab.inference_rotary_cache_v1 import inference_cache
    from slm_lab import object_evaluation,vision_evaluation
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    setup=json.loads((REPORTS/'plan.json').read_text());guard=FileBindings(setup['source_files_sha256'])
    model_path=ROOT/'checkpoints/fast-joint-v3'/arm/'model.pt';plan_path=ROOT/'reports/fast-joint-v3/plan.json'
    model,metadata=load(model_path,plan_path,arm,expected_checkpoint_sha256=file_hash(model_path),expected_plan_sha256=file_hash(plan_path),device='cpu')
    view=DenseNativeView(model);folder=REPORTS/arm;folder.mkdir()
    # Fix the CPU pool throughout the sequence, keeping CPU/FP32 explicit.
    original_threads=torch.set_num_threads;torch.set_num_threads=lambda count:None
    try:
        with inference_cache():
            for module,name,source,options in [(object_evaluation,'objects.json',REPORTS/'objects.jsonl',dict(max_examples=setup['object_examples'],max_new_tokens=32)),
                    (vision_evaluation,'vision.json',Path(setup['vision_source']),dict(max_examples=32,image_size=448))]:
                original=module.load_checkpoint;module.load_checkpoint=lambda path,device,training_state=False:(view,metadata)
                try:module.evaluate(model_path,source,folder/name,device='cpu',**options)
                finally:module.load_checkpoint=original
    finally:torch.set_num_threads=original_threads
    guard.check(full=True)
    atomic_json(folder/'complete.json',dict(arm=arm,inference_device='cpu',precision='fp32',CPU_threads=2,
        separate_development_protocol=True,not_comparable_to_GPU_bf16_scores=True,goal_achieved=False))


def parent():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU comparison must hide all GPUs')
    if REPORTS.exists():raise ValueError('Preserve previous visual comparison')
    from slm_lab.corpus import file_hash
    from slm_lab.diverse_images import atomic_json
    from slm_lab.object_evaluation import select
    from slm_lab.ocr_evaluation import normalized
    REPORTS.mkdir();source=ROOT/'data/object-qa-v1/valid.jsonl'
    rows=list(map(json.loads,source.open()));chosen=select(rows,32)
    for row in chosen:
        alternatives=[other for other in chosen if other['task']==row['task'] and other['group_id']!=row['group_id']
                      and normalized(other['response']).casefold()!=normalized(row['response']).casefold()]
        if not alternatives:raise ValueError('Every selected question must retain an independent differently labelled photo control')
        image=Path(row['image']);row['image']=str((image if image.is_absolute() else source.parent/image).resolve())
    path=REPORTS/'objects.jsonl';path.write_text(''.join(json.dumps(r)+'\n' for r in chosen))
    run=json.loads((ROOT/'reports/fast-joint-v3/plan.json').read_text());vision=Path(run['pulse_sources']['vision'])
    files=[Path(__file__),source,path,vision,ROOT/'reports/fast-joint-v3/plan.json',
        ROOT/'src/slm_lab/object_evaluation.py',ROOT/'src/slm_lab/vision_evaluation.py',
        ROOT/'src/slm_lab/inference_rotary_cache_v1.py']
    atomic_json(REPORTS/'plan.json',dict(observed_at=datetime.now(timezone.utc).isoformat(),object_examples=len(chosen),
        every_object_has_independent_different_target_control=True,vision_source=str(vision),
        source_files_sha256={str(p):file_hash(p) for p in files},role='evaluation_only',never_training_data=True,
        precision='cpu/fp32',separate_development_protocol=True,failed_GPU_source_preserved=True,goal_achieved=False))
    children=[]
    for arm in ['initial','step-1024']:
        log=(REPORTS/(arm+'.log')).open('xb')
        child=subprocess.Popen(['.venv/bin/python','-u',str(Path(__file__).resolve()),'--arm',arm],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
            env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2'},start_new_session=True)
        children.append(child);log.close()
    codes=[p.wait() for p in children]
    if codes!=[0,0]:raise ValueError('Both exact CPU visual workers must complete; preserve partial results')
    atomic_json(REPORTS/'complete.json',dict(both_matched_CPU_visual_arms_complete=True,
        official_GPU_and_full_endpoint_scores_unchanged=True,independent_score_comparison_pending=True,goal_achieved=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--arm',choices=['initial','step-1024']);a=p.parse_args()
    if a.arm:worker(a.arm)
    else:parent()
