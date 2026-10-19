"""Full unchanged warm-up code cases on CPU/FP32, alongside GPU/BF16 reports.

This changes both device and precision, so differences do not isolate precision
alone. No GPU, gradient, model update or benchmark source modification occurs.
"""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT/'reports/warmup88-code-CPU-FP32-v1'


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('This independent CPU evaluation must hide every GPU')
    if REPORTS.exists():
        raise ValueError('Preserve previous complete or partial inference comparisons')
    import torch
    from slm_lab.codenet_checkpoint_v1 import FileBindings
    from slm_lab.corpus import file_hash
    from slm_lab.diverse_images import atomic_json
    from slm_lab.fast_joint_inference_v2 import fresh
    from slm_lab.inference_rotary_cache_v1 import inference_cache
    from slm_lab.native_code_shards_v1 import evaluate_shard, reconcile
    from slm_lab.own_dense_actor_v1 import OwnDenseActor
    from slm_lab import code_quality
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    run = ROOT/'reports/verified-views-execution-v1'
    checkpoint = ROOT/'checkpoints/verified-views-execution-v1/warmup-actor.pt'
    plan_path = run/'plan.json'
    plan = json.loads(plan_path.read_text())
    sources = dict(run_plan_sha256=file_hash(plan_path),
        own_initial_dense_sha256=plan['initial_dense_settings']['initializer_sha256'],
        verified_views_admission_sha256=file_hash(ROOT/'data/verified-code-views-v1/admission.json'),
        causal_reward_admission_sha256=file_hash(ROOT/'reports/causal-code-rewards-v2/admission.json'))
    files = dict(plan['source_files_sha256'])
    for path in [Path(__file__),plan_path,checkpoint,ROOT/'src/slm_lab/native_code_shards_v1.py',
                 ROOT/'src/slm_lab/inference_rotary_cache_v1.py',
                 ROOT/'reports/verified-views-warmup88-readback-v1/complete.json',
                 run/'warmup/code-completion.json',run/'warmup/code-instruction.json']:
        files[str(path)] = file_hash(path)
    guard = FileBindings(files)
    REPORTS.mkdir()
    started = time.monotonic()
    atomic_json(REPORTS/'plan.json',dict(at=datetime.now(timezone.utc).isoformat(),PID=os.getpid(),
        source_files_sha256=files,checkpoint=str(checkpoint),generation_budget=512,CPU_threads=1,
        frames=['completion','instruction'],all_original_24_cases_per_frame=True,
        comparison_changes_both_device_and_precision=True,training_updates=0,goal_achieved=False))
    base,_ = fresh(plan,'cpu')
    model = OwnDenseActor(base)
    payload = torch.load(checkpoint,map_location='cpu',weights_only=True,mmap=True)
    model.import_export(payload,expected_sources=sources)
    del payload
    if model.counts()['actual_unique'] != 999706442:
        raise ValueError('Use the actual complete counted warm-up policy')
    metadata = dict(tokenizer_sha256=plan['initial_dense_settings']['tokenizer_sha256'],training_options=dict(image_size=448))
    original_loader,original_threads = code_quality.load_checkpoint,torch.set_num_threads
    code_quality.load_checkpoint = lambda *a,**kw:(model.policy,metadata)
    torch.set_num_threads = lambda count:None
    scores = {}
    try:
        with inference_cache():
            for frame in ['completion','instruction']:
                guard.check()
                evaluate_shard(model.policy,metadata,checkpoint,REPORTS,rank=0,world=1,
                               frame=frame,device='cpu',budget=512)
                report = reconcile([REPORTS/f'code-{frame}-shard-0.json'],REPORTS/f'code-{frame}.json')
                old = json.loads((run/'warmup'/f'code-{frame}.json').read_text())
                scores[frame] = dict(CPU_FP32={k:report[k] for k in ['syntax_pass_rate','functional_pass_at_1','efficiency_stress_pass_rate']},
                    original_GPU_BF16={k:old[k] for k in ['syntax_pass_rate','functional_pass_at_1','efficiency_stress_pass_rate']},
                    changed_generated_responses=sum(a['raw_response']!=b['raw_response'] for a,b in zip(report['results'],old['results'])))
    finally:
        code_quality.load_checkpoint,torch.set_num_threads = original_loader,original_threads
    if any(p.grad is not None for p in model.parameters()):
        raise ValueError('CPU inference must not produce training gradients')
    guard.check(full=True)
    atomic_json(REPORTS/'complete.json',dict(at=datetime.now(timezone.utc).isoformat(),scores=scores,
        actual_unique_parameters=model.counts()['actual_unique'],seconds=time.monotonic()-started,
        CPU_FP32_vs_GPU_BF16_not_precision_only_control=True,training_updates=0,
        held_outputs_require_independent_code_execution_readback=True,candidate_promoted=False,goal_achieved=False))
    print(json.dumps(scores),flush=True)


if __name__ == '__main__':
    main()
