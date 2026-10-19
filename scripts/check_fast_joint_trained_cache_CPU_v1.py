"""Full-size trained cache checks on fixed non-benchmark text/image controls."""
from datetime import datetime,timezone
import os
from pathlib import Path
import json
import time

ROOT=Path(__file__).resolve().parents[1]
OUTPUT=ROOT/'reports/fast-joint-step1024-cache-CPU-v1.json'


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU side-check must hide all GPUs')
    if OUTPUT.exists():raise ValueError('Preserve previous actual trained cache evidence')
    import torch
    from tokenizers import Tokenizer
    from slm_lab.corpus import file_hash
    from slm_lab.diverse_images import atomic_json
    from slm_lab.fast_joint_inference_v2 import load
    from slm_lab.inference_rotary_cache_v1 import inference_cache
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True);started=time.monotonic()
    checkpoint=ROOT/'checkpoints/fast-joint-v3/step-1024/model.pt';plan=ROOT/'reports/fast-joint-v3/plan.json'
    model,metadata=load(checkpoint,plan,'step-1024',expected_checkpoint_sha256=file_hash(checkpoint),expected_plan_sha256=file_hash(plan),device='cpu')
    tokenizer=Tokenizer.from_file(str(checkpoint.parent/'tokenizer.json'))
    ids=torch.tensor([[1,*tokenizer.encode('Read the following text and continue the description: a small tree beside a quiet road.').ids]])
    pixels=torch.zeros(1,3,448,448);records=[]
    with torch.no_grad():
        for name,image in [('text',None),('image',pixels)]:
            tokens=ids.clone();cached=[];state=model.network.prefill(tokens,image)
            for i in range(32):
                cached.append(state['logits'][:,-1].clone());nxt=state['logits'][:,-1].argmax(-1,keepdim=True)
                tokens=torch.cat((tokens,nxt),1)
                if i+1<32:state=model.network.decode(nxt,state['cache'])
            full=model(tokens,image)['logits'][:,ids.shape[1]-1:-1]
            actual=torch.stack(cached,dim=1)
            max_difference=float((full-actual).abs().max())
            if not torch.allclose(full,actual,atol=2e-4,rtol=2e-5):raise ValueError('Actual trained cached logits differ from full causal forward')
            with inference_cache():optimized=model.generate(ids,image,max_new_tokens=32,eos_token_id=None)
            if not torch.equal(tokens,optimized):raise ValueError('Actual trained optimized and original cache greedy tokens differ')
            records.append(dict(input=name,actual_full_size_parameters=model.counts()['total_unique'],fixed_horizon=32,
                cached_vs_full_max_absolute_logit_difference=max_difference,
                cached_vs_full_argmax_disagreements=int((full.argmax(-1)!=actual.argmax(-1)).sum()),
                original_vs_optimized_cached_tokens_exact=True))
    atomic_json(OUTPUT,dict(observed_at=datetime.now(timezone.utc).isoformat(),actual_trained_updates=1024,
        inference_device='cpu',inference_precision='fp32',records=records,
        all_original_cached_and_full_forward_logits_match_tolerance=True,all_optimized_and_original_greedy_tokens_exact=True,
        fixed_nonbenchmark_controls_not_held_quality=True,GPU_bf16_parity_not_checked_here=True,
        checkpoint_sha256=file_hash(checkpoint),source_plan_sha256=file_hash(plan),source_sha256=file_hash(__file__),
        seconds=time.monotonic()-started,goal_achieved=False))
    print(json.dumps(dict(seconds=time.monotonic()-started,records=records)),flush=True)


if __name__=='__main__':main()
