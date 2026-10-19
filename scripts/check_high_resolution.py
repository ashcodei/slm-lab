"""Check full-size mixed-precision image backward memory without updating weights."""
from pathlib import Path
import time
import torch
from slm_lab.diverse_images import atomic_json
from slm_lab.model import ModelConfig,TinyMultimodalLM

ROOT=Path(__file__).resolve().parents[1]


def main():
    torch.set_num_threads(1);torch.manual_seed(123);torch.cuda.set_device(0)
    device='cuda:0';config=ModelConfig.load(ROOT/'configs/model.json')
    free_before,total=torch.cuda.mem_get_info(device)
    with torch.device(device):model=TinyMultimodalLM(config)
    model.train();model.gradient_checkpointing=True;results=[]
    ids=torch.randint(0,config.vocab_size,(2,512),device=device)
    for size in [224,448]:
        model.zero_grad(set_to_none=True);torch.cuda.empty_cache()
        pixels=torch.randn(2,3,size,size,device=device);torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize();started=time.monotonic()
        with torch.autocast('cuda',dtype=torch.bfloat16):out=model(ids,pixels,labels=ids)
        out['loss'].backward();torch.cuda.synchronize()
        result={'image_size':size,'visual_tokens':config.image_token_count(size),
                'batch':2,'text_tokens_per_sample':512,'loss':float(out['loss']),
                'forward_backward_seconds':time.monotonic()-started,
                'peak_allocated_bytes':torch.cuda.max_memory_allocated(device),
                'peak_reserved_bytes':torch.cuda.max_memory_reserved(device),
                'position_gradient_finite':bool(torch.isfinite(model.vision.position.grad).all()),
                'position_gradient_norm':float(model.vision.position.grad.norm())}
        if not result['position_gradient_finite'] or not torch.isfinite(out['loss']):raise ValueError('Nonfinite readiness result')
        results.append(result);print(result,flush=True);del pixels,out
    parameters=model.parameter_counts();adam_states=8*parameters['total']
    report={'device':torch.cuda.get_device_name(0),'parameters':parameters,'results':results,
            'initial_free_gpu_bytes':free_before,'total_gpu_bytes':total,
            'estimated_additional_fp32_adam_moment_bytes':adam_states,
            'optimizer_steps':0,'checkpoint_saved':False,
            'scope':'Synthetic forward/backward resource check, contended with text pretraining; not model learning or capability evaluation.',
            'limitations':['Optimizer moments are estimated, not allocated; optimizer temporaries are additional.',
                           'First-call kernel startup and concurrent workloads make timings incomparable.',
                           '512-token examples; longer image-conditioned sequences can use more memory.',
                           'Mixed-precision backward on one device; full distributed stage remains unverified.']}
    atomic_json(ROOT/'reports/l40-high-resolution-readiness.json',report)

if __name__=='__main__':main()
