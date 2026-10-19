"""Bounded resource test of the configured architecture, not capability training."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import resource
import time
import torch
from .model import ModelConfig,TinyMultimodalLM


def benchmark(config,device,batch_size,length,steps,with_images,threads,output):
    if min(batch_size,length,steps,threads)<1 or length<2:
        raise ValueError('Positive dimensions and at least two text tokens are required.')
    if length>config.max_text_length:raise ValueError('Sequence length exceeds the model configuration.')
    device=torch.device(device)
    if device.type=='cuda' and not torch.cuda.is_available():
        raise ValueError('No CUDA device is usable in this instance.')
    torch.set_num_threads(threads);torch.manual_seed(42)
    started=time.monotonic();model=TinyMultimodalLM(config).to(device).train()
    model.gradient_checkpointing=True
    initialization=time.monotonic()-started
    ids=torch.randint(config.vocab_size,(batch_size,length),device=device)
    pixels=torch.randn(batch_size,3,config.image_size,config.image_size,device=device) if with_images else None
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-5)
    use_bf16=device.type=='cuda' and torch.cuda.is_bf16_supported()
    if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
    measurements=[]
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        if device.type=='cuda':torch.cuda.synchronize(device)
        begin=time.monotonic()
        with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=use_bf16):
            loss=model(ids,pixels,labels=ids)['loss']
        loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.0,error_if_nonfinite=True)
        vision_norm=float(model.vision.patch.weight.grad.norm()) if pixels is not None else None
        if not torch.isfinite(loss):raise ValueError('Nonfinite full-size loss.')
        optimizer.step()
        if device.type=='cuda':torch.cuda.synchronize(device)
        elapsed=time.monotonic()-begin
        measurements.append({'step':step+1,'loss':float(loss.detach()),'gradient_norm':float(norm),
            'vision_patch_gradient_norm':vision_norm,'elapsed_seconds':elapsed,
            'text_target_tokens_per_second':batch_size*(length-1)/elapsed})
        print(json.dumps(measurements[-1]),flush=True)
    result={'configuration':asdict(config),'parameters':model.parameter_counts(),'device':str(device),
        'batch_size':batch_size,'text_length':length,'with_images':with_images,'bf16':use_bf16,
        'gradient_checkpointing':True,'initialization_seconds':initialization,
        'process_peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None,
        'measurements':measurements,
        'scope':'Synthetic short-context resource test. No trained checkpoint or capability claim. '
        'CPU results must not be extrapolated as measured GPU throughput.'}
    Path(output).parent.mkdir(parents=True,exist_ok=True)
    Path(output).write_text(json.dumps(result,indent=2)+'\n')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default='configs/model.json')
    p.add_argument('--device',default='cpu');p.add_argument('--batch-size',type=int,default=1)
    p.add_argument('--length',type=int,default=128);p.add_argument('--steps',type=int,default=2)
    p.add_argument('--threads',type=int,default=8);p.add_argument('--with-images',action='store_true')
    p.add_argument('--output',required=True)
    a=p.parse_args()
    try:benchmark(ModelConfig.load(a.config),a.device,a.batch_size,a.length,a.steps,
                  a.with_images,a.threads,a.output)
    except ValueError as e:p.error(str(e))


if __name__=='__main__':main()
