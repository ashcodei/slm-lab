"""Bounded reuse of identical RoPE tables across layers during generation.

This preserves the original FP32 frequency/angle/trigonometric arithmetic and
every model weight. It avoids recomputing the same position table separately
for Q and K in every language layer. Training delegates to the original path.
"""
from collections import OrderedDict
from contextlib import contextmanager
import threading

import torch
from . import model as core

ORIGINAL=core.rotary
TABLES=OrderedDict()
LOCK=threading.RLock()
COUNTS=dict(hits=0,misses=0)
MAXIMUM_TABLES=64


def cached_rotary(x,theta,offset=0):
    if torch.is_grad_enabled():return ORIGINAL(x,theta,offset)
    dim=x.shape[-1];key=(str(x.device),x.dtype,float(theta),dim,int(offset),x.shape[-2])
    with LOCK:
        pair=TABLES.get(key)
        if pair is None:
            # A normal immutable tensor also remains safe if an inference-mode
            # caller creates it. No input/model tensor is retained in the cache.
            with torch.inference_mode(False),torch.no_grad():
                freq=theta**(-torch.arange(0,dim,2,device=x.device,dtype=torch.float32)/dim)
                angle=torch.arange(offset,offset+x.shape[-2],device=x.device,dtype=torch.float32)[:,None]*freq[None,:]
                pair=(angle.cos().to(x.dtype),angle.sin().to(x.dtype))
            TABLES[key]=pair;COUNTS['misses']+=1
            if len(TABLES)>MAXIMUM_TABLES:TABLES.popitem(last=False)
        else:TABLES.move_to_end(key);COUNTS['hits']+=1
    cos,sin=pair;a,b=x[...,0::2],x[...,1::2]
    return torch.stack((a*cos-b*sin,a*sin+b*cos),dim=-1).flatten(-2)


@contextmanager
def inference_cache():
    # Reentrant context serializes replacement/restoration within this process.
    # Grad-enabled callers always delegate to the unchanged original function.
    with LOCK:
        previous=core.rotary;core.rotary=cached_rotary
        try:yield COUNTS
        finally:core.rotary=previous
