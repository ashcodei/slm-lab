"""Observe unchanged cached greedy text/image outputs on their actual device."""
from contextlib import contextmanager
import hashlib

import numpy as np
import torch

from .evaluation_provenance import pixels_sha256
from .native_greedy_observer_v1 import signature


@contextmanager
def observe(model,callback,*,device,precision='bf16'):
    device=torch.device(device)
    if (device.type not in {'cpu','cuda'} or precision not in {'fp32','bf16'} or model.training
            or not callable(callback) or getattr(model,'_device_native_observer_active_v1',False)
            or any(p.device!=device for p in model.parameters())):
        raise ValueError('One complete eval-mode model on the declared actual device and explicit precision required')
    original=model.generate;before=signature(model);events=[]
    def generate(input_ids,*args,**kwargs):
        if (input_ids.ndim!=2 or input_ids.shape[0]!=1 or input_ids.device!=device or len(args)>1
                or kwargs.get('temperature',0)!=0 or kwargs.get('return_rollout',False)
                or kwargs.get('use_cache',True) is not True or type(kwargs.get('max_new_tokens')) is not int
                or kwargs['max_new_tokens']<1):raise ValueError('Explicit complete single-example cached greedy text/image generation required')
        pixels=args[0] if args else kwargs.get('pixel_values')
        enabled=torch.is_autocast_enabled(device.type)
        if (precision=='bf16' and (not enabled or torch.get_autocast_dtype(device.type)!=torch.bfloat16)
                or precision=='fp32' and enabled or pixels is not None and pixels.device!=device):
            raise ValueError('Actual generation precision and image device must match the declared profile')
        output=original(input_ids,*args,**kwargs)
        count=output.shape[1]-input_ids.shape[1] if isinstance(output,torch.Tensor) and output.ndim==2 else -1
        if (not isinstance(output,torch.Tensor) or output.ndim!=2 or output.shape[0]!=1 or output.device!=device
                or not torch.equal(output[:,:input_ids.shape[1]],input_ids)
                or not 1<=count<=kwargs['max_new_tokens'] or before!=signature(model)):
            raise ValueError('Actual unchanged output prefix, bounded native actions and unmodified parameters required')
        prefix=input_ids[0].cpu().tolist();actions=output[0,len(prefix):].cpu().tolist();eos=kwargs.get('eos_token_id')
        event=dict(case_position=len(events),original_input_token_ids=prefix,actual_generated_token_ids=actions,
            original_input_ids_sha256=hashlib.sha256(np.asarray(prefix,dtype='<u2').tobytes()).hexdigest(),
            input_pixels_sha256=None if pixels is None else pixels_sha256(pixels),
            requested_native_action_budget=kwargs['max_new_tokens'],actual_native_action_count=count,
            actual_stopped_on_EOS=bool(eos is not None and actions[-1]==eos),
            inference_device_type=device.type,inference_device_index=device.index,inference_precision=precision,
            greedy_native_output_tensor_returned_unchanged=True,generated_program_not_repaired=True)
        events.append(event);callback(event);return output
    model.generate=generate;model._device_native_observer_active_v1=True
    try:yield events
    finally:model.generate=original;model._device_native_observer_active_v1=False
