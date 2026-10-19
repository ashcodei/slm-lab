"""Training-only CPU bfloat16 representations of immutable own base matrices.

The original FP32 parameters stay registered and exported. Cached tensors are
detached casts of those same values, add no learned degrees of freedom, and
are explicitly counted as additional resident bytes/values. They are never
exported as another model or silently used outside CPU bfloat16 autocast.
The caller must keep the frozen base unchanged between forward and backward.
"""
from types import MethodType
import weakref

import torch
from torch.nn import functional as F

from .own_dense_actor_v1 import _FrozenLinear,_FrozenHead


def signature(parameter):
    return (id(parameter),parameter._version,parameter.data_ptr(),parameter.dtype,parameter.device,
            tuple(parameter.shape),tuple(parameter.stride()),parameter.requires_grad)


class CPUFrozenCastCache:
    def __init__(self):
        self.entries={};self.patched=[];self.hits=0;self.conversions=0

    def value(self,parameter):
        if parameter.device.type!='cpu' or parameter.dtype!=torch.float32:
            raise ValueError('Only original own FP32 CPU frozen matrices may be cast')
        key=id(parameter);state=signature(parameter);previous=self.entries.get(key)
        if previous is not None and previous[0]() is not parameter:previous=None
        if previous is not None and previous[0]() is parameter and previous[1]==state:
            self.hits+=1;return previous[2]
        result=parameter.detach().to(dtype=torch.bfloat16,device='cpu',copy=True)
        self.entries[key]=(weakref.ref(parameter),state,result);self.conversions+=1
        return result

    def attach(self,view):
        cache=self;known={id(module) for module,_ in self.patched}
        for module in view.modules():
            if type(module) not in {_FrozenLinear,_FrozenHead} or id(module) in known:continue
            original=module.forward
            def forward(current,hidden,_original=original):
                if not torch.is_autocast_enabled('cpu') or torch.get_autocast_dtype('cpu')!=torch.bfloat16 or hidden.device.type!='cpu':
                    return _original(hidden)
                layer=current.original if type(current) is _FrozenLinear else current
                weight=layer.weight
                if weight.device.type!='cpu' or weight.dtype!=torch.float32:return _original(hidden)
                bias=getattr(layer,'bias',None)
                return F.linear(hidden,cache.value(weight),None if bias is None else cache.value(bias))
            module.forward=MethodType(forward,module);self.patched.append((module,original));known.add(id(module))

    def assert_current(self):
        if any(ref() is None or signature(ref())!=state for ref,state,_ in self.entries.values()):
            raise ValueError('Frozen source changed while its cached cast was in use')

    def inventory(self):
        return dict(additional_training_only_cast_values=sum(t.numel() for _,_,t in self.entries.values()),
            additional_training_only_cast_bytes=sum(t.numel()*t.element_size() for _,_,t in self.entries.values()),
            cached_original_parameter_tensors=len(self.entries),actual_conversions=self.conversions,actual_cache_hits=self.hits,
            added_learned_parameters=0,cache_excluded_from_student_export=True)

    def close(self):
        for module,original in self.patched:module.forward=original
        self.patched.clear();self.entries.clear()
