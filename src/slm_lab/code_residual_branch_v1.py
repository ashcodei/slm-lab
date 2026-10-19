"""Explicit general/code routes over one frozen own backbone plus trainable residuals.

Adapter-style parameter sharing is established prior work (Houlsby et al., ICML
2019). This local residual placement/capacity is an unproven coding hypothesis,
not a claim of novelty, preserved automatic routing quality, or model skill.
"""
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .model import DecoderCache,TinyMultimodalLM


class CodeResidual(nn.Module):
    def __init__(self,hidden,width,device=None,dtype=None):
        super().__init__()
        self.down=nn.Linear(hidden,width,bias=False,device=device,dtype=dtype)
        self.up=nn.Linear(width,hidden,bias=False,device=device,dtype=dtype)
        nn.init.normal_(self.down.weight,std=.02)
        nn.init.zeros_(self.up.weight)

    def forward(self,hidden):
        return hidden+self.up(F.silu(self.down(hidden)))


class _CodeBlock(nn.Module):
    def __init__(self,original,residual):
        super().__init__();self.original=original;self.residual=residual

    def forward(self,hidden,mask=None,causal=False):
        return self.residual(self.original(hidden,mask,causal))

    @torch.no_grad()
    def cached(self,hidden,past=None,mask=None,causal=False):
        hidden,stored=self.original.cached(hidden,past,mask,causal)
        return self.residual(hidden),stored


@dataclass(frozen=True)
class CodeBranchCache(DecoderCache):
    parameter_signature:tuple


class _CodeView(TinyMultimodalLM):
    """Reuse unchanged own decoder methods; only the code block list differs."""
    def __init__(self,base,residuals):
        # Deliberately do not construct/reinitialize a second backbone.
        nn.Module.__init__(self)
        self.config=base.config
        self.embedding=base.embedding;self.vision=base.vision
        self.norm=base.norm;self.lm_head=base.lm_head;self.value_head=base.value_head
        self.blocks=nn.ModuleList(_CodeBlock(block,residual) for block,residual in zip(base.blocks,residuals))
        self.gradient_checkpointing=False

    def _signature(self):
        # Versions reject stale KV after optimizer/load operations; pointer,
        # dtype/device identity also reject movement or replaced parameters.
        return tuple((id(p),p._version,p.data_ptr(),p.dtype,p.device) for p in self.parameters())

    def _bind_cache(self,state):
        cache=state['cache']
        return {**state,'cache':CodeBranchCache(cache.layers,cache.image_tokens,cache.text_length,cache.owner,self._signature())}

    @torch.no_grad()
    def prefill(self,input_ids,pixel_values=None):
        return self._bind_cache(super().prefill(input_ids,pixel_values))

    @torch.no_grad()
    def decode(self,input_ids,cache):
        if not isinstance(cache,CodeBranchCache) or cache.owner!=id(self):
            raise ValueError('Code cache must belong to this code route instance')
        if cache.parameter_signature!=self._signature():
            raise ValueError('Code route parameters changed after cache creation')
        return self._bind_cache(super().decode(input_ids,cache))


class RoutedCodeResidual(nn.Module):
    """Explicit caller-selected routes; no hidden mutable global route switch."""
    def __init__(self,base,width=2048,parameter_limit=10**9):
        super().__init__()
        if type(base) is not TinyMultimodalLM:
            raise ValueError('The exact original own multimodal backbone implementation is required')
        if type(width) is not int or width<1 or type(parameter_limit) is not int or not 1<=parameter_limit<=10**9:
            raise ValueError('Explicit positive residual width and unique-parameter limit required')
        original=sum(p.numel() for p in base.parameters())
        added=2*base.config.hidden_size*width*base.config.num_layers
        if original+added>=parameter_limit:raise ValueError('All unique frozen/trainable parameters must fit the total model limit')
        self.base=base;self.width=width;self.parameter_limit=parameter_limit
        for parameter in self.base.parameters():parameter.requires_grad_(False)
        self.residuals=nn.ModuleList(CodeResidual(base.config.hidden_size,width,
            device=base.embedding.weight.device,dtype=base.embedding.weight.dtype) for _ in base.blocks)
        self.code=_CodeView(base,self.residuals)
        self.assert_partition()

    @property
    def config(self):return self.base.config

    def parameter_counts(self):
        return dict(total_unique=sum(p.numel() for p in self.parameters()),
            frozen_base=sum(p.numel() for p in self.base.parameters()),
            trainable_code=sum(p.numel() for p in self.residuals.parameters()))

    def assert_partition(self):
        if len(self.residuals)!=len(self.base.blocks) or len(self.code.blocks)!=len(self.base.blocks):
            raise ValueError('Every original decoder layer needs its explicit residual')
        for block,original,residual in zip(self.code.blocks,self.base.blocks,self.residuals):
            if block.original is not original or block.residual is not residual:
                raise ValueError('Code/general routes must share exactly the original backbone objects')
        for name in ['embedding','vision','norm','lm_head','value_head']:
            if getattr(self.code,name) is not getattr(self.base,name):raise ValueError('A second backbone/component is forbidden')
        if (any(p.requires_grad for p in self.base.parameters()) or not all(p.requires_grad for p in self.residuals.parameters())
                or sum(p.numel() for p in self.parameters())>=self.parameter_limit):
            raise ValueError('Frozen general weights and bounded trainable residuals required')

    def adapter_parameters(self):
        self.assert_partition();return list(self.residuals.parameters())

    def adapter_state(self):
        return {name:value.detach().cpu().clone() for name,value in self.residuals.state_dict().items()}

    def load_adapter_state(self,state):
        expected=self.residuals.state_dict()
        if (set(state)!=set(expected) or any(not isinstance(state[k],torch.Tensor) or state[k].shape!=expected[k].shape
                or state[k].dtype!=expected[k].dtype or not bool(torch.isfinite(state[k]).all()) for k in expected)):
            raise ValueError('Complete finite code-residual tensor inventory/shapes/dtypes required')
        self.residuals.load_state_dict(state,strict=True);self.assert_partition()

    def route(self,name):
        if name=='general':return self.base
        if name=='code':return self.code
        raise ValueError('Caller must explicitly select the general or code route')

    def forward(self,*args,route='general',**kwargs):
        return self.route(route)(*args,**kwargs)

    def generate(self,*args,route='general',**kwargs):
        return self.route(route).generate(*args,**kwargs)
