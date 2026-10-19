"""Compress an own learned code update while preserving a separate general view.

Low-rank weight updates are established prior work (LoRA). Here factors are
initialized from two of this project's own scratch-trained policies. Plain
weight-space SVD is only a hypothesis: retained matrix energy does not prove
that generated code or algorithmic correctness is preserved.
"""
import copy
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .model import Attention,Block,DecoderCache,TinyMultimodalLM


def matrix_names(layers):
    return [f'blocks.{i}.{name}.weight' for i in range(layers)
        for name in ['attn.q_proj','attn.k_proj','attn.v_proj','attn.o_proj','gate','up','down']]


def norm_names(layers):
    return [f'blocks.{i}.{name}.weight' for i in range(layers) for name in ['attn_norm','ffn_norm']]+['norm.weight']


def factorize(delta,rank,*,seed=783211,iterations=2):
    if (not isinstance(delta,torch.Tensor) or delta.device.type!='cpu' or delta.ndim!=2 or not torch.isfinite(delta).all()
            or type(rank) is not int or rank<1):
        raise ValueError('Finite own matrix difference and positive explicit rank required')
    rank=min(rank,min(delta.shape));oversample=min(min(delta.shape),rank+16)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        if oversample==min(delta.shape):
            left,singular,right=torch.linalg.svd(delta.float(),full_matrices=False)
            right=right.T
        else:
            left,singular,right=torch.svd_lowrank(delta.float(),q=oversample,niter=iterations)
    root=singular[:rank].clamp_min(0).sqrt()
    return dict(down=(root[:,None]*right[:,:rank].T).contiguous(),
        up=(left[:,:rank]*root[None,:]).contiguous())


class _Factors(nn.Module):
    def __init__(self,values,original):
        super().__init__()
        self.down=nn.Parameter(values['down'].detach().to(original.weight).clone())
        self.up=nn.Parameter(values['up'].detach().to(original.weight).clone())


class _DeltaLinear(nn.Module):
    def __init__(self,original,factors):
        super().__init__();self.original=original;self.factors=factors

    def forward(self,hidden):
        return self.original(hidden)+F.linear(F.linear(hidden,self.factors.down),self.factors.up)


class _DeltaAttention(Attention):
    def __init__(self,original,factors):
        nn.Module.__init__(self)
        self.heads,self.kv_heads,self.head_dim=original.heads,original.kv_heads,original.head_dim
        self.rope_theta=original.rope_theta
        for name,parts in zip(['q_proj','k_proj','v_proj','o_proj'],factors):
            setattr(self,name,_DeltaLinear(getattr(original,name),parts))


class _DeltaBlock(Block):
    def __init__(self,original,factors,norms):
        nn.Module.__init__(self)
        self.attn_norm,self.ffn_norm=norms
        self.attn=_DeltaAttention(original.attn,factors[:4])
        for name,parts in zip(['gate','up','down'],factors[4:]):
            setattr(self,name,_DeltaLinear(getattr(original,name),parts))


class _TiedHead(nn.Linear):
    def __init__(self,embedding):
        nn.Module.__init__(self);self.weight=embedding.weight;self.bias=None
        self.in_features=embedding.embedding_dim;self.out_features=embedding.num_embeddings


@dataclass(frozen=True)
class OwnDeltaCache(DecoderCache):
    parameter_signature:tuple


class _CodeView(TinyMultimodalLM):
    def __init__(self,base,parts):
        nn.Module.__init__(self)
        # This is the same frozen object, not a second backbone. Registering
        # the general view also makes every unique combined parameter visible
        # when an evaluator receives only the explicit code view.
        self.frozen_general=base;self.config=base.config
        self.embedding=parts['embedding'];self.lm_head=_TiedHead(self.embedding)
        self.vision=base.vision;self.value_head=base.value_head;self.norm=parts['norms'][-1]
        self.blocks=nn.ModuleList(_DeltaBlock(original,parts['linears'][i*7:(i+1)*7],parts['norms'][i*2:i*2+2])
            for i,original in enumerate(base.blocks))
        self.gradient_checkpointing=False

    def signature(self):
        return tuple((id(p),p._version,p.data_ptr(),p.dtype,p.device) for p in self.parameters())

    def bind(self,state):
        cache=state['cache']
        return {**state,'cache':OwnDeltaCache(cache.layers,cache.image_tokens,cache.text_length,cache.owner,self.signature())}

    @torch.no_grad()
    def prefill(self,input_ids,pixel_values=None):
        return self.bind(super().prefill(input_ids,pixel_values))

    @torch.no_grad()
    def decode(self,input_ids,cache):
        if not isinstance(cache,OwnDeltaCache) or cache.owner!=id(self) or cache.parameter_signature!=self.signature():
            raise ValueError('Own code cache must match this route and unchanged actual parameters')
        return self.bind(super().decode(input_ids,cache))


class OwnCodeDelta(nn.Module):
    def __init__(self,base,components,*,parameter_limit=10**9,reserved_parameters=786432):
        super().__init__()
        if type(base) is not TinyMultimodalLM or type(reserved_parameters) is not int or reserved_parameters<0:
            raise ValueError('Exact own general backbone and nonnegative extra-head parameter reserve required')
        names=matrix_names(base.config.num_layers);norms=norm_names(base.config.num_layers);state=base.state_dict()
        if (set(components)!=set(['matrices','embedding','norms']) or set(components['matrices'])!=set(names)
                or set(components['norms'])!=set(norms)):
            raise ValueError('Every original own decoder matrix, norm and tied code embedding required')
        extra=0
        for name in names:
            values=components['matrices'][name];out_size,in_size=state[name].shape
            if (set(values)!=set(['down','up']) or values['down'].ndim!=2 or values['up'].ndim!=2
                    or values['down'].shape[1]!=in_size or values['up'].shape[0]!=out_size
                    or values['down'].shape[0]!=values['up'].shape[1] or not 1<=values['down'].shape[0]<=min(out_size,in_size)
                    or any(t.dtype!=state[name].dtype or not bool(torch.isfinite(t).all()) for t in values.values())):
                raise ValueError('Complete finite shape-matched own code delta factors required')
            extra+=sum(t.numel() for t in values.values())
        for name,value in [('embedding.weight',components['embedding']),*components['norms'].items()]:
            if value.shape!=state[name].shape or value.dtype!=state[name].dtype or not bool(torch.isfinite(value).all()):
                raise ValueError('Exact finite own target embedding/norm tensors required')
            extra+=value.numel()
        original=sum(p.numel() for p in base.parameters())
        if type(parameter_limit) is not int or not original+extra+reserved_parameters<parameter_limit<=10**9:
            raise ValueError('All unique general/code weights and reserved vision head must fit under1B')
        self.base=base;self.parameter_limit=parameter_limit;self.reserved_parameters=reserved_parameters
        for parameter in base.parameters():parameter.requires_grad_(False)
        linears=[]
        for name in names:
            module=base.get_submodule(name.removesuffix('.weight'))
            linears.append(_Factors(components['matrices'][name],module))
        embedding=nn.Embedding(base.config.vocab_size,base.config.hidden_size,device=base.embedding.weight.device,dtype=base.embedding.weight.dtype)
        with torch.no_grad():embedding.weight.copy_(components['embedding'])
        norm_modules=[]
        for name in norms:
            module=copy.deepcopy(base.get_submodule(name.removesuffix('.weight')))
            module.weight.requires_grad_(True)
            with torch.no_grad():module.weight.copy_(components['norms'][name])
            norm_modules.append(module)
        self.parts=nn.ModuleDict(dict(linears=nn.ModuleList(linears),embedding=embedding,norms=nn.ModuleList(norm_modules)))
        self.code=_CodeView(base,self.parts)
        self.assert_partition()

    @property
    def config(self):return self.base.config

    def counts(self):
        total=sum(p.numel() for p in self.parameters())
        return dict(total_unique=total,frozen_general=sum(p.numel() for p in self.base.parameters()),
            trainable_code=sum(p.numel() for p in self.parts.parameters()),
            reserved_extra_head=self.reserved_parameters,total_with_reserved_head=total+self.reserved_parameters)

    def assert_partition(self):
        if (self.code.frozen_general is not self.base or self.code.vision is not self.base.vision
                or self.code.value_head is not self.base.value_head or self.code.lm_head.weight is not self.code.embedding.weight
                or any(p.requires_grad for p in self.base.parameters())
                or not all(p.requires_grad for p in self.parts.parameters())
                or sum(p.numel() for p in self.code.parameters())!=sum(p.numel() for p in self.parameters())
                or self.counts()['total_with_reserved_head']>=self.parameter_limit):
            raise ValueError('One actual frozen own backbone, tied code embedding and bounded trainable code parts required')

    def code_parameters(self):
        self.assert_partition();return list(self.parts.parameters())

    def route(self,name):
        if name=='general':return self.base
        if name=='code':return self.code
        raise ValueError('Explicit general or code route required; automatic intent routing is unproven')

    def forward(self,*args,route='general',**kwargs):
        return self.route(route)(*args,**kwargs)

    def generate(self,*args,route='general',**kwargs):
        return self.route(route).generate(*args,**kwargs)
