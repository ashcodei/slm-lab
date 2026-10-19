"""One own dense backbone with a bounded policy adapter and separate critic.

The original dense parameters keep their weights and trainability metadata.
Functional detaches freeze them while preserving derivatives through hidden
states to earlier adapters. This implements a policy graph, not useful RLHF.
"""
import copy
from dataclasses import asdict
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

from .own_aligned_assistant_v1 import _ZeroFactors
from .own_code_delta_v1 import _DeltaBlock,matrix_names,norm_names
from .own_dense_growth_v1 import DenseBackbone,OwnDenseAssistant
from .whole_human_sft_evidence_v1 import tensor_digest

FORMAT='own-single-dense-backbone-policy-adapter-separate-critic-v1'


class _FrozenLinear(nn.Module):
    def __init__(self,original):
        super().__init__();self.original=original

    def forward(self,hidden):
        bias=self.original.bias
        return F.linear(hidden,self.original.weight.detach(),None if bias is None else bias.detach())


class _FrozenEmbedding(nn.Module):
    def __init__(self,original):
        super().__init__();self.weight=original.weight

    def forward(self,ids):return F.embedding(ids,self.weight.detach())


class _FrozenHead(_FrozenEmbedding):
    def forward(self,hidden):return F.linear(hidden,self.weight.detach())


class _FrozenVision(nn.Module):
    def __init__(self,original):
        super().__init__();self.original=original

    def forward(self,pixels,gradient_checkpointing=False):
        with torch.no_grad():return self.original(pixels,False)


def frozen_block(block):
    attn=block.attn
    attention=SimpleNamespace(heads=attn.heads,kv_heads=attn.kv_heads,head_dim=attn.head_dim,rope_theta=attn.rope_theta,
        **{n:_FrozenLinear(getattr(attn,n)) for n in ('q_proj','k_proj','v_proj','o_proj')})
    return SimpleNamespace(attn=attention,**{n:_FrozenLinear(getattr(block,n)) for n in ('gate','up','down')})


class DensePolicyView(DenseBackbone):
    def __init__(self,base,parts):
        nn.Module.__init__(self)
        # Register every counted parameter, including auxiliary heads, even
        # when this native view is passed alone to an evaluator.
        self.base,self.parts=base,parts;network=base.network;self.config=base.config
        self.embedding=_FrozenEmbedding(network.embedding);self.lm_head=_FrozenHead(network.embedding)
        self.vision=_FrozenVision(network.vision);self.value_head=parts['critic'];self.norm=parts['norms'][-1]
        self.blocks=nn.ModuleList(_DeltaBlock(frozen_block(block),parts['linears'][i*7:(i+1)*7],parts['norms'][i*2:i*2+2])
            for i,block in enumerate(network.blocks))
        self.gradient_checkpointing=False

    def train(self,mode=True):
        super().train(mode);self.base.eval();return self


def budget(base,rank):
    if type(base) is not OwnDenseAssistant or type(rank) is not int or rank<1:
        raise ValueError('Exact own dense model and positive explicit policy rank required')
    base.assert_inventory();network=base.network
    matrices=[network.get_parameter(n) for n in matrix_names(base.config.num_layers)]
    if any(rank>min(p.shape) for p in matrices):raise ValueError('Policy rank must fit every actual dense matrix')
    actor=sum(rank*sum(p.shape) for p in matrices)+sum(network.get_parameter(n).numel() for n in norm_names(base.config.num_layers))
    return dict(frozen_dense_and_all_heads=base.counts()['total_unique'],actor=actor,
        separate_policy_critic=base.config.hidden_size,total_unique=base.counts()['total_unique']+actor+base.config.hidden_size)


class OwnDenseActor(nn.Module):
    def __init__(self,base,*,rank=16,seed=949331,parameter_limit=10**9):
        super().__init__();expected=budget(base,rank)
        if (type(seed) is not int or seed<1 or type(parameter_limit) is not int
                or not expected['total_unique']<parameter_limit<=10**9):
            raise ValueError('Every actual backbone/head/actor/separate critic value must fit strictly under1B')
        self.base=base;self.settings=dict(rank=rank,seed=seed,parameter_limit=parameter_limit);network=base.network
        device=network.embedding.weight.device;devices=[] if device.type=='cpu' else [device.index]
        with torch.random.fork_rng(devices=devices):
            torch.random.default_generator.manual_seed(seed)
            for d in devices:torch.cuda.default_generators[d].manual_seed(seed)
            self.parts=nn.ModuleDict(dict(
                linears=nn.ModuleList(_ZeroFactors(network.get_submodule(n.removesuffix('.weight')),rank)
                    for n in matrix_names(base.config.num_layers)),
                norms=nn.ModuleList(copy.deepcopy(network.get_submodule(n.removesuffix('.weight')))
                    for n in norm_names(base.config.num_layers)),
                critic=nn.Linear(base.config.hidden_size,1,bias=False,device=device,dtype=network.embedding.weight.dtype)))
            nn.init.zeros_(self.parts['critic'].weight)
        self.policy=DensePolicyView(base,self.parts);self.expected_counts=expected;self.assert_shared()

    @property
    def config(self):return self.base.config

    def counts(self):return {**self.expected_counts,'actual_unique':sum(p.numel() for p in self.parameters())}

    def assert_shared(self):
        self.base.assert_inventory();old={id(p) for p in self.base.parameters()};added={id(p) for p in self.parts.parameters()}
        if (old&added or self.policy.base is not self.base or self.policy.parts is not self.parts
                or self.policy.embedding.weight is not self.base.network.embedding.weight
                or self.policy.lm_head.weight is not self.base.network.embedding.weight
                or any(not p.requires_grad for p in self.parts.parameters())
                or self.counts()['actual_unique']!=self.expected_counts['total_unique']
                or sum(p.numel() for p in self.policy.parameters())!=self.expected_counts['total_unique']
                or self.expected_counts['total_unique']>=self.settings['parameter_limit']):
            raise ValueError('Exactly one shared dense backbone, one table and all bounded policy/critic/head values required')

    def train(self,mode=True):
        super().train(mode);self.base.eval();return self

    def forward(self,*args,**kwargs):return self.policy(*args,**kwargs)
    def generate(self,*args,**kwargs):return self.policy.generate(*args,**kwargs)

    def named_selected(self,groups):
        if not groups or len(set(groups))!=len(groups) or any(g not in ('actor','critic') for g in groups):
            raise ValueError('Explicit distinct actor/critic optimizer groups required')
        return [(n,p) for n,p in self.parts.named_parameters() if ('critic' if n.startswith('critic.') else 'actor') in groups]

    def export(self,sources):
        self.assert_shared()
        if (not sources or any(not isinstance(k,str) or not isinstance(v,str) or len(v)!=64
                or any(c not in '0123456789abcdef' for c in v) for k,v in sources.items())):
            raise ValueError('Explicit complete dense and human-source SHA bindings required')
        return dict(format=FORMAT,settings=self.settings,config=asdict(self.config),actual_counts=self.counts(),sources=dict(sources),
            frozen_dense_sha256=tensor_digest(self.base.named_parameters()),
            parts={n:t.detach().to(device='cpu',copy=True) for n,t in self.parts.state_dict().items()},
            candidate_promoted=False,goal_achieved=False)

    def validate_export(self,payload,*,expected_sources):
        if (payload.get('format')!=FORMAT or payload.get('settings')!=self.settings or payload.get('config')!=asdict(self.config)
                or payload.get('actual_counts')!=self.counts() or payload.get('sources')!=expected_sources or not expected_sources
                or payload.get('frozen_dense_sha256')!=tensor_digest(self.base.named_parameters())
                or payload.get('candidate_promoted') is not False or payload.get('goal_achieved') is not False):
            raise ValueError('Exact actor computation/counts/frozen dense features/sources required before mutation')
        live=self.parts.state_dict();saved=payload.get('parts')
        if (not isinstance(saved,dict) or saved.keys()!=live.keys()
                or any(not isinstance(t,torch.Tensor) or t.shape!=live[n].shape or t.dtype!=live[n].dtype
                    or not bool(torch.isfinite(t).all()) for n,t in saved.items())):
            raise ValueError('Every actual finite policy/critic tensor with original shape/dtype required')

    def import_export(self,payload,*,expected_sources):
        self.validate_export(payload,expected_sources=expected_sources)
        self.parts.load_state_dict(payload['parts'],strict=True);self.assert_shared()
