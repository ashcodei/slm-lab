"""Fold our own code factors and grow a dense model with neutral residuals.

Weight merging and function-preserving growth are established techniques.
FP32 factor merging changes arithmetic order; native quality/parity must be
measured. Added pre-norm blocks start with zero attention/FFN output matrices,
so their initial residual function is the identity. No external weights or
old general backbone is retained in the resulting single dense model.
"""
from contextlib import nullcontext
from dataclasses import asdict,dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .model import DecoderCache,ModelConfig,RMSNorm,TinyMultimodalLM
from .own_aligned_assistant_v1 import ByteWordHead
from .own_aligned_assistant_v2 import ExpandedByteWordHead
from .own_code_delta_v1 import OwnCodeDelta,matrix_names,norm_names

FORMAT='own-folded-code-grown-dense-text-image-heads-v1'


@dataclass(frozen=True)
class DenseCache(DecoderCache):
    parameter_signature:tuple


class DenseBackbone(TinyMultimodalLM):
    def signature(self):return tuple((id(p),p._version,p.data_ptr(),p.dtype,p.device) for p in self.parameters())

    def bind(self,state):
        c=state['cache']
        return {**state,'cache':DenseCache(c.layers,c.image_tokens,c.text_length,c.owner,self.signature())}

    @torch.no_grad()
    def prefill(self,*args,**kwargs):return self.bind(super().prefill(*args,**kwargs))

    @torch.no_grad()
    def decode(self,input_ids,cache):
        if (type(cache) is not DenseCache or cache.owner!=id(self) or cache.parameter_signature!=self.signature()):
            raise ValueError('Dense cache must match this actual model and unchanged parameters')
        return self.bind(super().decode(input_ids,cache))


def new_backbone(config, seed):
    if type(seed) is not int:raise ValueError('Explicit integer own initialization seed required')
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        return DenseBackbone(config)


@torch.no_grad()
def fold_code(language, *, seed=948431):
    if type(language) is not OwnCodeDelta or next(language.parameters()).device.type!='cpu':
        raise ValueError('Exact existing own CPU code computation required; no external or moved source')
    language.assert_partition();folded=new_backbone(language.config,seed)
    if next(folded.parameters()).dtype!=language.base.embedding.weight.dtype:
        raise ValueError('Original own FP32 construction and code weight dtype must agree')
    state=folded.state_dict();base=language.base.state_dict()
    # Copy every unchanged vision/critic parameter, then the trained own code
    # vocabulary/norms. No source module or tensor is registered in the result.
    for name,value in base.items():state[name].copy_(value)
    state['embedding.weight'].copy_(language.parts['embedding'].weight)
    for name,module in zip(norm_names(language.config.num_layers),language.parts['norms']):state[name].copy_(module.weight)
    for name,parts in zip(matrix_names(language.config.num_layers),language.parts['linears']):
        state[name].copy_(base[name]+parts.up@parts.down)
    if folded.embedding.weight is not folded.lm_head.weight:raise ValueError('One actual tied own code vocabulary table required')
    return folded.eval()


def depth_layout(original_layers,target_layers):
    if (type(original_layers) is not int or type(target_layers) is not int or not 1<=original_layers<=target_layers):
        raise ValueError('Positive original depth and no shrinking in the declared growth required')
    # Evenly interleave added blocks after existing blocks. Original blocks
    # always retain order; -1 denotes a new identity-at-initialization block.
    result=[]
    for old in range(original_layers):
        result.append(old)
        extras=((old+1)*(target_layers-original_layers)//original_layers)-(old*(target_layers-original_layers)//original_layers)
        result.extend([-1]*extras)
    if len(result)!=target_layers:raise ValueError('Every original and added block must be counted')
    return result


@torch.no_grad()
def grow(folded,target_layers=35,*,seed=948531):
    if type(folded) is not DenseBackbone or next(folded.parameters()).device.type!='cpu':
        raise ValueError('Actual CPU dense own folded network required')
    layout=depth_layout(folded.config.num_layers,target_layers)
    grown=new_backbone(ModelConfig(**{**asdict(folded.config),'num_layers':target_layers}),seed)
    before=folded.state_dict();after=grown.state_dict()
    for name,tensor in before.items():
        if not name.startswith('blocks.'):after[name].copy_(tensor)
    for position,old in enumerate(layout):
        block=grown.blocks[position]
        if old>=0:block.load_state_dict(folded.blocks[old].state_dict(),strict=True)
        else:
            block.attn.o_proj.weight.zero_();block.down.weight.zero_()
            block.attn_norm.weight.fill_(1.);block.ffn_norm.weight.fill_(1.)
    return grown.eval(),layout


class OwnDenseAssistant(nn.Module):
    def __init__(self,network,*,seed=948631,reward_hidden=128,word_temporal=96,pointing_rank=256,
                 router_hidden=64,detector_hidden=64,word_expansion=5,parameter_limit=10**9):
        super().__init__()
        if type(network) is not DenseBackbone or any(type(v) is not int or v<1 for v in
                (seed,reward_hidden,word_temporal,pointing_rank,router_hidden,detector_hidden,word_expansion)):
            raise ValueError('Actual own dense network and explicit positive bounded head dimensions required')
        if pointing_rank>network.config.hidden_size:raise ValueError('Pointing rank must fit the actual hidden representation')
        self.network=network;c=network.config;h,v=c.hidden_size,c.vision_hidden_size
        self.settings=dict(seed=seed,reward_hidden=reward_hidden,word_temporal=word_temporal,pointing_rank=pointing_rank,
            router_hidden=router_hidden,detector_hidden=detector_hidden,word_expansion=word_expansion,parameter_limit=parameter_limit)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            self.heads=nn.ModuleDict(dict(
                reward=nn.Sequential(RMSNorm(h,c.norm_eps),nn.Linear(h,reward_hidden,bias=False),nn.SiLU(),nn.Linear(reward_hidden,1,bias=False)),
                word=ExpandedByteWordHead(ByteWordHead(v,c.norm_eps,word_temporal),word_expansion),
                pointing=nn.ModuleDict(dict(query=nn.Linear(h,pointing_rank,bias=False),patch=nn.Linear(h,pointing_rank,bias=False))),
                router=nn.Sequential(nn.Linear(h,router_hidden),nn.SiLU(),nn.Linear(router_hidden,3)),
                detector=nn.Sequential(nn.Conv2d(v,v,3,padding=1,groups=v),nn.Conv2d(v,detector_hidden,1),nn.SiLU(),nn.Conv2d(detector_hidden,5,1))))
        self.assert_inventory()

    @property
    def config(self):return self.network.config

    def counts(self):
        network=sum(p.numel() for p in self.network.parameters());heads={n:sum(p.numel() for p in module.parameters()) for n,module in self.heads.items()}
        return dict(dense_text_image_critic=network,heads=heads,total_unique=sum(p.numel() for p in self.parameters()),
                    remaining_below_1B=10**9-network-sum(heads.values()))

    def assert_inventory(self):
        counts=self.counts();limit=self.settings['parameter_limit']
        if (type(limit) is not int or not counts['total_unique']<limit<=10**9
                or counts['total_unique']!=counts['dense_text_image_critic']+sum(counts['heads'].values())
                or self.network.embedding.weight is not self.network.lm_head.weight
                or any(isinstance(module,OwnCodeDelta) for module in self.modules())):
            raise ValueError('Every actual dense/head value, one tied table and no retained old code/general model must fit under1B')

    def forward(self,*args,**kwargs):return self.network(*args,**kwargs)
    def generate(self,*args,**kwargs):return self.network.generate(*args,**kwargs)

    def language_named_parameters(self):
        return [(n,p) for n,p in self.network.named_parameters() if not n.startswith(('vision.','value_head.'))]

    def future_actor_parameter_budget(self,rank=16):
        if type(rank) is not int or rank<1:raise ValueError('Explicit positive future actor rank required')
        extra=sum(rank*sum(self.network.get_parameter(n).shape) for n in matrix_names(self.config.num_layers))
        extra+=sum(self.network.get_parameter(n).numel() for n in norm_names(self.config.num_layers))
        return dict(actual_dense_model=self.counts()['total_unique'],proposed_actor_rank=rank,
            proposed_actor_values=extra,proposed_total=self.counts()['total_unique']+extra,
            actor_not_instantiated_or_trained=True,fits_under_1B=self.counts()['total_unique']+extra<10**9)

    def patch_grid(self,pixels):
        c=self.config;vision=self.network.vision
        if pixels.ndim!=4 or pixels.shape[1]!=3 or any(side<c.patch_size or side>c.max_image_size or side%c.patch_size for side in pixels.shape[-2:]):
            raise ValueError('Actual bounded patch-aligned complete RGB image required')
        h,w=(side//c.patch_size for side in pixels.shape[-2:]);base=c.image_size//c.patch_size
        position=vision.position
        if (h,w)!=(base,base):
            position=F.interpolate(position.float().reshape(1,base,base,c.vision_hidden_size).permute(0,3,1,2),
                                   size=(h,w),mode='bicubic',align_corners=False).flatten(2).transpose(1,2).to(position.dtype)
        x=vision.patch(pixels).flatten(2).transpose(1,2)+position
        for block in vision.blocks:
            x=checkpoint(block,x,use_reentrant=False) if self.network.gradient_checkpointing and self.training else block(x)
        return vision.norm(x).reshape(pixels.shape[0],h,w,c.vision_hidden_size)

    def word_logits(self,pixels):return self.heads['word'](self.patch_grid(pixels))
    def detection_logits(self,pixels):return self.heads['detector'](self.patch_grid(pixels).permute(0,3,1,2))

    def preference_score(self,input_ids,attention_mask=None,*,detach_features=True):
        mask=torch.ones_like(input_ids,dtype=torch.bool) if attention_mask is None else attention_mask.bool()
        if mask.shape!=input_ids.shape or not bool(mask.any(1).all()) or bool((mask[:,1:]&~mask[:,:-1]).any()):
            raise ValueError('Complete nonempty right-padded original response mask required')
        with torch.no_grad() if detach_features else nullcontext():
            hidden=self.network(input_ids,attention_mask=mask,return_hidden=True,return_logits=False)['hidden']
        final=hidden[torch.arange(len(hidden),device=hidden.device),mask.sum(1)-1]
        with torch.autocast(device_type=hidden.device.type,enabled=False):return self.heads['reward'](final.float()).squeeze(-1)

    def point_logits(self,input_ids,pixels,attention_mask=None):
        mask=torch.ones_like(input_ids,dtype=torch.bool) if attention_mask is None else attention_mask.bool()
        content=mask&(input_ids!=0)&(input_ids!=1)&(input_ids!=2)
        if mask.shape!=input_ids.shape or not bool(content.any(1).all()):raise ValueError('Complete original content-bearing query required')
        with torch.no_grad():
            hidden=self.network(input_ids,attention_mask=mask,return_hidden=True,return_logits=False)['hidden']
            query=(hidden.float()*content[:,:,None]).sum(1)/content.sum(1,keepdim=True)
        patches=self.network.vision(pixels,self.network.gradient_checkpointing)
        q=self.heads['pointing']['query'](query.to(patches.dtype));k=self.heads['pointing']['patch'](patches)
        return (k.float()*q.float()[:,None]).sum(-1)/math.sqrt(self.settings['pointing_rank'])

    def export(self,sources):
        self.assert_inventory()
        if not sources or any(not isinstance(k,str) or not isinstance(v,str) or len(v)!=64 or any(c not in '0123456789abcdef' for c in v) for k,v in sources.items()):
            raise ValueError('Exact named own source digests required')
        states={name:{k:t.detach().to(device='cpu',copy=True) for k,t in module.state_dict().items() if k!='lm_head.weight'}
                for name,module in [('network',self.network),('heads',self.heads)]}
        states['network']['lm_head.weight']=states['network']['embedding.weight']
        return dict(format=FORMAT,config=asdict(self.config),settings=self.settings,actual_counts=self.counts(),sources=dict(sources),
                    **states,candidate_promoted=False,goal_achieved=False)

    def validate_export(self,payload,*,expected_sources):
        if (payload.get('format')!=FORMAT or payload.get('config')!=asdict(self.config) or payload.get('settings')!=self.settings
                or payload.get('actual_counts')!=self.counts() or payload.get('sources')!=expected_sources or not expected_sources
                or payload.get('candidate_promoted') is not False or payload.get('goal_achieved') is not False):
            raise ValueError('Exact complete own dense model/head/source specification required before mutation')
        for name,module in [('network',self.network),('heads',self.heads)]:
            live=module.state_dict();saved=payload.get(name)
            if (not isinstance(saved,dict) or saved.keys()!=live.keys()
                    or any(not isinstance(t,torch.Tensor) or t.shape!=live[k].shape or t.dtype!=live[k].dtype or not bool(torch.isfinite(t).all()) for k,t in saved.items())):
                raise ValueError('Every finite dense/head tensor with exact shape/dtype required before mutation')
        if not torch.equal(payload['network']['embedding.weight'],payload['network']['lm_head.weight']):
            raise ValueError('Saved vocabulary and language head must be one equal tied table')

    def import_export(self,payload,*,expected_sources):
        self.validate_export(payload,expected_sources=expected_sources)
        self.network.load_state_dict(payload['network'],strict=True);self.heads.load_state_dict(payload['heads'],strict=True)
        self.assert_inventory()
