"""An original implementation initialized from scratch.

Dense GQA/RoPE/SwiGLU decoder with a bidirectional patch encoder and image prefix.
This is a research baseline, not a claim of frontier capability or novelty.
"""
from dataclasses import dataclass, asdict
import json
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 32768
    hidden_size: int = 1536
    intermediate_size: int = 4096
    num_layers: int = 24
    num_heads: int = 24
    num_kv_heads: int = 8
    max_text_length: int = 4096
    vision_hidden_size: int = 512
    vision_intermediate_size: int = 2048
    vision_layers: int = 12
    vision_heads: int = 8
    image_size: int = 224
    patch_size: int = 16
    pool_size: int = 2
    rope_theta: float = 10000.0
    norm_eps: float = 1e-6

    def __post_init__(self):
        ints = [v for k,v in asdict(self).items() if k not in {'rope_theta','norm_eps'}]
        if any(not isinstance(v,int) or isinstance(v,bool) or v <= 0 for v in ints):
            raise ValueError('All dimensions must be positive integers.')
        if self.hidden_size % self.num_heads or self.num_heads % self.num_kv_heads:
            raise ValueError('Hidden size and grouped attention heads are incompatible.')
        if self.hidden_size // self.num_heads % 2:
            raise ValueError('RoPE head dimension must be even.')
        if self.vision_hidden_size % self.vision_heads:
            raise ValueError('Vision size must be divisible by vision heads.')
        if self.image_size % self.patch_size or (self.image_size // self.patch_size) % self.pool_size:
            raise ValueError('Image, patch and spatial pooling sizes are incompatible.')
        if self.rope_theta <= 0 or self.norm_eps <= 0:
            raise ValueError('RoPE scale and normalization epsilon must be positive.')

    @property
    def image_tokens(self):
        return (self.image_size // self.patch_size // self.pool_size) ** 2

    @property
    def max_image_size(self):return 2*self.image_size

    def image_token_count(self,size):
        if (not isinstance(size,int) or isinstance(size,bool) or not self.image_size<=size<=self.max_image_size
            or size%(self.patch_size*self.pool_size)):
            raise ValueError(f'Image resolution must be a multiple of {self.patch_size*self.pool_size} '
                             f'between {self.image_size} and {self.max_image_size}.')
        return (size//self.patch_size//self.pool_size)**2

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text()))


class RMSNorm(nn.Module):
    def __init__(self, dim, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        y = x.float() * torch.rsqrt(x.float().square().mean(-1,keepdim=True) + self.eps)
        return y.to(x.dtype) * self.weight.to(x.dtype)


def rotary(x, theta, offset=0):
    # Frequencies are computed in fp32 even under mixed precision.
    dim = x.shape[-1]
    freq = theta ** (-torch.arange(0,dim,2,device=x.device,dtype=torch.float32) / dim)
    angle = torch.arange(offset,offset+x.shape[-2],device=x.device,dtype=torch.float32)[:,None] * freq[None,:]
    cos,sin = angle.cos().to(x.dtype),angle.sin().to(x.dtype)
    a,b = x[...,0::2],x[...,1::2]
    return torch.stack((a*cos-b*sin,a*sin+b*cos),dim=-1).flatten(-2)


class Attention(nn.Module):
    def __init__(self, dim, heads, kv_heads, rope_theta=None):
        super().__init__()
        self.heads,self.kv_heads,self.head_dim = heads,kv_heads,dim//heads
        self.rope_theta = rope_theta
        self.q_proj = nn.Linear(dim,dim,bias=False)
        self.k_proj = nn.Linear(dim,self.head_dim*kv_heads,bias=False)
        self.v_proj = nn.Linear(dim,self.head_dim*kv_heads,bias=False)
        self.o_proj = nn.Linear(dim,dim,bias=False)

    def forward(self, x, mask=None, causal=False):
        b,t,_ = x.shape
        q = self.q_proj(x).view(b,t,self.heads,self.head_dim).transpose(1,2)
        k = self.k_proj(x).view(b,t,self.kv_heads,self.head_dim).transpose(1,2)
        v = self.v_proj(x).view(b,t,self.kv_heads,self.head_dim).transpose(1,2)
        if self.rope_theta is not None:
            q,k = rotary(q,self.rope_theta),rotary(k,self.rope_theta)
        # CUDA SDPA can retain grouped KV heads in fused attention. The CPU
        # fallback expands heads explicitly for portability.
        grouped = self.kv_heads != self.heads
        native_gqa = grouped and x.device.type == 'cuda'
        if grouped and not native_gqa:
            repeat = self.heads // self.kv_heads
            k,v = k.repeat_interleave(repeat,1),v.repeat_interleave(repeat,1)
        y = F.scaled_dot_product_attention(q,k,v,attn_mask=mask,is_causal=causal,
            dropout_p=0.0,enable_gqa=native_gqa)
        return self.o_proj(y.transpose(1,2).contiguous().view(b,t,-1))


    @torch.no_grad()
    def cached(self, x, past=None, mask=None, causal=False):
        b,t,_=x.shape
        q=self.q_proj(x).view(b,t,self.heads,self.head_dim).transpose(1,2)
        k=self.k_proj(x).view(b,t,self.kv_heads,self.head_dim).transpose(1,2)
        v=self.v_proj(x).view(b,t,self.kv_heads,self.head_dim).transpose(1,2)
        offset=0 if past is None else past[0].shape[2]
        if self.rope_theta is not None:
            q,k=rotary(q,self.rope_theta,offset),rotary(k,self.rope_theta,offset)
        if past is not None:
            if past[0].dtype!=k.dtype:raise ValueError('Cache and decode must use the same precision context.')
            k,v=torch.cat((past[0],k),dim=2),torch.cat((past[1],v),dim=2)
        stored=(k,v)
        grouped=self.kv_heads!=self.heads
        native=grouped and x.device.type=='cuda'
        if grouped and not native:
            repeat=self.heads//self.kv_heads
            k,v=k.repeat_interleave(repeat,1),v.repeat_interleave(repeat,1)
        y=F.scaled_dot_product_attention(q,k,v,attn_mask=mask,is_causal=causal,
                                        dropout_p=0.0,enable_gqa=native)
        return self.o_proj(y.transpose(1,2).contiguous().view(b,t,-1)),stored


class Block(nn.Module):
    def __init__(self, dim, intermediate, heads, kv_heads, eps, rope_theta=None):
        super().__init__()
        self.attn_norm,self.ffn_norm = RMSNorm(dim,eps),RMSNorm(dim,eps)
        self.attn = Attention(dim,heads,kv_heads,rope_theta)
        self.gate = nn.Linear(dim,intermediate,bias=False)
        self.up = nn.Linear(dim,intermediate,bias=False)
        self.down = nn.Linear(intermediate,dim,bias=False)

    def forward(self, x, mask=None, causal=False):
        x = x + self.attn(self.attn_norm(x),mask,causal)
        z = self.ffn_norm(x)
        return x + self.down(F.silu(self.gate(z))*self.up(z))


    @torch.no_grad()
    def cached(self,x,past=None,mask=None,causal=False):
        attention,stored=self.attn.cached(self.attn_norm(x),past,mask,causal)
        x=x+attention;z=self.ffn_norm(x)
        return x+self.down(F.silu(self.gate(z))*self.up(z)),stored


@dataclass(frozen=True)
class DecoderCache:
    layers: tuple
    image_tokens: int
    text_length: int
    owner: int

    @property
    def stored_tokens(self):return self.image_tokens+self.text_length

    @property
    def bytes(self):return sum(t.numel()*t.element_size() for pair in self.layers for t in pair)


class VisionEncoder(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.config = c
        self.patch = nn.Conv2d(3,c.vision_hidden_size,c.patch_size,stride=c.patch_size,bias=False)
        patches = (c.image_size//c.patch_size)**2
        self.position = nn.Parameter(torch.zeros(1,patches,c.vision_hidden_size))
        self.blocks = nn.ModuleList(Block(c.vision_hidden_size,c.vision_intermediate_size,
            c.vision_heads,c.vision_heads,c.norm_eps) for _ in range(c.vision_layers))
        self.norm = RMSNorm(c.vision_hidden_size,c.norm_eps)
        self.project = nn.Linear(c.vision_hidden_size,c.hidden_size,bias=False)

    def forward(self, pixels, gradient_checkpointing=False):
        c = self.config
        if pixels.ndim != 4 or pixels.shape[1]!=3 or pixels.shape[2]!=pixels.shape[3]:
            raise ValueError('Images must have shape [batch,3,side,side].')
        size=pixels.shape[2];c.image_token_count(size)
        side=size//c.patch_size;position=self.position
        if size!=c.image_size:
            # Interpolate our learned spatial grid in fp32, preserving checkpoint
            # tensor shapes and the exact base-resolution path. No extra weights.
            base=c.image_size//c.patch_size
            grid=position.float().reshape(1,base,base,c.vision_hidden_size).permute(0,3,1,2)
            position=F.interpolate(grid,size=(side,side),mode='bicubic',align_corners=False)
            position=position.flatten(2).transpose(1,2).to(self.position.dtype)
        x = self.patch(pixels).flatten(2).transpose(1,2) + position
        for block in self.blocks:
            if gradient_checkpointing and self.training:
                x = checkpoint(block,x,use_reentrant=False)
            else:
                x = block(x)
        x = self.norm(x).transpose(1,2).reshape(pixels.shape[0],c.vision_hidden_size,side,side)
        x = F.avg_pool2d(x,c.pool_size,c.pool_size).flatten(2).transpose(1,2)
        return self.project(x)


class TinyMultimodalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        c = config
        self.embedding = nn.Embedding(c.vocab_size,c.hidden_size)
        self.vision = VisionEncoder(c)
        self.blocks = nn.ModuleList(Block(c.hidden_size,c.intermediate_size,c.num_heads,
            c.num_kv_heads,c.norm_eps,c.rope_theta) for _ in range(c.num_layers))
        self.norm = RMSNorm(c.hidden_size,c.norm_eps)
        self.lm_head = nn.Linear(c.hidden_size,c.vocab_size,bias=False)
        self.lm_head.weight = self.embedding.weight
        self.value_head = nn.Linear(c.hidden_size,1,bias=False)
        self.gradient_checkpointing = False
        self.apply(self._initialize)
        nn.init.normal_(self.vision.position,std=0.02)
        # Zero initial critic avoids a spurious initial value scale in PPO.
        nn.init.zeros_(self.value_head.weight)

    @staticmethod
    def _initialize(module):
        if isinstance(module,(nn.Linear,nn.Embedding,nn.Conv2d)):
            nn.init.normal_(module.weight,std=0.02)
            if getattr(module,'bias',None) is not None:
                nn.init.zeros_(module.bias)

    def forward(self, input_ids, pixel_values=None, attention_mask=None, labels=None,
                return_hidden=False,return_logits=True):
        if labels is not None and not return_logits:raise ValueError('Token supervision requires logits.')
        if input_ids.ndim != 2 or not 1 <= input_ids.shape[1] <= self.config.max_text_length:
            raise ValueError('Text must be [batch,length] within max_text_length.')
        b,t = input_ids.shape
        x = self.embedding(input_ids)
        prefix = 0
        if pixel_values is not None:
            if pixel_values.shape[0] != b:
                raise ValueError('Text and image batch sizes must match.')
            visual = self.vision(pixel_values,self.gradient_checkpointing)
            prefix = visual.shape[1]
            x = torch.cat((visual,x),dim=1)
        if attention_mask is not None:
            if attention_mask.shape != input_ids.shape:
                raise ValueError('attention_mask must have the text shape.')
            if bool(attention_mask.bool().all()):
                # Packed text has no padding; use the causal SDPA fast path
                # instead of allocating a quadratic mask for every minibatch.
                attention_mask = None
        # Vision features cannot see text. Text can see the whole image prefix
        # and preceding text; future text is never visible.
        mask = None
        if prefix or attention_mask is not None:
            s = t+prefix
            mask = torch.ones(s,s,device=x.device,dtype=torch.bool).tril()
            if prefix:
                mask[:prefix,:prefix] = True
            if attention_mask is not None:
                if attention_mask.shape != input_ids.shape:
                    raise ValueError('attention_mask must have the text shape.')
                valid = torch.cat((torch.ones(b,prefix,device=x.device,dtype=torch.bool),
                                   attention_mask.bool()),dim=1)
                mask = mask[None,None,:,:] & valid[:,None,None,:]
        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                x = checkpoint(block,x,mask,mask is None,use_reentrant=False)
            else:
                x = block(x,mask,mask is None)
        hidden = self.norm(x[:,prefix:])
        out = {'values':self.critic(hidden)}
        if return_logits:
            logits = self.lm_head(hidden);out['logits']=logits
        if return_hidden:
            out['hidden'] = hidden
        if labels is not None:
            if labels.shape != input_ids.shape:
                raise ValueError('labels must have the text shape.')
            target = labels[:,1:].contiguous()
            if t < 2 or not torch.any(target != -100):
                raise ValueError('Loss requires at least one unmasked next-token target.')
            out['loss'] = F.cross_entropy(logits[:,:-1].float().reshape(-1,self.config.vocab_size),
                                          target.reshape(-1),ignore_index=-100)
        return out

    def critic(self, hidden):
        # Reward differences can be much smaller than one BF16 scalar bin.
        # Keep this small projection in FP32 while the backbone uses autocast.
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            return F.linear(hidden.float(), self.value_head.weight.float()).squeeze(-1)

    @torch.no_grad()
    def prefill(self,input_ids,pixel_values=None):
        if self.training:raise ValueError('Cached inference requires eval mode.')
        if input_ids.ndim!=2 or not 1<=input_ids.shape[1]<=self.config.max_text_length:
            raise ValueError('Text must be [batch,length] within max_text_length.')
        x=self.embedding(input_ids);prefix=0
        if pixel_values is not None:
            if pixel_values.shape[0]!=input_ids.shape[0]:raise ValueError('Text and image batch sizes must match.')
            visual=self.vision(pixel_values);prefix=visual.shape[1];x=torch.cat((visual,x),dim=1)
        mask=None
        if prefix:
            mask=torch.ones(x.shape[1],x.shape[1],device=x.device,dtype=torch.bool).tril()
            mask[:prefix,:prefix]=True
        layers=[]
        for block in self.blocks:
            x,pair=block.cached(x,mask=mask,causal=mask is None);layers.append(pair)
        hidden=self.norm(x[:,-1:])
        return {'logits':self.lm_head(hidden),'values':self.critic(hidden),
                'cache':DecoderCache(tuple(layers),prefix,input_ids.shape[1],id(self))}

    @torch.no_grad()
    def decode(self,input_ids,cache):
        if self.training:raise ValueError('Cached inference requires eval mode.')
        if not isinstance(cache,DecoderCache) or cache.owner!=id(self):
            raise ValueError('Cache must belong to this model instance.')
        if input_ids.ndim!=2 or input_ids.shape[1]<1 or cache.text_length+input_ids.shape[1]>self.config.max_text_length:
            raise ValueError('Cached text length must fit max_text_length.')
        if len(cache.layers)!=len(self.blocks) or any(pair[0].shape[:2]!=(input_ids.shape[0],self.config.num_kv_heads)
                or pair[0].shape[2]!=cache.stored_tokens or pair[1].shape!=pair[0].shape for pair in cache.layers):
            raise ValueError('Cache batch, heads or stored lengths are inconsistent.')
        x=self.embedding(input_ids);t=input_ids.shape[1];mask=None
        if t>1:
            # Cached queries start after all stored image/text keys. SDPA's
            # default non-square causal mask is top-left aligned, so use the
            # explicit positions for multi-token decoding.
            q=torch.arange(cache.stored_tokens,cache.stored_tokens+t,device=x.device)
            k=torch.arange(cache.stored_tokens+t,device=x.device)
            mask=k[None,:]<=q[:,None]
        layers=[]
        for block,past in zip(self.blocks,cache.layers):
            x,pair=block.cached(x,past,mask=mask);layers.append(pair)
        hidden=self.norm(x)
        return {'logits':self.lm_head(hidden),'values':self.critic(hidden),
                'cache':DecoderCache(tuple(layers),cache.image_tokens,cache.text_length+t,id(self))}

    @torch.no_grad()
    def generate(self, input_ids, pixel_values=None, max_new_tokens=32, temperature=0.0,
                 eos_token_id=None,use_cache=True,return_rollout=False):
        if input_ids.ndim!=2 or input_ids.shape[1]<1 or max_new_tokens<1 or input_ids.shape[1]+max_new_tokens>self.config.max_text_length:
            raise ValueError('Generation length must fit max_text_length.')
        if temperature<0:raise ValueError('Temperature must be nonnegative.')
        if return_rollout and temperature<=0:raise ValueError('Sampling probabilities require a positive temperature.')
        was_training=self.training;self.eval()
        try:
            tokens=input_ids;done=torch.zeros(input_ids.shape[0],device=input_ids.device,dtype=torch.bool)
            sampled_logps=[];sampled_values=[];response_masks=[]
            state=self.prefill(tokens,pixel_values) if use_cache else None
            for step in range(max_new_tokens):
                out=state if use_cache else self(tokens,pixel_values)
                logits=out['logits'][:,-1].float()
                logps=(logits/temperature).log_softmax(-1) if temperature else None
                nxt=torch.multinomial(logps.exp(),1) if temperature else logits.argmax(-1,keepdim=True)
                if return_rollout:
                    active=~done
                    sampled_logps.append(torch.where(active,logps.gather(1,nxt).squeeze(-1),0.))
                    sampled_values.append(torch.where(active,out['values'][:,-1].float(),0.))
                    response_masks.append(active)
                if eos_token_id is not None:
                    nxt=torch.where(done[:,None],eos_token_id,nxt);done|=nxt[:,0]==eos_token_id
                tokens=torch.cat((tokens,nxt),dim=1)
                if eos_token_id is not None and done.all():break
                if use_cache and step+1<max_new_tokens:state=self.decode(nxt,state['cache'])
            if return_rollout:
                return {'tokens':tokens,'logps':torch.stack(sampled_logps,1),
                        'values':torch.stack(sampled_values,1),'response_mask':torch.stack(response_masks,1)}
            return tokens
        finally:self.train(was_training)

    def parameter_counts(self):
        total = sum(p.numel() for p in self.parameters())
        vision = sum(p.numel() for p in self.vision.parameters())
        return {'total':total,'vision':vision,'language_and_critic':total-vision,
                'inference_without_critic':total-self.config.hidden_size}
