"""Parameter-free, fixed-per-request RoPE profiles for own native models.

Plain extrapolation retains the original rotary arithmetic. PI and the YaRN
paper's rotation-ratio ramp are experimental alternatives, not quality claims.
The distinct cache wrapper rejects caches made before installation or by a
different profile/model. No global rotary function or live process is patched.
"""
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
from types import MethodType

import torch
from torch.nn import functional as F

from . import model as core
from .own_dense_growth_v1 import OwnDenseAssistant

PROTOCOL = 'own-fixed-text-context-rope-profiles-v1'


@dataclass(frozen=True)
class Profile:
    mode: str
    original_text_length: int
    maximum_text_length: int
    rope_theta: float = 10000.
    alpha: float = 1.
    beta: float = 32.

    def __post_init__(self):
        if (self.mode not in ('extrapolation', 'linear_PI', 'YaRN')
                or any(type(n) is not int for n in (self.original_text_length, self.maximum_text_length))
                or not 2 <= self.original_text_length < self.maximum_text_length <= 32768
                or any(type(v) not in (int, float) or not math.isfinite(v) for v in (self.rope_theta, self.alpha, self.beta))
                or self.rope_theta <= 0 or not 0 < self.alpha < self.beta):
            raise ValueError('An explicit finite bounded fixed RoPE profile is required')

    @property
    def factor(self): return self.maximum_text_length / self.original_text_length

    @property
    def fingerprint(self):
        raw = json.dumps(dict(protocol=PROTOCOL, **asdict(self)), sort_keys=True, separators=(',', ':'))
        return hashlib.sha256(raw.encode()).hexdigest()


def rotary(x, profile, offset=0):
    if type(profile) is not Profile or type(offset) is not int or offset < 0:
        raise ValueError('One fixed profile and actual nonnegative cached position required')
    if profile.mode == 'extrapolation':
        return core.rotary(x, profile.rope_theta, offset)
    dim = x.shape[-1]
    frequencies = profile.rope_theta ** (-torch.arange(0, dim, 2, device=x.device, dtype=torch.float32) / dim)
    if profile.mode == 'linear_PI':
        frequencies = frequencies / profile.factor
        amplitude = 1.
    else:
        rotations = profile.original_text_length * frequencies / (2 * math.pi)
        ramp = ((rotations - profile.alpha) / (profile.beta - profile.alpha)).clamp(0, 1)
        frequencies = frequencies * (ramp + (1 - ramp) / profile.factor)
        amplitude = 1 + .1 * math.log(profile.factor)
    positions = torch.arange(offset, offset + x.shape[-2], device=x.device, dtype=torch.float32)
    angles = positions[:, None] * frequencies[None, :]
    cosine = (angles.cos() * amplitude).to(x.dtype)
    sine = (angles.sin() * amplitude).to(x.dtype)
    a, b = x[..., 0::2], x[..., 1::2]
    return torch.stack((a * cosine - b * sine, a * sine + b * cosine), dim=-1).flatten(-2)


def attention(self, x, mask=None, causal=False):
    b, t, _ = x.shape
    q = self.q_proj(x).view(b, t, self.heads, self.head_dim).transpose(1, 2)
    k = self.k_proj(x).view(b, t, self.kv_heads, self.head_dim).transpose(1, 2)
    v = self.v_proj(x).view(b, t, self.kv_heads, self.head_dim).transpose(1, 2)
    q, k = rotary(q, self._text_context_profile), rotary(k, self._text_context_profile)
    grouped = self.kv_heads != self.heads
    native = grouped and x.device.type == 'cuda'
    if grouped and not native:
        repeat = self.heads // self.kv_heads
        k, v = k.repeat_interleave(repeat, 1), v.repeat_interleave(repeat, 1)
    y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=causal, dropout_p=0., enable_gqa=native)
    return self.o_proj(y.transpose(1, 2).contiguous().view(b, t, -1))


@torch.no_grad()
def cached_attention(self, x, past=None, mask=None, causal=False):
    b, t, _ = x.shape
    q = self.q_proj(x).view(b, t, self.heads, self.head_dim).transpose(1, 2)
    k = self.k_proj(x).view(b, t, self.kv_heads, self.head_dim).transpose(1, 2)
    v = self.v_proj(x).view(b, t, self.kv_heads, self.head_dim).transpose(1, 2)
    offset = 0 if past is None else past[0].shape[2]
    q, k = rotary(q, self._text_context_profile, offset), rotary(k, self._text_context_profile, offset)
    if past is not None:
        if past[0].dtype != k.dtype: raise ValueError('Cache and decode precision must remain identical')
        k, v = torch.cat((past[0], k), dim=2), torch.cat((past[1], v), dim=2)
    stored = (k, v)
    grouped = self.kv_heads != self.heads
    native = grouped and x.device.type == 'cuda'
    if grouped and not native:
        repeat = self.heads // self.kv_heads
        k, v = k.repeat_interleave(repeat, 1), v.repeat_interleave(repeat, 1)
    y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=causal, dropout_p=0., enable_gqa=native)
    return self.o_proj(y.transpose(1, 2).contiguous().view(b, t, -1)), stored


@dataclass(frozen=True)
class ContextCache:
    native: object
    profile_fingerprint: str
    installation_token: object

    @property
    def bytes(self): return self.native.bytes
    @property
    def text_length(self): return self.native.text_length
    @property
    def image_tokens(self): return self.native.image_tokens
    @property
    def stored_tokens(self): return self.native.stored_tokens
    @property
    def layers(self): return self.native.layers
    @property
    def owner(self): return self.native.owner


def wrap(self, result):
    return {**result, 'cache': ContextCache(result['cache'], self._text_context_profile.fingerprint, self._text_context_token)}


@torch.no_grad()
def prefill(self, *args, **kwargs):
    return wrap(self, self._text_context_original_prefill(*args, **kwargs))


@torch.no_grad()
def decode(self, ids, cache):
    if (type(cache) is not ContextCache or cache.profile_fingerprint != self._text_context_profile.fingerprint
            or cache.installation_token is not self._text_context_token):
        raise ValueError('Only this installation and fixed profile may use its cached rotated keys')
    return wrap(self, self._text_context_original_decode(ids, cache.native))


def forbidden_export(self, *args, **kwargs):
    raise ValueError('A context-extended model requires its distinct profile-aware checkpoint format')


def guarded_preference_score(self, ids, *args, **kwargs):
    profile = self._text_context_profile
    if profile.mode != 'extrapolation' or ids.shape[1] > profile.original_text_length:
        raise ValueError('The inherited reward feature basis is unverified for changed positional features or longer responses')
    return self._text_context_original_preference_score(ids, *args, **kwargs)


def install(model, profile):
    if (type(model) is not OwnDenseAssistant or type(profile) is not Profile
            or hasattr(model, '_text_context_profile') or model.config.max_text_length != profile.original_text_length
            or model.config.rope_theta != profile.rope_theta):
        raise ValueError('An unmodified owned native and a matching explicit original context are required')
    network = model.network
    if any(b.attn.rope_theta != profile.rope_theta
            or getattr(b.attn.forward, '__func__', None) is not core.Attention.forward
            or getattr(b.attn.cached, '__func__', None) is not core.Attention.cached for b in network.blocks):
        raise ValueError('Every original native attention operator must match before installing a profile')
    before = [(id(p), p.data_ptr(), p._version, p.requires_grad) for p in model.parameters()]
    counts = model.counts(); config = replace(model.config, max_text_length=profile.maximum_text_length)
    network._text_context_profile = profile; network._text_context_token = object()
    network._text_context_original_prefill = network.prefill
    network._text_context_original_decode = network.decode
    for block in network.blocks:
        block.attn._text_context_profile = profile
        block.attn.forward = MethodType(attention, block.attn)
        block.attn.cached = MethodType(cached_attention, block.attn)
    network.prefill = MethodType(prefill, network); network.decode = MethodType(decode, network)
    network.config = config; network.vision.config = config
    model._text_context_profile = profile
    model._text_context_original_export = model.export
    model.export = MethodType(forbidden_export, model)
    model._text_context_original_preference_score = model.preference_score
    model.preference_score = MethodType(guarded_preference_score, model)
    validate(model)
    if model.counts() != counts or before != [(id(p), p.data_ptr(), p._version, p.requires_grad) for p in model.parameters()]:
        raise ValueError('A positional experiment must add or change no registered value or parameter metadata')
    return model


def validate(model):
    profile = getattr(model, '_text_context_profile', None)
    if (type(model) is not OwnDenseAssistant or type(profile) is not Profile
            or model.config.max_text_length != profile.maximum_text_length or model.config.rope_theta != profile.rope_theta
            or model.network.vision.config != model.config or model.network._text_context_profile != profile
            or any(b.attn._text_context_profile != profile or b.attn.rope_theta != profile.rope_theta
                or getattr(b.attn.forward, '__func__', None) is not attention
                or getattr(b.attn.cached, '__func__', None) is not cached_attention for b in model.network.blocks)
            or getattr(model.network.prefill, '__func__', None) is not prefill
            or getattr(model.network.decode, '__func__', None) is not decode
            or getattr(model.export, '__func__', None) is not forbidden_export
            or getattr(model.preference_score, '__func__', None) is not guarded_preference_score):
        raise ValueError('Every native forward/cache/checkpoint path must retain its one fixed positional profile')
    model.assert_inventory()
    return profile
