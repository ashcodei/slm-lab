"""Own alignment and vision extensions around the existing single backbone.

This implements an experimental architecture, not a capability claim. The
original general/code/grounding routes remain available. New language and
vision factors share original weights and the original tied vocabulary table.
Only explicitly selected extension parameters belong in a new optimizer.
"""
import copy
from contextlib import nullcontext
from dataclasses import asdict

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .model import RMSNorm, TinyMultimodalLM
from .own_assistant_v1 import OwnAssistant
from .own_code_delta_v1 import (
    OwnDeltaCache, _DeltaBlock, _DeltaLinear, _TiedHead, matrix_names, norm_names,
)

FORMAT = 'own-single-backbone-alignment-vision-extensions-v1'
GROUPS = dict(language=('language_linears', 'language_norms'),
              vision=('vision_linears', 'vision_norms', 'vision_projection'),
              critic=('critic',), reward=('reward',), word=('word',))


def byte_ctc_loss(logits, texts):
    """Native byte values0..255, blank256; never silently discard long labels."""
    if logits.ndim != 3 or logits.shape[-1] != 257 or len(texts) != logits.shape[0]:
        raise ValueError('Whole byte-word logits and one complete source transcript per image required')
    targets = [list(text.encode('utf-8')) for text in texts]
    if any(not row or len(row) + sum(a == b for a, b in zip(row, row[1:])) > logits.shape[1] for row in targets):
        raise ValueError('Whole UTF-8 transcript and repeated bytes must fit the actual CTC time axis')
    packed = torch.tensor([v for row in targets for v in row], dtype=torch.long, device=logits.device)
    return F.ctc_loss(logits.float().log_softmax(-1).transpose(0, 1), packed,
        torch.full((len(targets),), logits.shape[1], dtype=torch.long),
        torch.tensor([len(row) for row in targets], dtype=torch.long), blank=256, zero_infinity=False)


def decode_bytes(path):
    result, previous = [], None
    for value in path:
        value = int(value)
        if not 0 <= value <= 256:
            raise ValueError('Actual byte or blank class required')
        if value != 256 and value != previous:
            result.append(value)
        previous = value
    raw = bytes(result)
    try:
        return raw.decode('utf-8'), True
    except UnicodeDecodeError:
        return raw.decode('utf-8', errors='replace'), False


class _ZeroFactors(nn.Module):
    def __init__(self, original, rank):
        super().__init__()
        self.down = nn.Parameter(torch.empty(rank, original.in_features,
            dtype=original.weight.dtype, device=original.weight.device))
        self.up = nn.Parameter(torch.zeros(original.out_features, rank,
            dtype=original.weight.dtype, device=original.weight.device))
        nn.init.normal_(self.down, std=.02)


class ByteWordHead(nn.Module):
    """Byte CTC on a supplied word image; this does not detect text in photos."""
    def __init__(self, visual, eps, temporal=96):
        super().__init__()
        self.norm = RMSNorm(visual, eps)
        self.local = nn.Conv2d(visual, visual, 3, padding=1, groups=visual)
        self.postnorm = RMSNorm(visual, eps)
        self.height = nn.Linear(visual, 1)
        self.temporal = nn.GRU(visual, temporal, batch_first=True, bidirectional=True)
        self.classifier = nn.Linear(2 * temporal, 257)

    def forward(self, patches):
        if patches.ndim != 4 or patches.shape[-1] != self.norm.weight.numel():
            raise ValueError('Whole word patch grid must be [batch,height,width,visual]')
        x = self.norm(patches).permute(0, 3, 1, 2)
        x = self.postnorm(self.local(x).permute(0, 2, 3, 1))
        weights = self.height(x).softmax(1)
        features, _ = self.temporal((x * weights).sum(1))
        return self.classifier(features)


class _VisionView(nn.Module):
    def __init__(self, original, parts):
        super().__init__()
        self.original = original
        self.config = original.config
        self.patch, self.position = original.patch, original.position
        self.blocks = nn.ModuleList(_DeltaBlock(block,
            parts['vision_linears'][i*7:(i+1)*7], parts['vision_norms'][i*2:i*2+2])
            for i, block in enumerate(original.blocks))
        self.norm = parts['vision_norms'][-1]
        self.project = _DeltaLinear(original.project, parts['vision_projection'])

    def patch_grid(self, pixels, gradient_checkpointing=False, *, word_image=False):
        c = self.config
        if pixels.ndim != 4 or pixels.shape[0] < 1 or pixels.shape[1] != 3:
            raise ValueError('Actual RGB image batch required')
        height, width = pixels.shape[-2:]
        if word_image:
            if any(side < c.patch_size or side > c.max_image_size or side % c.patch_size
                   for side in (height, width)):
                raise ValueError('Whole word image sides must be bounded positive patch multiples')
        else:
            if height != width:
                raise ValueError('Multimodal image prefix requires an original square image')
            c.image_token_count(height)
        h, w = height // c.patch_size, width // c.patch_size
        base = c.image_size // c.patch_size
        position = self.position
        if (h, w) != (base, base):
            # The original position grid stays frozen: no CUDA interpolation
            # backward is used. Rectangular interpolation is only for words.
            position = F.interpolate(position.float().reshape(1, base, base, c.vision_hidden_size)
                .permute(0, 3, 1, 2), size=(h, w), mode='bicubic', align_corners=False)
            position = position.flatten(2).transpose(1, 2).to(self.position.dtype)
        x = self.patch(pixels).flatten(2).transpose(1, 2) + position
        for block in self.blocks:
            x = checkpoint(block, x, use_reentrant=False) if gradient_checkpointing and self.training else block(x)
        return self.norm(x).reshape(pixels.shape[0], h, w, c.vision_hidden_size)

    def forward(self, pixels, gradient_checkpointing=False):
        x = self.patch_grid(pixels, gradient_checkpointing).permute(0, 3, 1, 2)
        x = F.avg_pool2d(x, self.config.pool_size, self.config.pool_size).flatten(2).transpose(1, 2)
        return self.project(x)


class _AlignedView(TinyMultimodalLM):
    def __init__(self, original, parts):
        nn.Module.__init__(self)
        # Register the complete inventory even when this view is passed alone
        # to a native evaluator. These references share objects, not copies.
        self.original, self.extensions = original, parts
        base = original.language.base
        self.config = base.config
        self.embedding = base.embedding
        self.lm_head = _TiedHead(base.embedding)
        self.value_head = parts['critic']
        self.vision = _VisionView(base.vision, parts)
        self.norm = parts['language_norms'][-1]
        self.blocks = nn.ModuleList(_DeltaBlock(block,
            parts['language_linears'][i*7:(i+1)*7], parts['language_norms'][i*2:i*2+2])
            for i, block in enumerate(base.blocks))
        self.gradient_checkpointing = False

    def signature(self):
        return tuple((id(p), p._version, p.data_ptr(), p.dtype, p.device) for p in self.parameters())

    def bind(self, state):
        cache = state['cache']
        return {**state, 'cache': OwnDeltaCache(cache.layers, cache.image_tokens,
            cache.text_length, cache.owner, self.signature())}

    @torch.no_grad()
    def prefill(self, input_ids, pixel_values=None):
        return self.bind(super().prefill(input_ids, pixel_values))

    @torch.no_grad()
    def decode(self, input_ids, cache):
        if (not isinstance(cache, OwnDeltaCache) or cache.owner != id(self)
                or cache.parameter_signature != self.signature()):
            raise ValueError('Aligned cache must match this route and unchanged actual parameters')
        return self.bind(super().decode(input_ids, cache))


def _norms(base, names):
    result = nn.ModuleList(copy.deepcopy(base.get_submodule(name.removesuffix('.weight'))) for name in names)
    for parameter in result.parameters():
        parameter.requires_grad_(True)
    return result


def parameter_budget(original, *, rank=16, reward_hidden=128, word_temporal=96):
    """Read actual shapes before allocation; count every distinct old weight."""
    if type(original) is not OwnAssistant:
        raise ValueError('The existing exact one-backbone own assistant is required')
    original.assert_shared()
    if any(type(v) is not int or v < 1 for v in (rank, reward_hidden, word_temporal)):
        raise ValueError('Positive explicit extension dimensions required')
    base = original.language.base
    def linears(native, names):
        total = 0
        for name in names:
            module = native.get_submodule(name.removesuffix('.weight'))
            if rank > min(module.weight.shape):
                raise ValueError('Extension rank must fit every actual original matrix')
            total += rank * sum(module.weight.shape)
        return total
    h, v, t = base.config.hidden_size, base.config.vision_hidden_size, word_temporal
    extra = dict(language_factors=linears(base, matrix_names(base.config.num_layers)),
        language_norms=sum(base.get_parameter(n).numel() for n in norm_names(base.config.num_layers)),
        vision_factors=linears(base.vision, matrix_names(base.config.vision_layers)),
        vision_norms=sum(base.vision.get_parameter(n).numel() for n in norm_names(base.config.vision_layers)),
        vision_projection=linears(base.vision, ['project.weight']), critic=h,
        reward=h + h*reward_hidden + reward_hidden,
        word=2*v + 10*v + v + 1 + 2*(3*t*v + 3*t*t + 6*t) + 2*t*257 + 257)
    return dict(original=original.counts()['total_unique'], added=extra,
                total_unique=original.counts()['total_unique'] + sum(extra.values()))


class OwnAlignedAssistant(nn.Module):
    def __init__(self, original, *, rank=16, reward_hidden=128, word_temporal=96,
                 seed=947131, parameter_limit=10**9):
        super().__init__()
        expected = parameter_budget(original, rank=rank, reward_hidden=reward_hidden, word_temporal=word_temporal)
        if (type(parameter_limit) is not int or not expected['total_unique'] < parameter_limit <= 10**9
                or type(seed) is not int):
            raise ValueError('Complete original/code/grounding/extensions must fit strictly below1B')
        self.original = original
        self.settings = dict(rank=rank, reward_hidden=reward_hidden, word_temporal=word_temporal,
                             seed=seed, parameter_limit=parameter_limit)
        base = original.language.base
        if base.embedding.weight is not base.lm_head.weight:
            raise ValueError('Actual original tied vocabulary table required')
        devices = [] if base.embedding.weight.device.type == 'cpu' else [base.embedding.weight.device.index]
        # Preserve caller RNG without touching any original parameter or its
        # trainability metadata; the established factories require that metadata.
        with torch.random.fork_rng(devices=devices):
            torch.random.default_generator.manual_seed(seed)
            for device in devices:
                torch.cuda.default_generators[device].manual_seed(seed)
            def factors(native, names):
                return nn.ModuleList(_ZeroFactors(native.get_submodule(n.removesuffix('.weight')), rank) for n in names)
            h, v, eps = base.config.hidden_size, base.config.vision_hidden_size, base.config.norm_eps
            self.parts = nn.ModuleDict(dict(
                language_linears=factors(base, matrix_names(base.config.num_layers)),
                language_norms=_norms(base, norm_names(base.config.num_layers)),
                vision_linears=factors(base.vision, matrix_names(base.config.vision_layers)),
                vision_norms=_norms(base.vision, norm_names(base.config.vision_layers)),
                vision_projection=_ZeroFactors(base.vision.project, rank),
                critic=nn.Linear(h, 1, bias=False),
                reward=nn.Sequential(RMSNorm(h, eps), nn.Linear(h, reward_hidden, bias=False),
                    nn.SiLU(), nn.Linear(reward_hidden, 1, bias=False)),
                word=ByteWordHead(v, eps, word_temporal)))
            nn.init.zeros_(self.parts['critic'].weight)
            # .to only traverses new tensors; it never moves/reversions the base.
            self.parts.to(device=base.embedding.weight.device, dtype=base.embedding.weight.dtype)
        self.aligned = _AlignedView(original, self.parts)
        self.expected_budget = expected
        self.assert_shared()

    @property
    def config(self):
        return self.original.config

    def counts(self):
        return dict(original=self.original.counts(), extension_unique=sum(p.numel() for p in self.parts.parameters()),
                    total_unique=sum(p.numel() for p in self.parameters()))

    def assert_shared(self):
        self.original.assert_shared()
        base = self.original.language.base
        old = {id(p) for p in self.original.parameters()}
        added = {id(p) for p in self.parts.parameters()}
        counts = self.counts()
        if (self.aligned.original is not self.original or self.aligned.extensions is not self.parts
                or self.aligned.embedding is not base.embedding or self.aligned.lm_head.weight is not base.embedding.weight
                or self.aligned.vision.original is not base.vision
                or old & added or any(not p.requires_grad for p in self.parts.parameters())
                or counts['total_unique'] != self.expected_budget['total_unique']
                or counts['total_unique'] != self.original.counts()['total_unique'] + counts['extension_unique']
                or counts['total_unique'] >= self.settings['parameter_limit']
                or sum(p.numel() for p in self.aligned.parameters()) != counts['total_unique']):
            raise ValueError('One shared own backbone/vocabulary table and every distinct bounded extension required')

    def extension_parameters(self, *groups):
        self.assert_shared()
        if not groups or len(set(groups)) != len(groups) or any(g not in GROUPS for g in groups):
            raise ValueError('Explicit distinct language/vision/reward/critic/word optimizer groups required')
        return [p for group in groups for name in GROUPS[group] for p in self.parts[name].parameters()]

    def route(self, route):
        if route == 'aligned':
            return self.aligned
        if route in ('general', 'code'):
            return self.original.language.route(route)
        raise ValueError('Explicit original general/code or new aligned language route required')

    def forward(self, *args, route='aligned', **kwargs):
        self.assert_shared()
        if route == 'grounding':
            return self.original(*args, route=route, **kwargs)
        return self.route(route)(*args, **kwargs)

    def generate(self, *args, route='aligned', **kwargs):
        self.assert_shared()
        return self.route(route).generate(*args, **kwargs)

    def preference_score(self, input_ids, *, attention_mask=None, detach_features=True):
        """Own learned score at each complete response's actual last token."""
        self.assert_shared()
        if input_ids.ndim != 2 or input_ids.shape[1] < 1 or type(detach_features) is not bool:
            raise ValueError('Complete batched source response and explicit feature-gradient policy required')
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        mask = attention_mask.bool()
        if mask.shape != input_ids.shape or not bool(mask.any(1).all()) or bool((mask[:, 1:] & ~mask[:, :-1]).any()):
            raise ValueError('Complete nonempty right-padded source response mask required')
        with torch.no_grad() if detach_features else nullcontext():
            hidden = self.aligned(input_ids, attention_mask=mask, return_hidden=True, return_logits=False)['hidden']
        final = hidden[torch.arange(input_ids.shape[0], device=hidden.device), mask.sum(1) - 1]
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            # Modules remain FP32 during supported mixed-precision training.
            return self.parts['reward'](final.float()).squeeze(-1)

    def word_logits(self, pixel_values):
        self.assert_shared()
        return self.parts['word'](self.aligned.vision.patch_grid(pixel_values,
            self.aligned.gradient_checkpointing, word_image=True))

    def extension_state(self):
        self.assert_shared()
        return {name: t.detach().cpu().clone() for name, t in self.parts.state_dict().items()}

    def validate_extensions(self, state):
        """Validate the entire new-only state without changing live tensors."""
        self.assert_shared()
        live = self.parts.state_dict()
        if (not isinstance(state, dict) or set(state) != set(live)
                or any(not isinstance(t, torch.Tensor) or t.shape != live[n].shape or t.dtype != live[n].dtype
                    or not bool(torch.isfinite(t).all()) for n, t in state.items())):
            raise ValueError('Every exact finite new-only extension tensor required before mutation')

    def load_extensions(self, state):
        self.validate_extensions(state)
        self.parts.load_state_dict(state, strict=True)

    def export(self, sources):
        if not sources or any(not isinstance(k, str) or not isinstance(v, str) or len(v) != 64
                              or any(c not in '0123456789abcdef' for c in v) for k, v in sources.items()):
            raise ValueError('Exact named source artifact digests required')
        return dict(format=FORMAT, config=asdict(self.config), settings=self.settings,
                    actual_counts=self.counts(), sources=dict(sources), extensions=self.extension_state(),
                    extension_training_not_established=True, candidate_promoted=False, goal_achieved=False)

    def validate_export(self, payload, *, expected_sources):
        if (payload.get('format') != FORMAT or payload.get('config') != asdict(self.config)
                or payload.get('settings') != self.settings or payload.get('actual_counts') != self.counts()
                or payload.get('sources') != expected_sources or not expected_sources
                or payload.get('candidate_promoted') is not False or payload.get('goal_achieved') is not False):
            raise ValueError('Exact whole architecture and external source bindings required before mutation')
        self.validate_extensions(payload['extensions'])

    def import_export(self, payload, *, expected_sources):
        self.validate_export(payload, expected_sources=expected_sources)
        self.load_extensions(payload['extensions'])
