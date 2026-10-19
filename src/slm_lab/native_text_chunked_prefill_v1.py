"""Bound native text prefill queries without dropping any prefix or KV history.

This is ordinary causal chunked prefill, not a new model architecture. It is
installed on one explicit own actor only; training/full rescoring, cached
decode, image prefill and every learned parameter retain their original paths.
Chunking can change FP32 arithmetic. The native sampler still independently
rescores every emitted action with its unchanged full-forward probability guard.
"""
from types import MethodType, MappingProxyType

import torch

from .own_dense_actor_v1 import OwnDenseActor, DensePolicyView
from .own_dense_growth_v1 import DenseBackbone, DenseCache

PROTOCOL = 'own-native-complete-text-causal-chunked-prefill-FP32-v1'


class NativeTextChunkedPrefill:
    def __init__(self, actor, *, chunk_tokens, purpose):
        if (type(actor) is not OwnDenseActor or type(actor.policy) is not DensePolicyView
                or type(chunk_tokens) is not int or not 1 <= chunk_tokens <= 256
                or purpose not in ['CPU-mechanics', 'GPU-runtime']
                or getattr(actor.policy.prefill, '__func__', None) is not DenseBackbone.prefill):
            raise ValueError('One explicit own actor, original prefill and bounded positive chunk size are required')
        device = next(actor.parameters()).device
        if purpose == 'CPU-mechanics':
            if device.type != 'cpu' or torch.cuda.is_initialized() or actor.counts()['actual_unique'] >= 10**6:
                raise ValueError('Small prefill mechanics must remain CUDA-hidden and under one million parameters')
        elif (device.type != 'cuda' or actor.counts()['actual_unique'] != 999706442
                or actor.base.counts()['total_unique'] != 984399690
                or not torch.are_deterministic_algorithms_enabled()
                or torch.backends.cuda.matmul.allow_tf32):
            raise ValueError('Actual full-size GPU prefill requires the unchanged deterministic FP32 own actor')
        actor.assert_shared()
        self.actor, self.policy = actor, actor.policy
        self.device = device
        self.settings = MappingProxyType(dict(protocol=PROTOCOL, chunk_tokens=chunk_tokens, purpose=purpose))
        self.original = self.policy.prefill
        self.inventory = self._inventory()
        self.last_observation = None
        controller = self
        def installed(policy, input_ids, pixel_values=None):
            return controller.prefill(policy, input_ids, pixel_values)
        self.function = installed
        self.policy.prefill = MethodType(installed, self.policy)
        self.assert_current()

    def _inventory(self):
        return tuple((n, id(p), tuple(p.shape), p.dtype, p.device) for n, p in self.actor.named_parameters())

    def assert_current(self):
        if (self.actor.policy is not self.policy or self._inventory() != self.inventory
                or getattr(self.policy.prefill, '__self__', None) is not self.policy
                or getattr(self.policy.prefill, '__func__', None) is not self.function
                or any(p.dtype != torch.float32 or p.device != self.device for p in self.actor.parameters())):
            raise ValueError('The explicit local prefill method and complete original FP32 parameter inventory must remain')

    @torch.no_grad()
    def prefill(self, policy, input_ids, pixel_values):
        self.assert_current()
        if policy is not self.policy or policy.training or torch.is_autocast_enabled(self.device.type):
            raise ValueError('Native chunked prefill requires its own FP32 eval policy')
        if pixel_values is not None:
            # Preserve the existing bidirectional image-prefix semantics exactly.
            self.last_observation = dict(protocol=PROTOCOL, original_image_prefill_used=True)
            return self.original(input_ids, pixel_values=pixel_values)
        if (not isinstance(input_ids, torch.Tensor) or input_ids.dtype != torch.long
                or input_ids.device != self.device or input_ids.ndim != 2 or not 1 <= input_ids.shape[0] <= 8
                or not 1 <= input_ids.shape[1] <= policy.config.max_text_length):
            raise ValueError('All complete native text IDs and one through eight logical histories must fit unchanged')
        before = policy.signature()
        length = input_ids.shape[1]
        chunk = self.settings['chunk_tokens']
        layers = [None] * len(policy.blocks)
        query_lengths, key_lengths, mask_elements = [], [], []
        for start in range(0, length, chunk):
            end = min(length, start + chunk)
            x = policy.embedding(input_ids[:, start:end])
            mask = None
            if start:
                # SDPA's rectangular is_causal mask is upper-left aligned.
                # Explicit absolute positions permit all past keys and only
                # previous/current keys inside the actual new query chunk.
                q = torch.arange(start, end, device=self.device)
                k = torch.arange(end, device=self.device)
                mask = k[None, :] <= q[:, None]
            for index, block in enumerate(policy.blocks):
                x, layers[index] = block.cached(x, layers[index], mask=mask, causal=start == 0)
            query_lengths.append(end - start)
            key_lengths.append(end)
            mask_elements.append(0 if mask is None else mask.numel())
        hidden = policy.norm(x[:, -1:])
        cache = DenseCache(tuple(layers), 0, length, id(policy), policy.signature())
        expected_KV = (2 * input_ids.shape[0] * len(policy.blocks) * policy.config.num_kv_heads
            * (policy.config.hidden_size // policy.config.num_heads) * length * 4)
        if (policy.signature() != before or cache.bytes != expected_KV
                or any(pair[0].shape[2] != length or pair[1].shape != pair[0].shape for pair in layers)):
            raise ValueError('Every original history and every layer must retain the complete exact native KV geometry')
        self.last_observation = dict(protocol=PROTOCOL, purpose=self.settings['purpose'],
            chunk_tokens=chunk, complete_original_prefix_tokens=length, actual_logical_histories=input_ids.shape[0],
            actual_chunk_query_lengths=query_lengths, actual_chunk_key_lengths=key_lengths,
            actual_explicit_causal_mask_elements=mask_elements, actual_complete_native_KV_bytes=cache.bytes,
            maximum_attention_query_key_pairs_per_head=max(q * k for q, k in zip(query_lengths, key_lengths)),
            corresponding_unchunked_query_key_pairs_per_head=length**2,
            no_prefix_history_or_attempt_dropped=True, measured_GPU_memory_or_speedup=False,
            model_updates=0, new_model_quality_gain=False)
        return dict(logits=policy.lm_head(hidden), values=policy.critic(hidden), cache=cache)

    def close(self):
        self.assert_current()
        self.policy.prefill = self.original


def install(actor, *, chunk_tokens=128, purpose):
    return NativeTextChunkedPrefill(actor, chunk_tokens=chunk_tokens, purpose=purpose)
