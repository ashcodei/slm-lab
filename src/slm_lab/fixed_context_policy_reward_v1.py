"""Extend only the own policy's positions, preserving the original reward view.

Both computations share exactly one backbone's weights. The policy retains
unfused adapters and its trained critic; the frozen human reward retains its
original native operators and context limit. Longer reward inputs abstain.
"""
from dataclasses import asdict, replace
from pathlib import Path
from types import MethodType

import torch
from tokenizers import Tokenizer

from . import model as core
from .code_residual_checkpoint_v1 import _cpu, _digest
from .corpus import file_hash
from .deterministic_positions_v1 import PROTOCOL as VISION_PROTOCOL, install as install_vision
from .fixed_text_context_v1 import (PROTOCOL, Profile, attention, cached_attention,
    decode, forbidden_export, prefill)
from .native_human_reward_v1 import FrozenSharedNativeReward, signature, sources_ok
from .own_dense_actor_v1 import OwnDenseActor
from .own_dense_growth_v1 import OwnDenseAssistant, new_backbone
from .own_policy_reward_standalone_v1 import inventory
from .whole_human_sft_evidence_v1 import tensor_digest

FORMAT = 'own-fixed-context-policy-original-shared-human-reward-v1'
MAXIMUM_BYTES = 5 * 1024**3


def unchanged_reward_view(base, profile):
    if (base.config.max_text_length != profile.original_text_length or base.config.rope_theta != profile.rope_theta
            or hasattr(base, '_text_context_profile') or hasattr(base.network, '_text_context_profile')
            or any(b.attn.rope_theta != profile.rope_theta
                or getattr(b.attn.forward, '__func__', None) is not core.Attention.forward
                or getattr(b.attn.cached, '__func__', None) is not core.Attention.cached for b in base.network.blocks)):
        raise ValueError('The original frozen native reward positional computation must remain unchanged')


class FixedContextPolicyReward(torch.nn.Module):
    """Owned research graph; actor-only optimization leaves the reward fixed."""
    def __init__(self, actor, reward, profile, *, allow_mechanics=False):
        super().__init__()
        if type(actor) is not OwnDenseActor or type(reward) is not FrozenSharedNativeReward or type(profile) is not Profile:
            raise ValueError('An actual own actor, learned shared-native reward and fixed profile are required')
        inventory(actor, reward, allow_mechanics=allow_mechanics)
        unchanged_reward_view(actor.base, profile)
        policy = actor.policy
        if (hasattr(policy, '_text_context_profile') or policy.config != actor.base.config
                or any(getattr(b.attn.forward, '__func__', None) is not core.Attention.forward
                    or getattr(b.attn.cached, '__func__', None) is not core.Attention.cached for b in policy.blocks)):
            raise ValueError('Every original owned policy operator must match before changing its positions')
        before = signature(actor.parameters()); counts = actor.counts()
        self.actor, self.reward, self.profile = actor, reward, profile
        self.allow_mechanics = allow_mechanics
        policy._text_context_profile = profile; policy._text_context_token = object()
        policy._text_context_original_prefill = policy.prefill
        policy._text_context_original_decode = policy.decode
        for block in policy.blocks:
            block.attn._text_context_profile = profile
            block.attn.forward = MethodType(attention, block.attn)
            block.attn.cached = MethodType(cached_attention, block.attn)
        policy.prefill = MethodType(prefill, policy); policy.decode = MethodType(decode, policy)
        # Do not mutate the frozen vision/native config or its reward features.
        policy.config = replace(actor.base.config, max_text_length=profile.maximum_text_length)
        actor._fixed_context_original_export = actor.export
        actor.export = MethodType(forbidden_export, actor)
        self.eval(); self.check()
        if before != signature(actor.parameters()) or counts != actor.counts():
            raise ValueError('Policy positions must not add/change any parameter value or metadata')

    @property
    def config(self): return self.actor.policy.config
    @property
    def policy(self): return self.actor.policy
    @property
    def base(self): return self.actor.base

    def train(self, mode=True):
        self.training = mode; self.actor.train(mode); self.reward.eval()
        return self

    def check(self):
        self.actor.assert_shared(); self.reward.check(); unchanged_reward_view(self.base, self.profile)
        p = self.policy
        if (self.reward.base is not self.base or self.base.training
                or p.config.max_text_length != self.profile.maximum_text_length
                or p._text_context_profile != self.profile
                or any(b.attn._text_context_profile != self.profile
                    or getattr(b.attn.forward, '__func__', None) is not attention
                    or getattr(b.attn.cached, '__func__', None) is not cached_attention for b in p.blocks)
                or getattr(p.prefill, '__func__', None) is not prefill or getattr(p.decode, '__func__', None) is not decode
                or getattr(self.actor.export, '__func__', None) is not forbidden_export
                or sum(v.numel() for v in self.parameters()) != self.actor.counts()['actual_unique']):
            raise ValueError('One fixed policy profile and unchanged shared reward computation are required')

    def counts(self):
        return dict(**self.actor.counts(), additional_reward_encoder_values=0,
            shared_backbone=True, policy_max_text_length=self.config.max_text_length,
            reward_max_text_length=self.base.config.max_text_length)

    def forward(self, *args, **kwargs): self.check(); return self.actor(*args, **kwargs)
    def generate(self, *args, **kwargs): self.check(); return self.actor.generate(*args, **kwargs)

    @torch.no_grad()
    def preference_score(self, ids):
        self.check()
        if ids.ndim != 2 or ids.shape[1] > self.base.config.max_text_length:
            raise ValueError('The complete original reward input must fit its unchanged verified context; never crop it')
        return self.reward.score(ids)

    def word_logits(self, pixels): self.check(); return self.base.word_logits(pixels)


def export(model, *, sources, provenance, completed_updates, tokenizer_sha256):
    if type(model) is not FixedContextPolicyReward: raise ValueError('An actual fixed-context shared policy/reward graph required')
    model.check(); counts = inventory(model.actor, model.reward, allow_mechanics=model.allow_mechanics)
    if (model.training or any(p.device.type != 'cpu' for p in model.parameters())
            or type(completed_updates) is not int or completed_updates < 0 or not isinstance(provenance, dict)
            or not isinstance(sources, dict) or not sources or any(not isinstance(k, str) or not _digest(v) for k,v in sources.items())
            or sources.get('tokenizer_sha256') != tokenizer_sha256 or not _digest(tokenizer_sha256)
            or any(sources.get(k) != v for k,v in model.reward.provenance['sources'].items())):
        raise ValueError('Complete CPU eval-mode source graph and exact original reward/tokenizer identities required')
    before = signature(model.parameters()); rng = torch.get_rng_state().clone()
    result = dict(format=FORMAT, complete_base=model.base.export(sources),
        original_policy_adapter=model.actor._fixed_context_original_export(sources),
        original_reward_provenance=_cpu(model.reward.provenance), original_counts=counts, actual_counts=model.counts(),
        policy_parameters_sha256=tensor_digest(model.actor.named_parameters()),
        reward_parameters_sha256=tensor_digest(model.reward.named_parameters()),
        fixed_policy_profile=asdict(model.profile), profile_fingerprint=model.profile.fingerprint,
        text_context_protocol=PROTOCOL, vision_position_protocol=VISION_PROTOCOL,
        sources=dict(sources), tokenizer_sha256=tokenizer_sha256, completed_updates=completed_updates,
        provenance=_cpu(provenance), original_frozen_reward_positional_computation_preserved=True,
        policy_factors_unfused=True, longer_reward_inputs_forbidden=True, additional_reward_encoder_values=0,
        old_checkpoint_or_teacher_or_optimizer_needed_for_inference=False,
        full_size_extended_GPU_capacity_or_quality_proved=False, candidate_promoted=False, goal_achieved=False)
    if before != signature(model.parameters()) or not torch.equal(rng, torch.get_rng_state()):
        raise ValueError('Export must preserve all source parameters and caller CPU RNG')
    return result


def load(path, *, expected_checkpoint_sha256, expected_sources, allow_mechanics=False):
    path = Path(path)
    if path.stat().st_size > MAXIMUM_BYTES or file_hash(path) != expected_checkpoint_sha256:
        raise ValueError('An exact bounded complete fixed-context policy/reward package required')
    raw = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    if (raw.get('format') != FORMAT or raw.get('text_context_protocol') != PROTOCOL
            or raw.get('vision_position_protocol') != VISION_PROTOCOL or raw.get('sources') != expected_sources
            or raw.get('original_frozen_reward_positional_computation_preserved') is not True
            or raw.get('policy_factors_unfused') is not True or raw.get('longer_reward_inputs_forbidden') is not True
            or raw.get('additional_reward_encoder_values') != 0
            or raw.get('old_checkpoint_or_teacher_or_optimizer_needed_for_inference') is not False
            or raw.get('full_size_extended_GPU_capacity_or_quality_proved') is not False
            or raw.get('candidate_promoted') is not False or raw.get('goal_achieved') is not False
            or type(raw.get('completed_updates')) is not int or raw['completed_updates'] < 0):
        raise ValueError('Exact original computation/source identities and research flags are required')
    profile = Profile(**raw['fixed_policy_profile']); base_payload = raw['complete_base']
    policy_payload = raw['original_policy_adapter']; provenance = raw['original_reward_provenance']
    if (raw['profile_fingerprint'] != profile.fingerprint or not sources_ok(provenance.get('sources'))
            or any(expected_sources.get(k) != v for k,v in provenance['sources'].items())
            or base_payload['config'] != policy_payload['config']
            or base_payload['config']['max_text_length'] != profile.original_text_length
            or base_payload['config']['rope_theta'] != profile.rope_theta):
        raise ValueError('Original reward geometry and exact policy-only positional semantics must agree')
    token_path = path.parent / 'tokenizer.json'
    if file_hash(token_path) != raw.get('tokenizer_sha256') or expected_sources.get('tokenizer_sha256') != raw['tokenizer_sha256']:
        raise ValueError('The entire original packaged vocabulary bytes are required')
    tokenizer = Tokenizer.from_file(str(token_path))
    if (tokenizer.get_vocab_size() != base_payload['config']['vocab_size']
            or [tokenizer.token_to_id(s) for s in ['<pad>','<bos>','<eos>']] != [0,1,2]):
        raise ValueError('One complete own native tokenizer and original boundaries required')
    if raw['original_counts'].get('mechanics_only') and not allow_mechanics:
        raise ValueError('Small CPU mechanics cannot impersonate the full current model')
    with torch.random.fork_rng(devices=[]):
        base = install_vision(OwnDenseAssistant(new_backbone(core.ModelConfig(**base_payload['config']), 1002191),
            **base_payload['settings'])).eval()
        base.import_export(base_payload, expected_sources=expected_sources)
        actor = OwnDenseActor(base, **policy_payload['settings']).eval()
        actor.import_export(policy_payload, expected_sources=expected_sources)
        reward = FrozenSharedNativeReward(base, provenance, allow_mechanics=allow_mechanics)
        model = FixedContextPolicyReward(actor, reward, profile, allow_mechanics=allow_mechanics)
    if (model.counts() != raw['actual_counts'] or inventory(actor,reward,allow_mechanics=allow_mechanics) != raw['original_counts']
            or tensor_digest(actor.named_parameters()) != raw['policy_parameters_sha256']
            or tensor_digest(reward.named_parameters()) != raw['reward_parameters_sha256']
            or file_hash(path) != expected_checkpoint_sha256 or file_hash(token_path) != raw['tokenizer_sha256']):
        raise ValueError('All actual native/adapter/critic/reward values and source bytes must reconstruct exactly')
    return model, dict(actual_counts=model.counts(), profile=asdict(profile), profile_fingerprint=profile.fingerprint,
        original_reward_feature_basis_preserved=True, no_second_encoder_or_old_checkpoint_loaded=True,
        full_size_extended_GPU_capacity_or_model_quality_proved=False, completed_updates=raw['completed_updates'])
