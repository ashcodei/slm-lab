"""Cold-load all own native weights with an explicit fixed RoPE profile.

The original-context weight payload and extended computation are separate
fields in a distinct format. No ancestor model or outside weights are loaded.
This native research codec does not serialize an unfused policy/reward graph.
"""
from dataclasses import asdict
from pathlib import Path

import torch
from tokenizers import Tokenizer

from .corpus import file_hash
from .deterministic_positions_v1 import PROTOCOL as VISION_PROTOCOL, install as install_vision
from .fixed_text_context_v1 import PROTOCOL, Profile, install, validate
from .model import ModelConfig
from .own_dense_growth_v1 import OwnDenseAssistant, new_backbone

FORMAT = 'own-complete-native-fixed-text-context-v1'
MAXIMUM_BYTES = 5 * 1024**3


def export(model, *, sources, tokenizer_sha256, completed_updates):
    profile = validate(model)
    if (type(completed_updates) is not int or completed_updates < 0
            or not isinstance(tokenizer_sha256, str) or len(tokenizer_sha256) != 64
            or any(p.device.type != 'cpu' or p.dtype != torch.float32 or p.grad is not None
                or not bool(torch.isfinite(p).all()) for p in model.parameters())):
        raise ValueError('Complete finite FP32 CPU native, clear gradients and an explicit cursor/tokenizer are required')
    rng = torch.get_rng_state().clone()
    payload = model._text_context_original_export(sources)
    payload['config'] = {**payload['config'], 'max_text_length': profile.original_text_length}
    result = dict(format=FORMAT, complete_original_context_native=payload, text_context_protocol=PROTOCOL,
        fixed_text_context_profile=asdict(profile), profile_fingerprint=profile.fingerprint,
        vision_position_protocol=VISION_PROTOCOL, actual_counts=model.counts(), sources=dict(sources),
        tokenizer_sha256=tokenizer_sha256, completed_updates=completed_updates,
        original_short_context_reward_feature_arithmetic_preserved=profile.mode == 'extrapolation',
        inherited_reward_use_for_changed_positions_or_longer_context_forbidden=True,
        parameter_values_added_or_changed_by_context_installation=0,
        full_size_extended_context_GPU_capacity_or_quality_proved=False,
        old_checkpoint_or_teacher_needed_for_inference=False, candidate_promoted=False, goal_achieved=False)
    if not torch.equal(rng, torch.get_rng_state()):
        raise ValueError('Export must preserve the caller CPU generator')
    return result


def load(path, *, expected_checkpoint_sha256, expected_sources):
    path = Path(path)
    if path.stat().st_size > MAXIMUM_BYTES or file_hash(path) != expected_checkpoint_sha256:
        raise ValueError('One bounded complete profile-aware checkpoint with its exact SHA is required')
    raw = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    if (raw.get('format') != FORMAT or raw.get('text_context_protocol') != PROTOCOL
            or raw.get('vision_position_protocol') != VISION_PROTOCOL
            or raw.get('sources') != expected_sources or not expected_sources
            or type(raw.get('completed_updates')) is not int or raw['completed_updates'] < 0
            or raw.get('old_checkpoint_or_teacher_needed_for_inference') is not False
            or raw.get('candidate_promoted') is not False or raw.get('goal_achieved') is not False
            or raw.get('inherited_reward_use_for_changed_positions_or_longer_context_forbidden') is not True
            or raw.get('full_size_extended_context_GPU_capacity_or_quality_proved') is not False):
        raise ValueError('Exact complete native/profile identity and explicit research scope are required')
    profile = Profile(**raw['fixed_text_context_profile'])
    payload = raw['complete_original_context_native']
    if (raw.get('profile_fingerprint') != profile.fingerprint
            or payload['config']['max_text_length'] != profile.original_text_length
            or payload['config']['rope_theta'] != profile.rope_theta
            or raw.get('original_short_context_reward_feature_arithmetic_preserved') is not (profile.mode == 'extrapolation')
            or raw.get('parameter_values_added_or_changed_by_context_installation') != 0):
        raise ValueError('The exact fixed positional operator and original weight geometry must agree')
    token_path = path.parent / 'tokenizer.json'
    if file_hash(token_path) != raw.get('tokenizer_sha256'):
        raise ValueError('The original packaged tokenizer checksum must match before parsing')
    tokenizer = Tokenizer.from_file(str(token_path))
    if (tokenizer.get_vocab_size() != payload['config']['vocab_size']
            or [tokenizer.token_to_id(s) for s in ('<pad>', '<bos>', '<eos>')] != [0, 1, 2]):
        raise ValueError('The entire own native vocabulary and boundary IDs are required')
    with torch.random.fork_rng(devices=[]):
        model = install_vision(OwnDenseAssistant(new_backbone(ModelConfig(**payload['config']), 1001181),
            **payload['settings'])).eval()
        model.import_export(payload, expected_sources=expected_sources)
        install(model, profile)
    if (model.counts() != raw['actual_counts'] or model.counts()['total_unique'] >= 10**9
            or model.network.embedding.weight is not model.network.lm_head.weight
            or file_hash(path) != expected_checkpoint_sha256 or file_hash(token_path) != raw['tokenizer_sha256']):
        raise ValueError('Every native value/count and all original package bytes must remain exact')
    return model, dict(actual_counts=model.counts(), profile=asdict(profile), profile_fingerprint=profile.fingerprint,
        completed_updates=raw['completed_updates'], native_all_weights_loaded=True,
        no_old_checkpoint_teacher_or_optimizer_loaded=True, full_size_extended_GPU_capacity_or_quality_proved=False,
        inherited_reward_feature_compatibility_for_original_context=profile.mode == 'extrapolation')
