"""A complete unfused coding policy with no training checkpoint dependency.

The exact dense native, policy factors/norms and unused separate critic are
stored once. No matrix folding or reward-feature compatibility is assumed.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path

import torch
from tokenizers import Tokenizer

from .corpus import file_hash
from .deterministic_positions_v1 import PROTOCOL, install, patch_grid, vision_forward
from .model import ModelConfig
from .own_dense_actor_v1 import OwnDenseActor, DensePolicyView
from .own_dense_growth_v1 import OwnDenseAssistant, new_backbone

FORMAT = 'own-complete-unfused-coding-policy-standalone-v1'


def export(model, *, sources, tokenizer_sha256, completed_updates, provenance):
    if (type(model) is not OwnDenseActor or type(model.policy) is not DensePolicyView
            or model.training or model.policy.training or type(completed_updates) is not int or completed_updates < 0
            or not isinstance(provenance, dict) or not provenance
            or not isinstance(tokenizer_sha256, str) or len(tokenizer_sha256) != 64
            or any(c not in '0123456789abcdef' for c in tokenizer_sha256)
            or getattr(model.base.network.vision.forward, '__func__', None) is not vision_forward
            or getattr(model.base.patch_grid, '__func__', None) is not patch_grid
            or any(p.device.type != 'cpu' or p.dtype != torch.float32 or p.grad is not None
                or not bool(torch.isfinite(p).all()) for p in model.parameters())):
        raise ValueError('Complete eval-mode own FP32 CPU coding policy, exact geometry, tokenizer and explicit source cursor required')
    model.assert_shared()
    if model.counts()['actual_unique'] >= 10**9:
        raise ValueError('Every actual native/policy/critic/auxiliary value must fit strictly under1B')
    return dict(format=FORMAT, complete_dense=model.base.export(sources), policy=model.export(sources),
        completed_updates=completed_updates, provenance=dict(provenance), tokenizer_sha256=tokenizer_sha256,
        actual_counts=model.counts(), position_interpolation=PROTOCOL,
        policy_factors_not_merged=True, every_original_native_and_policy_critic_value_included=True,
        old_checkpoint_optimizer_or_training_reference_needed_for_inference=False,
        compatible_trained_reward_model_or_critic_quality_certified=False,
        candidate_promoted=False, goal_achieved=False)


def load(path, *, expected_checkpoint_sha256, expected_sources, seed=1003381):
    path = Path(path)
    if file_hash(path) != expected_checkpoint_sha256:
        raise ValueError('The exact complete standalone policy bytes are required')
    raw = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    if (raw.get('format') != FORMAT or raw.get('position_interpolation') != PROTOCOL
            or raw.get('policy_factors_not_merged') is not True
            or raw.get('every_original_native_and_policy_critic_value_included') is not True
            or raw.get('old_checkpoint_optimizer_or_training_reference_needed_for_inference') is not False
            or raw.get('compatible_trained_reward_model_or_critic_quality_certified') is not False
            or raw.get('candidate_promoted') is not False or raw.get('goal_achieved') is not False
            or type(raw.get('completed_updates')) is not int or raw['completed_updates'] < 0
            or not isinstance(raw.get('provenance'), dict) or not raw['provenance']):
        raise ValueError('The complete unfused coding policy, source cursor and original qualifications are required')
    payload, policy = raw['complete_dense'], raw['policy']
    tokenizer_path = path.parent/'tokenizer.json'
    if file_hash(tokenizer_path) != raw.get('tokenizer_sha256'):
        raise ValueError('The original exact packaged tokenizer is required')
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    if (tokenizer.get_vocab_size() != payload['config']['vocab_size']
            or [tokenizer.token_to_id(t) for t in ['<pad>','<bos>','<eos>']] != [0,1,2]
            or policy['config'] != payload['config']):
        raise ValueError('One actual matching native vocabulary, geometry and boundary convention are required')
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        base = install(OwnDenseAssistant(new_backbone(ModelConfig(**payload['config']), seed), **payload['settings'])).eval()
        base.import_export(payload, expected_sources=expected_sources)
        model = OwnDenseActor(base, **policy['settings']).eval()
        model.import_export(policy, expected_sources=expected_sources)
    if (model.counts() != raw['actual_counts'] or model.counts()['actual_unique'] >= 10**9
            or asdict(model.config) != payload['config']
            or file_hash(path) != expected_checkpoint_sha256
            or file_hash(tokenizer_path) != raw['tokenizer_sha256']):
        raise ValueError('All exact source bytes, actual unique parameters and complete native geometry must agree')
    return model, dict(actual_counts=model.counts(), completed_updates=raw['completed_updates'],
        tokenizer_sha256=raw['tokenizer_sha256'], provenance=raw['provenance'],
        no_ancestor_checkpoint_optimizer_or_training_reference_loaded=True,
        unfused_policy_operations_preserved=True, model_quality_not_proven_by_loading=True,
        compatible_trained_reward_model_or_critic_quality_certified=False)


def main():
    from .own_native_query_v1 import query
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--prompt', required=True)
    parser.add_argument('--image')
    parser.add_argument('--image-size', type=int)
    parser.add_argument('--raw-prompt', action='store_true')
    parser.add_argument('--max-new-tokens', type=int, default=128)
    parser.add_argument('--precision', choices=['fp32','bf16'], default='bf16')
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--json', action='store_true')
    args = parser.parse_args()
    if args.threads < 1 or (args.image_size is not None and args.image is None):
        parser.error('Positive CPU thread count and an image for an explicit image size are required')
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    path = Path(args.checkpoint).resolve()
    raw = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    source_identities = dict(raw['policy']['sources'])
    del raw
    model, metadata = load(path, expected_checkpoint_sha256=file_hash(path), expected_sources=source_identities)
    result = query(model, Tokenizer.from_file(str(path.parent/'tokenizer.json')), args.prompt,
        max_new_tokens=args.max_new_tokens, raw_prompt=args.raw_prompt, image=args.image,
        image_size=args.image_size, precision=args.precision)
    result.update(checkpoint_sha256=file_hash(path), actual_registered_parameters=metadata['actual_counts']['actual_unique'],
        completed_checkpoint_updates=metadata['completed_updates'], no_ancestor_checkpoint_needed=True)
    print(json.dumps(result, indent=2) if args.json else result['response'])


if __name__ == '__main__': main()
