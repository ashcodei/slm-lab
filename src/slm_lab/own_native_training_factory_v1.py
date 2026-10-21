"""Build a validated CPU native inside an already active training process.

The user-facing CPU query loader rejects any initialized CUDA runtime. A
training/recovery process instead needs an independent CPU graph before its
explicit device transfer. All checksum/source/value/count checks remain.
"""
from pathlib import Path

import torch

from .corpus import file_hash
from .own_dense_standalone_v1 import FORMAT as DENSE_FORMAT,load as load_dense
from .own_actor_dense_fold_v1 import FORMAT as FOLDED_FORMAT,load as load_folded


def load_package_for_training(package):
    if package['format'] not in [DENSE_FORMAT,FOLDED_FORMAT]:
        raise ValueError('An explicitly inspected complete own native format is required')
    loader=load_dense if package['format']==DENSE_FORMAT else load_folded
    before=torch.get_rng_state().clone()
    with torch.random.fork_rng(devices=[]):
        model,metadata=loader(package['checkpoint'],expected_checkpoint_sha256=package['checkpoint_sha256'],
            expected_sources=package['source_identities'])
    actual=sum(p.numel() for p in model.parameters())
    if (any(p.device.type!='cpu' for p in model.parameters()) or actual>=10**9
            or actual!=package['declared_counts']['total_unique'] or model.counts()!=package['declared_counts']
            or model.network.embedding.weight is not model.network.lm_head.weight
            or file_hash(Path(package['checkpoint']).parent/'tokenizer.json')!=package['tokenizer_sha256']
            or not torch.equal(before,torch.get_rng_state())):
        raise ValueError('Every exact own CPU value/count/tokenizer/tied table and caller CPU RNG must remain valid')
    return model.eval(),dict(metadata,actual_registered_parameters=actual,
        construction_on_CPU_before_explicit_training_device_transfer=True,
        inherited_reward_head_quality_after_language_updates_not_certified=True)
