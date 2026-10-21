"""Recover exact frozen own native values from a persistent complete policy.

The extra trained coding factors remain in their original package and do not
initialize this full-language comparison. No old scratch ancestor is needed.
"""
from pathlib import Path

import torch

from .corpus import file_hash
from .own_coding_policy_standalone_v1 import load
from .whole_human_sft_evidence_v1 import tensor_digest

CHECKPOINT='014b31377e83da55614ce3bbee620db28e2b022e13b00e3293f12ad700ea5cbf'
FROZEN='4c11e022880ca0f9993a9262f20e9bcc02aa4af6cb11015a1009770a53bddd0b'


def recover(root):
    path=Path(root)/'checkpoints/public-coding-research-v1/plain_tokens/model.pt'
    if file_hash(path)!=CHECKPOINT:raise ValueError('The exact physically preserved completed policy is required')
    raw=torch.load(path,map_location='cpu',weights_only=True,mmap=True);sources=dict(raw['policy']['sources']);del raw
    rng=torch.get_rng_state().clone();policy,metadata=load(path,expected_checkpoint_sha256=CHECKPOINT,expected_sources=sources)
    base=policy.base
    if (base.counts()['total_unique']!=984399690 or policy.counts()['actual_unique']!=999706442
            or tensor_digest(base.named_parameters())!=FROZEN or not torch.equal(rng,torch.get_rng_state())
            or any(p.device.type!='cpu' for p in base.parameters())):
        raise ValueError('Every frozen original native value/count and caller CPU RNG must reproduce exactly')
    del policy
    return base.eval(),dict(source_checkpoint=str(path),source_checkpoint_sha256=CHECKPOINT,
        recovered_native_tensor_sha256=FROZEN,actual_recovered_native_parameters=984399690,
        original_trained_coding_policy_parameters=999706442,coding_adapters_not_used_as_initializer=True,
        lost_scratch_container_not_recreated_byte_for_byte=True,no_external_weights_or_new_neural_updates=True,
        source_identities=sources,original_full_native_values_reconstructed_exactly=True)
