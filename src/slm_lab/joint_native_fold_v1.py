"""Fold jointly trained language adapters while retaining the native critic.

Joint CE/DPO/vision/word objectives never train the separate research critic.
Copying that zero critic over an earlier learned native critic would discard
unrelated learned values. Preserve the actual native critic and every current
vision/word/head value; merge only the selected language factors and norms.
"""
import torch

from .own_dense_actor_v1 import OwnDenseActor
from .own_dense_actor_fold_v1 import fold_actor, signature
from .whole_human_sft_evidence_v1 import tensor_digest

FORMAT = 'own-joint-language-fold-retaining-original-native-critic-and-current-vision-v1'


@torch.no_grad()
def fold(model, *, sources, working_budget_bytes):
    if (type(model) is not OwnDenseActor
            or any(not torch.equal(p, torch.zeros_like(p)) for p in model.parts['critic'].parameters())):
        raise ValueError('The joint actor must retain its unused zero research critic; a trained RLcritic needs a different export contract')
    before = signature(model)
    native_critic = tensor_digest(model.base.network.value_head.named_parameters())
    result, receipt = fold_actor(model, sources=sources, working_budget_bytes=working_budget_bytes)
    result.network.value_head.weight.copy_(model.base.network.value_head.weight)
    excluded = {'network.' + n for n, _ in model.base.language_named_parameters()}
    untouched = [(n, p) for n, p in model.base.named_parameters() if n not in excluded]
    actual = dict(result.named_parameters())
    if (native_critic != tensor_digest(result.network.value_head.named_parameters()) or signature(model) != before
            or any(not torch.equal(p, actual[n]) for n, p in untouched)):
        raise ValueError('Every original native critic and current visual/word/auxiliary value must survive the nonmutating language fold')
    receipt.update(format=FORMAT, original_language_fold_format=receipt['format'],
        policy_critic_folded_into_existing_dense_value_head=False, original_native_critic_preserved=True,
        unused_zero_research_critic_excluded=True, current_joint_visual_word_and_auxiliary_values_preserved=True,
        original_native_critic_sha256=native_critic, folded_parameters_sha256=tensor_digest(result.named_parameters()),
        complete_joint_training_and_GPU_cold_quality_not_proved=True)
    return result, receipt
