"""Package the complete native base only when the research policy is identity.

Visual training changes existing base vision/pointing/word values. Its frozen
zero-residual actor is a research training view, not an extra inference model.
Reject any learned policy delta, changed policy normalization or critic before
exporting the complete base. Actual cold logits/readouts remain separate gates.
"""
import math

import torch

from .joint_visual_private_checkpoint_v2 import SOURCE_KEYS
from .own_actor_vision_prefix_split_backward_v1 import OwnActorVisionPrefixSplitBackward
from .own_code_delta_v1 import norm_names
from .own_dense_standalone_v1 import export as export_dense

PROFILE = 'own-complete-nine-family-visual-continuation-native-base-v1'


def assert_identity(view):
    if type(view) is not OwnActorVisionPrefixSplitBackward:
        raise ValueError('The exact shared own visual research view is required')
    view.assert_shared()
    actor = view.actor
    if any(p.grad is not None for p in view.parameters()):
        raise ValueError('A complete clear-gradient native export boundary is required')
    with torch.no_grad():
        if (any(not bool(torch.isfinite(parts.down).all()) or not bool(torch.isfinite(parts.up).all())
                or bool((parts.up != 0).any()) for parts in actor.parts['linears'])
                or bool((actor.parts['critic'].weight != 0).any())
                or not bool(torch.isfinite(actor.parts['critic'].weight).all())):
            raise ValueError('Only the finite unchanged zero-residual/zero-research-critic policy can export as the native base')
        for name, module in zip(norm_names(actor.config.num_layers), actor.parts['norms']):
            if not torch.equal(module.weight, actor.base.network.get_parameter(name)):
                raise ValueError('Every frozen policy normalization must equal its native base normalization')
    return dict(actual_research_actor_parameters=actor.counts()['actual_unique'],
        actual_complete_native_parameters=actor.base.counts()['total_unique'],
        existing_visual_word_pointing_and_all_other_native_heads_retained=True,
        all_policy_residual_up_factors_exactly_zero=True, all_policy_norms_equal_native_base=True,
        research_only_zero_critic_not_part_of_native_export=True,
        original_native_critic_and_all_other_native_values_retained=True,
        no_learned_policy_delta_or_critic_discarded=True,
        actual_native_logits_word_readout_and_cold_load_parity_still_required=True)


def export(view, *, sources, tokenizer_sha256, mode, completed, planned, original_epoch_entries):
    if (set(sources) != SOURCE_KEYS or mode not in ['mechanics', 'capacity', 'train']
            or any(type(n) is not int for n in [completed, planned, original_epoch_entries])
            or not 0 <= completed <= planned <= original_epoch_entries
            or planned < 1 or any(not isinstance(v, str) or len(v) != 64
                or any(c not in '0123456789abcdef' for c in v) for v in sources.values())):
        raise ValueError('Exact full source SHA bindings, explicit mode and whole source cursors are required')
    identity = assert_identity(view)
    if mode != 'mechanics' and (identity['actual_research_actor_parameters'] != 999706442
            or identity['actual_complete_native_parameters'] != 984399690
            or original_epoch_entries != 4733 or mode == 'train' and planned != 4733):
        raise ValueError('Production visual exports require the complete actual under1B graph and original4733step source epoch')
    provenance = dict(profile=PROFILE, mode=mode, actual_completed_source_entries=completed,
        planned_source_entries=planned, original_complete_epoch_entries=original_epoch_entries,
        whole_production_source_epoch_completed=mode == 'train' and completed == planned == original_epoch_entries,
        exact_policy_identity=identity, external_pretrained_model_or_neural_teacher=False,
        full_source_runtime_GPU_capacity_and_quality_not_proved_by_export=True,
        generated_or_all_round_quality_not_established=True)
    return export_dense(view.actor.base, sources=sources, provenance=provenance,
        completed_updates=completed, tokenizer_sha256=tokenizer_sha256)
