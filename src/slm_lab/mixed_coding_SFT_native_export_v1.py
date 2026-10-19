"""Complete mixed coding actor to self-contained under1B dense candidate.

Adapter merging changes floating-point arithmetic order. Native cold equality
does not assert hot-policy equivalence; actual folded candidate assessments
remain necessary. Preserve the original dense critic: this stage trains actor
weights only and does not train its separate research critic.
"""
import copy

import torch

from .mixed_coding_SFT_training_controller_v1 import TRAINING
from .mixed_coding_SFT_data_v1 import FAMILIES, PROFILES
from .deterministic_positions_v1 import vision_forward, patch_grid
from .own_dense_actor_fold_v1 import fold_actor
from .own_dense_actor_v1 import OwnDenseActor
from .own_dense_standalone_v1 import export as export_dense
from .parallel_mixed_coding_SFT_stage_v1 import FORMAT as STATE
from .parallel_function_repair_stage_v1 import value_digest
from .whole_human_sft_evidence_v1 import tensor_digest

PROFILE = 'own-complete-source-mixed-coding-SFT-dense-candidate-v1'
FULL = dict(updates=5256, source_presentations=84084, complete_context_tokens=42271575,
    response_EOS_targets=21316734, family_presentations=dict(zip(FAMILIES, [65217, 11163, 7704])))


def export(model, state, receipt, *, sources, tokenizer_sha256, working_budget_bytes, allow_mechanics=False):
    if (type(model) is not OwnDenseActor or receipt.get('format') != TRAINING or receipt.get('complete') is not True
            or state.get('format') != STATE or state.get('mode') != receipt.get('mode')
            or state.get('profile') != receipt.get('profile') or receipt.get('profile') not in PROFILES
            or state.get('world_size') != 4 or receipt.get('world') != 4 or state.get('groups') != ['actor']
            or state.get('completed') != state.get('planned') or state.get('completed') != receipt.get('actual_complete_updates')
            or len(state.get('metrics', [])) != state.get('completed')
            or receipt.get('actual_research_actor_parameters') != model.counts()['actual_unique']
            or receipt.get('actual_native_parameters') != model.base.counts()['total_unique']
            or receipt.get('every_original_complete_source_and_CE_target_retained') is not True
            or receipt.get('candidate_promoted') is not False or receipt.get('goal_achieved') is not False):
        raise ValueError('The exact complete own actor-only whole-source training state and controller receipt are required')
    if (getattr(model.base.network.vision.forward, '__func__', None) is not vision_forward
            or getattr(model.base.patch_grid, '__func__', None) is not patch_grid):
        raise ValueError('The exact original deterministic native vision/word position protocol is required')
    mode, budget = state['mode'], receipt['original_complete_training_budget']
    if mode == 'train':
        if (budget != FULL or receipt['actual_complete_updates'] != 5256 or receipt['production_updates'] != 5256
                or model.counts()['actual_unique'] != 999706442 or model.base.counts()['total_unique'] != 984399690):
            raise ValueError('Full production export requires every original84084presentation/21316734target/5256update')
    elif mode != 'mechanics' or allow_mechanics is not True or receipt.get('production_updates') != 0:
        raise ValueError('Only explicit small CPU mechanics can export without the complete production scope')
    model.validate_export(state['model'], expected_sources=sources)
    if (tensor_digest(model.parts.named_parameters()) != receipt['actual_final_actor_sha256']
            or tensor_digest(state['model']['parts'].items()) != receipt['actual_final_actor_sha256']
            or any(not torch.equal(value.detach().cpu(), state['model']['parts'][name]) for name, value in model.parts.state_dict().items())
            or value_digest(state['optimizer']) != receipt['actual_final_Adam_sha256']
            or any(not torch.equal(p, torch.zeros_like(p)) for p in model.parts['critic'].parameters())):
        raise ValueError('All actual completed actor/Adam values and the unused zero research critic must match')
    native, fold = fold_actor(model, sources=sources, working_budget_bytes=working_budget_bytes)
    with torch.no_grad(): native.network.value_head.weight.copy_(model.base.network.value_head.weight)
    if not torch.equal(native.network.value_head.weight, model.base.network.value_head.weight):
        raise ValueError('The original dense critic must survive actor-only coding continuation')
    provenance = dict(profile=PROFILE, mode=mode, training_profile=state['profile'],
        complete_training_budget=copy.deepcopy(budget), actual_completed_updates=state['completed'],
        whole_production_source_schedule_completed=mode == 'train' and budget == FULL,
        completed_training_actor_sha256=receipt['actual_final_actor_sha256'],
        completed_training_Adam_sha256=receipt['actual_final_Adam_sha256'],
        original_dense_critic_preserved=True, unused_zero_research_critic_excluded=True,
        final_native_parameters_sha256=tensor_digest(native.named_parameters()),
        original_nonmutating_fold_observation=fold, actor_to_folded_arithmetic_order_changes=True,
        hot_policy_FP32_BF16_or_generation_equivalence_not_assumed=True,
        no_external_model_weights_or_live_teacher_API=True,
        public_precomputed_teacher_text_in_SFT=True, generated_synthetic_tests_never_used_as_RL_oracle=True,
        actual_full_candidate_native_cold_and_quality_assessments_still_required=True)
    payload = export_dense(native, sources=sources, provenance=provenance, completed_updates=state['completed'],
        tokenizer_sha256=tokenizer_sha256)
    return native, payload, provenance
