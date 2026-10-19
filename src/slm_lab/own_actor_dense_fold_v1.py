"""Fold our own trained policy factors into one independently stored dense model.

For each linear operator, W*x + U*(D*x) becomes (W + U*D)*x. Policy
normalizations replace the original unused base normalizations; its critic
replaces the unused base critic. This changes floating-point arithmetic, so
native text/image/generation drift must be measured before production use.
"""
import copy
from dataclasses import asdict

import torch

from .corpus import file_hash
from .deterministic_positions_v1 import PROTOCOL,install,patch_grid,vision_forward
from .model import ModelConfig
from .own_code_delta_v1 import matrix_names,norm_names
from .own_dense_actor_v1 import OwnDenseActor
from .own_dense_growth_v1 import OwnDenseAssistant,new_backbone

FORMAT='own-complete-policy-folded-dense-student-v1'


def fold(actor):
    if type(actor) is not OwnDenseActor:raise ValueError('An actual own dense actor is required')
    actor.assert_shared()
    if (getattr(actor.base.network.vision.forward,'__func__',None) is not vision_forward
            or getattr(actor.base.patch_grid,'__func__',None) is not patch_grid
            or any(p.device.type!='cpu' or p.dtype!=torch.float32 or p.grad is not None
                   or not bool(torch.isfinite(p).all()) for p in actor.parameters())):
        raise ValueError('Own native deterministic-position FP32 CPU model with clear gradients and finite values required')
    names=matrix_names(actor.config.num_layers);norms=norm_names(actor.config.num_layers)
    for name,parts in zip(names,actor.parts['linears']):
        weight=actor.base.network.get_parameter(name);rank=actor.settings['rank']
        if parts.up.shape!=(weight.shape[0],rank) or parts.down.shape!=(rank,weight.shape[1]):
            raise ValueError('Every original actual policy matrix shape/rank must match before folding')
    # deepcopy preserves the installed position methods, shared embedding/head
    # tie, and every own vision/word/reward/router/pointing/detector tensor.
    # It copies only the base, never the source actor or its policy graph.
    candidate=copy.deepcopy(actor.base).eval()
    with torch.no_grad():
        for name,parts in zip(names,actor.parts['linears']):
            candidate.network.get_parameter(name).copy_(actor.base.network.get_parameter(name)+parts.up@parts.down)
        for name,module in zip(norms,actor.parts['norms']):candidate.network.get_parameter(name).copy_(module.weight)
        candidate.network.value_head.weight.copy_(actor.parts['critic'].weight)
    candidate.assert_inventory()
    if any(not bool(torch.isfinite(p).all()) for p in candidate.parameters()):
        raise ValueError('Folded actual weights must remain finite after every FP32 factor product')
    if (candidate.counts()!=actor.base.counts() or candidate.counts()['total_unique']>=10**9
            or {p.data_ptr() for p in candidate.parameters()}&{p.data_ptr() for p in actor.parameters()}
            or any(type(m) is OwnDenseActor for m in candidate.modules())
            or candidate.network.embedding.weight is not candidate.network.lm_head.weight):
        raise ValueError('One independently stored under1B dense model with all heads and no retained source actor required')
    receipt=dict(source_actor_counts=actor.counts(),folded_dense_counts=candidate.counts(),
        actual_neural_values_removed=actor.counts()['actual_unique']-candidate.counts()['total_unique'],
        original_policy_linear_matrices_folded=len(names),policy_norms_copied=len(norms),
        actual_source_policy_critic_copied_to_single_native_critic=True,
        all_original_visual_and_auxiliary_heads_copied=True,
        independent_student_storage_and_one_tied_vocabulary=True,
        position_interpolation=PROTOCOL,folding_arithmetic='FP32: original base plus own up @ down',
        numerical_and_generated_quality_parity_not_assumed=True,new_model_training_updates=0,goal_achieved=False)
    return candidate,receipt


def export(model,*,sources,fold_receipt,completed_updates):
    if (type(model) is not OwnDenseAssistant or type(completed_updates) is not int or completed_updates<0
            or model.counts()!=fold_receipt['folded_dense_counts'] or fold_receipt['position_interpolation']!=PROTOCOL):
        raise ValueError('Exact full folded own model and explicit training cursor required')
    return dict(format=FORMAT,complete_dense=model.export(sources),fold_receipt=fold_receipt,
        completed_updates=completed_updates,position_interpolation=PROTOCOL,
        old_base_or_actor_checkpoint_or_teacher_needed_for_inference=False,
        optimizer_or_training_reference_included=False,candidate_promoted=False,goal_achieved=False)


def load(path,*,expected_checkpoint_sha256,expected_sources,seed=972117):
    if file_hash(path)!=expected_checkpoint_sha256:raise ValueError('Exact complete folded student bytes required')
    raw=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
    if (raw.get('format')!=FORMAT or raw.get('position_interpolation')!=PROTOCOL
            or raw.get('old_base_or_actor_checkpoint_or_teacher_needed_for_inference') is not False
            or raw.get('optimizer_or_training_reference_included') is not False
            or raw.get('candidate_promoted') is not False or raw.get('goal_achieved') is not False
            or type(raw.get('completed_updates')) is not int or raw['completed_updates']<0):
        raise ValueError('A complete independently usable own folded student is required')
    payload=raw['complete_dense']
    model=install(OwnDenseAssistant(new_backbone(ModelConfig(**payload['config']),seed),**payload['settings'])).eval()
    model.import_export(payload,expected_sources=expected_sources)
    if model.counts()!=raw['fold_receipt']['folded_dense_counts'] or file_hash(path)!=expected_checkpoint_sha256:
        raise ValueError('Every folded native value, architecture setting and actual unique count must agree')
    return model,dict(fold_receipt=raw['fold_receipt'],actual_counts=model.counts(),completed_updates=raw['completed_updates'],
        no_old_model_or_actor_or_training_reference_loaded=True,generated_capability_not_proven_by_loading=True)
