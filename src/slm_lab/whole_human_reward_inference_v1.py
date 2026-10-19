"""Attach a completed human-ranking head to the same actual own backbone.

Only the three reward tensors are copied. Inference must explicitly call
frozen_general_reward, whose feature route matches training; the older
model.preference_score method uses a different representation.
"""
import json
from pathlib import Path

import torch

from .codenet_checkpoint_v1 import FileBindings
from .corpus import file_hash
from .point_grounding_evaluation_v1 import state_digest
from .whole_human_reward_v1 import FEATURE_ROUTE, general_features, scores

FORMAT = 'own-whole-human-ranking-frozen-general-reward-v1'


def expected_sources(plan, plan_sha256, feature_admission_sha256):
    return dict(run_plan_sha256=plan_sha256, source_manifest_sha256=plan['source_manifest_sha256'],
        source_admission_sha256=plan['source_admission_sha256'], frozen_encoder_sha256=plan['frozen_encoder_sha256'],
        feature_admission_sha256=feature_admission_sha256)


def export(model, plan, plan_sha256, feature_admission_sha256, arm, completed):
    if (arm not in plan['arms'] or completed != plan['training_options']['steps']
            or model.counts() != plan['initial_model_plan']['actual_counts']):
        raise ValueError('Exact complete original human-ranking arm/update/model inventory required')
    return dict(format=FORMAT, sources=expected_sources(plan, plan_sha256, feature_admission_sha256),
        feature_route=FEATURE_ROUTE, actual_counts=model.counts(), arm=arm, actual_human_reward_updates=completed,
        training_options={**plan['training_options'], 'learning_rate':plan['arms'][arm]['learning_rate']},
        source_exposure_pairs=plan['source_population'], reward={n:t.detach().cpu().clone() for n,t in model.parts['reward'].state_dict().items()},
        frozen_encoder_sha256=plan['frozen_encoder_sha256'], candidate_promoted=False, goal_achieved=False)


def attach(model, checkpoint, run_plan, feature_admission, arm, *, expected_checkpoint_sha256,
           expected_plan_sha256, expected_feature_admission_sha256):
    checkpoint, run_plan, feature_admission = map(Path, (checkpoint, run_plan, feature_admission))
    guard = FileBindings({str(checkpoint):expected_checkpoint_sha256, str(run_plan):expected_plan_sha256,
                          str(feature_admission):expected_feature_admission_sha256})
    plan = json.loads(run_plan.read_text()); payload = torch.load(checkpoint, map_location='cpu', weights_only=True, mmap=True)
    admission = json.loads(feature_admission.read_text()); live = model.parts['reward'].state_dict()
    options = {**plan['training_options'], 'learning_rate':plan['arms'][arm]['learning_rate']} if arm in plan['arms'] else None
    if (payload.get('format') != FORMAT or payload.get('arm') != arm or options is None
            or payload.get('sources') != expected_sources(plan, expected_plan_sha256, expected_feature_admission_sha256)
            or payload.get('feature_route') != FEATURE_ROUTE or payload.get('training_options') != options
            or payload.get('actual_human_reward_updates') != plan['training_options']['steps']
            or payload.get('source_exposure_pairs') != plan['source_population']
            or payload.get('actual_counts') != model.counts() or payload['actual_counts'] != plan['initial_model_plan']['actual_counts']
            or payload.get('frozen_encoder_sha256') != plan['frozen_encoder_sha256']
            or payload.get('candidate_promoted') is not False or payload.get('goal_achieved') is not False
            or admission.get('feature_route') != FEATURE_ROUTE or admission.get('train_only_head_optimization_admitted') is not True
            or admission.get('complete_original_train_pairs') != plan['source_population']
            or admission.get('complete_original_held_pairs') != plan['held_source_population']
            or admission.get('feature_sources') != {k:v for k,v in expected_sources(plan, expected_plan_sha256, expected_feature_admission_sha256).items() if k != 'feature_admission_sha256'}):
        raise ValueError('Exact completed full-source human reward and frozen-feature provenance required before mutation')
    saved = payload.get('reward')
    if (not isinstance(saved, dict) or saved.keys() != live.keys()
            or any(not isinstance(t, torch.Tensor) or t.dtype != live[n].dtype or t.shape != live[n].shape
                   or not bool(torch.isfinite(t).all()) for n,t in saved.items())):
        raise ValueError('Every exact finite original-shape reward head tensor required before mutation')
    if state_digest(model.original.language.base) != plan['frozen_encoder_sha256']:
        raise ValueError('Reward features require the exact same fixed original general encoder')
    guard.check(full=True)
    model.parts['reward'].load_state_dict(saved, strict=True)
    return dict(actual_counts=model.counts(), actual_human_reward_updates=payload['actual_human_reward_updates'],
        reward_feature_route=FEATURE_ROUTE, reward_entrypoint='slm_lab.whole_human_reward_inference_v1.frozen_general_reward',
        additional_backbones=0, changed_parameter_tensors=len(live), policy_alignment_not_established=True)


def frozen_general_reward(model, input_ids, attention_mask=None):
    return scores(model, general_features(model, input_ids, attention_mask))
