"""Source-bound complete training inputs, fused base and dense human policy.

Production inputs must be completed dense-foundation and shared-word arms.
The exported policy adds no retained source/reference backbone. A common
source-bound fused base is stored once and reused by all policy arms.
"""
import json
from pathlib import Path

import torch

from .codenet_checkpoint_v1 import FileBindings
from .corpus import file_hash
from .dense_word_folding_v1 import fold_words
from .model import ModelConfig
from .own_dense_actor_v1 import OwnDenseActor
from .own_dense_growth_v1 import OwnDenseAssistant,new_backbone
from .whole_human_sft_evidence_v1 import tensor_digest

BASE_FORMAT='own-complete-dense-foundation-shared-word-fused-base-v1'
INITIAL_FORMAT='own-dense-human-preference-initial-policy-v1'
FORMAT='own-dense-complete-human-preference-policy-v1'


def base_sources(inputs):
    return {key:inputs[key] for key in ('foundation_checkpoint_sha256','foundation_plan_sha256','word_checkpoint_sha256','word_plan_sha256')}


def prepare_base(inputs):
    from .dense_foundation_inference_v1 import load as load_foundation
    from .shared_word_inference_v1 import load as load_word,sources as word_sources
    if inputs['foundation_arm']=='initial' or inputs['word_arm']=='initial':
        raise ValueError('Only complete production-trained source arms may initialize full human preference training')
    base,_=load_foundation(inputs['foundation_checkpoint'],inputs['foundation_plan'],inputs['foundation_arm'],
        expected_checkpoint_sha256=inputs['foundation_checkpoint_sha256'],expected_plan_sha256=inputs['foundation_plan_sha256'],device='cpu')
    word,_=load_word(inputs['word_checkpoint'],inputs['word_plan'],inputs['word_arm'],
        expected_checkpoint_sha256=inputs['word_checkpoint_sha256'],expected_plan_sha256=inputs['word_plan_sha256'],device='cpu')
    raw=torch.load(inputs['word_checkpoint'],map_location='cpu',weights_only=True,mmap=True)
    plan=json.loads(Path(inputs['word_plan']).read_text())
    receipt=fold_words(base,word,raw['aligned_model'],expected_sources=word_sources(plan,inputs['word_plan_sha256']))
    return base,receipt


def export_base(base,inputs,receipt):
    if receipt['actual_counts']!=base.counts():raise ValueError('Actual complete source-folded base inventory required')
    return dict(format=BASE_FORMAT,model=base.export(base_sources(inputs)),folding_receipt=receipt,
        complete_trained_source_epochs_required=True,candidate_promoted=False,goal_achieved=False)


def fresh(plan,device='cpu'):
    path=Path(plan['base_checkpoint']);guard=FileBindings({str(path):plan['base_checkpoint_sha256']})
    raw=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
    if (raw.get('format')!=BASE_FORMAT or raw.get('folding_receipt')!=plan['base_folding_receipt']
            or raw.get('complete_trained_source_epochs_required') is not True
            or raw.get('candidate_promoted') is not False or raw.get('goal_achieved') is not False):
        raise ValueError('Exact complete-source fused model and integration provenance required')
    saved=raw['model'];config=ModelConfig(**saved['config'])
    base=OwnDenseAssistant(new_backbone(config,948531),**saved['settings']).eval()
    base.import_export(saved,expected_sources=base_sources(plan['base_inputs']))
    if base.counts()!=plan['actual_dense_counts'] or tensor_digest(base.named_parameters())!=plan['frozen_dense_sha256']:
        raise ValueError('Every actual frozen dense/vision/word/head value must match the common base')
    model=OwnDenseActor(base,**plan['actor_settings']).eval()
    if model.counts()!=plan['actual_counts'] or tensor_digest(model.parts.named_parameters())!=plan['initial_actor_parts_sha256']:
        raise ValueError('Actual complete under1B policy inventory and identical fresh adapters/critic required')
    guard.check(full=True)
    return model.to(device),dict(tokenizer_sha256=plan['tokenizer_sha256'],actual_counts=model.counts(),
        training_options=dict(image_size=config.max_image_size),human_preference_source='original public human-ranked other-model responses',
        own_reference_is_same_frozen_counted_dense_backbone=True,policy_critic_training_not_established=True)


def options(plan,arm):
    if arm!='initial' and arm not in plan['arms']:raise ValueError('Exact predeclared complete human preference arm required')
    profile=dict(learning_rate=0.,DPO_weight=0) if arm=='initial' else plan['arms'][arm]
    return {**plan['training_options'],**profile}


def sources(plan,plan_sha256):
    return dict(run_plan_sha256=plan_sha256,frozen_base_checkpoint_sha256=plan['base_checkpoint_sha256'],
        human_admission_sha256=plan['human_admission_sha256'],human_manifest_sha256=plan['human_manifest_sha256'])


def export(model,plan,plan_sha256,arm,completed):
    initial=arm=='initial';setting=options(plan,arm)
    if completed!=(0 if initial else setting['steps']) or model.counts()!=plan['actual_counts']:
        raise ValueError('Every actual complete human source update and bounded policy parameter required')
    return dict(format=INITIAL_FORMAT if initial else FORMAT,policy=model.export(sources(plan,plan_sha256)),
        arm=arm,run_plan_sha256=plan_sha256,actual_human_preference_updates=completed,training_options=setting,
        source_exposure_budget={} if initial else plan['source_exposure_budget'][arm],
        candidate_promoted=False,goal_achieved=False)


def load(checkpoint,run_plan,arm,*,expected_checkpoint_sha256,expected_plan_sha256,device='cpu'):
    checkpoint,run_plan=map(Path,(checkpoint,run_plan));guard=FileBindings({str(checkpoint):expected_checkpoint_sha256,str(run_plan):expected_plan_sha256})
    plan=json.loads(run_plan.read_text());raw=torch.load(checkpoint,map_location='cpu',weights_only=True,mmap=True)
    initial=arm=='initial';setting=options(plan,arm)
    if (raw.get('format')!=(INITIAL_FORMAT if initial else FORMAT) or raw.get('arm')!=arm or raw.get('run_plan_sha256')!=expected_plan_sha256
            or raw.get('actual_human_preference_updates')!=(0 if initial else setting['steps']) or raw.get('training_options')!=setting
            or raw.get('source_exposure_budget')!=({} if initial else plan['source_exposure_budget'][arm])
            or raw.get('candidate_promoted') is not False or raw.get('goal_achieved') is not False):
        raise ValueError('Exact complete human policy/arm/source/update provenance required')
    model,metadata=fresh(plan,'cpu');payload=raw['policy']
    model.validate_export(payload,expected_sources=sources(plan,expected_plan_sha256))
    if (payload['frozen_dense_sha256']!=plan['frozen_dense_sha256']
            or not torch.equal(model.parts['critic'].weight,payload['parts']['critic.weight'])
            or (initial and tensor_digest(payload['parts'].items())!=plan['initial_actor_parts_sha256'])):
        raise ValueError('Same frozen base and untouched separate critic/initial adapters required before import')
    model.import_export(payload,expected_sources=sources(plan,expected_plan_sha256));guard.check(full=True)
    return model.eval().to(device),dict(metadata,human_preference_policy=dict(arm=arm,updates=raw['actual_human_preference_updates']),
        helpfulness_harmlessness_correctness_and_high_end_quality_not_established_by_loading=True)
