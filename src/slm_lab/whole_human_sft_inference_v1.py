"""Completed whole human SFT extensions, with one source-bound own backbone."""
import json
from pathlib import Path

import torch

from .codenet_checkpoint_v1 import FileBindings
from .corpus import file_hash
from .own_aligned_assistant_v2 import OwnAlignedAssistantV2
from .own_onpolicy_inference_v1 import load as load_code, attach_original_head

FORMAT='own-whole-human-dialogue-SFT-trained-extensions-v1'
INITIAL_FORMAT='own-whole-human-dialogue-SFT-initial-extensions-v1'


def fresh(plan, device='cpu'):
    old_path=Path(plan['own_code_plan']);old=json.loads(old_path.read_text())
    language,metadata=load_code(plan['own_code_checkpoint'],old_path,'CE_control',plan['own_code_reference'],
        expected_checkpoint_sha256=plan['own_code_checkpoint_sha256'],expected_plan_sha256=plan['own_code_plan_sha256'],
        expected_reference_sha256=plan['own_code_reference_sha256'],device='cpu')
    original,attachment=attach_original_head(language,old)
    model=OwnAlignedAssistantV2(original,**plan['model_initialization']).eval().to(device)
    if model.counts()!=plan['actual_counts']:
        raise ValueError('Complete actual own source-bound under1B model inventory required')
    return model,dict(metadata,tokenizer_sha256=old['tokenizer_sha256'],training_options=dict(image_size=448),
        external_model_or_teacher_loaded=False,actual_counts=model.counts(),candidate_promoted=False)


def sources(plan, plan_sha256, arm):
    if arm!='initial' and arm not in plan['arms']:
        raise ValueError('Exact predeclared whole-source SFT arm required')
    return dict(run_plan_sha256=plan_sha256,source_admission_sha256=plan['source_admission_sha256'],
        source_manifest_sha256=plan['source_manifest_sha256'],own_code_checkpoint_sha256=plan['own_code_checkpoint_sha256'])


def export_initial(model, plan, plan_sha256):
    return dict(format=INITIAL_FORMAT,aligned_model=model.export(sources(plan,plan_sha256,'initial')),
        run_plan_sha256=plan_sha256,arm='initial',actual_human_SFT_updates=0,
        training_options={**plan['training_options'],'learning_rate':0.},
        actual_source_exposure_budget=dict(source_exposures=0,response_EOS_targets=0),candidate_promoted=False,goal_achieved=False)


def export(model, plan, plan_sha256, arm, completed):
    if completed!=plan['training_options']['steps'] or model.counts()!=plan['actual_counts']:
        raise ValueError('Every original fixed human SFT update and whole model count required')
    state=model.export(sources(plan,plan_sha256,arm));state['extension_training_not_established']=False
    return dict(format=FORMAT,aligned_model=state,run_plan_sha256=plan_sha256,arm=arm,
        actual_human_SFT_updates=completed,training_options={**plan['training_options'],'learning_rate':plan['arms'][arm]['learning_rate']},
        actual_source_exposure_budget=plan['source_exposure_budget'],candidate_promoted=False,goal_achieved=False)


def load(checkpoint, run_plan, arm, *, expected_checkpoint_sha256, expected_plan_sha256, device='cpu'):
    checkpoint,run_plan=map(Path,(checkpoint,run_plan))
    guard=FileBindings({str(checkpoint):expected_checkpoint_sha256,str(run_plan):expected_plan_sha256})
    plan=json.loads(run_plan.read_text());payload=torch.load(checkpoint,map_location='cpu',weights_only=True,mmap=True)
    if arm!='initial' and arm not in plan['arms']:
        raise ValueError('Exact initial or predeclared whole-source SFT arm required')
    initial=arm=='initial';steps=0 if initial else plan['training_options']['steps']
    rate=0. if initial else plan['arms'][arm]['learning_rate']
    budget=dict(source_exposures=0,response_EOS_targets=0) if initial else plan['source_exposure_budget']
    if (payload.get('format')!=(INITIAL_FORMAT if initial else FORMAT) or payload.get('run_plan_sha256')!=expected_plan_sha256 or payload.get('arm')!=arm
            or payload.get('actual_human_SFT_updates')!=steps
            or payload.get('training_options')!={**plan['training_options'],'learning_rate':rate}
            or payload.get('actual_source_exposure_budget')!=budget
            or payload.get('candidate_promoted') is not False or payload.get('goal_achieved') is not False):
        raise ValueError('Exact completed whole-human SFT source/arm/budget provenance required')
    model,metadata=fresh(plan,'cpu')
    model.import_export(payload['aligned_model'],expected_sources=sources(plan,expected_plan_sha256,arm))
    guard.check(full=True)
    return model.eval().to(device),dict(metadata,human_SFT_training=dict(arm=arm,updates=payload['actual_human_SFT_updates']),
        genuine_human_ranked_reward_DPO_PPO_not_yet_trained=True,high_end_capability_not_established=True)
