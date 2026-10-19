"""Source-bound full-word vision extensions on one existing own code backbone."""
import json
from pathlib import Path

import torch

from .codenet_checkpoint_v1 import FileBindings
from .corpus import file_hash
from .own_aligned_assistant_v3 import OwnAlignedAssistantV3
from .shared_word_stage_v1 import PREFIXES
from .whole_human_sft_evidence_v1 import tensor_digest
from .whole_human_sft_inference_v1 import fresh as original_factory

INITIAL_FORMAT = 'own-shared-vision-byte-native-word-initial-v1'
FORMAT = 'own-shared-vision-byte-native-word-trained-v1'


def fresh(plan, device='cpu'):
    prior,metadata = original_factory(plan['initial_model_plan'],'cpu')
    original = prior.original;del prior
    model = OwnAlignedAssistantV3(original,**plan['initial_model_plan']['model_initialization']).eval().to(device)
    if model.counts()!=plan['actual_counts']: raise ValueError('Actual complete shared under1B inventory required')
    return model,dict(metadata,shared_word_experiment=True,language_anchor='original_own_code',
        later_mixed_human_program_language_extensions_not_attached=True,external_teacher_or_encoder_loaded=False)


def sources(plan, plan_sha256):
    return dict(run_plan_sha256=plan_sha256,word_admission_sha256=plan['source_admission_sha256'],
                word_manifest_sha256=plan['source_manifest_sha256'],
                own_code_checkpoint_sha256=plan['initial_model_plan']['own_code_checkpoint_sha256'])


def options(plan, arm):
    if arm!='initial' and arm not in plan['arms']: raise ValueError('Exact initial or predeclared full-word arm required')
    return {**plan['training_options'],'learning_rate':0. if arm=='initial' else plan['arms'][arm]['learning_rate']}


def export(model, plan, plan_sha256, arm, completed):
    initial = arm=='initial'; setting = options(plan,arm)
    if completed!=(0 if initial else setting['steps']) or model.counts()!=plan['actual_counts']:
        raise ValueError('Complete original word-source optimizer budget and actual unique parameters required')
    value = model.export(sources(plan,plan_sha256));value['extension_training_not_established']=initial
    return dict(format=INITIAL_FORMAT if initial else FORMAT,aligned_model=value,run_plan_sha256=plan_sha256,
        arm=arm,actual_shared_word_updates=completed,training_options=setting,
        actual_source_exposure_budget={} if initial else plan['source_exposure_budget'],
        candidate_promoted=False,goal_achieved=False)


def load(checkpoint, run_plan, arm, *, expected_checkpoint_sha256, expected_plan_sha256, device='cpu'):
    checkpoint,run_plan = map(Path,(checkpoint,run_plan))
    guard = FileBindings({str(checkpoint):expected_checkpoint_sha256,str(run_plan):expected_plan_sha256})
    plan = json.loads(run_plan.read_text());raw = torch.load(checkpoint,map_location='cpu',weights_only=True,mmap=True)
    initial = arm=='initial';setting = options(plan,arm)
    if (raw.get('format')!=(INITIAL_FORMAT if initial else FORMAT) or raw.get('run_plan_sha256')!=expected_plan_sha256
            or raw.get('arm')!=arm or raw.get('actual_shared_word_updates')!=(0 if initial else setting['steps'])
            or raw.get('training_options')!=setting or raw.get('actual_source_exposure_budget')!=({} if initial else plan['source_exposure_budget'])
            or raw.get('candidate_promoted') is not False or raw.get('goal_achieved') is not False
            or raw['aligned_model'].get('extension_training_not_established') is not initial):
        raise ValueError('Exact completed full-word or initial source/arm/exposure provenance required')
    model,metadata = fresh(plan,'cpu');saved = raw['aligned_model']['extensions'];live = model.parts.state_dict()
    model.validate_export(raw['aligned_model'],expected_sources=sources(plan,expected_plan_sha256))
    if (tensor_digest(live.items())!=plan['initial_parts_digest']
            or (initial and tensor_digest(saved.items())!=plan['initial_parts_digest'])
            or any(not torch.equal(t,saved[name]) for name,t in live.items() if not name.startswith(PREFIXES))):
        raise ValueError('Fresh exact initialization and every fixed language/reward/critic tensor required before import')
    model.import_export(raw['aligned_model'],expected_sources=sources(plan,expected_plan_sha256));guard.check(full=True)
    return model.eval().to(device),dict(metadata,shared_word_training=dict(arm=arm,updates=raw['actual_shared_word_updates']),
        full_photo_reading_and_useful_multimodal_quality_not_established_by_loading=True)
