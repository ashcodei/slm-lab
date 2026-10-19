"""Source-bound initial/completed code-initialized human/program models."""
import json
from pathlib import Path

import torch

from .codenet_checkpoint_v1 import FileBindings
from .corpus import file_hash
from .own_aligned_assistant_v3 import OwnAlignedAssistantV3
from .whole_human_sft_evidence_v1 import tensor_digest
from .whole_human_sft_inference_v1 import fresh as original_factory

INITIAL_FORMAT = 'own-code-initialized-human-program-initial-v1'
FORMAT = 'own-code-initialized-human-program-trained-v1'


def fresh(plan, device='cpu'):
    prior, metadata = original_factory(plan['initial_model_plan'],'cpu')
    original = prior.original; del prior
    model = OwnAlignedAssistantV3(original,**plan['initial_model_plan']['model_initialization']).eval().to(device)
    if model.counts() != plan['actual_counts']: raise ValueError('Actual complete source-bound code-initialized under1B inventory required')
    return model,dict(metadata,code_initialized_alignment=True,external_model_or_teacher_loaded=False)


def sources(plan, plan_sha256):
    return dict(run_plan_sha256=plan_sha256,human_source_admission_sha256=plan['human_source_admission_sha256'],
        program_source_admission_sha256=plan['program_source_admission_sha256'],
        original_own_code_sha256=plan['initial_model_plan']['own_code_checkpoint_sha256'])


def options(plan, arm):
    if arm != 'initial' and arm not in plan['arms']: raise ValueError('Exact predeclared initial or mixed-alignment arm required')
    return {**plan['training_options'],'learning_rate':0. if arm=='initial' else plan['training_options']['learning_rate'],
            'reference_KL_alpha':0. if arm=='initial' else plan['arms'][arm]['reference_KL_alpha']}


def export(model, plan, plan_sha256, arm, completed):
    initial = arm=='initial'
    if (model.counts() != plan['actual_counts'] or completed != (0 if initial else plan['training_options']['steps'])):
        raise ValueError('Complete original mixed-alignment source update budget required')
    setting = options(plan,arm); value = model.export(sources(plan,plan_sha256))
    value['extension_training_not_established'] = initial
    return dict(format=INITIAL_FORMAT if initial else FORMAT,aligned_model=value,run_plan_sha256=plan_sha256,
        arm=arm,actual_mixed_alignment_updates=completed,training_options=setting,
        actual_source_exposure_budget={} if initial else plan['source_exposure_budget'],candidate_promoted=False,goal_achieved=False)


def load(checkpoint, run_plan, arm, *, expected_checkpoint_sha256, expected_plan_sha256, device='cpu'):
    checkpoint,run_plan = map(Path,(checkpoint,run_plan))
    guard = FileBindings({str(checkpoint):expected_checkpoint_sha256,str(run_plan):expected_plan_sha256})
    plan = json.loads(run_plan.read_text()); raw = torch.load(checkpoint,map_location='cpu',weights_only=True,mmap=True)
    initial = arm=='initial'; setting = options(plan,arm)
    if (raw.get('format') != (INITIAL_FORMAT if initial else FORMAT) or raw.get('run_plan_sha256') != expected_plan_sha256
            or raw.get('arm') != arm or raw.get('actual_mixed_alignment_updates') != (0 if initial else plan['training_options']['steps'])
            or raw.get('training_options') != setting or raw.get('actual_source_exposure_budget') != ({} if initial else plan['source_exposure_budget'])
            or raw.get('candidate_promoted') is not False or raw.get('goal_achieved') is not False
            or raw['aligned_model'].get('extension_training_not_established') is not initial):
        raise ValueError('Exact completed mixed-alignment or initial source/arm/budget provenance required')
    model,metadata = fresh(plan,'cpu'); saved = raw['aligned_model']['extensions']; live = model.parts.state_dict()
    model.validate_export(raw['aligned_model'],expected_sources=sources(plan,expected_plan_sha256))
    if (tensor_digest(live.items()) != plan['initial_parts_digest']
            or (initial and tensor_digest(saved.items()) != plan['initial_parts_digest'])
            or any(not torch.equal(t,saved[name]) for name,t in live.items() if not name.startswith(('language_linears.','language_norms.')))):
        raise ValueError('Fresh exact own code initialization and every fixed unused new tensor required before mutation')
    model.import_export(raw['aligned_model'],expected_sources=sources(plan,expected_plan_sha256)); guard.check(full=True)
    return model.eval().to(device),dict(metadata,mixed_alignment=dict(arm=arm,updates=raw['actual_mixed_alignment_updates']),
        genuine_human_ranked_RLHF_not_proven_by_this_SFT=True,high_end_capability_not_established=True)
