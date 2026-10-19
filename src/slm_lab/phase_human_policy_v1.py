"""Own policy adapters above a complete source-bound phase-trained foundation.

No old foundation, word merge or external model can silently replace the
declared base. The base supplies both inference and the frozen DPO reference.
"""
import json
from pathlib import Path

import torch

from .codenet_checkpoint_v1 import FileBindings
from .corpus import file_hash
from .fast_joint_inference_v2 import load as load_foundation
from .own_dense_actor_v1 import OwnDenseActor
from .whole_human_sft_evidence_v1 import tensor_digest

FORMAT = 'own-complete-phase-foundation-human-policy-v1'


def validate_foundation(plan, raw):
    if (raw.get('arm') != 'trained' or raw.get('completed_updates') != plan['planned_steps']
            or raw.get('whole_source_epochs_completed') != 1
            or plan.get('all_selected_family_records_once') is not True
            or plan.get('all_original_sources_once') is not False
            or plan.get('later_reference_is_own_completed_foundation') is not True
            or raw.get('candidate_promoted') is not False or raw.get('goal_achieved') is not False):
        raise ValueError('The complete declared selected-family source epoch is required; no partial or older foundation')


def fresh(plan, device='cpu'):
    inputs = plan['foundation_inputs']
    checkpoint, source_plan = Path(inputs['checkpoint']), Path(inputs['plan'])
    guard = FileBindings({str(checkpoint):inputs['checkpoint_sha256'], str(source_plan):inputs['plan_sha256']})
    source = json.loads(source_plan.read_text())
    raw = torch.load(checkpoint, map_location='cpu', weights_only=True, mmap=True)
    validate_foundation(source, raw)
    del raw
    base, metadata = load_foundation(checkpoint, source_plan, 'trained',
        expected_checkpoint_sha256=inputs['checkpoint_sha256'],
        expected_plan_sha256=inputs['plan_sha256'], device='cpu')
    if (base.counts() != plan['actual_dense_counts']
            or tensor_digest(base.named_parameters()) != plan['frozen_dense_sha256']):
        raise ValueError('Every declared trained foundation, vision, word and head value must match')
    model = OwnDenseActor(base, **plan['actor_settings']).eval()
    if (model.counts() != plan['actual_counts']
            or tensor_digest(model.parts.named_parameters()) != plan['initial_actor_parts_sha256']):
        raise ValueError('Exact under1B fresh actor, shared own base and separate unused critic required')
    guard.check(full=True)
    return model.to(device), dict(metadata, actual_counts=model.counts(),
        own_reference_is_same_frozen_counted_dense_backbone=True,
        human_preference_source='public human-ranked other-model responses',
        fresh_human_judgments_of_own_outputs=False)


def sources(plan, digest):
    return dict(run_plan_sha256=digest,
        complete_foundation_checkpoint_sha256=plan['foundation_inputs']['checkpoint_sha256'],
        complete_foundation_plan_sha256=plan['foundation_inputs']['plan_sha256'],
        human_admission_sha256=plan['human_admission_sha256'],
        human_manifest_sha256=plan['human_manifest_sha256'])


def options(plan, arm):
    if arm not in ('chosen_CE', 'human_DPO'):
        raise ValueError('Exact matched chosen-CE or genuine human-ranking DPO arm required')
    return dict(plan['training_options'], DPO_weight=int(arm == 'human_DPO'))


def export(model, plan, digest, arm, completed):
    if (arm not in ('initial', 'chosen_CE', 'human_DPO') or type(completed) is not int
            or completed != (0 if arm == 'initial' else plan['training_options']['steps'])
            or model.counts() != plan['actual_counts']):
        raise ValueError('Exact initial or full-epoch matched human policy required')
    return dict(format=FORMAT, policy=model.export(sources(plan, digest)),
        run_plan_sha256=digest, arm=arm, completed_updates=completed,
        full_human_epoch_completed=arm != 'initial',
        actual_global_human_pairs=0 if arm == 'initial' else plan['human_train_pairs'],
        candidate_promoted=False, goal_achieved=False)


def load(checkpoint, run_plan, arm, *, expected_checkpoint_sha256, expected_plan_sha256, device='cpu'):
    checkpoint, run_plan = Path(checkpoint), Path(run_plan)
    guard = FileBindings({str(checkpoint):expected_checkpoint_sha256, str(run_plan):expected_plan_sha256})
    plan = json.loads(run_plan.read_text())
    raw = torch.load(checkpoint, map_location='cpu', weights_only=True, mmap=True)
    completed = 0 if arm == 'initial' else plan['training_options']['steps']
    if (arm not in ('initial', 'chosen_CE', 'human_DPO') or raw.get('format') != FORMAT
            or raw.get('arm') != arm or raw.get('run_plan_sha256') != expected_plan_sha256
            or raw.get('completed_updates') != completed
            or raw.get('full_human_epoch_completed') != (arm != 'initial')
            or raw.get('actual_global_human_pairs') != (0 if arm == 'initial' else plan['human_train_pairs'])
            or raw.get('candidate_promoted') is not False or raw.get('goal_achieved') is not False):
        raise ValueError('Exact source-bound initial or complete human policy export required')
    model, metadata = fresh(plan, 'cpu')
    payload = raw['policy']
    if (not torch.equal(payload['parts']['critic.weight'], model.parts['critic'].weight)
            or (arm == 'initial' and tensor_digest(payload['parts'].items()) != plan['initial_actor_parts_sha256'])):
        raise ValueError('Untouched separate critic and exact declared initial adapter values required')
    model.import_export(payload, expected_sources=sources(plan, expected_plan_sha256))
    guard.check(full=True)
    return model.eval().to(device), dict(metadata, human_policy_arm=arm, human_policy_updates=completed,
        useful_own_answer_human_alignment_and_high_end_quality_not_proven_by_loading=True)
