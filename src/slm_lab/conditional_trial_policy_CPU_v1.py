"""Student-only native exports for an explicit partial-foundation CPU trial.

This does not masquerade as the complete human-DPO policy or a complete code
source epoch. The immutable own step2048 native base remains external to the
small private adapter artifact, and every loaded student value is counted.
"""
import json
from pathlib import Path

import torch

from .codenet_checkpoint_v1 import FileBindings
from .fast_joint_inference_v2 import load as load_base
from .own_dense_actor_v1 import OwnDenseActor
from .whole_human_sft_evidence_v1 import tensor_digest

FORMAT='own-partial-foundation-qualified-prefix-CPU-trial-policy-v1'
ARMS=('anchor_CE','anchor_execution')


def sources(plan,digest):
    return dict(trial_run_plan_sha256=digest,own_partial_foundation_checkpoint_sha256=plan['base_inputs']['checkpoint_sha256'],
        own_partial_foundation_plan_sha256=plan['base_inputs']['plan_sha256'],qualified_public_prefix_sha256=plan['qualified_episode_binding_sha256'])


def fresh(plan):
    inputs=plan['base_inputs']
    if inputs.get('arm')!='step-2048' or plan.get('actual_base_updates')!=2048 or plan.get('is_complete_human_DPO_or_source_epoch') is not False:
        raise ValueError('Explicit partial own foundation CPU trial identity required')
    base,metadata=load_base(inputs['checkpoint'],inputs['plan'],inputs['arm'],
        expected_checkpoint_sha256=inputs['checkpoint_sha256'],expected_plan_sha256=inputs['plan_sha256'],device='cpu')
    model=OwnDenseActor(base,**plan['actor_settings']).eval()
    if model.counts()!=plan['actual_counts'] or tensor_digest(model.parts.named_parameters())!=plan['initial_parts_sha256']:
        raise ValueError('Exact own under1B neutral starting actor and all values required')
    return model,dict(metadata,partial_foundation_CPU_trial_not_complete_human_policy=True)


def export(model,plan,digest,arm,completed):
    if (arm not in ('initial',*ARMS) or type(completed) is not int
            or completed!=(0 if arm=='initial' else plan['planned_updates_each_arm'])
            or model.counts()!=plan['actual_counts']):raise ValueError('Exact bounded initial or final CPU trial endpoint required')
    if arm=='initial' and tensor_digest(model.parts.named_parameters())!=plan['initial_parts_sha256']:
        raise ValueError('Initial trial endpoint must be exactly the own initial policy')
    return dict(format=FORMAT,policy=model.export(sources(plan,digest)),arm=arm,completed_trial_updates=completed,
        run_plan_sha256=digest,actual_base_updates=2048,
        private_training_reference_included=False,complete_human_alignment_or_source_epoch=False,
        partial_training_source_diagnostic_not_final_candidate=True,candidate_promoted=False,goal_achieved=False)


def load(checkpoint,run_plan,arm,*,expected_checkpoint_sha256,expected_plan_sha256):
    if arm not in ('initial',*ARMS):raise ValueError('Explicit initial, chosen-CE or execution CPU-trial arm required')
    checkpoint,run_plan=Path(checkpoint),Path(run_plan)
    guard=FileBindings({str(checkpoint):expected_checkpoint_sha256,str(run_plan):expected_plan_sha256})
    plan=json.loads(run_plan.read_text());raw=torch.load(checkpoint,map_location='cpu',weights_only=True,mmap=True)
    if (raw.get('format')!=FORMAT or raw.get('arm')!=arm or raw.get('run_plan_sha256')!=expected_plan_sha256
            or raw.get('completed_trial_updates')!=(0 if arm=='initial' else plan['planned_updates_each_arm'])
            or raw.get('actual_base_updates')!=2048 or raw.get('private_training_reference_included') is not False
            or raw.get('complete_human_alignment_or_source_epoch') is not False
            or raw.get('partial_training_source_diagnostic_not_final_candidate') is not True
            or raw.get('candidate_promoted') is not False or raw.get('goal_achieved') is not False):
        raise ValueError('Exact student-only partial-foundation CPU trial export required')
    model,metadata=fresh(plan)
    if not torch.equal(raw['policy']['parts']['critic.weight'],model.parts['critic'].weight):
        raise ValueError('Unused private critic must remain exact')
    model.import_export(raw['policy'],expected_sources=sources(plan,expected_plan_sha256));guard.check(full=True)
    return model.eval(),dict(metadata,actual_trial_arm=arm,actual_trial_updates=raw['completed_trial_updates'],
        exported_unique_values=model.counts()['actual_unique'],training_reference_not_loaded=True)
