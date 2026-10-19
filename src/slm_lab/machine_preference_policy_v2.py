"""Source-bound whole-program preferences after our complete own code/RL arm."""
import json
import math
from pathlib import Path

import torch

from .broad_anchored_policy_v3 import load as load_code_policy
from .codenet_checkpoint_v1 import FileBindings
from .frozen_own_actor_reference_v1 import FrozenOwnActorReference
from .whole_human_sft_evidence_v1 import tensor_digest

FORMAT = 'own-complete-code-actor-full-execution-preference-epoch-v2'
ARMS = ('chosen_CE', 'machine_DPO')


def reference_sources(plan):
    inputs=plan['starting_policy_inputs']
    return dict(own_starting_code_checkpoint_sha256=inputs['checkpoint_sha256'],
                own_starting_code_plan_sha256=inputs['plan_sha256'])


def fresh(plan, device='cpu'):
    inputs=plan['starting_policy_inputs']
    model,metadata=load_code_policy(inputs['checkpoint'],inputs['plan'],'anchor_execution',
        expected_checkpoint_sha256=inputs['checkpoint_sha256'],expected_plan_sha256=inputs['plan_sha256'],device='cpu')
    if (model.counts()!=plan['actual_counts']
            or tensor_digest(model.base.named_parameters())!=plan['frozen_dense_sha256']
            or tensor_digest(model.parts.named_parameters())!=plan['starting_actor_parts_sha256']):
        raise ValueError('Exact complete own code/RL actor, shared foundation and all counted student values required')
    return model.to(device).eval(),dict(metadata,starting_code_policy_is_complete=True,
        starting_policy_not_selected_as_quality_winner_by_loading=True)


def make_reference(model, plan):
    reference=FrozenOwnActorReference(model,sources=reference_sources(plan))
    if reference.binding()!=plan['own_reference_binding']:
        raise ValueError('The frozen own reference must include the actual nonzero starting code actor')
    return reference


def options(plan, arm):
    if arm not in ARMS:raise ValueError('A declared matched machine preference arm is required')
    return dict(plan['training_options'],method=arm)


def sources(plan, digest):
    return dict(run_plan_sha256=digest,**reference_sources(plan),
        machine_admission_sha256=plan['machine_admission_sha256'],machine_manifest_sha256=plan['machine_manifest_sha256'])


def export(model, plan, digest, arm, completed):
    population=plan['original_machine_pairs'];width=plan['training_options']['global_batch_size']
    if (arm not in ('initial',*ARMS) or type(completed) is not int
            or plan['planned_steps']!=math.ceil(population/width)
            or completed!=(0 if arm=='initial' else plan['planned_steps'])
            or model.counts()!=plan['actual_counts']):
        raise ValueError('An exact initial endpoint or complete unrepeated machine preference source epoch is required')
    if arm=='initial' and tensor_digest(model.parts.named_parameters())!=plan['starting_actor_parts_sha256']:
        raise ValueError('The initial machine endpoint must equal the complete own starting code actor')
    return dict(format=FORMAT,policy=model.export(sources(plan,digest)),arm=arm,
        completed_updates=completed,run_plan_sha256=digest,
        full_declared_machine_source_epoch_completed=arm!='initial',
        original_machine_pairs_consumed=0 if arm=='initial' else population,
        human_preferences=False,private_own_reference_weights_included_in_export=False,
        candidate_promoted=False,goal_achieved=False)


def load(checkpoint, run_plan, arm, *, expected_checkpoint_sha256, expected_plan_sha256, device='cpu'):
    checkpoint,run_plan=Path(checkpoint),Path(run_plan)
    guard=FileBindings({str(checkpoint):expected_checkpoint_sha256,str(run_plan):expected_plan_sha256})
    plan=json.loads(run_plan.read_text());raw=torch.load(checkpoint,map_location='cpu',weights_only=True,mmap=True)
    completed=0 if arm=='initial' else plan['planned_steps']
    if (arm not in ('initial',*ARMS) or raw.get('format')!=FORMAT or raw.get('arm')!=arm
            or raw.get('run_plan_sha256')!=expected_plan_sha256 or raw.get('completed_updates')!=completed
            or raw.get('full_declared_machine_source_epoch_completed')!=(arm!='initial')
            or raw.get('original_machine_pairs_consumed')!=(0 if arm=='initial' else plan['original_machine_pairs'])
            or raw.get('human_preferences') is not False
            or raw.get('private_own_reference_weights_included_in_export') is not False
            or raw.get('candidate_promoted') is not False or raw.get('goal_achieved') is not False):
        raise ValueError('Exact complete machine source identity, method and student-only export required')
    if plan['planned_steps']!=math.ceil(plan['original_machine_pairs']/plan['training_options']['global_batch_size']):
        raise ValueError('All original machine pairs must fit exactly one declared unrepeated source epoch')
    model,metadata=fresh(plan,'cpu');payload=raw['policy']
    if not torch.equal(payload['parts']['critic.weight'],model.parts['critic'].weight):
        raise ValueError('Unused separate critic must remain exact')
    if arm=='initial' and tensor_digest(payload['parts'].items())!=plan['starting_actor_parts_sha256']:
        raise ValueError('Initial policy parts differ from the declared complete starting code actor')
    model.import_export(payload,expected_sources=sources(plan,expected_plan_sha256));guard.check(full=True)
    return model.to(device).eval(),dict(metadata,machine_preference_arm=arm,machine_preference_updates=completed,
        private_training_reference_not_loaded_for_inference=True,
        useful_coding_or_general_quality_not_proven_by_loading=True)
