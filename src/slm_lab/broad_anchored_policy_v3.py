"""Source-bound code policies starting at our complete human-DPO checkpoint."""
from pathlib import Path

import torch

from .codenet_checkpoint_v1 import FileBindings
from .frozen_own_actor_reference_v1 import FrozenOwnActorReference
from .phase_human_policy_v3 import load as load_human_policy
from .whole_human_sft_evidence_v1 import tensor_digest

FORMAT = 'own-complete-code-first-human-policy-broad-anchored-code-epoch-v3'
ARMS = ('anchor_CE', 'anchor_execution')


def reference_sources(plan):
    inputs = plan['starting_policy_inputs']
    return dict(own_starting_policy_checkpoint_sha256=inputs['checkpoint_sha256'],
                own_starting_policy_plan_sha256=inputs['plan_sha256'])


def fresh(plan, device='cpu'):
    inputs = plan['starting_policy_inputs']
    model, metadata = load_human_policy(inputs['checkpoint'], inputs['plan'], 'human_DPO',
        expected_checkpoint_sha256=inputs['checkpoint_sha256'], expected_plan_sha256=inputs['plan_sha256'], device='cpu')
    if (model.counts() != plan['actual_counts']
            or tensor_digest(model.base.named_parameters()) != plan['frozen_dense_sha256']
            or tensor_digest(model.parts.named_parameters()) != plan['starting_actor_parts_sha256']):
        raise ValueError('Exact own complete human policy, shared foundation and all counted student values required')
    return model.to(device).eval(), dict(metadata, starting_human_policy_is_complete=True,
        starting_human_policy_not_selected_as_winner_by_loading=True)


def make_reference(model, plan):
    reference = FrozenOwnActorReference(model, sources=reference_sources(plan))
    if reference.binding() != plan['own_reference_binding']:
        raise ValueError('Exact frozen own starting actor snapshot required')
    return reference


def sources(plan, digest):
    return dict(run_plan_sha256=digest, **reference_sources(plan),
        APPS_execution_admission_sha256=plan['APPS_execution_admission_sha256'],
        documented_probe_admission_sha256=plan['documented_probe_admission_sha256'])


def export(model, plan, digest, arm, completed):
    if (arm not in ('initial', *ARMS) or type(completed) is not int
            or completed != (0 if arm == 'initial' else plan['planned_steps'])
            or model.counts() != plan['actual_counts']):
        raise ValueError('Exact initial or complete declared code-source epoch required')
    if arm == 'initial' and tensor_digest(model.parts.named_parameters()) != plan['starting_actor_parts_sha256']:
        raise ValueError('Initial code endpoint must be the exact own starting human policy')
    return dict(format=FORMAT, policy=model.export(sources(plan, digest)), arm=arm,
        completed_updates=completed, run_plan_sha256=digest,
        full_declared_code_source_epoch_completed=arm != 'initial',
        original_coding_sources_consumed=0 if arm == 'initial' else plan['original_coding_sources'],
        private_own_reference_weights_included_in_export=False,
        candidate_promoted=False, goal_achieved=False)


def load(checkpoint, run_plan, arm, *, expected_checkpoint_sha256, expected_plan_sha256, device='cpu'):
    checkpoint, run_plan = Path(checkpoint), Path(run_plan)
    guard = FileBindings({str(checkpoint):expected_checkpoint_sha256, str(run_plan):expected_plan_sha256})
    import json
    plan = json.loads(run_plan.read_text())
    raw = torch.load(checkpoint, map_location='cpu', weights_only=True, mmap=True)
    completed = 0 if arm == 'initial' else plan['planned_steps']
    if (arm not in ('initial', *ARMS) or raw.get('format') != FORMAT or raw.get('arm') != arm
            or raw.get('completed_updates') != completed or raw.get('run_plan_sha256') != expected_plan_sha256
            or raw.get('full_declared_code_source_epoch_completed') != (arm != 'initial')
            or raw.get('original_coding_sources_consumed') != (0 if arm == 'initial' else plan['original_coding_sources'])
            or raw.get('private_own_reference_weights_included_in_export') is not False
            or raw.get('candidate_promoted') is not False or raw.get('goal_achieved') is not False):
        raise ValueError('Exact initial/full source-epoch identity and student-only export required')
    model, metadata = fresh(plan, 'cpu')
    payload = raw['policy']
    if not torch.equal(payload['parts']['critic.weight'], model.parts['critic'].weight):
        raise ValueError('Unused separate critic must remain exact')
    if arm == 'initial' and tensor_digest(payload['parts'].items()) != plan['starting_actor_parts_sha256']:
        raise ValueError('Initial policy parts must equal the declared own human-policy endpoint')
    model.import_export(payload, expected_sources=sources(plan, expected_plan_sha256))
    guard.check(full=True)
    return model.to(device).eval(), dict(metadata, native_code_arm=arm, native_code_updates=completed,
        private_training_reference_not_loaded_for_inference=True,
        useful_coding_or_high_end_capability_not_proven_by_loading=True)
