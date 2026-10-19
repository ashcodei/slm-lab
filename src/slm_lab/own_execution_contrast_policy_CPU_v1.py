"""Native student-only IO for the bounded own-negative CPU experiment.

The original own partial foundation remains source-bound. The exported actor
does not carry negative examples, cached reference logits, Adam or a teacher.
This format cannot claim complete human alignment or a full code source epoch.
"""
import json
from pathlib import Path

import torch

from .codenet_checkpoint_v1 import FileBindings
from .fast_joint_inference_v2 import load as load_base
from .own_dense_actor_v1 import OwnDenseActor
from .whole_human_sft_evidence_v1 import tensor_digest

FORMAT = 'own-partial-foundation-machine-execution-contrast-CPU-trial-policy-v1'
ARMS = ('anchor_CE', 'anchor_contrast')


def sources(plan, digest):
    return dict(run_plan_sha256=digest, own_partial_foundation_checkpoint_sha256=plan['base_inputs']['checkpoint_sha256'],
        own_partial_foundation_plan_sha256=plan['base_inputs']['plan_sha256'],
        original_train_pair_manifest_sha256=plan['pair_manifest_sha256'], own_initial_reference_cache_sha256=plan['own_initial_reference_cache_sha256'])


def fresh(plan):
    inputs = plan['base_inputs']
    if (inputs.get('arm') != 'step-2048' or plan.get('actual_base_updates') != 2048
            or plan.get('pairs') != 20 or plan.get('original_train_sources') != 4
            or plan.get('planned_updates_each_arm') != 40 or plan.get('complete_human_alignment_or_source_epoch') is not False):
        raise ValueError('Exact partial own foundation and20-pair/two-pass CPU trial required')
    base, metadata = load_base(inputs['checkpoint'], inputs['plan'], inputs['arm'],
        expected_checkpoint_sha256=inputs['checkpoint_sha256'], expected_plan_sha256=inputs['plan_sha256'], device='cpu')
    actor = OwnDenseActor(base, **plan['actor_settings']).eval()
    if actor.counts() != plan['actual_counts'] or tensor_digest(actor.parts.named_parameters()) != plan['initial_parts_sha256']:
        raise ValueError('Exact complete neutral own starting actor/counts required')
    return actor, dict(metadata, bounded_own_execution_contrast_CPU_trial_not_human_alignment=True)


def export(actor, plan, digest, arm, completed):
    if (arm not in ('initial', *ARMS) or type(completed) is not int
            or completed != (0 if arm == 'initial' else 40) or actor.counts() != plan['actual_counts']):
        raise ValueError('Exact declared initial or40-update student endpoint required')
    if arm == 'initial' and tensor_digest(actor.parts.named_parameters()) != plan['initial_parts_sha256']:
        raise ValueError('Actual initial actor must remain unchanged')
    return dict(format=FORMAT, policy=actor.export(sources(plan, digest)), arm=arm, completed_trial_updates=completed,
        run_plan_sha256=digest, actual_base_updates=2048, machine_execution_labels_not_human_rankings=True,
        own_initial_reference_cache_or_negative_examples_included=False,
        complete_human_alignment_or_source_epoch=False, candidate_promoted=False, goal_achieved=False)


def load(checkpoint, run_plan, arm, *, expected_checkpoint_sha256, expected_plan_sha256):
    if arm not in ('initial', *ARMS): raise ValueError('Explicit own-negative CPU-trial arm required')
    checkpoint, run_plan = Path(checkpoint), Path(run_plan)
    guard = FileBindings({str(checkpoint): expected_checkpoint_sha256, str(run_plan): expected_plan_sha256})
    plan = json.loads(run_plan.read_text()); raw = torch.load(checkpoint, map_location='cpu', weights_only=True, mmap=True)
    if (raw.get('format') != FORMAT or raw.get('arm') != arm or raw.get('run_plan_sha256') != expected_plan_sha256
            or raw.get('completed_trial_updates') != (0 if arm == 'initial' else 40) or raw.get('actual_base_updates') != 2048
            or raw.get('machine_execution_labels_not_human_rankings') is not True
            or raw.get('own_initial_reference_cache_or_negative_examples_included') is not False
            or raw.get('complete_human_alignment_or_source_epoch') is not False
            or raw.get('candidate_promoted') is not False or raw.get('goal_achieved') is not False):
        raise ValueError('Exact bounded partial-foundation machine-labelled CPU student identity required')
    actor, metadata = fresh(plan)
    if not torch.equal(actor.parts['critic'].weight, raw['policy']['parts']['critic.weight']):
        raise ValueError('Unused separate critic must remain unchanged')
    actor.import_export(raw['policy'], expected_sources=sources(plan, expected_plan_sha256)); guard.check(full=True)
    return actor.eval(), dict(metadata, actual_trial_arm=arm, actual_trial_updates=raw['completed_trial_updates'],
        exported_unique_values=actor.counts()['actual_unique'], teacher_or_training_examples_not_loaded=True)
