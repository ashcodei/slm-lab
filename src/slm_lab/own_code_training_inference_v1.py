"""Reference-bound trained code components, with explicit original provenance."""
from pathlib import Path

import torch

from .codenet_checkpoint_v1 import FileBindings
from .corpus import file_hash
from .own_code_delta_inference_v1 import load as load_initial_format
from .own_code_training_stage_v1 import components


def export_payload(model, initial, plan, plan_sha256, completed):
    if completed != plan['training_options']['steps'] or model.counts() != plan['actual_model_counts']:
        raise ValueError('Complete sealed own-code budget and actual model inventory required')
    model.assert_partition()
    return dict(initial, components=components(model), own_code_training=dict(
        format='own-original-generation-repair-with-sampled-CodeNet-v1',
        run_plan_sha256=plan_sha256, initial_component_checkpoint_sha256=plan['initial_component_checkpoint_sha256'],
        original_base_checkpoint_sha256=plan['base_checkpoint_sha256'],
        data_admissions_sha256=plan['data_admissions_sha256'], actual_steps=completed,
        training_options=plan['training_options'], general_vision_and_critic_frozen=True,
        machine_execution_targets_not_genuine_human_preferences=True,
        sampled_variety_not_full_prior_source_pass=True, candidate_promoted=False, goal_achieved=False))


def load(checkpoint, base_checkpoint, tokenizer, initial_plan, training_plan, *, expected_checkpoint_sha256,
         expected_base_sha256, expected_target_sha256, expected_initial_plan_sha256,
         expected_training_plan_sha256, device='cpu'):
    checkpoint, training_plan = Path(checkpoint), Path(training_plan)
    guard = FileBindings({str(checkpoint): expected_checkpoint_sha256,
                          str(training_plan): expected_training_plan_sha256})
    import json
    plan = json.loads(training_plan.read_text())
    state = torch.load(checkpoint, map_location='cpu', weights_only=True, mmap=True)
    expected = dict(format='own-original-generation-repair-with-sampled-CodeNet-v1',
        run_plan_sha256=expected_training_plan_sha256,
        initial_component_checkpoint_sha256=plan['initial_component_checkpoint_sha256'],
        original_base_checkpoint_sha256=expected_base_sha256,
        data_admissions_sha256=plan['data_admissions_sha256'], actual_steps=plan['training_options']['steps'],
        training_options=plan['training_options'], general_vision_and_critic_frozen=True,
        machine_execution_targets_not_genuine_human_preferences=True,
        sampled_variety_not_full_prior_source_pass=True, candidate_promoted=False, goal_achieved=False)
    if (state.get('own_code_training') != expected
            or plan['base_checkpoint_sha256'] != expected_base_sha256
            or plan['initial_plan_sha256'] != expected_initial_plan_sha256
            or plan['own_code_target_sha256'] != expected_target_sha256):
        raise ValueError('Exact completed own training/source/component/reference provenance required')
    model, metadata = load_initial_format(checkpoint, base_checkpoint, tokenizer, initial_plan,
        expected_checkpoint_sha256=expected_checkpoint_sha256, expected_base_sha256=expected_base_sha256,
        expected_target_sha256=expected_target_sha256, expected_plan_sha256=expected_initial_plan_sha256, device=device)
    if model.counts() != plan['actual_model_counts']:
        raise ValueError('Actual trained unique parameter inventory differs from the sealed run')
    guard.check(full=True)
    return model, dict(metadata, own_code_training=expected)
