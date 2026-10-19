"""Completed own execution-feedback components, distinct from the815update format."""
from dataclasses import asdict
import json
from pathlib import Path
import re

import torch
from tokenizers import Tokenizer

from .codenet_checkpoint_v1 import FileBindings
from .corpus import file_hash
from .own_code_delta_v1 import OwnCodeDelta
from .own_code_training_stage_v1 import components
from .training import load_checkpoint

FORMAT = 'own-onpolicy-execution-trained-code-v1'


def options(plan, arm):
    if arm not in plan['arms']:
        raise ValueError('Exact predeclared matched training arm required')
    return {**plan['training_options'], 'DPO_alpha': plan['arms'][arm]['DPO_alpha']}


def provenance(plan, plan_sha256, arm, reference_sha256):
    return dict(format=FORMAT, run_plan_sha256=plan_sha256, arm=arm,
        initial_policy_manifest_sha256=plan['initial_manifest_sha256'],
        initial_code_checkpoint_sha256=plan['initial_code_checkpoint_sha256'],
        original_base_checkpoint_sha256=plan['base_checkpoint_sha256'], tokenizer_sha256=plan['tokenizer_sha256'],
        data_admissions_sha256=plan['data_admissions_sha256'], reference_scores_sha256=reference_sha256,
        actual_steps=options(plan, arm)['steps'], training_options=options(plan, arm),
        general_vision_and_critic_frozen=True, machine_execution_not_genuine_human_preferences=True,
        sampled_variety_not_full_source_pass=True, candidate_promoted=False, goal_achieved=False)


def export_payload(model, plan, plan_sha256, arm, reference_sha256, completed):
    if completed != options(plan, arm)['steps'] or model.counts() != plan['actual_model_counts']:
        raise ValueError('Complete exact own execution-feedback arm, budget and counts required')
    model.assert_partition()
    return dict(format=FORMAT, config=asdict(model.config), components=components(model),
        reserved_parameters=model.reserved_parameters, actual_counts=model.counts(),
        own_onpolicy_training=provenance(plan, plan_sha256, arm, reference_sha256),
        candidate_promoted=False, goal_achieved=False)


def load(checkpoint, run_plan, arm, reference_scores, *, expected_checkpoint_sha256,
         expected_plan_sha256, expected_reference_sha256, device='cpu'):
    if any(not isinstance(v, str) or not re.fullmatch('[0-9a-f]{64}', v) for v in
           [expected_checkpoint_sha256, expected_plan_sha256, expected_reference_sha256]):
        raise ValueError('Exact externally bound trained/source/reference digests required')
    checkpoint, run_plan, reference_scores = map(Path, [checkpoint, run_plan, reference_scores])
    guard = FileBindings({str(checkpoint): expected_checkpoint_sha256, str(run_plan): expected_plan_sha256,
        str(reference_scores): expected_reference_sha256})
    plan = json.loads(run_plan.read_text())
    guard = FileBindings({**{str(p): d for p, d in guard.files.items()},
        plan['base_checkpoint']: plan['base_checkpoint_sha256'], plan['tokenizer']: plan['tokenizer_sha256']})
    state = torch.load(checkpoint, map_location='cpu', weights_only=True, mmap=True)
    if (state.get('format') != FORMAT or state.get('own_onpolicy_training') != provenance(plan, expected_plan_sha256, arm, expected_reference_sha256)
            or state.get('candidate_promoted') is not False or state.get('goal_achieved') is not False
            or state['actual_counts'] != plan['actual_model_counts']):
        raise ValueError('Actual completed own arm/component/reference/source provenance required')
    reference = json.loads(reference_scores.read_text())
    if reference['policy_sha256'] != plan['initial_manifest_sha256'] or reference['human_preferences'] is not False:
        raise ValueError('Exact frozen own initial policy reference required')
    base, metadata = load_checkpoint(plan['base_checkpoint'], 'cpu', training_state=False)
    if (asdict(base.config) != state['config'] or metadata['tokenizer_sha256'] != plan['tokenizer_sha256']
            or Tokenizer.from_file(plan['tokenizer']).get_vocab_size() != base.config.vocab_size):
        raise ValueError('Exact native original backbone and own tokenizer required')
    model = OwnCodeDelta(base, state['components'], reserved_parameters=state['reserved_parameters']).eval().to(device)
    if model.counts() != state['actual_counts']:
        raise ValueError('Actual complete original/code/reserved-head parameter budget must match')
    guard.check(full=True)
    return model, dict(own_onpolicy_training=state['own_onpolicy_training'], tokenizer_sha256=plan['tokenizer_sha256'],
        actual_model_counts=model.counts(), original_backbones_loaded=1, external_model_loaded=False,
        full_dense_code_source_or_initial_component_weights_loaded=False, candidate_promoted=False)


def attach_original_head(model, plan):
    from .own_assistant_v1 import attach_grounding
    manifest = Path(plan['initial_manifest'])
    FileBindings({str(manifest): plan['initial_manifest_sha256']})
    spec = json.loads(manifest.read_text())
    def path(key):
        value = Path(spec[key]); return value if value.is_absolute() else manifest.parent / value
    return attach_grounding(model, path('grounding_head'), path('base_checkpoint'), path('tokenizer'), path('grounding_plan'),
        expected_head_sha256=spec['grounding_head_sha256'], expected_base_sha256=spec['base_checkpoint_sha256'],
        expected_tokenizer_sha256=spec['tokenizer_sha256'], expected_plan_sha256=spec['grounding_plan_sha256'],
        expected_data_admission_sha256=spec['grounding_data_admission_sha256'])
