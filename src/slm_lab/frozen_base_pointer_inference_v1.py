"""Load the two learned grounding matrices with one immutable own backbone."""
from dataclasses import asdict
import json
from pathlib import Path

import torch

from .codenet_checkpoint_v1 import FileBindings
from .corpus import file_hash
from .frozen_base_pointer_stage_v1 import make_head, projections, assert_partition
from .training import load_checkpoint

FORMAT = 'own-frozen-base-pointer-head-v1'


def export_payload(head, options, sources, completed, frozen_base_digest):
    assert_partition(head)
    if completed != options['steps'] or options['vision_learning_rate'] != 0.:
        raise ValueError('Complete original head-only source epoch required')
    return dict(format=FORMAT, projections=projections(head), config=asdict(head.base.config), head_rank=head.rank,
        training_options=options, sources=sources, actual_steps=completed,
        actual_inventory=head.trainable_inventory(), frozen_base_digest=frozen_base_digest,
        entire_original_backbone_frozen=True, original_human_attention_not_boxes_or_preferences=True,
        candidate_promoted=False, goal_achieved=False)


def load(checkpoint, base_checkpoint, tokenizer, run_plan, *, expected_head_sha256, expected_base_sha256,
         expected_tokenizer_sha256, expected_plan_sha256, expected_data_admission_sha256, device='cpu'):
    checkpoint, base_checkpoint, tokenizer, run_plan = map(Path, [checkpoint, base_checkpoint, tokenizer, run_plan])
    guard = FileBindings({str(checkpoint): expected_head_sha256, str(base_checkpoint): expected_base_sha256,
                          str(tokenizer): expected_tokenizer_sha256, str(run_plan): expected_plan_sha256})
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    plan = json.loads(run_plan.read_text())
    expected_sources = dict(base_checkpoint_sha256=expected_base_sha256, tokenizer_sha256=expected_tokenizer_sha256,
        run_plan_sha256=expected_plan_sha256, data_admission_sha256=expected_data_admission_sha256)
    if (state.get('format') != FORMAT or state['sources'] != expected_sources
            or state['training_options'] != plan['training_options'] or state['actual_steps'] != plan['training_options']['steps']
            or state['training_options']['vision_learning_rate'] != 0.
            or state['entire_original_backbone_frozen'] is not True
            or state['candidate_promoted'] is not False or state['goal_achieved'] is not False
            or state['head_rank'] != plan['head_rank']
            or plan['base_checkpoint_sha256'] != expected_base_sha256
            or plan['data_admission_sha256'] != expected_data_admission_sha256):
        raise ValueError('Exact completed frozen-backbone/head/source/tokenizer/budget provenance required')
    base, metadata = load_checkpoint(base_checkpoint, 'cpu', training_state=False)
    if asdict(base.config) != state['config'] or metadata['tokenizer_sha256'] != expected_tokenizer_sha256:
        raise ValueError('Original own complete multimodal backbone/config/tokenizer required')
    head = make_head(base, rank=state['head_rank'])
    current = head.state_dict()
    if (set(state['projections']) != {'query_projection.weight', 'patch_projection.weight'}
            or any(tensor.shape != current[name].shape or tensor.dtype != current[name].dtype
                   or not bool(torch.isfinite(tensor).all()) for name, tensor in state['projections'].items())):
        raise ValueError('Both finite original-dimension projection matrices required')
    head.query_projection.load_state_dict({'weight': state['projections']['query_projection.weight']})
    head.patch_projection.load_state_dict({'weight': state['projections']['patch_projection.weight']})
    if head.trainable_inventory() != state['actual_inventory']:
        raise ValueError('Actual unique frozen-base/head inventory differs from the saved artifact')
    guard.check(full=True); head.eval().to(device)
    return head, dict(tokenizer_sha256=expected_tokenizer_sha256, base_checkpoint_sha256=expected_base_sha256,
        head_checkpoint_sha256=expected_head_sha256, run_plan_sha256=expected_plan_sha256,
        data_admission_sha256=expected_data_admission_sha256, actual_inventory=head.trainable_inventory(),
        additional_parameters=head.extra_parameters, entire_original_backbone_frozen=True,
        training_options=state['training_options'], candidate_promoted=False)
