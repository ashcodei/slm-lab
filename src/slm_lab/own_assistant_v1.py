"""Explicit general/code/grounding routes sharing one original own backbone.

The artifact remains experimental. A complete source-bound component is not
automatically a qualified policy, and structural parity is not capability.
"""
from dataclasses import asdict
import json
from pathlib import Path

import torch
from torch import nn

from .codenet_checkpoint_v1 import FileBindings
from .corpus import file_hash
from .frozen_base_pointer_inference_v1 import FORMAT as HEAD_FORMAT
from .frozen_base_pointer_stage_v1 import make_head, assert_partition, PROJECTIONS
from .own_code_delta_v1 import OwnCodeDelta
from .point_grounding_evaluation_v1 import state_digest
from .point_grounding_head_v1 import PointGroundingHead

MANIFEST_FORMAT = 'own-single-backbone-code-and-grounding-references-v1'


class OwnAssistant(nn.Module):
    def __init__(self, language, grounding, *, parameter_limit=10**9):
        super().__init__()
        if type(language) is not OwnCodeDelta or type(grounding) is not PointGroundingHead:
            raise ValueError('Exact own language/code and original grounding architectures required')
        language.assert_partition(); assert_partition(grounding)
        if grounding.base is not language.base:
            raise ValueError('A second backbone is forbidden; both routes must share the identical base object')
        ids = {id(p) for p in language.parameters()}
        extras = [p for name, p in grounding.named_parameters() if name in PROJECTIONS]
        actual = sum(p.numel() for p in language.parameters()) + sum(p.numel() for p in extras)
        if (any(id(p) in ids for p in extras) or grounding.extra_parameters != language.reserved_parameters
                or type(parameter_limit) is not int or not actual < parameter_limit <= 10**9):
            raise ValueError('Both distinct head matrices must fit the reserved slot and the complete model must stay below1B')
        self.language = language; self.grounding = grounding; self.parameter_limit = parameter_limit
        self.assert_shared()

    @property
    def config(self):
        return self.language.config

    def counts(self):
        return dict(shared_original_backbone=sum(p.numel() for p in self.language.base.parameters()),
            additional_code=sum(p.numel() for p in self.language.parts.parameters()),
            additional_grounding=self.grounding.extra_parameters,
            total_unique=sum(p.numel() for p in self.parameters()))

    def assert_shared(self):
        self.language.assert_partition(); assert_partition(self.grounding)
        counts = self.counts()
        if (self.language.base is not self.grounding.base
                or self.language.code.vision is not self.grounding.base.vision
                or counts['total_unique'] != counts['shared_original_backbone'] + counts['additional_code'] + counts['additional_grounding']
                or counts['total_unique'] >= self.parameter_limit):
            raise ValueError('Exactly one actual own backbone and complete unique parameter accounting required')

    def forward(self, *args, route='general', **kwargs):
        self.assert_shared()
        if route == 'grounding':
            return self.grounding(*args, **kwargs)
        return self.language(*args, route=route, **kwargs)

    def generate(self, *args, route='general', **kwargs):
        if route not in ['general', 'code']:
            raise ValueError('Generation requires an explicit general or code route')
        self.assert_shared()
        return self.language.generate(*args, route=route, **kwargs)

    @torch.no_grad()
    def locate(self, input_ids, pixel_values, *, attention_mask=None, valid_cells=None):
        self.assert_shared()
        return self.grounding(input_ids, pixel_values, attention_mask=attention_mask, valid_cells=valid_cells)


def attach_grounding(language, checkpoint, base_checkpoint, tokenizer, run_plan, *, expected_head_sha256,
                     expected_base_sha256, expected_tokenizer_sha256, expected_plan_sha256,
                     expected_data_admission_sha256, parameter_limit=10**9):
    """Validate every reference before attaching two matrices to an existing base.

    No backbone is instantiated, loaded into, moved or restored here. The
    already loaded base's actual canonical tensor digest must match the head.
    """
    if type(language) is not OwnCodeDelta or language.training:
        raise ValueError('An already-loaded exact own conditional-code evaluation model is required')
    language.assert_partition()
    checkpoint, base_checkpoint, tokenizer, run_plan = map(Path, [checkpoint, base_checkpoint, tokenizer, run_plan])
    guard = FileBindings({str(checkpoint): expected_head_sha256, str(base_checkpoint): expected_base_sha256,
                          str(tokenizer): expected_tokenizer_sha256, str(run_plan): expected_plan_sha256})
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    plan = json.loads(run_plan.read_text())
    sources = dict(base_checkpoint_sha256=expected_base_sha256, tokenizer_sha256=expected_tokenizer_sha256,
        run_plan_sha256=expected_plan_sha256, data_admission_sha256=expected_data_admission_sha256)
    rank = state.get('head_rank')
    if (state.get('format') != HEAD_FORMAT or state.get('sources') != sources
            or state['training_options'] != plan['training_options'] or state['actual_steps'] != plan['training_options']['steps']
            or state['training_options']['vision_learning_rate'] != 0.
            or state['entire_original_backbone_frozen'] is not True or state['candidate_promoted'] is not False
            or state['goal_achieved'] is not False or type(rank) is not int or not 1 <= rank <= language.config.hidden_size
            or rank != plan['head_rank'] or state['config'] != asdict(language.config)
            or plan['base_checkpoint_sha256'] != expected_base_sha256 or plan['data_admission_sha256'] != expected_data_admission_sha256
            or state['frozen_base_digest'] != state_digest(language.base)):
        raise ValueError('Actual shared original base and complete head/source/tokenizer/budget provenance must match')
    shape = (rank, language.config.hidden_size)
    dtype = language.base.embedding.weight.dtype
    extra = 2 * rank * language.config.hidden_size
    total = language.counts()['total_unique'] + extra
    expected_inventory = dict(total_unique_parameters=language.counts()['frozen_general'] + extra,
        actual_trainable_parameters=extra, actual_additional_head_parameters=extra,
        trainable_parameter_names=list(PROJECTIONS))
    if (set(state['projections']) != set(PROJECTIONS)
            or any(t.shape != shape or t.dtype != dtype or not bool(torch.isfinite(t).all()) for t in state['projections'].values())
            or state['actual_inventory'] != expected_inventory
            or extra != language.reserved_parameters or type(parameter_limit) is not int or not total < parameter_limit <= 10**9):
        raise ValueError('Complete finite head matrices and an actual combined sub1B budget required')
    if len({p.device for p in language.parameters()}) != 1:
        raise ValueError('Original general/code model must already occupy one explicit inference device')
    guard.check(full=True)
    # The head constructor initializes CPU matrices. Preserve the caller's CPU
    # RNG, overwrite those matrices, then move only them to the existing base
    # device. The original base and existing code cache retain their versions.
    original_checkpointing = language.base.gradient_checkpointing
    try:
        with torch.random.fork_rng(devices=[]):
            head = make_head(language.base, rank=rank)
    finally:
        # Preserve the serving model's metadata as well as its tensor bytes.
        language.base.gradient_checkpointing = original_checkpointing
        for parameter in language.base.parameters():
            parameter.requires_grad_(False)
    head.query_projection.load_state_dict({'weight': state['projections']['query_projection.weight']})
    head.patch_projection.load_state_dict({'weight': state['projections']['patch_projection.weight']})
    device = language.base.embedding.weight.device
    head.query_projection.to(device); head.patch_projection.to(device)
    if head.trainable_inventory() != state['actual_inventory']:
        raise ValueError('Actual shared-base/head parameter inventory differs from the completed artifact')
    combined = OwnAssistant(language, head, parameter_limit=parameter_limit).eval()
    return combined, dict(actual_counts=combined.counts(), base_checkpoint_sha256=expected_base_sha256,
        head_checkpoint_sha256=expected_head_sha256, head_plan_sha256=expected_plan_sha256,
        tokenizer_sha256=expected_tokenizer_sha256, full_backbones_loaded_by_attachment=0,
        automatic_routing_unproven=True, candidate_promoted=False)


def load_manifest(manifest, *, expected_manifest_sha256, device='cpu'):
    """Load one complete own general/code base, then attach its validated head."""
    manifest = Path(manifest).resolve()
    FileBindings({str(manifest): expected_manifest_sha256})
    spec = json.loads(manifest.read_text())
    if spec.get('format') != MANIFEST_FORMAT or spec.get('candidate_promoted') is not False or spec.get('goal_achieved') is not False:
        raise ValueError('Explicit source-bound experimental single-backbone manifest required')
    def path(name):
        value = Path(spec[name])
        return value if value.is_absolute() else manifest.parent / value
    names = ['base_checkpoint', 'tokenizer', 'code_checkpoint', 'code_initial_plan', 'code_training_plan', 'grounding_head', 'grounding_plan']
    files = {str(manifest): expected_manifest_sha256}
    for name in names:
        key = str(path(name).resolve()); digest = spec[name + '_sha256']
        if key in files and files[key] != digest:
            raise ValueError('Conflicting artifact references are forbidden')
        files[key] = digest
    guard = FileBindings(files)
    from .own_code_training_inference_v1 import load
    language, code_metadata = load(path('code_checkpoint'), path('base_checkpoint'), path('tokenizer'),
        path('code_initial_plan'), path('code_training_plan'), expected_checkpoint_sha256=spec['code_checkpoint_sha256'],
        expected_base_sha256=spec['base_checkpoint_sha256'], expected_target_sha256=spec['own_code_target_sha256'],
        expected_initial_plan_sha256=spec['code_initial_plan_sha256'], expected_training_plan_sha256=spec['code_training_plan_sha256'], device=device)
    combined, metadata = attach_grounding(language, path('grounding_head'), path('base_checkpoint'), path('tokenizer'), path('grounding_plan'),
        expected_head_sha256=spec['grounding_head_sha256'], expected_base_sha256=spec['base_checkpoint_sha256'],
        expected_tokenizer_sha256=spec['tokenizer_sha256'], expected_plan_sha256=spec['grounding_plan_sha256'],
        expected_data_admission_sha256=spec['grounding_data_admission_sha256'])
    if combined.counts() != spec['actual_counts']:
        raise ValueError('Every actual unique original/code/head parameter must match the declared complete model')
    guard.check(full=True)
    return combined, dict(metadata, code_metadata=code_metadata, manifest_sha256=expected_manifest_sha256,
                          genuine_final_policy_alignment_or_high_end_capability_not_established=True)
