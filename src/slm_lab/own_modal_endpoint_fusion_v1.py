"""Combine source-bound own dense text and visual values without extra weights.

Copy the complete learned vision/word/pointing branches onto a new text-model
copy. Text-only computation and word features are retained independently;
their coupled image-language interpretation requires joint learning/testing.
No source model, inference cache, trainability flag or RNG is changed.
"""
import copy
from dataclasses import asdict
from pathlib import Path

import torch

from .codenet_checkpoint_v1 import FileBindings
from .corpus import file_hash
from .deterministic_positions_v1 import vision_forward, patch_grid
from .own_dense_growth_v1 import OwnDenseAssistant
from .whole_human_sft_evidence_v1 import tensor_digest

FORMAT = 'own-single-dense-text-vision-word-endpoint-fusion-v1'
VISUAL_PREFIXES = ('network.vision.', 'heads.word.', 'heads.pointing.')


def signature(model):
    return tuple((n, id(p), p.data_ptr(), p._version, p.requires_grad, str(p.dtype), str(p.device))
        for n, p in model.named_parameters())


def binding(model, identity):
    if (type(identity) is not dict or set(identity) != {'parameters_sha256', 'tokenizer_sha256', 'sources'}
            or type(identity['sources']) is not dict or not identity['sources']
            or any(type(v) is not str or len(v) != 64 or any(c not in '0123456789abcdef' for c in v)
                for v in [identity['parameters_sha256'], identity['tokenizer_sha256'], *identity['sources'].values()])
            or any(type(k) is not str or not k for k in identity['sources'])):
        raise ValueError('Every actual own model, tokenizer and complete original source binding is required')
    files = {p: h for p, h in identity['sources'].items() if Path(p).is_absolute()}
    if not files: raise ValueError('Actual immutable source files must accompany semantic model bindings')
    guard = FileBindings(files)
    if tensor_digest(model.named_parameters()) != identity['parameters_sha256']:
        raise ValueError('Every actually loaded own source value must match its independently supplied identity')
    return guard


def combined_sources(text_identity, visual_identity):
    result = {str(Path(__file__).resolve()): file_hash(__file__)}
    for prefix, identity in [('text_source', text_identity), ('visual_source', visual_identity)]:
        for name, digest in identity['sources'].items():
            key = name if Path(name).is_absolute() else prefix + ':' + name
            if key in result and result[key] != digest:
                raise ValueError('An immutable source file cannot have conflicting identities across the two endpoints')
            result[key] = digest
    return result


@torch.no_grad()
def fuse(text, visual, *, text_identity, visual_identity, working_budget_bytes):
    if type(text) is not OwnDenseAssistant or type(visual) is not OwnDenseAssistant:
        raise ValueError('Two exact own complete dense model sources are required')
    for model in [text, visual]:
        model.assert_inventory()
        if (model.training or any(p.device.type != 'cpu' or p.dtype != torch.float32 or p.grad is not None for p in model.parameters())
                or any(m._forward_hooks or m._forward_pre_hooks or m._backward_hooks for m in model.modules())
                or getattr(model.network.vision.forward, '__func__', None) is not vision_forward
                or getattr(model.patch_grid, '__func__', None) is not patch_grid):
            raise ValueError('Complete eval-mode CPU FP32 sources with canonical deterministic vision and clear gradients are required')
    required = 4 * sum(p.numel() for p in text.parameters()) + 4 * max(p.numel() for p in text.parameters()) + 128 * 1024**2
    if type(working_budget_bytes) is not int or working_budget_bytes < required:
        raise ValueError('The complete separate output and source-check working memory must be explicitly funded')
    text_identity, visual_identity = copy.deepcopy(text_identity), copy.deepcopy(visual_identity)
    guards = [binding(text, text_identity), binding(visual, visual_identity)]
    output_sources = combined_sources(text_identity, visual_identity)
    if (asdict(text.config) != asdict(visual.config) or text.counts() != visual.counts()
            or {k: v for k, v in text.settings.items() if k != 'seed'} != {k: v for k, v in visual.settings.items() if k != 'seed'}
            or text_identity['tokenizer_sha256'] != visual_identity['tokenizer_sha256']):
        raise ValueError('Identical complete own vocabulary, architecture, vision and auxiliary head geometry are required')
    left, right = dict(text.named_parameters()), dict(visual.named_parameters())
    if left.keys() != right.keys() or any(left[n].shape != right[n].shape or not bool(torch.isfinite(left[n]).all())
            or not bool(torch.isfinite(right[n]).all()) for n in left):
        raise ValueError('Every original source parameter must have its exact finite counterpart')
    source_signatures = signature(text), signature(visual)
    rng = torch.get_rng_state().clone()
    output = copy.deepcopy(text)
    copied = [n for n in left if n.startswith(VISUAL_PREFIXES)]
    for name in copied: output.get_parameter(name).copy_(right[name])
    output.assert_inventory()
    if (signature(text) != source_signatures[0] or signature(visual) != source_signatures[1]
            or not torch.equal(torch.get_rng_state(), rng)
            or tensor_digest(text.named_parameters()) != text_identity['parameters_sha256']
            or tensor_digest(visual.named_parameters()) != visual_identity['parameters_sha256']
            or {p.data_ptr() for p in output.parameters()} & {p.data_ptr() for model in [text, visual] for p in model.parameters()}
            or any(not torch.equal(p, right[n] if n in copied else left[n]) for n, p in output.named_parameters())):
        raise ValueError('Every source value/metadata/RNG and every exact independently owned output value must remain')
    for guard in guards: guard.check(full=True)
    return output, dict(format=FORMAT, text_identity=copy.deepcopy(text_identity), visual_identity=copy.deepcopy(visual_identity),
        complete_output_sources=output_sources,
        actual_counts=output.counts(), actual_copied_visual_tensor_names=copied,
        actual_output_parameters_sha256=tensor_digest(output.named_parameters()), declared_working_bytes=working_budget_bytes,
        required_additional_working_bytes=required, retained_extra_source_models_or_neural_parameters=0,
        all_other_original_text_model_and_auxiliary_values_preserved=True, all_source_parameters_metadata_and_CPU_RNG_unchanged=True,
        shared_image_language_alignment_and_model_quality_still_require_joint_training_and_evaluation=True,
        full_size_GPU_capacity_or_quality_gain=False, candidate_promoted=False, goal_achieved=False)
