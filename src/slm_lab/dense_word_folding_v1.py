"""Fold source-bound shared vision/word updates into one own dense model.

This offline FP32 operation adds no parameters or second encoder. Matrix
merging changes floating-point order, so native output/quality parity must be
measured separately. It does not certify a trained word source is useful.
"""
from dataclasses import asdict

import torch

from .own_aligned_assistant_v3 import OwnAlignedAssistantV3
from .own_code_delta_v1 import matrix_names,norm_names
from .own_dense_growth_v1 import OwnDenseAssistant
from .whole_human_sft_evidence_v1 import tensor_digest

FORMAT='own-dense-folded-shared-word-vision-attachment-v1'
VISION_FIELDS=('hidden_size','vision_hidden_size','vision_intermediate_size','vision_layers','vision_heads',
               'image_size','patch_size','pool_size','rope_theta','norm_eps')


def fold_words(target,source,payload,*,expected_sources):
    """Validate the entire source and candidate copy plan before any mutation."""
    if type(target) is not OwnDenseAssistant or type(source) is not OwnAlignedAssistantV3:
        raise ValueError('Exact one-backbone dense target and own code-initialized shared-word source required')
    target.assert_inventory();source.assert_word_contract()
    source.validate_export(payload,expected_sources=expected_sources)
    if (any(p.device.type!='cpu' or p.dtype!=torch.float32 for p in (*target.parameters(),*source.parameters()))
            or any(getattr(target.config,n)!=getattr(source.config,n) for n in VISION_FIELDS)
            or target.heads['word'].expansion!=source.parts['word'].expansion):
        raise ValueError('Offline real FP32 CPU models with identical vision and word geometry required')
    live=source.parts.state_dict();saved=payload['extensions']
    if any(not torch.equal(t,saved[n]) for n,t in live.items()):
        raise ValueError('Every actually loaded source extension must equal the bound source payload')
    target_vision=target.network.vision;original=source.original.language.base.vision
    reference=original.state_dict();before=target_vision.state_dict()
    if (reference.keys()!=before.keys() or any(t.shape!=before[n].shape or t.dtype!=before[n].dtype
            or not bool(torch.isfinite(t).all()) or not torch.equal(t,before[n]) for n,t in reference.items())):
        raise ValueError('Every original vision tensor must match before a single word-update fold; reject double folding')
    target_word=target.heads['word'].state_dict();source_word=source.parts['word'].state_dict()
    if (target_word.keys()!=source_word.keys() or any(t.shape!=target_word[n].shape or t.dtype!=target_word[n].dtype
            for n,t in source_word.items())):
        raise ValueError('Exact whole byte-word head shapes and computation required')
    # Read/compute every result first. Overflow, missing tensors or bad geometry
    # cannot leave a partly attached target behind.
    copies={};merges=[]
    for i,name in enumerate(matrix_names(target.config.vision_layers)):
        factor=source.parts['vision_linears'][i]
        copies['network.vision.'+name]=reference[name]+factor.up.detach()@factor.down.detach()
        merges.append(name)
    for i,name in enumerate(norm_names(target.config.vision_layers)):
        copies['network.vision.'+name]=source.parts['vision_norms'][i].weight.detach().clone()
    factor=source.parts['vision_projection']
    copies['network.vision.project.weight']=reference['project.weight']+factor.up.detach()@factor.down.detach()
    for name,t in source_word.items():copies['heads.word.'+name]=t.detach().clone()
    for name,t in copies.items():
        parameter=target.get_parameter(name)
        if t.shape!=parameter.shape or t.dtype!=parameter.dtype or not bool(torch.isfinite(t).all()):
            raise ValueError('Every complete fused vision/word tensor must be finite and exact-shaped before mutation')
    source_signature=[(id(p),p._version,p.data_ptr(),p.requires_grad) for p in source.parameters()]
    selected={id(target.get_parameter(n)) for n in copies}
    fixed_signature=[(id(p),p._version,p.data_ptr(),p.requires_grad) for p in target.parameters() if id(p) not in selected]
    language_before=tensor_digest(target.language_named_parameters())
    counts=target.counts();rng=torch.get_rng_state().clone()
    with torch.no_grad():
        for name,t in copies.items():target.get_parameter(name).copy_(t)
    target.assert_inventory()
    if (source_signature!=[(id(p),p._version,p.data_ptr(),p.requires_grad) for p in source.parameters()]
            or fixed_signature!=[(id(p),p._version,p.data_ptr(),p.requires_grad) for p in target.parameters() if id(p) not in selected]
            or target.counts()!=counts or not torch.equal(rng,torch.get_rng_state())
            or language_before!=tensor_digest(target.language_named_parameters())):
        raise ValueError('One counted target, every language/fixed/source parameter and RNG must stay exact')
    return dict(format=FORMAT,source_export_format=payload['format'],source_bindings=dict(expected_sources),
        source_vision_config={n:asdict(source.config)[n] for n in VISION_FIELDS},word_frame_expansion=source.parts['word'].expansion,
        merged_original_vision_matrices=merges,merged_vision_projection=True,
        copied_vision_norms=len(norm_names(target.config.vision_layers)),copied_word_tensors=len(source_word),
        actual_updated_dense_tensor_names=list(copies),actual_counts=counts,
        language_values_and_fixed_parameter_metadata_exact=True,source_parameter_metadata_and_CPU_RNG_exact=True,
        retained_extra_vision_encoders_or_adapter_factors=0,
        FP32_merge_changes_arithmetic_order=True,neural_native_parity_and_held_reading_quality_not_inferred=True,
        candidate_promoted=False,goal_achieved=False)
