"""Blend already reduced task gradients with optional primary-task projection.

The projection is inspired by PCGrad (Yu et al., NeurIPS2020), with a fixed
primary direction instead of its symmetric/random task treatment:
https://arxiv.org/abs/2001.06782 . This priority variant is an experimental
hypothesis, not a new-method, accuracy or monotonic AdamW-loss guarantee.
Global task reductions and original target denominators belong upstream;
gradient clipping and the actual optimizer step belong downstream.
"""
import math

import torch

FORMAT = 'own-primary-task-gradient-projection-and-complete-blend-v1'


def dot(left, right):
    if left is None or right is None: return 0.
    value = float((left.double() * right.double()).sum())
    if not math.isfinite(value): raise ValueError('Every complete gradient interaction must remain finite')
    return value


def combine(primary, secondary, *, primary_names, secondary_weight, project_conflicts):
    if (type(primary) is not dict or type(secondary) is not dict or not primary or list(primary) != list(secondary)
            or type(primary_names) is not list or not primary_names or len(set(primary_names)) != len(primary_names)
            or not set(primary_names) <= set(primary) or type(project_conflicts) is not bool
            or type(secondary_weight) not in [int, float] or not math.isfinite(secondary_weight) or secondary_weight <= 0):
        raise ValueError('Ordered complete task gradients, explicit primary parameters and a finite positive secondary weight are required')
    tensors = []
    for name in primary:
        if type(name) is not str or not name: raise ValueError('Every actual parameter must retain its complete name')
        p, s = primary[name], secondary[name]
        if p is not None and name not in primary_names: raise ValueError('A primary task cannot acquire undeclared modality gradients')
        for value in [p, s]:
            if value is not None:
                if (type(value) is not torch.Tensor or value.dtype not in [torch.float32, torch.float64]
                        or value.requires_grad or not bool(torch.isfinite(value).all())):
                    raise ValueError('Only complete finite detached FP32/FP64 task gradients are eligible')
                tensors.append(value)
        if p is not None and s is not None and (p.shape != s.shape or p.dtype != s.dtype or p.device != s.device):
            raise ValueError('Every shared parameter must retain its exact gradient geometry')
    if not tensors or len({(v.dtype, v.device) for v in tensors}) != 1:
        raise ValueError('At least one actual complete task gradient on one exact device/dtype is required')
    before = math.fsum(dot(primary[n], secondary[n]) for n in primary_names)
    norm = math.fsum(dot(primary[n], primary[n]) for n in primary_names)
    secondary_norm = math.fsum(dot(secondary[n], secondary[n]) for n in primary_names)
    cosine = before / (math.sqrt(norm) * math.sqrt(secondary_norm)) if norm > 0 and secondary_norm > 0 else None
    coefficient = before / norm if project_conflicts and before < 0 and norm > 0 else 0.
    if not math.isfinite(coefficient): raise ValueError('The complete task projection coefficient must remain finite')
    result, after_terms, blend_terms = {}, [], []
    for name in primary:
        p, s = primary[name], secondary[name]
        projected = s
        if coefficient and p is not None:
            projected = -coefficient * p if s is None else s - coefficient * p
        if p is None and projected is None: result[name] = None
        elif p is None: result[name] = projected * secondary_weight
        elif projected is None: result[name] = p.clone()
        else: result[name] = p + secondary_weight * projected
        if result[name] is not None and not bool(torch.isfinite(result[name]).all()):
            raise ValueError('The complete mixed gradient must remain finite before any optimizer mutation')
        if name in primary_names:
            after_terms.append(dot(p, projected)); blend_terms.append(dot(p, result[name]))
    return result, dict(format=FORMAT, project_conflicts=project_conflicts, actual_projection_applied=coefficient != 0.,
        original_primary_secondary_dot=before, original_primary_squared_norm=norm,
        original_secondary_primary_subspace_squared_norm=secondary_norm, original_shared_parameter_gradient_cosine=cosine,
        actual_projection_coefficient=coefficient, actual_primary_projected_secondary_dot=math.fsum(after_terms),
        actual_primary_blended_gradient_dot=math.fsum(blend_terms), secondary_weight=secondary_weight,
        actual_primary_gradient_tensors=sum(v is not None for v in primary.values()),
        actual_secondary_gradient_tensors=sum(v is not None for v in secondary.values()),
        actual_blended_gradient_tensors=sum(v is not None for v in result.values()),
        all_original_sparse_None_dispositions_retained_unless_projection_requires_a_component=True,
        source_task_tensors_not_mutated=True, upstream_global_task_reduction_required=True,
        AdamW_or_finite_step_loss_improvement_not_guaranteed=True, new_model_quality_gain=False)
