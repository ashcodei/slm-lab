"""Bounded Gloo SUM of locally global-normalized joint task gradients.

Broadcast root's reduced bytes before task projection or Adam so every actual
replica sees one canonical result, including signed zero. Caller retains all
original global component denominators; no second world average is applied. This
is a correctness primitive, not measured full-size GPU throughput evidence.
"""
import torch
from torch import distributed as dist

from .parallel_function_repair_stage_v1 import value_digest
from .shared_joint_checkpoint_v1 import agree, control, sha
from .whole_human_sft_evidence_v1 import tensor_digest


def reduce(gradients, *, named_parameters, task_sha256, global_denominators, bucket_values, control_group=None):
    _, world = control(control_group); error = specification = None
    try:
        if (type(gradients) is not dict or type(named_parameters) is not list or not named_parameters
                or list(gradients) != [n for n, _ in named_parameters]
                or len({n for n, _ in named_parameters}) != len(named_parameters)
                or len({id(p) for _, p in named_parameters}) != len(named_parameters)
                or not sha(task_sha256) or type(global_denominators) is not dict or not global_denominators
                or any(type(k) is not str or not k or type(v) is not int or v < 1 for k, v in global_denominators.items())
                or type(bucket_values) is not int or bucket_values < 1 or not 1 <= world <= 4):
            raise ValueError('Complete ordered joint gradients, an original task identity/denominator and bounded transport are required')
        devices = {p.device for _, p in named_parameters}
        if len(devices) != 1 or any(p.dtype != torch.float32 for _, p in named_parameters):
            raise ValueError('One actual FP32 joint parameter device is required')
        for name, parameter in named_parameters:
            value = gradients[name]
            if (type(name) is not str or not name or value is not None and
                    (not isinstance(value, torch.Tensor) or value.requires_grad or value.dtype != parameter.dtype
                    or value.device != parameter.device or value.shape != parameter.shape or not bool(torch.isfinite(value).all()))):
                raise ValueError('Each local task gradient must be finite, detached and match its original parameter')
        specification = value_digest(dict(task_sha256=task_sha256, global_denominators=global_denominators,
            bucket_values=bucket_values, world=world,
            layout=[(n, list(p.shape), str(p.dtype), p.device.type) for n, p in named_parameters]))
    except Exception as exc: error = type(exc).__name__ + ': ' + str(exc)
    agree(error, specification, control_group, world)
    presence = torch.tensor([int(gradients[n] is not None) for n, _ in named_parameters], dtype=torch.int64)
    if world > 1: dist.all_reduce(presence, op=dist.ReduceOp.MAX, group=control_group)
    result = {n: torch.empty_like(p, memory_format=torch.contiguous_format) if presence[i] else None
        for i, (n, p) in enumerate(named_parameters)}
    segments = []; pieces = []; used = 0; buckets = 0; largest = 0
    def flush():
        nonlocal segments, pieces, used, buckets, largest
        if not used: return
        flat = torch.cat(pieces)
        if world > 1:
            dist.all_reduce(flat, op=dist.ReduceOp.SUM, group=control_group)
            root = 0 if control_group is None else dist.get_global_rank(control_group, 0)
            dist.broadcast(flat, src=root, group=control_group)
        offset = 0
        for name, start, amount in segments:
            result[name].reshape(-1)[start:start + amount].copy_(flat[offset:offset + amount])
            offset += amount
        largest = max(largest, used); buckets += 1; segments = []; pieces = []; used = 0
    for index, (name, parameter) in enumerate(named_parameters):
        if not presence[index]: continue
        local = gradients[name]
        # Small source views are transferred independently. No whole-model
        # gradient concatenation or whole-leaf host copy is required.
        for start in range(0, parameter.numel(), bucket_values):
            amount = min(bucket_values, parameter.numel() - start)
            if used + amount > bucket_values: flush()
            value = torch.zeros(amount, dtype=torch.float32) if local is None else local.reshape(-1)[start:start + amount].to(device='cpu', copy=True)
            segments.append((name, start, amount)); pieces.append(value); used += amount
    flush()
    if any(v is not None and not bool(torch.isfinite(v).all()) for v in result.values()):
        error = 'The complete globally reduced task gradient must remain finite'
    else: error = None
    identity = tensor_digest((n, v) for n, v in result.items() if v is not None)
    agree(error, identity, control_group, world)
    return result, dict(task_sha256=task_sha256, original_global_denominators=dict(global_denominators),
        inputs_already_divided_by_original_global_component_denominators=True, actual_world_size=world,
        gradient_transport='Gloo_host_SUM_then_root_byte_broadcast' if world > 1 else 'serial_complete_copy',
        second_world_average=False, actual_active_parameter_tensors=int(presence.sum()),
        complete_output_gradient_sha256=identity, actual_buckets=buckets, largest_host_bucket_values=largest,
        declared_host_bucket_values=bucket_values, all_actual_reduced_gradient_bytes_agree=True,
        source_gradients_not_mutated=True, production_source_admission_or_GPU_speed_not_proved=True)
