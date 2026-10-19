"""Complete generic-image CE in bounded microbatches, normalized globally.

Only existing vision leaves receive gradients. All original response/EOS
targets survive; padding alone is removed between microbatches. The caller
must SUM parameter gradients once, clip, and commit its optimizer. This is
not full-size CUDA capacity, recovery, source admission or quality evidence.
"""
from contextlib import nullcontext
import hashlib
import json
import math

import torch
from torch import distributed as dist

from .deterministic_positions_v1 import vision_forward
from .own_actor_vision_prefix_split_backward_v1 import OwnActorVisionPrefixSplitBackward


def native_microbatches(lengths, *, prefix_tokens, maximum_examples=4, maximum_padded_tokens=4096):
    if (not isinstance(lengths, list) or len(lengths) > 64
            or any(type(n) is not int or not 2 <= n <= 4096 for n in lengths)
            or type(prefix_tokens) is not int or prefix_tokens < 1
            or type(maximum_examples) is not int or not 1 <= maximum_examples <= 64
            or type(maximum_padded_tokens) is not int or maximum_padded_tokens < 1):
        raise ValueError('Complete bounded native lengths and explicit positive microbatch limits required')
    if any(n + prefix_tokens > maximum_padded_tokens for n in lengths):
        raise ValueError('A complete original source exceeds the token limit; clipping or omission is forbidden')
    chunks = []
    begin = 0
    while begin < len(lengths):
        end = begin + 1
        while (end < len(lengths) and end - begin < maximum_examples
                and (max(lengths[begin:end + 1]) + prefix_tokens) * (end + 1 - begin) <= maximum_padded_tokens):
            end += 1
        chunks.append(list(range(begin, end)))
        begin = end
    return chunks


def named_native_visual_selected(view):
    return [('vision.' + name, parameter) for name, parameter in view.actor.base.network.vision.named_parameters()]


def accumulate_global_native_visual(view, batch, *, global_positions, local_positions,
        global_context_lengths, global_response_EOS_targets, maximum_examples=4,
        maximum_padded_tokens=4096, native_CE_weight=1., native_logits_dtype=torch.float32,
        amp=None, group=None):
    initialized = dist.is_initialized()
    if initialized and dist.get_backend(group) != 'gloo':
        raise ValueError('Actual explicit Gloo control group required')
    if not initialized and group is not None:
        raise ValueError('An uninitialized control group is not a serial call')
    world = dist.get_world_size(group) if initialized else 1
    rank = dist.get_rank(group) if initialized else 0

    def gather(value):
        if world == 1:
            return [value]
        result = [None] * world
        dist.all_gather_object(result, value, group=group)
        return result

    def coordinated(error, phase):
        errors = gather(error)
        if any(value is not None for value in errors):
            raise ValueError('Coordinated native visual ' + phase + ' rejection: ' + repr(errors))

    error = specification = None
    try:
        if (type(view) is not OwnActorVisionPrefixSplitBackward or not view.training or not view.native.training
                or not 1 <= world <= 4 or not isinstance(global_positions, list) or not 1 <= len(global_positions) <= 64
                or len(set(global_positions)) != len(global_positions)
                or any(type(p) is not int or p < 0 for p in global_positions)
                or local_positions != global_positions[rank::world]
                or len(global_context_lengths) != len(global_positions)
                or len(global_response_EOS_targets) != len(global_positions)
                or any(type(v) is not int or v < 1 for v in global_response_EOS_targets)
                or type(native_CE_weight) not in [int, float] or not math.isfinite(native_CE_weight) or native_CE_weight <= 0
                or native_logits_dtype not in [torch.float32, torch.bfloat16] or amp is not None and not callable(amp)):
            raise ValueError('Exact own view and complete global source/context/target/precision settings required')
        view.assert_shared()
        prefix = view.actor.config.image_token_count(448)
        # Validate every global source against the limits on every rank,
        # including sources assigned to a currently empty tail rank.
        native_microbatches(global_context_lengths, prefix_tokens=prefix,
            maximum_examples=maximum_examples, maximum_padded_tokens=maximum_padded_tokens)
        if max(global_context_lengths) > view.actor.config.max_text_length:
            raise ValueError('The actual native decoder must fit every complete original context')
        selected = [p for _, p in named_native_visual_selected(view)]
        chosen = {id(p) for p in selected}
        if not selected or any(p.grad is not None for p in view.parameters()):
            raise ValueError('Every gradient must start clear before the explicit vision-only backward')
        device = next(view.parameters()).device
        if any(p.device != device or p.dtype != torch.float32 for p in view.parameters()):
            raise ValueError('One consistent own FP32-master replica with optional explicit AMP required')
        if device.type == 'cuda' and (getattr(view.native.vision.forward, '__func__', None) is not vision_forward
                or not torch.are_deterministic_algorithms_enabled()
                or torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32):
            raise ValueError('CUDA requires the existing deterministic position operator and original precision controls')
        fixed = [p for p in view.parameters() if id(p) not in chosen]
        fixed_state = [(id(p), p._version, p.data_ptr(), p.requires_grad) for p in fixed]
        selected_state = [(id(p), p._version, p.data_ptr(), p.requires_grad) for p in selected]
        local_lengths = global_context_lengths[rank::world]
        local_targets = global_response_EOS_targets[rank::world]
        chunks = native_microbatches(local_lengths, prefix_tokens=prefix,
            maximum_examples=maximum_examples, maximum_padded_tokens=maximum_padded_tokens)
        if not local_positions:
            if batch is not None:
                raise ValueError('Empty ranks must have no fabricated images or native targets')
        else:
            if (not isinstance(batch, dict) or set(batch) != {'input_ids', 'labels', 'attention_mask', 'pixel_values'}
                    or batch['input_ids'].ndim != 2 or batch['input_ids'].dtype != torch.long
                    or batch['labels'].shape != batch['input_ids'].shape or batch['labels'].dtype != torch.long
                    or batch['attention_mask'].shape != batch['input_ids'].shape or batch['attention_mask'].dtype != torch.bool
                    or len(batch['input_ids']) != len(local_positions)
                    or tuple(batch['pixel_values'].shape) != (len(local_positions), 3, 448, 448)
                    or batch['pixel_values'].dtype != torch.float32
                    or any(value.device != device for value in batch.values())
                    or not bool(torch.isfinite(batch['pixel_values']).all())
                    or bool((batch['pixel_values'].abs() > 1).any())):
                raise ValueError('All complete actual native inputs and canonical normalized448px RGB images required')
            for ids, labels, mask, length, wanted in zip(batch['input_ids'], batch['labels'], batch['attention_mask'], local_lengths, local_targets):
                targets = (labels != -100).nonzero().flatten()
                if (int(mask.sum()) != length or not bool(mask[:length].all()) or bool(mask[length:].any())
                        or int(ids[0]) != 1 or int(ids[length-1]) != 2
                        or bool(((ids[1:length-1] < 4) | (ids[1:length-1] >= view.actor.config.vocab_size)).any())
                        or not len(targets) or int(targets[0]) < 1 or len(targets) != wanted
                        or targets.tolist() != list(range(int(targets[0]), length))
                        or not torch.equal(ids[targets], labels[targets])
                        or bool((ids[length:] != 0).any())):
                    raise ValueError('Every complete original context, contiguous response and genuine EOS target must match')
        specification = hashlib.sha256(json.dumps(dict(global_positions=global_positions,
            global_context_lengths=global_context_lengths, global_response_EOS_targets=global_response_EOS_targets,
            maximum_examples=maximum_examples, maximum_padded_tokens=maximum_padded_tokens, visual_prefix_tokens=prefix,
            native_CE_weight=native_CE_weight, native_logits_dtype=str(native_logits_dtype)), sort_keys=True).encode()).hexdigest()
    except Exception as exc:
        error = type(exc).__name__ + ': ' + str(exc)
    coordinated(error, 'source/native preflight')
    if any(value != specification for value in gather(specification)):
        raise ValueError('Coordinated native visual global source/normalization/settings mismatch')
    amp = amp or nullcontext
    ce_total = 0.
    observed_targets = observed_contexts = calls = 0
    micro_records = []
    error = None
    try:
        for slots in chunks:
            width = max(local_lengths[i] for i in slots)
            actual = {name:value[slots, :width] if name != 'pixel_values' else value[slots]
                for name, value in batch.items()}
            with amp():
                logits = view.native(**{k:actual[k] for k in ['input_ids', 'attention_mask', 'pixel_values']})['logits']
            if logits.dtype != native_logits_dtype:
                raise ValueError('Actual native precision differs from the declared backward graph')
            labels = actual['labels'][:, 1:]
            mask = labels != -100
            ce = torch.nn.functional.cross_entropy(logits[:, :-1].float()[mask], labels[mask], reduction='sum')
            if not bool(torch.isfinite(ce)):
                raise ValueError('Every complete actual microbatch CE must be finite')
            (ce * native_CE_weight / sum(global_response_EOS_targets)).backward(inputs=selected)
            ce_total += float(ce.detach())
            observed_targets += int(mask.sum())
            observed_contexts += int(actual['attention_mask'].sum())
            calls += 1
            micro_records.append(dict(local_slots=slots, original_positions=[local_positions[i] for i in slots],
                complete_context_lengths=[local_lengths[i] for i in slots], response_EOS_targets=int(mask.sum()),
                padded_decoder_tokens=(width + prefix) * len(slots)))
            del ce, logits, actual
        if (observed_targets != sum(local_targets) or observed_contexts != sum(local_lengths)
                or bool(local_positions) and any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in selected)
                or not local_positions and any(p.grad is not None for p in selected)):
            raise ValueError('Every original local context/target and finite vision gradient must reach backward')
        if (fixed_state != [(id(p), p._version, p.data_ptr(), p.requires_grad) for p in fixed]
                or selected_state != [(id(p), p._version, p.data_ptr(), p.requires_grad) for p in selected]
                or any(p.grad is not None for p in fixed)):
            raise ValueError('Weights, trainability and all nonvision gradients must remain untouched before the caller commits')
    except Exception as exc:
        error = type(exc).__name__ + ': ' + str(exc)
    coordinated(error, 'complete local backward')
    return dict(rank=rank, world=world, global_positions=global_positions, local_positions=local_positions,
        global_original_context_tokens=sum(global_context_lengths), global_response_EOS_targets=sum(global_response_EOS_targets),
        local_original_context_tokens=observed_contexts, local_response_EOS_targets=observed_targets,
        local_native_CE_sum=ce_total, native_CE_weight=native_CE_weight, actual_native_forward_calls=calls,
        actual_microbatches=micro_records, native_logits_dtype=str(native_logits_dtype) if calls else None,
        one_parameter_gradient_SUM_still_required=True, optimizer_committed=False,
        whole_original_sources_and_response_EOS_targets_preserved=True, only_padding_trimmed=True,
        fixed_language_actor_critic_word_pointing_other_heads_untouched=True,
        full_size_CUDA_capacity_recovery_and_quality_pending=True)
