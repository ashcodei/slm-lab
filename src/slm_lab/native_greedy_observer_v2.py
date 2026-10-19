"""Capture unchanged native actions, including positional image inputs.

The first observer accepted text-only calls. Native image evaluators also
pass pixels positionally. Preserve that public generator calling convention
and the original code-action journal schema, and record image input bytes.
"""
from contextlib import contextmanager
import hashlib

import torch

from .evaluation_provenance import pixels_sha256
from .native_greedy_observer_v1 import signature


@contextmanager
def observe(model, callback):
    if (model.training or not callable(callback)
            or getattr(model, '_native_greedy_observer_active_v1', False)
            or getattr(model, '_native_greedy_observer_active_v2', False)):
        raise ValueError('One eval-mode native generator and explicit observation callback required')
    original = model.generate
    before = signature(model)
    calls = []

    def generate(input_ids, *args, **kwargs):
        if (input_ids.ndim != 2 or input_ids.shape[0] != 1 or len(args) > 1
                or args and 'pixel_values' in kwargs
                or kwargs.get('temperature', 0) != 0 or kwargs.get('return_rollout', False)
                or kwargs.get('use_cache', True) is not True
                or type(kwargs.get('max_new_tokens')) is not int or kwargs['max_new_tokens'] < 1):
            raise ValueError('Explicit single-example cached greedy native generation required')
        pixels = args[0] if args else kwargs.get('pixel_values')
        if pixels is not None and (not isinstance(pixels, torch.Tensor)
                or pixels.ndim != 4 or pixels.shape[0] != 1 or pixels.device != input_ids.device):
            raise ValueError('Complete single-example image tensor on the native input device required')
        prefix = input_ids.detach().cpu().tolist()[0]
        image_hash = None if pixels is None else pixels_sha256(pixels)
        output = original(input_ids, *args, **kwargs)
        if (not isinstance(output, torch.Tensor) or output.ndim != 2 or output.shape[0] != 1
                or output.device != input_ids.device
                or not torch.equal(output[:, :len(prefix)], input_ids)
                or not 1 <= output.shape[1] - len(prefix) <= kwargs['max_new_tokens']
                or signature(model) != before
                or pixels is not None and pixels_sha256(pixels) != image_hash):
            raise ValueError('Unchanged prefix, pixels, bounded actions and unmodified parameters required')
        actions = output[0, len(prefix):].detach().cpu().tolist()
        eos = kwargs.get('eos_token_id')
        event = dict(case_position=len(calls), original_input_token_ids=prefix,
            actual_generated_token_ids=actions,
            complete_input_ids_sha256=hashlib.sha256(
                input_ids.detach().cpu().numpy().astype('<u2').tobytes()).hexdigest(),
            input_pixels_sha256=image_hash,
            requested_native_action_budget=kwargs['max_new_tokens'], actual_native_action_count=len(actions),
            actual_stopped_on_EOS=bool(eos is not None and actions[-1] == eos),
            greedy_native_output_tensor_returned_unchanged=True,
            selected_action_log_probabilities_not_retained=True)
        calls.append(event)
        callback(event)
        return output

    model._native_greedy_observer_active_v2 = True
    model.generate = generate
    try:
        yield calls
    finally:
        model.generate = original
        model._native_greedy_observer_active_v2 = False
