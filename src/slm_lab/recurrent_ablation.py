"""Budget and measurement helpers for the separate own-weight recurrence trial."""
import hashlib
import random
import re

import torch

from .code_quality import execute, efficiency
from .code_tasks import TASKS, request


def budgets(model, recurrent_steps, depth, length):
    if type(recurrent_steps) is not int or recurrent_steps < 1 or length < 2:
        raise ValueError('Positive update and next-token budgets are required')
    one = model.execution_layers(1)
    repeated = model.execution_layers(depth)
    if recurrent_steps * repeated % one:
        raise ValueError('Use an exactly matching integer layer-execution budget')
    control_steps = recurrent_steps * repeated // one
    return {
        'recurrent_steps': recurrent_steps,
        'equal_target_control_steps': recurrent_steps,
        'equal_layer_control_steps': control_steps,
        'targets_per_update': length - 1,
        'recurrent_target_exposures': recurrent_steps * (length - 1),
        'equal_layer_control_target_exposures': control_steps * (length - 1),
        'matched_decoder_layer_token_executions': recurrent_steps * repeated * length,
        'recurrent_decoder_layers': repeated,
        'control_decoder_layers': one,
        'scope': 'Decoder layer/token executions match exactly; optimizer, head, attention and wall time do not.',
    }


def window_order(size, count, seed):
    if not 0 < count <= size:
        raise ValueError('Require enough unique original training windows')
    return random.Random(seed).sample(range(size), count)


def language_parameters(model, train_gates):
    """Train the whole language backbone, retaining original vision and critic."""
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(not name.startswith('vision.') and name != 'value_head.weight'
                                 and (name != 'recurrence_gates' or train_gates))
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def state_digest(state):
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode()); digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def frozen_digest(model):
    return state_digest({name: value for name, value in model.state_dict().items()
                         if name.startswith('vision.') or name == 'value_head.weight'})


@torch.no_grad()
def assess_completion(model, tokenizer, device, max_new_tokens, notify=None):
    """Use the unchanged local worker/cases on all 24 completion tasks.

    This trial explicitly uses a 256-token budget. It is not interchangeable
    with the separate main-model assessment's 512-token budget.
    """
    model.eval(); rows = []
    for language in ('Python', 'JavaScript'):
        for task in TASKS:
            name, args, specification, scaling, py, js = task
            prefix = (f'def {name}({args}):\n    """{specification}"""\n' if language == 'Python'
                      else f'function {name}({args}) {{\n  // {specification}\n')
            ids = torch.tensor([[tokenizer.token_to_id('<bos>')] + tokenizer.encode(prefix).ids], device=device)
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                generated = model.generate(ids, max_new_tokens=max_new_tokens,
                                           eos_token_id=tokenizer.token_to_id('<eos>'), use_cache=True)
            raw = tokenizer.decode(generated[0, ids.shape[1]:].tolist())
            code = prefix + raw.split('```')[0]
            if language == 'Python': code = re.split(r'\n(?=File:)', code)[0]
            candidate = execute(request(task, language, code))
            reference = execute(request(task, language, py if language == 'Python' else js, True))
            if not reference.get('functional_passed') or not reference.get('efficiency_completed'):
                raise RuntimeError('Unchanged original coding reference failed: ' + str(reference))
            passed, measurement = efficiency(candidate, reference, scaling)
            rows.append({'language': language, 'task': name, 'raw_response': raw, 'code': code,
                         'candidate': candidate, 'reference_stress': reference['stress'],
                         'efficiency_stress_passed': passed, 'efficiency_measurement': measurement})
            if notify: notify(language, name, len(rows))
    return {
        'prompt_mode': 'completion', 'generation_budget': max_new_tokens,
        'decoding': 'greedy cached BF16', 'tasks': len(rows),
        'syntax_passes': sum(bool(row['candidate'].get('syntax_valid')) for row in rows),
        'functional_passes': sum(bool(row['candidate'].get('functional_passed')) for row in rows),
        'efficiency_passes': sum(row['efficiency_stress_passed'] for row in rows),
        'results': rows,
        'scope': 'Same 12 local cases per language and finite empirical stress checks; no general/all-code or asymptotic claim.',
    }
