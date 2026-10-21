"""Evaluation-only specification intervention on fixed correct code targets.

These protected local references must never become optimization examples.
Conflicting specifications retain the original name and argument signature;
their different lengths and retained name hints limit causal interpretation.
"""
import torch

from .code_tasks import TASKS
from .inference_rotary_cache_v1 import inference_cache
from .native_code_action_readback_v1 import prompt_for


def comparison(task, language, frame):
    if task not in TASKS or language not in {'Python', 'JavaScript'}:
        raise ValueError('An unchanged complete protected local task is required')
    peers = [other for other in TASKS if other[1] == task[1]]
    other = peers[(peers.index(task) + 1) % len(peers)]
    if other == task or other[2] == task[2]:
        raise ValueError('A distinct full specification with the same arguments is required')
    case = dict(task=task[0], specification=task[2], language=language)
    original = prompt_for(case, frame)
    if original.count(task[2]) != 1:
        raise ValueError('Exactly one complete original specification is required')
    conflicting = original.replace(task[2], other[2])
    reference = task[4 if language == 'Python' else 5]
    if frame == 'instruction':
        response = reference
    elif language == 'Python':
        header, response = reference.split('\n', 1)
        if header.replace(' ', '') != f'def{task[0]}({task[1]}):'.replace(' ', ''):
            raise ValueError('The complete original Python header must reconstruct')
    else:
        header, response = reference.split('{', 1)
        if header.replace(' ', '') != f'function{task[0]}({task[1]})'.replace(' ', ''):
            raise ValueError('The complete original JavaScript header must reconstruct')
    return dict(task=task[0], language=language, frame=frame,
                original_prompt=original, conflicting_prompt=conflicting,
                conflicting_specification_task=other[0], response=response,
                original_reference=reference,
                intervention_changes_only_one_full_specification=True,
                name_hint_and_argument_signature_retained=True,
                evaluation_only_optimization_forbidden=True)


@torch.no_grad()
def score(model, prompt_ids, response_ids):
    """FP32 CPU log probabilities after the complete prompt and gold prefix.

    Every response token, including its actual supplied EOS, is scored once.
    Prompt tokens are context only. No new free-running output is generated.
    """
    if (model.training or any(p.device.type != 'cpu' or p.dtype != torch.float32
                              for p in model.parameters())
            or not prompt_ids or prompt_ids[0] != 1 or 2 in prompt_ids
            or not response_ids or response_ids[-1] != 2 or 2 in response_ids[:-1]
            or any(type(i) is not int or not 0 <= i < model.config.vocab_size
                   for i in prompt_ids + response_ids)
            or len(prompt_ids) + len(response_ids) > model.config.max_text_length):
        raise ValueError('An eval FP32 CPU model and complete native prompt/response/EOS are required')
    ids = torch.tensor([prompt_ids + response_ids], dtype=torch.long)
    with inference_cache():
        full = model(ids[:, :-1])['logits']
    if full.shape != (1, ids.shape[1] - 1, model.config.vocab_size):
        raise ValueError('Every original causal text position must be retained')
    logits = full[0, len(prompt_ids)-1:].float()
    target = torch.tensor(response_ids, dtype=torch.long)
    selected = logits.gather(1, target[:, None]).squeeze(1)
    nll = torch.logsumexp(logits, -1) - selected
    if not bool(torch.isfinite(nll).all()):
        raise ValueError('Every complete response probability must be finite')
    top1 = logits.argmax(-1)
    # Strictly greater scores define the best tied rank; top1 uses torch's
    # explicit lowest-index tie rule and is reported independently.
    ranks = (logits > selected[:, None]).sum(-1) + 1
    ties = (logits == selected[:, None]).sum(-1)
    wrong = (top1 != target).nonzero().flatten()
    return dict(prompt_tokens=len(prompt_ids), response_EOS_targets=len(response_ids),
                token_NLL=nll.tolist(), target_best_tied_rank=ranks.tolist(),
                target_tied_values=ties.tolist(), teacher_prefix_top1_ids=top1.tolist(),
                mean_token_NLL=float(nll.mean()), summed_token_NLL=float(nll.double().sum()),
                gold_prefix_top1_correct=int((top1 == target).sum()),
                first_gold_prefix_top1_divergence=None if not len(wrong) else int(wrong[0]),
                EOS_NLL=float(nll[-1]), inference_device='cpu', inference_precision='fp32',
                no_training_or_free_generation=True,
                exact_CUDA_BF16_probability_parity_not_established=True)
