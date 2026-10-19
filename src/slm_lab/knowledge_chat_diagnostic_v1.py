"""Matched plain/chat framing diagnostic with complete raw token evidence.

The chat form uses the source curriculum's User/Assistant frame. This is a
post-hoc diagnostic prompted by a native knowledge regression, not a replacement
benchmark or a format chosen to promote a policy on held scores.
"""
from collections import defaultdict
import math

import torch

from .knowledge_evaluation import letter, interval


def question_prompt(row, mode):
    options = '\n'.join(chr(65+i)+'. '+text for i,text in enumerate(row['options']))
    plain = f'The following is a multiple-choice question about {row["domain"]}.\n\n{row["question"]}\n\n{options}\n\nAnswer with one letter only.\nAnswer:'
    if mode == 'plain': return plain
    if mode == 'chat': return 'User: '+plain+'\nAssistant:'
    raise ValueError('Explicit original plain or source-curriculum chat frame required')


def summarize(records):
    count = len(records)
    if not count: raise ValueError('Complete nonempty knowledge predictions required')
    generated = sum(r['generated_correct'] for r in records); likelihood = sum(r['likelihood_correct'] for r in records)
    compliant = sum(r['generated_index'] is not None for r in records)
    return dict(questions=count, generated_correct=generated, likelihood_correct=likelihood,
        format_compliant=compliant, generated_accuracy=generated/count, likelihood_accuracy=likelihood/count,
        format_compliance=compliant/count, chance_accuracy=math.fsum(r['chance_accuracy'] for r in records)/count,
        generated_accuracy_95ci=interval(generated,count), likelihood_accuracy_95ci=interval(likelihood,count))


@torch.no_grad()
def evaluate_rows(model, tokenizer, rows, mode, *, max_new_tokens=32, checked=lambda:None, record=lambda row:None):
    if (not rows or tokenizer.get_vocab_size() != model.config.vocab_size
            or [tokenizer.token_to_id(t) for t in ('<pad>', '<bos>', '<eos>')] != [0, 1, 2]):
        raise ValueError('Complete native questions and original own tokenizer required')
    model.eval(); device = next(model.parameters()).device
    if device.type != 'cpu': raise ValueError('This separate matched diagnostic uses CPU/FP32 only')
    results = []; groups = defaultdict(list)
    for row in rows:
        checked(); prompt = question_prompt(row, mode); tokens = [1, *tokenizer.encode(prompt).ids]
        if len(tokens)+max_new_tokens > model.config.max_text_length:
            raise ValueError('Every original complete question must fit its unchanged generation budget')
        ids = torch.tensor([tokens], device=device); initial = model.prefill(ids); candidates = []; likelihoods = []
        for i in range(len(row['options'])):
            candidate = tokenizer.encode(' '+chr(65+i)).ids
            if not candidate: raise ValueError('Every answer letter needs its entire native token sequence')
            state = initial; values = []
            for position, token in enumerate(candidate):
                values.append(float(state['logits'][0, -1].float().log_softmax(-1)[token]))
                if position+1 < len(candidate): state = model.decode(torch.tensor([[token]], device=device), state['cache'])
            likelihoods.append(sum(values))
            candidates.append(dict(token_ids=candidate, per_token_logprobs=values))
        generated = model.generate(ids, max_new_tokens=max_new_tokens, eos_token_id=2, use_cache=True)
        emitted = generated[0, len(tokens):].tolist(); answer = tokenizer.decode(emitted)
        prediction = letter(answer, len(row['options'])); likelihood = max(range(len(likelihoods)), key=lambda i:likelihoods[i])
        gold = int(row['answer_index'])
        result = dict(id=row['id'], domain=row['domain'], answer_index=gold, prompt=prompt, prompt_ids=tokens,
            generated_ids=emitted, generated_response=answer, generated_index=prediction,
            generated_correct=prediction == gold, likelihood_index=likelihood, likelihood_correct=likelihood == gold,
            candidate_scores=likelihoods, candidate_token_evidence=candidates, chance_accuracy=1/len(likelihoods), context_tokens=len(tokens))
        record(result); results.append(result); groups[row['domain']].append(result)
    return dict(overall=summarize(results), by_domain={name:summarize(items) for name,items in groups.items()}, predictions=results,
        frame=mode, inference_device_type='cpu', inference_precision='FP32', generation_budget=max_new_tokens,
        post_hoc_format_diagnostic=True, native_benchmark_results_not_replaced=True,
        training_or_selection_uses_of_these_held_rows=False, candidate_promoted=False, goal_achieved=False)
