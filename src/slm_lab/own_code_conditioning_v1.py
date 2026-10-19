"""Train-only whole-response likelihood controls; no gradients or selection.

Other complete questions may differ in length/position and can conceivably
admit related programs. These are mispaired-context controls, not verified
incorrect answers or human preferences. Teacher forcing is not code execution.
"""
from collections import Counter
import hashlib
import json
import math

import torch
from torch.nn import functional as F


def sequences(rows, tokenizer, *, max_text_length=4096):
    if not rows or any(r.get('split') != 'train' or r.get('publisher_split') != 'train' for r in rows):
        raise ValueError('Only original publisher-train questions/targets may enter this diagnostic')
    if len({r['id'] for r in rows}) != len(rows) or any(r['mode'] not in ['stdin', 'function'] for r in rows):
        raise ValueError('Unique original source IDs and explicit original interfaces required')
    generic = [1, *tokenizer.encode('User: Write a Python solution. Return only the complete code.\nAssistant:').ids]
    output = []
    for target in rows:
        body = [*tokenizer.encode(target['original_human_program']).ids, 2]
        if len(body) != target['target_tokens']:
            raise ValueError('Every original whole human response and EOS target must reconstruct')
        questions = [r for r in rows if r['mode'] == target['mode']]
        if len(questions) < 2:
            raise ValueError('At least two original complete questions in every interface group required')
        for question in [*questions, None]:
            prefix = question['prefix_ids'] if question is not None else generic
            ids = [*prefix, *body]; labels = [-100] * len(prefix) + body
            if len(ids) > max_text_length or ids[0] != 1 or ids[-1] != 2:
                raise ValueError('Every complete original question and target must fit without clipping')
            output.append(dict(target_id=target['id'], target_mode=target['mode'],
                question_id=question['id'] if question is not None else None,
                condition=('proper' if question and question['id'] == target['id'] else 'other_question' if question else 'no_problem'),
                input_ids=ids, labels=labels, prefix_tokens=len(prefix), target_tokens=len(body),
                sequence_sha256=hashlib.sha256(json.dumps([ids, labels], separators=(',', ':')).encode()).hexdigest()))
    return output


@torch.no_grad()
def score(model, sequence, *, device, checked=lambda: None):
    if model.training:
        raise ValueError('Likelihood controls require evaluation mode without any updates')
    ids, labels = sequence['input_ids'], sequence['labels']; start = sequence['prefix_tokens']
    if (type(start) is not int or not 1 <= start < len(ids) or len(ids) > model.config.max_text_length
            or labels != [-100] * start + ids[start:] or ids[0] != 1 or ids[-1] != 2
            or sequence['target_tokens'] != len(ids) - start
            or hashlib.sha256(json.dumps([ids, labels], separators=(',', ':')).encode()).hexdigest() != sequence['sequence_sha256']):
        raise ValueError('Whole source-bound input, response/EOS and contiguous original mask required')
    checked()
    inputs = torch.tensor([ids], device=device); expected = torch.tensor(ids[start:], device=device)
    out = model(input_ids=inputs, labels=torch.tensor([labels], device=device))
    logits = out['logits'][0, start - 1:-1].float()
    values = F.cross_entropy(logits, expected, reduction='none')
    if (not bool(torch.isfinite(values).all()) or not torch.isclose(out['loss'].float(), values.mean(), rtol=1e-6, atol=1e-6)):
        raise ValueError('Every actual response/EOS position and native shifted mean must agree')
    top = logits.argmax(-1); correct = (top == expected)
    checked()
    return dict(target_id=sequence['target_id'], question_id=sequence['question_id'],
        target_mode=sequence['target_mode'], condition=sequence['condition'],
        sequence_sha256=sequence['sequence_sha256'], prefix_tokens=start, target_tokens=len(expected),
        per_target_NLL=values.cpu().tolist(), teacher_argmax_ids=top.cpu().tolist(), target_ids=expected.cpu().tolist(),
        whole_response_EOS_mean_NLL=float(values.mean()), whole_response_EOS_sum_NLL=float(values.sum()),
        teacher_argmax_correct=int(correct.sum()), first_target_correct=bool(correct[0]),
        first_32_target_mean_NLL=float(values[:32].mean()), first_target_NLL=float(values[0]),
        all_target_log_probabilities_not_free_generation=True)


def summarize(records, rows, *, routes=('general', 'code')):
    source = {r['id']: r for r in rows}
    if len(source) != len(rows) or any(r['split'] != 'train' or r['publisher_split'] != 'train' for r in rows):
        raise ValueError('Every actual unique original train question/target required')
    result = {}
    expected_pairs = {(r['id'], q['id']) for r in rows for q in rows if r['mode'] == q['mode']} | {(r['id'], None) for r in rows}
    for route in routes:
        part = [r for r in records if r['route'] == route]
        if len(part) != len(expected_pairs) or {(r['target_id'], r['question_id']) for r in part} != expected_pairs:
            raise ValueError('Every original same-interface question/target matrix cell and no-problem control exactly once required')
        by_target = {}; proper = []; blank = []; wrong = []; first = []; ranks = []; unique_best = 0; tied_best = 0; total_tokens = 0; total_NLL = 0.; total_correct = 0
        for row in part:
            n = row['target_tokens']; values = row['per_target_NLL']; target = row['target_ids']; top = row['teacher_argmax_ids']
            if (n != source[row['target_id']]['target_tokens'] or len(values) != n or len(top) != n or len(target) != n
                    or any(not math.isfinite(v) for v in values) or abs(sum(values) - row['whole_response_EOS_sum_NLL']) > 1e-3
                    or abs(sum(values) / n - row['whole_response_EOS_mean_NLL']) > 1e-5
                    or row['teacher_argmax_correct'] != sum(a == b for a, b in zip(top, target))):
                raise ValueError('Every actual target, argmax and mean/sum score must reconstruct')
            by_target.setdefault(row['target_id'], []).append(row)
        for identity, cells in by_target.items():
            right = next(r for r in cells if r['question_id'] == identity)
            empty = next(r for r in cells if r['question_id'] is None)
            others = [r for r in cells if r['question_id'] not in {identity, None}]
            wrong_mean = sum(r['whole_response_EOS_mean_NLL'] for r in others) / len(others)
            proper.append(right['whole_response_EOS_mean_NLL']); blank.append(empty['whole_response_EOS_mean_NLL']); wrong.append(wrong_mean)
            first.append(right['first_target_NLL'])
            ranks.append(1 + sum(r['whole_response_EOS_mean_NLL'] < right['whole_response_EOS_mean_NLL'] for r in others))
            equal = sum(r['whole_response_EOS_mean_NLL'] == right['whole_response_EOS_mean_NLL'] for r in others)
            unique_best += ranks[-1] == 1 and equal == 0
            tied_best += ranks[-1] == 1 and equal > 0
            total_tokens += right['target_tokens']; total_NLL += right['whole_response_EOS_sum_NLL']; total_correct += right['teacher_argmax_correct']
        result[route] = dict(original_train_targets=len(rows), complete_matrix_and_control_scores=len(part),
            proper_equal_program_mean_NLL=sum(proper) / len(proper), no_problem_equal_program_mean_NLL=sum(blank) / len(blank),
            other_question_equal_program_mean_NLL=sum(wrong) / len(wrong),
            mean_no_problem_minus_proper_NLL=sum(b - p for b, p in zip(blank, proper)) / len(proper),
            mean_other_question_minus_proper_NLL=sum(w - p for w, p in zip(wrong, proper)) / len(proper),
            proper_lower_than_mean_other_question=sum(p < w for p, w in zip(proper, wrong)),
            proper_lower_than_no_problem=sum(p < b for p, b in zip(proper, blank)),
            matching_question_best_likelihood_in_same_interface=sum(r == 1 for r in ranks),
            matching_question_unique_best_likelihood=unique_best, matching_question_tied_best_likelihood=tied_best,
            matching_question_rank_counts=dict(Counter(ranks)), proper_token_mean_NLL=total_NLL / total_tokens,
            proper_teacher_argmax_accuracy=total_correct / total_tokens, complete_original_response_EOS_targets=total_tokens,
            not_greedy_execution_or_held_accuracy=True)
    if any(r['route'] not in routes for r in records):
        raise ValueError('No unaccounted route scores allowed')
    return result
