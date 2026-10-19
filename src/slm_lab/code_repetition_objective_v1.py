"""Training-only repetition candidates and stable sequence unlikelihood.

Based on Welleck et al., https://arxiv.org/abs/1908.04319, section 5.2.
The long n-gram threshold and original-code repetition exclusions are local,
unvalidated adaptations. This is neither a correctness oracle nor human feedback.
"""
from collections import Counter
import random

import torch


def select_rollout_rows(selection, *, number=128, budget=128, seed=910901):
    if (set(selection) != {'train','validation'} or type(number) is not int or number<1
            or type(budget) is not int or budget<1 or type(seed) is not int):
        raise ValueError('Explicit original roles and positive rollout budget required')
    train,validation=selection['train'],selection['validation']
    if (not train or any(r['role']!='train' for r in train)
            or any(r['role']!='validation' for r in validation)
            or len({r['id'] for r in train+validation})!=len(train+validation)
            or {r['group_id'] for r in train}&{r['group_id'] for r in validation}):
        raise ValueError('Unique original training rows and disjoint validation groups required')
    if any(not 1<r['target_tokens']<r['context_tokens']<=4096 for r in train):
        raise ValueError('Whole original training programs and EOS required')
    eligible=[r for r in train if r['context_tokens']-r['target_tokens']+budget<=4096]
    languages=sorted({r['language'] for r in train})
    if len(eligible)<number or len(languages)>number or {r['language'] for r in eligible}!=set(languages):
        raise ValueError('Declared budget must cover every original training language without clipping prompts')
    chosen=[min((r for r in eligible if r['language']==lang),key=lambda r:r['id']) for lang in languages]
    ids={r['id'] for r in chosen}
    rng=random.Random(seed);other=eligible.copy();rng.shuffle(other)
    chosen.extend([r for r in other if r['id'] not in ids][:number-len(chosen)])
    rng.shuffle(chosen)
    return chosen


def repetition_negative_positions(generated, reference, *, n=16, eos=2):
    """Later repeated n-gram endings, excluding EOS and reference repetitions.

    Only generated response tokens are scanned. A window occurring at least
    twice in the complete original *training* answer is exempt. This reduces
    false penalties on valid code but cannot recognize every valid alternative.
    """
    if (type(n) is not int or n<2 or type(eos) is not int or not reference
            or reference[-1]!=eos or any(type(t) is not int or t<0 for t in generated+reference)
            or eos in generated[:-1]):
        raise ValueError('Explicit complete original reference/EOS and bounded raw generated tokens required')
    counts=Counter(tuple(reference[i:i+n]) for i in range(len(reference)-n+1))
    seen=set();positions=[]
    for end in range(n-1,len(generated)):
        window=tuple(generated[end-n+1:end+1])
        if generated[end]!=eos and window in seen and counts[window]<2:
            positions.append(end)
        seen.add(window)
    return positions


def sequence_unlikelihood(logits, input_ids, prompt_tokens, negative_positions, *, eos=2):
    """Mean -log(1-p(token)) on selected response positions, with causal shift.

    logsumexp(all)-logsumexp(other tokens) stays finite and differentiable even
    for a negative token with nearly unit probability; no probability clamp.
    """
    if (logits.ndim!=3 or logits.shape[0]!=1 or logits.shape[2]<2
            or input_ids.shape!=logits.shape[:2] or input_ids.dtype!=torch.long
            or type(prompt_tokens) is not int or not 1<=prompt_tokens<input_ids.shape[1]
            or any(type(i) is not int or not 0<=i<input_ids.shape[1]-prompt_tokens for i in negative_positions)
            or sorted(set(negative_positions))!=negative_positions):
        raise ValueError('One complete causal sequence and unique ordered generated-response positions required')
    if not negative_positions:
        return logits[:,:0].float().sum()
    positions=torch.tensor(negative_positions,device=input_ids.device)+prompt_tokens
    targets=input_ids[0,positions]
    if bool(((targets==eos)|(targets<0)|(targets>=logits.shape[2])).any()):
        raise ValueError('True EOS and out-of-vocabulary targets cannot be repetition negatives')
    selected=logits[0,positions-1].float()
    if not bool(torch.isfinite(selected).all()):
        raise ValueError('Finite original policy logits required')
    other=selected.clone().scatter(1,targets[:,None],float('-inf'))
    return (torch.logsumexp(selected,dim=-1)-torch.logsumexp(other,dim=-1)).mean()
