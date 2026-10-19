"""Source-only generation controls; none of these measurements is a benchmark score."""
from torch import nn

from .code_residual_branch_v1 import _CodeView, _CodeBlock
from .model import TinyMultimodalLM


class OriginalBaseView(TinyMultimodalLM):
    """Expose the original shared modules, without allocating another backbone."""
    def __init__(self, code):
        if type(code) is not _CodeView or any(type(b) is not _CodeBlock for b in code.blocks):
            raise ValueError('The original own residual inference view is required')
        nn.Module.__init__(self)
        self.config = code.config
        self.embedding = code.embedding
        self.vision = code.vision
        self.norm = code.norm
        self.lm_head = code.lm_head
        self.value_head = code.value_head
        self.blocks = nn.ModuleList(block.original for block in code.blocks)
        self.gradient_checkpointing = False
        if any(p.requires_grad for p in self.parameters()):
            raise ValueError('The shared original backbone must already be frozen')


def select_controls(selection, budget=1024):
    """Shortest whole answer per declared role/language; no generated-score selection."""
    if set(selection) != {'train', 'validation'} or type(budget) is not int or budget < 1:
        raise ValueError('Explicit original pilot train/validation roles and positive budget required')
    if ({r['group_id'] for r in selection['train']}
            & {r['group_id'] for r in selection['validation']}):
        raise ValueError('Original training and validation groups must remain disjoint')
    result = []
    for role in ['train', 'validation']:
        rows = selection[role]
        if any(r['role'] != role for r in rows):
            raise ValueError('Never relabel an original source role')
        for language in ['C++', 'Python', 'D', 'Java']:
            candidates = [r for r in rows if r['language'] == language
                and 1 < r['target_tokens'] < r['context_tokens'] <= 4096
                and r['context_tokens'] - r['target_tokens'] + budget <= 4096]
            if not candidates:
                raise ValueError('Every declared language needs a complete original source control')
            result.append(min(candidates, key=lambda r: (r['target_tokens'], r['id'])))
    if len({r['id'] for r in result}) != 8:
        raise ValueError('Eight distinct complete original source controls required')
    return result


def reference_agreement(generated, reference, eos):
    if not reference or reference[-1] != eos:
        raise ValueError('Complete original reference must include its true EOS')
    prefix = 0
    for actual, expected in zip(generated, reference):
        if actual != expected:
            break
        prefix += 1
    return dict(first_reference_token_match=bool(generated and generated[0] == reference[0]),
        matching_reference_prefix_tokens=prefix,
        exact_complete_reference_token_match=generated == reference,
        terminal_EOS=bool(generated and generated[-1] == eos),
        unique_contiguous_token_4grams=len({tuple(generated[i:i + 4]) for i in range(len(generated) - 3)}))
