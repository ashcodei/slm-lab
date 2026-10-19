"""Compare real cached text/visual logits with the same model's full forward.

This is an inference diagnostic, never a model-quality benchmark or training
admission. Coverage is stated in text positions and actual vocabulary values.
"""
from contextlib import nullcontext

import torch

from . import model as core
from .inference_rotary_cache_v1 import ORIGINAL,inference_cache


def signature(model):return [(id(p),p._version,p.data_ptr(),p.requires_grad) for p in model.parameters()]


def chunks(length,width):
    result=[]
    while length:
        n=min(length,width);result.append(n);length-=n
    return result


@torch.inference_mode()
def compare(model,ids,*,pixels=None,paths,maximum_error=.002,record=lambda row:None):
    if (model.training or ids.ndim!=2 or len(ids)!=1 or not paths or core.rotary is not ORIGINAL
            or any(p.grad is not None for p in model.parameters())):
        raise ValueError('Single actual evaluation-only native example and unchanged original rotary oracle required')
    frozen=signature(model);input_before=ids.clone();image_before=pixels.clone() if pixels is not None else None
    oracle=model(ids,pixels)['logits']
    if oracle.shape[:2]!=ids.shape or not bool(torch.isfinite(oracle).all()):raise ValueError('Complete finite real full-forward oracle required')
    reports=[]
    for spec in paths:
        prefix=spec['prefix'];pieces=spec['chunks']
        if (type(prefix) is not int or not 1<=prefix<=ids.shape[1] or any(type(n) is not int or n<1 for n in pieces)
                or prefix+sum(pieces)!=ids.shape[1] or type(spec['rotary_tables']) is not bool):
            raise ValueError('Every planned full original continuation must reach its stated final text position')
        observations=[]
        def compare_logits(logits,start):
            for a in range(0,logits.shape[1],64):
                actual=logits[:,a:a+64].float();expected=oracle[:,start+a:start+a+actual.shape[1]].float()
                if actual.shape!=expected.shape or not bool(torch.isfinite(actual).all()):raise ValueError('Every actual cached vocabulary logit required')
                errors=(actual-expected).abs().amax(-1)[0].tolist()
                mismatch=(actual.argmax(-1)!=expected.argmax(-1))[0].tolist()
                for i,(error,different) in enumerate(zip(errors,mismatch)):
                    observations.append(dict(original_text_position=start+a+i,maximum_vocabulary_logit_error=error,
                        full_vocabulary_greedy_argmax_different=different))
        with inference_cache() if spec['rotary_tables'] else nullcontext():
            state=model.prefill(ids[:,:prefix],pixels);compare_logits(state['logits'],prefix-1);position=prefix
            expected_image_tokens=model.config.image_token_count(pixels.shape[-1]) if pixels is not None else 0
            if state['cache'].text_length!=prefix or state['cache'].image_tokens!=expected_image_tokens:
                raise ValueError('Actual whole visual prefix and original text cursor required')
            for n in pieces:
                state=model.decode(ids[:,position:position+n],state['cache']);compare_logits(state['logits'],position);position+=n
                if state['cache'].text_length!=position:raise ValueError('Every actual cache cursor must advance by the complete native input chunk')
        if [r['original_text_position'] for r in observations]!=list(range(prefix-1,ids.shape[1])):
            raise ValueError('No original compared text position may be omitted or repeated')
        maximum=max(r['maximum_vocabulary_logit_error'] for r in observations)
        row=dict(spec=spec,actual_text_length=ids.shape[1],actual_image_tokens=expected_image_tokens,
            actual_compared_text_positions=len(observations),actual_vocabulary=oracle.shape[-1],
            actual_vocabulary_values_compared=len(observations)*oracle.shape[-1],
            maximum_absolute_logit_error=maximum,
            actual_greedy_argmax_disagreements=sum(r['full_vocabulary_greedy_argmax_different'] for r in observations),
            within_predeclared_absolute_bound=maximum<=maximum_error,per_position_observations=observations)
        reports.append(row);record(row)
        del state
    if (signature(model)!=frozen or not torch.equal(ids,input_before)
            or pixels is not None and not torch.equal(pixels,image_before) or core.rotary is not ORIGINAL):
        raise ValueError('Native inference comparison changed model, original inputs or rotary implementation')
    return dict(actual_paths=reports,original_full_forward_neural_oracle_computed=True,
        actual_model_and_inputs_unchanged=True,maximum_absolute_logit_bound=maximum_error,
        all_paths_within_bound=all(r['within_predeclared_absolute_bound'] for r in reports),
        all_compared_position_greedy_argmaxes_equal=all(not r['actual_greedy_argmax_disagreements'] for r in reports),
        no_gradient_or_optimizer_step=True,not_generated_program_quality_or_all_devices_parity=True)
