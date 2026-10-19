"""Capture actual greedy native actions without changing the generator result."""
from contextlib import contextmanager
import hashlib

import torch


def signature(model):
    return [(id(p),p._version,p.data_ptr(),p.requires_grad) for p in model.parameters()]


@contextmanager
def observe(model,callback):
    if model.training or not callable(callback) or getattr(model,'_native_greedy_observer_active_v1',False):
        raise ValueError('One eval-mode native generator and an explicit observation callback required')
    original=model.generate;calls=[];before=signature(model)
    def generate(input_ids,**kwargs):
        if (input_ids.ndim!=2 or input_ids.shape[0]!=1 or kwargs.get('temperature',0)!=0
                or kwargs.get('return_rollout',False) or kwargs.get('use_cache',True) is not True
                or type(kwargs.get('max_new_tokens')) is not int or kwargs['max_new_tokens']<1):
            raise ValueError('Explicit single-example cached greedy tensor generation required')
        prefix=input_ids.detach().cpu().tolist()[0]
        output=original(input_ids,**kwargs)
        if (not isinstance(output,torch.Tensor) or output.ndim!=2 or output.shape[0]!=1
                or not torch.equal(output[:,:len(prefix)],input_ids)
                or not 1<=output.shape[1]-len(prefix)<=kwargs['max_new_tokens'] or signature(model)!=before):
            raise ValueError('Actual unchanged prefix, bounded actions and unmodified model parameters required')
        actions=output[0,len(prefix):].detach().cpu().tolist();eos=kwargs.get('eos_token_id')
        event=dict(case_position=len(calls),original_input_token_ids=prefix,actual_generated_token_ids=actions,
            complete_input_ids_sha256=hashlib.sha256(input_ids.detach().cpu().numpy().astype('<u2').tobytes()).hexdigest(),
            requested_native_action_budget=kwargs['max_new_tokens'],actual_native_action_count=len(actions),
            actual_stopped_on_EOS=bool(eos is not None and actions[-1]==eos),
            greedy_native_output_tensor_returned_unchanged=True,selected_action_log_probabilities_not_retained=True)
        calls.append(event);callback(event)
        return output
    model._native_greedy_observer_active_v1=True;model.generate=generate
    try:yield calls
    finally:model.generate=original;model._native_greedy_observer_active_v1=False
