"""Actual CPU/FP32 native sampling and full-forward probability observations.

This diagnostic has no optimizer, backward pass, truncated teacher response or
probability-tolerance admission. Cached/full differences are measurements only;
this is not a numerical admission for a later GPU policy-gradient experiment.
"""
import math

import torch


@torch.no_grad()
def sample(model, tokenizer, prefix_ids, response_prefix_ids, *, horizon, sampler, seed, progress=lambda _:None):
    parameters=list(model.parameters())
    if (not parameters or model.training or any(p.device.type!='cpu' or p.dtype!=torch.float32 or p.grad is not None for p in parameters)
            or sampler not in ['greedy','full_vocabulary_temperature1'] or type(seed) is not int or seed<0
            or type(horizon) is not int or horizon<1 or not prefix_ids or prefix_ids[0]!=1
            or len(prefix_ids)+horizon>model.config.max_text_length or tokenizer.get_vocab_size()!=model.config.vocab_size
            or [tokenizer.token_to_id(t) for t in ['<pad>','<bos>','<eos>']]!=[0,1,2]
            or any(type(t) is not int or not 0<=t<model.config.vocab_size for t in prefix_ids+response_prefix_ids)
            or response_prefix_ids and prefix_ids[-len(response_prefix_ids):]!=response_prefix_ids):
        raise ValueError('Exact unchanged native CPU/FP32 evaluation policy, full rollout budget and own boundaries required')
    signature=[(id(p),p._version,p.data_ptr(),p.requires_grad) for p in parameters]
    generator=torch.Generator(device='cpu');generator.manual_seed(seed)
    initial_global_rng=torch.get_rng_state().clone()
    with torch.autocast('cpu',enabled=False):
        state=model.prefill(torch.tensor([prefix_ids],dtype=torch.long));actions=[];behavior=[]
        for position in range(horizon):
            logits=state['logits'][0,-1].float()
            if not bool(torch.isfinite(logits).all()):raise ValueError('Finite actual full-vocabulary native logits required')
            action=int(logits.argmax()) if sampler=='greedy' else int(torch.multinomial(logits.softmax(-1),1,generator=generator))
            actions.append(action);behavior.append(float(logits.log_softmax(-1)[action]))
            if len(actions)%32==0:progress(dict(generated_tokens=len(actions),horizon=horizon))
            if action==2 or position+1==horizon:break
            state=model.decode(torch.tensor([[action]],dtype=torch.long),state['cache'])
        del state
        ids=torch.tensor([prefix_ids+actions],dtype=torch.long)
        logits=model(input_ids=ids,attention_mask=torch.ones_like(ids,dtype=torch.bool))['logits'][0,len(prefix_ids)-1:-1].float()
        full=logits.log_softmax(-1).gather(1,torch.tensor(actions,dtype=torch.long)[:,None]).flatten()
    if (len(full)!=len(actions) or not bool(torch.isfinite(full).all())
            or any(not math.isfinite(x) for x in behavior)
            or signature!=[(id(p),p._version,p.data_ptr(),p.requires_grad) for p in parameters]
            or any(p.grad is not None for p in parameters) or not torch.equal(initial_global_rng,torch.get_rng_state())):
        raise ValueError('Finite complete behavior observations, fixed weights and unchanged global RNG required')
    return dict(actual_generated_action_ids=actions,actual_cached_logps=behavior,actual_full_forward_logps=full.tolist(),
        cached_full_max_absolute_difference=float((full-torch.tensor(behavior,dtype=torch.float32)).abs().max()),
        actual_generated_action_tokens=len(actions),stopped_on_generated_EOS=actions[-1]==2,
        supplied_public_reference_prefix_ids=list(response_prefix_ids),
        complete_candidate_response=tokenizer.decode(response_prefix_ids+actions),
        actual_generated_suffix=tokenizer.decode(actions),sampler=sampler,seed=seed,
        original_native_prefix_ids=list(prefix_ids),horizon=horizon,
        inference_device_type='cpu',inference_precision='FP32',
        actual_sampled_tokens_not_reference_tokens=True,model_weights_and_global_RNG_unchanged=True,
        no_policy_gradient_numerical_admission=True,model_updates=0,goal_achieved=False)
