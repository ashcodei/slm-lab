"""Explicit full-horizon shape/gradient stress, never a training rollout.

Natural EOS cannot prove maximum response allocation. This control uses a
complete original train prompt plus declared fixed non-EOS actions, then
checks cached/full causal scores, original-encoder reward and actor/critic
gradients. It never steps Adam or claims human feedback for its fixed actions.
"""
import torch

from .dense_ppo_cached_v2 import SAMPLER
from .dense_ppo_control_v1 import DenseControlledPPOStage
from .dense_ppo_v1 import action_outputs,prompt_source
from .dense_ppo_reward_v1 import signature


def stress(kernel,position):
    if type(kernel) is not DenseControlledPPOStage:raise ValueError('The exact configured own-response PPO/critic kernel is required')
    model,reward=kernel.model,kernel.reward;source=prompt_source(kernel.data,position)
    prompt=source['prompt_ids'];horizon=kernel.options['max_new_tokens'];vocab=model.config.vocab_size
    if vocab<=4 or len(prompt)+horizon+1>min(model.config.max_text_length,reward.encoder.config.max_text_length):
        raise ValueError('The complete longest prompt and full action/scoring budget must fit')
    actions=[4+i%(vocab-4) for i in range(horizon)];before=signature(model.parameters());reward.check()
    completed=kernel.completed;adam_steps=[float(s['step']) for s in kernel.optimizer.state.values()]
    kernel.optimizer.zero_grad(set_to_none=True)
    try:
        with torch.autocast(device_type=kernel.device.type,enabled=False):
            model.eval();sampled=[]
            with torch.no_grad():
                cached=model.policy.prefill(torch.tensor([prompt],device=kernel.device))
                for i,action in enumerate(actions):
                    sampled.append(float(cached['logits'][0,-1].float().log_softmax(-1)[action]))
                    if i+1<horizon:cached=model.policy.decode(torch.tensor([[action]],device=kernel.device),cached['cache'])
                del cached
                ids=torch.tensor([prompt+actions],device=kernel.device)
                causal,_=action_outputs(model,ids,len(prompt),kernel.amp)
                reference,_=action_outputs(model.base,ids,len(prompt),kernel.amp)
                score=float(reward.score(torch.tensor([prompt+actions+[2]],device=kernel.device)))
                cached_logps=torch.tensor(sampled,device=kernel.device)
                if (not torch.allclose(cached_logps,causal,atol=SAMPLER['probability_atol'],rtol=SAMPLER['probability_rtol'])
                        or not bool(torch.isfinite(reference).all())):
                    raise ValueError('Actual complete full-horizon cached and full-causal FP32 probabilities must agree')
                difference=float((cached_logps-causal).abs().max())
            model.train();model.policy.gradient_checkpointing=True
            logps,values=action_outputs(model,ids,len(prompt),kernel.amp)
            # This is a synthetic capacity loss, explicitly separate from PPO
            # and human rewards. Both real actor and critic gradients allocate.
            loss=-logps.mean()+(values-1).square().mean()
            loss.backward(inputs=kernel.parameters)
            if (any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in kernel.parameters)
                    or any(p.grad is not None for p in kernel.fixed)):
                raise ValueError('Full actual source/horizon requires finite actor/critic-only gradients')
            reward.check()
        if before!=signature(model.parameters()) or kernel.completed!=completed or adam_steps!=[float(s['step']) for s in kernel.optimizer.state.values()]:
            raise ValueError('Synthetic capacity check must not update any weight, Adam step or source cursor')
        return dict(original_source=source,full_fixed_action_tokens=horizon,actual_policy_context_tokens=len(prompt)+horizon,
            actual_reward_context_tokens=len(prompt)+horizon+1,actual_prefills=1,actual_decodes=horizon-1,
            max_absolute_cache_full_logp_difference=difference,original_encoder_reward_score=score,
            actor_critic_finite_gradients=True,synthetic_capacity_loss=float(loss.detach()),
            no_optimizer_step=True,fixed_actions_not_training_rollouts_or_human_judgments=True)
    finally:kernel.optimizer.zero_grad(set_to_none=True)
