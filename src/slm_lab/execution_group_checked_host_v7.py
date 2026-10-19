"""Original numerical guard with captured observations and coordinated errors.

No tolerance is relaxed. Every replica exchanges local forward/backward errors
before any gradient reduction. Intended first for disposable diagnostics.
"""
import math

import torch
from torch import distributed as dist

from .execution_group_host_reduce_v6 import HostReducedCausalExecutionGroup
from .execution_group_controlled_v2 import advantages, trajectory_objective


class CoordinatedLocalUpdateError(ValueError):pass


class CheckedHostExecutionGroup(HostReducedCausalExecutionGroup):
    def update(self):
        if self.completed>=len(self.order):raise ValueError('Every original train function already consumed once')
        if self.signature(self.fixed)!=self.fixed_signature:raise ValueError('Frozen base/heads/critic metadata changed')
        position=self.order[self.completed];row=self.rows[position];source_id=row['parent_source']['id']
        sequence,labels=self.pack.sequence(position);start=next(i for i,y in enumerate(labels) if y!=-100);prefix=sequence[:start]
        if len(prefix)+self.max_new_tokens>self.model.config.max_text_length:raise ValueError('Entire prompt/action horizon must fit')
        local=self.sample(prefix,source_id);groups=self.gather(local);trajectories=[t for group in groups for t in group]
        rewards=[t['actual_execution_reward']['reward'] for t in trajectories];relative=advantages(rewards)
        total_actions=sum(len(t['response_ids']) for t in trajectories);all_zero=max(rewards)==0
        public_CE=self.method=='verified_solution' or all_zero;scale=int(self.method=='execution_hybrid')
        device=next(self.model.parameters()).device;local_metrics=dict(policy_surrogate=0.,reference_KL=0.,public_target_CE=0.)
        self.parity_observations=[];differences=[];self.model.train();self.model.policy.gradient_checkpointing=True;self.optimizer.zero_grad(set_to_none=True)
        try:
            local_error=None
            try:
                with torch.autocast(device_type=device.type,enabled=False):
                    for index,t in enumerate(local):
                        ids=torch.tensor([prefix+t['response_ids']],device=device)
                        logits=self.model(ids)['logits'][0,len(prefix)-1:-1]
                        with torch.no_grad():reference=self.model.base(ids)['logits'][0,len(prefix)-1:-1]
                        full=logits.detach().float().log_softmax(-1).gather(1,torch.tensor(t['response_ids'],device=device)[:,None]).squeeze(1)
                        actual=torch.tensor(t['actual_cached_behavior_logps'],device=device)
                        differences.append(float((full-actual).abs().max()))
                        observation=dict(completed_before=self.completed,original_train_position=position,source_id=source_id,
                            local_sample=index,complete_prompt_ids=prefix,response_ids=t['response_ids'],
                            cached_behavior_logps=t['actual_cached_behavior_logps'],full_behavior_logps=full.detach().cpu().tolist(),
                            max_absolute_action_logp_difference=float((full-actual).abs().max()),
                            original_allclose_passed=bool(torch.allclose(full,actual,atol=1e-4,rtol=1e-5)))
                        if getattr(self,'capture_parity_logits',False) or not observation['original_allclose_passed']:
                            observation['actual_full_FP32_logits']=logits.detach().cpu().clone()
                        self.parity_observations.append(observation)
                        if not observation['original_allclose_passed']:
                            raise ValueError('Cached and full actual behavior differ before update: '+str(observation['max_absolute_action_logp_difference']))
                        loss,metric=trajectory_objective(logits,reference,t['response_ids'],t['actual_cached_behavior_logps'],
                            relative[self.rank*self.group_size+index],total_actions,policy_scale=scale,KL_weight=self.KL_weight)
                        loss.backward(inputs=self.parameters)
                        for key in metric:local_metrics[key]+=metric[key]
                        del loss,logits,reference
                    if public_CE:
                        batch=self.encoder.batch([dict(packed_code_index=position)],'response',device)
                        ce=self.model(**batch)['loss']/self.world;ce.backward(inputs=self.parameters)
                        local_metrics['public_target_CE']=float(ce.detach());del ce
                if (any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in self.parameters)
                        or any(p.grad is not None for p in self.fixed)):
                    raise ValueError('Every selected actor and no fixed leaf needs finite gradients')
            except Exception as error:
                self.optimizer.zero_grad(set_to_none=True)
                local_error=dict(type=type(error).__name__,message=str(error),replica_rank=self.rank)
            errors=self.gather(local_error)
            if any(error is not None for error in errors):
                self.optimizer.zero_grad(set_to_none=True)
                self.coordinated_errors=errors
                raise CoordinatedLocalUpdateError(str(errors))
            if self.world>1:
                # These gradients already have complete-global denominators.
                flat=torch.cat([p.grad.reshape(-1) for p in self.parameters]);host=flat.cpu()
                dist.all_reduce(host,group=self.gradient_group);flat.copy_(host);del host
                offset=0
                for p in self.parameters:p.grad.copy_(flat[offset:offset+p.numel()].view_as(p));offset+=p.numel()
                del flat
            norm=float(torch.nn.utils.clip_grad_norm_(self.parameters,1.,error_if_nonfinite=True))
            self.optimizer.step();self.optimizer.zero_grad(set_to_none=True)
        except BaseException:self.optimizer.zero_grad(set_to_none=True);raise
        metric_groups=self.gather(local_metrics)
        sums={k:math.fsum(r[k] for r in metric_groups) for k in local_metrics}
        cache_differences=[x for group in self.gather(differences) for x in group]
        record=dict(step=self.completed+1,original_train_position=position,source_id=source_id,complete_prompt_ids=prefix,
            trajectories=trajectories,global_sampled_action_tokens=total_actions,global_rewards=rewards,global_advantages=relative,
            global_group_size=len(trajectories),all_global_execution_rewards_zero=all_zero,
            complete_public_target_used=bool(public_CE),policy_scale=scale,KL_weight=self.KL_weight,**sums,
            cached_full_logp_max_absolute_differences=cache_differences,gradient_norm_before_clip=norm,
            complete_public_target_sha256=row['unchanged_retained_target_sha256'],actual_counts=self.model.counts(),
            genuine_human_preference_RLHF=False,external_teacher_or_extra_reward_encoder=False,goal_achieved=False)
        self.completed+=1;self.metrics.append(record);return record
