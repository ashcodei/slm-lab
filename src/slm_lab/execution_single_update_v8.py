"""Single fresh-group gradient with calibrated rounding and coordinated errors.

The original archived trial remains unchanged. A declared legacy prefix is
validated through the original stage before this new objective is adopted.
"""
import copy
import math

import torch
from torch import distributed as dist

from .code_probability_guard_v8 import load as load_calibration
from .codenet_checkpoint_v1 import capture_rng
from .execution_group_checked_host_v7 import CheckedHostExecutionGroup,CoordinatedLocalUpdateError
from .execution_group_controlled_v2 import advantages
from .execution_group_host_capacity_v6 import equal
from .single_update_code_policy_v8 import objective

OBJECTIVE='fresh_single_update_group_score_function_plus_own_reference_KL_v8'


def trajectory_objective(*args,**kwargs):
    return objective(*args,**kwargs,freshly_sampled_single_update=True)


class SingleUpdateExecutionGroup(CheckedHostExecutionGroup):
    def __init__(self,model,optimizer,root,*,legacy_updates,**kwargs):
        if type(legacy_updates) is not int or not 0<=legacy_updates<173:
            raise ValueError('Explicit inherited original-objective source prefix required')
        calibration=load_calibration(root,model)
        self.legacy_updates=legacy_updates
        self.cache_bound=calibration['bound'];self.cache_calibration_sha256=calibration['sha256']
        super().__init__(model,optimizer,root,**kwargs)

    def settings(self):
        return dict(super().settings(),policy_objective=OBJECTIVE,legacy_objective_updates=self.legacy_updates,
            calibrated_absolute_cache_logp_bound=self.cache_bound,cache_calibration_sha256=self.cache_calibration_sha256)

    @classmethod
    def from_legacy(cls,model,optimizer,root,state,expected_sources,*,legacy_updates,**kwargs):
        if state.get('private_replica_state',{}).get('completed')!=legacy_updates:
            raise ValueError('Exact declared inherited source cursor required before migration')
        previous=CheckedHostExecutionGroup(model,optimizer,root,**kwargs)
        previous.resume(state,expected_sources)
        RNG=capture_rng(next(model.parameters()).device)
        current=cls(model,optimizer,root,legacy_updates=legacy_updates,**kwargs)
        current.completed=previous.completed;current.metrics=copy.deepcopy(previous.metrics)
        if not equal(RNG,capture_rng(next(model.parameters()).device)):
            raise ValueError('Objective migration must preserve every private RNG exactly')
        return current

    def resume(self,state,sources,*,allow_completed=False):
        if state.get('settings')!=self.settings():raise ValueError('Exact declared objective, numerical calibration and inherited prefix required')
        for row in state.get('private_replica_state',{}).get('metrics',[])[self.legacy_updates:]:
            if (row.get('policy_objective')!=OBJECTIVE or row.get('legacy_objective_updates')!=self.legacy_updates
                    or row.get('calibrated_absolute_cache_logp_bound')!=self.cache_bound
                    or row.get('cache_calibration_sha256')!=self.cache_calibration_sha256
                    or row.get('fresh_policy_unchanged_until_single_optimizer_commit') is not True):
                raise ValueError('Every post-migration update must retain actual objective/calibration provenance')
            for trajectory in row['trajectories']:
                full=trajectory.get('actual_full_behavior_logps',[]);cached=trajectory['actual_cached_behavior_logps']
                if (len(full)!=len(cached) or any(not math.isfinite(p) or p>0 for p in full)
                        or not torch.allclose(torch.tensor(full),torch.tensor(cached),atol=self.cache_bound,rtol=0.)):
                    raise ValueError('Every actual cached/full score must obey its admitted runtime guard')
        return super().resume(state,sources,allow_completed=allow_completed)

    def update(self):
        if self.completed>=len(self.order):raise ValueError('Every original train function already consumed once')
        if self.signature(self.fixed)!=self.fixed_signature:raise ValueError('Frozen base/heads/critic metadata changed')
        position=self.order[self.completed];row=self.rows[position];source_id=row['parent_source']['id']
        sequence,labels=self.pack.sequence(position);start=next(i for i,y in enumerate(labels) if y!=-100);prefix=sequence[:start]
        if len(prefix)+self.max_new_tokens>self.model.config.max_text_length:raise ValueError('Entire prompt/action horizon must fit')
        sampling_signature=self.model.policy.signature();local=self.sample(prefix,source_id);groups=self.gather(local);trajectories=[t for group in groups for t in group]
        rewards=[t['actual_execution_reward']['reward'] for t in trajectories];relative=advantages(rewards)
        total_actions=sum(len(t['response_ids']) for t in trajectories);all_zero=max(rewards)==0
        public_CE=self.method=='verified_solution' or all_zero;scale=int(self.method=='execution_hybrid')
        device=next(self.model.parameters()).device;local_metrics=dict(policy_surrogate=0.,reference_KL=0.,public_target_CE=0.)
        self.parity_observations=[];differences=[];self.model.train();self.model.policy.gradient_checkpointing=True;self.optimizer.zero_grad(set_to_none=True)
        try:
            local_error=None
            try:
                if self.model.policy.signature()!=sampling_signature:raise ValueError('Fresh sampling policy changed before its only gradient update')
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
                            calibrated_allclose_passed=bool(torch.allclose(full,actual,atol=self.cache_bound,rtol=0.)))
                        if getattr(self,'capture_parity_logits',False) or not observation['calibrated_allclose_passed']:
                            observation['actual_full_FP32_logits']=logits.detach().cpu().clone()
                        self.parity_observations.append(observation)
                        if not observation['calibrated_allclose_passed']:
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
            score_groups=self.gather([r['full_behavior_logps'] for r in self.parity_observations])
            full_scores=[scores for group in score_groups for scores in group]
            if len(full_scores)!=len(trajectories):raise ValueError('Every actual full-pass trajectory score must remain accounted for')
            for trajectory,scores in zip(trajectories,full_scores):trajectory['actual_full_behavior_logps']=scores
            if self.model.policy.signature()!=sampling_signature:raise ValueError('Only one unchanged-policy update is allowed per freshly sampled group')
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
            policy_objective=OBJECTIVE,legacy_objective_updates=self.legacy_updates,
            calibrated_absolute_cache_logp_bound=self.cache_bound,cache_calibration_sha256=self.cache_calibration_sha256,
            fresh_policy_unchanged_until_single_optimizer_commit=True,
            genuine_human_preference_RLHF=False,external_teacher_or_extra_reward_encoder=False,goal_achieved=False)
        self.completed+=1;self.metrics.append(record);return record
