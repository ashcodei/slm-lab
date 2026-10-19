"""Matched public-target CE and globally normalized execution-group RL.

Each arm may distribute one original prompt's group across replicas. Variable
sample lengths share one actual-action denominator; complete target CE is
divided by the replica count before gradient SUM. Microbatch backward avoids
retaining every full trajectory graph. No production-quality gain is implied.
"""
import math
import random

import torch
from torch import distributed as dist

from .execution_group_RL_v1 import ExecutionGroupLearner
from .own_dense_actor_checkpoint_v1 import make_state,restore_state
from .whole_human_sft_evidence_v1 import tensor_digest

FORMAT='own-matched-global-execution-RL-public-target-control-v2'


def advantages(rewards):
    if not rewards or any(not math.isfinite(r) or not 0<=r<=1 for r in rewards):raise ValueError('Actual finite example fractions required')
    values=torch.tensor(rewards,dtype=torch.float64)
    return ((values-values.mean())/(values.std(unbiased=False)+1e-6)).tolist()


def trajectory_objective(logits,reference,actions,old,advantage,global_actions,*,policy_scale,KL_weight,clip=.2):
    if (logits.ndim!=2 or logits.shape!=reference.shape or len(actions)!=len(logits) or len(old)!=len(actions)
            or not len(actions) or type(global_actions) is not int or global_actions<len(actions)
            or not math.isfinite(advantage) or policy_scale not in (0,1)
            or not math.isfinite(KL_weight) or KL_weight<0 or not 0<clip<1):
        raise ValueError('One actual full-vocabulary trajectory and complete group normalization required')
    logp=logits.float().log_softmax(-1);logq=reference.detach().float().log_softmax(-1)
    target=torch.tensor(actions,dtype=torch.long,device=logits.device)
    behavior=torch.tensor(old,dtype=torch.float32,device=logits.device).detach()
    if not bool(torch.isfinite(behavior).all()):raise ValueError('Finite actual behavior scores required')
    current=logp.gather(1,target[:,None]).squeeze(1);ratio=(current-behavior).exp()
    surrogate=-torch.minimum(ratio*advantage,ratio.clamp(1-clip,1+clip)*advantage).sum()/global_actions
    KL=(logq.exp()*(logq-logp)).sum()/global_actions
    loss=policy_scale*surrogate+KL_weight*KL
    if not bool(torch.isfinite(loss)):raise ValueError('Actual global objective must be finite')
    return loss,dict(policy_surrogate=float(surrogate.detach()),reference_KL=float(KL.detach()))


class ControlledExecutionGroup(ExecutionGroupLearner):
    def __init__(self,model,optimizer,root,*,method,seed=957117,group_size=4,max_new_tokens=512,
                 KL_weight=.01,world=1,rank=0,gradient_group=None,control_group=None):
        super().__init__(model,optimizer,root,group_size=group_size,max_new_tokens=max_new_tokens,KL_weight=KL_weight)
        if (method not in ('verified_solution','execution_hybrid') or type(seed) is not int or seed<1
                or type(world) is not int or world<1 or type(rank) is not int or not 0<=rank<world
                or not math.isfinite(KL_weight) or KL_weight<0):raise ValueError('Explicit matched objective and replica profile required')
        if world>1 and (not dist.is_initialized() or dist.get_world_size(gradient_group)!=world
                         or dist.get_world_size(control_group)!=world):raise ValueError('Exact initialized arm replicas required')
        self.method,self.seed,self.world,self.rank=method,seed,world,rank
        self.gradient_group,self.control_group=gradient_group,control_group
        self.order=list(range(len(self.pack)));random.Random(seed).shuffle(self.order)
        self.completed=0;self.metrics=[]
        self.fixed=[p for p in model.parameters() if id(p) not in {id(x) for x in self.parameters}]
        self.fixed_signature=self.signature(self.fixed);self.frozen_digest=tensor_digest(model.base.named_parameters())

    @staticmethod
    def signature(parameters):return [(id(p),p._version,p.data_ptr(),p.requires_grad) for p in parameters]

    def settings(self):return dict(method=self.method,seed=self.seed,local_group_size=self.group_size,
        max_new_tokens=self.max_new_tokens,KL_weight=self.KL_weight,replicas=self.world,replica_rank=self.rank,
        public_target_weight=1.,gradient_clip=1.,precision='FP32_TF32_disabled',whole_source_population=len(self.pack))

    def gather(self,value):
        if self.world==1:return [value]
        values=[None]*self.world;dist.all_gather_object(values,value,group=self.control_group);return values

    def sample(self,prefix,source_id):
        device=next(self.model.parameters()).device;rows=[];self.model.eval()
        with torch.no_grad(),torch.autocast(device_type=device.type,enabled=False):
            for _ in range(self.group_size):
                state=self.model.policy.prefill(torch.tensor([prefix],device=device));actions=[];old=[]
                for i in range(self.max_new_tokens):
                    logits=state['logits'][0,-1].float();token=int(torch.multinomial(logits.softmax(-1),1))
                    actions.append(token);old.append(float(logits.log_softmax(-1)[token]))
                    if token==2 or i+1==self.max_new_tokens:break
                    state=self.model.policy.decode(torch.tensor([[token]],device=device),state['cache'])
                del state
                response=self.tokenizer.decode(actions)
                rows.append(dict(response_ids=actions,actual_cached_behavior_logps=old,own_response=response,
                    actual_execution_reward=self.reward.score(source_id,response),temperature=1.,full_vocabulary_sampling=True))
        return rows

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
        differences=[];self.model.train();self.model.policy.gradient_checkpointing=True;self.optimizer.zero_grad(set_to_none=True)
        try:
            with torch.autocast(device_type=device.type,enabled=False):
                for index,t in enumerate(local):
                    ids=torch.tensor([prefix+t['response_ids']],device=device)
                    logits=self.model(ids)['logits'][0,len(prefix)-1:-1]
                    with torch.no_grad():reference=self.model.base(ids)['logits'][0,len(prefix)-1:-1]
                    full=logits.detach().float().log_softmax(-1).gather(1,torch.tensor(t['response_ids'],device=device)[:,None]).squeeze(1)
                    actual=torch.tensor(t['actual_cached_behavior_logps'],device=device)
                    differences.append(float((full-actual).abs().max()))
                    if not torch.allclose(full,actual,atol=1e-4,rtol=1e-5):raise ValueError('Cached and full actual behavior differ before update')
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
            if self.world>1:
                # These gradients already have complete-global denominators.
                flat=torch.cat([p.grad.reshape(-1) for p in self.parameters]);dist.all_reduce(flat,group=self.gradient_group)
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

    def cursor(self,completed):return dict(completed_updates=completed,source_population=len(self.order),source_order=self.order,
        original_functions_consumed=completed,replicas=self.world,replica_rank=self.rank,seed=self.seed)

    def state(self,sources):
        if self.signature(self.fixed)!=self.fixed_signature:raise ValueError('Frozen base/head metadata changed')
        core=make_state(self.model,self.optimizer,groups=('actor',),sources=sources,completed=self.completed,
            planned=len(self.order),cursor=self.cursor(self.completed),metrics=self.metrics,device=next(self.model.parameters()).device)
        if core['model']['frozen_dense_sha256']!=self.frozen_digest:raise ValueError('Frozen reference/base bytes changed')
        return dict(format=FORMAT,settings=self.settings(),private_replica_state=core,
            private_state_is_not_whole_distributed_checkpoint=True,goal_achieved=False)

    def resume(self,state,sources,*,allow_completed=False):
        core=state.get('private_replica_state',{});completed=core.get('completed')
        if (state.get('format')!=FORMAT or state.get('settings')!=self.settings()
                or state.get('private_state_is_not_whole_distributed_checkpoint') is not True
                or type(completed) is not int or not 0<completed<=len(self.order)
                or len(core.get('metrics',[]))!=completed or state.get('goal_achieved') is not False):
            raise ValueError('Exact private replica method, group, source order and state required')
        for i,r in enumerate(core['metrics']):
            position=self.order[i];sequence,labels=self.pack.sequence(position);start=next(j for j,y in enumerate(labels) if y!=-100)
            rows=r.get('trajectories',[]);rewards=[t['actual_execution_reward']['reward'] for t in rows]
            all_zero=bool(rewards) and max(rewards)==0
            if (r.get('step')!=i+1 or r.get('original_train_position')!=position or r.get('complete_prompt_ids')!=sequence[:start]
                    or r.get('source_id')!=self.rows[position]['parent_source']['id']
                    or r.get('complete_public_target_sha256')!=self.rows[position]['unchanged_retained_target_sha256']
                    or len(rows)!=self.world*self.group_size or r.get('global_rewards')!=rewards
                    or r.get('global_advantages')!=advantages(rewards)
                    or r.get('all_global_execution_rewards_zero')!=all_zero or r.get('KL_weight')!=self.KL_weight
                    or r.get('complete_public_target_used')!=(self.method=='verified_solution' or all_zero)
                    or r.get('global_sampled_action_tokens')!=sum(len(t['response_ids']) for t in rows)
                    or r.get('policy_scale')!=int(self.method=='execution_hybrid')):
                raise ValueError('Actual whole source draws, complete groups and fallback decisions must reconstruct')
            if any(not math.isfinite(r.get(k,float('nan'))) for k in ('policy_surrogate','reference_KL','public_target_CE','gradient_norm_before_clip')):
                raise ValueError('Actual finite objective and gradient records required')
            for t in rows:
                if (not 1<=len(t['response_ids'])<=self.max_new_tokens or len(t['actual_cached_behavior_logps'])!=len(t['response_ids'])
                        or self.tokenizer.decode(t['response_ids'])!=t['own_response']
                        or any(not math.isfinite(p) or p>0 for p in t['actual_cached_behavior_logps'])):
                    raise ValueError('Actual bounded own actions and behavior records required')
        self.completed,self.metrics=restore_state(self.model,self.optimizer,core,groups=('actor',),expected_sources=sources,
            planned=len(self.order),expected_cursor=self.cursor(completed),device=next(self.model.parameters()).device,allow_completed=allow_completed)
