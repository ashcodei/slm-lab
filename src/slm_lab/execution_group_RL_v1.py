"""Experimental own-code group RL with a verified-public-target fallback.

Relative execution rewards alone are flat when every sample fails. In that
case this hybrid explicitly adds complete original public-target CE, while
retaining the same counted backbone as its KL reference. It is not RLHF and
has no production controller or established quality gain yet.
"""
import gzip
import json
import math
from pathlib import Path

import torch
from torch.nn import functional as F
from tokenizers import Tokenizer

from .codenet_token_pack_v1 import CompleteCodePack,packed_encoder
from .own_dense_actor_v1 import OwnDenseActor
from .public_example_code_reward_v1 import PublicExampleCodeReward


def group_objective(logits,references,actions,behavior_logps,rewards,*,clip=.2,KL_weight=.01):
    if (not 2<=len(rewards)<=16 or not len(logits)==len(references)==len(actions)==len(behavior_logps)==len(rewards)
            or not 0<clip<1 or not math.isfinite(KL_weight) or KL_weight<0
            or any(not math.isfinite(r) or not 0<=r<=1 for r in rewards)):
        raise ValueError('Complete own-response group and finite train-example rewards required')
    values=torch.tensor(rewards,dtype=torch.float64);advantages=(values-values.mean())/(values.std(unbiased=False)+1e-6)
    denominator=sum(len(a) for a in actions)
    if not denominator:raise ValueError('Every own sample needs at least one actual action')
    policy=KL=0.
    for index,(new,reference,target,old) in enumerate(zip(logits,references,actions,behavior_logps)):
        if new.ndim!=2 or new.shape!=reference.shape or len(target)!=len(new) or len(old)!=len(new) or not len(new):
            raise ValueError('Every actual action needs its whole-vocabulary policy/reference and cached behavior probability')
        logp=new.float().log_softmax(-1);logq=reference.detach().float().log_softmax(-1)
        target=torch.tensor(target,dtype=torch.long,device=new.device)
        old=torch.tensor(old,dtype=torch.float32,device=new.device).detach()
        if not bool(torch.isfinite(old).all()) or not bool(torch.isfinite(logp).all()):raise ValueError('Finite actual action probabilities required')
        current=logp.gather(1,target[:,None]).squeeze(1);ratio=(current-old).exp();adv=advantages[index].to(device=new.device,dtype=torch.float32)
        policy=policy-torch.minimum(ratio*adv,ratio.clamp(1-clip,1+clip)*adv).sum()/denominator
        KL=KL+(logq.exp()*(logq-logp)).sum()/denominator
    objective=policy+KL_weight*KL
    if not bool(torch.isfinite(objective)):raise ValueError('Actual group RL objective must be finite')
    return objective,dict(rewards=list(rewards),group_advantages=advantages.tolist(),sampled_action_tokens=denominator,
        policy_surrogate=float(policy.detach()),whole_vocabulary_reference_KL=float(KL.detach()),
        all_execution_rewards_zero=max(rewards)==0,verified_public_target_fallback_required=max(rewards)==0,
        token_global_policy_normalization=True,KL_weight=KL_weight)


class ExecutionGroupLearner:
    """A real train-only step; later controlled scheduling/recovery is required."""
    def __init__(self,model,optimizer,root,*,group_size=4,max_new_tokens=512,public_fallback_weight=1.,KL_weight=.01):
        if type(model) is not OwnDenseActor or not 2<=group_size<=16 or type(max_new_tokens) is not int or max_new_tokens<1:
            raise ValueError('Counted own actor and explicit bounded group/action budget required')
        if any(p.dtype!=torch.float32 for p in model.parameters()):raise ValueError('This own-code sampler and gradient experiment are explicitly FP32')
        if not math.isfinite(public_fallback_weight) or public_fallback_weight<=0:raise ValueError('Positive complete public-target fallback coefficient required')
        self.model,self.optimizer=model,optimizer;self.root=Path(root).resolve();self.reward=PublicExampleCodeReward(root)
        folder=self.root/'data/documented-doctest-verified-v1';self.pack=CompleteCodePack(folder,'train','training')
        self.tokenizer=Tokenizer.from_file(str(folder/'tokenizer.json'));self.encoder=packed_encoder(self.pack,self.tokenizer,model.config)
        with gzip.open(folder/'original-rows.jsonl.gz','rt') as f:self.rows=list(map(json.loads,f))
        self.parameters=[p for _,p in model.named_selected(('actor',))]
        if [id(p) for g in optimizer.param_groups for p in g['params']]!=[id(p) for p in self.parameters]:
            raise ValueError('Only the actual counted actor parameters may enter code-RL Adam')
        self.group_size=group_size;self.max_new_tokens=max_new_tokens;self.fallback_weight=public_fallback_weight;self.KL_weight=KL_weight

    def step(self,position):
        if type(position) is not int or not 0<=position<len(self.pack):raise ValueError('An admitted original training function is required')
        device=next(self.model.parameters()).device;row=self.rows[position];source_id=row['parent_source']['id']
        seq,labels=self.pack.sequence(position);start=next(i for i,x in enumerate(labels) if x!=-100);prefix=seq[:start]
        if len(prefix)+self.max_new_tokens>self.model.config.max_text_length:raise ValueError('Whole original prompt and entire action horizon must fit')
        trajectories=[];self.model.eval()
        with torch.no_grad(),torch.autocast(device_type=device.type,enabled=False):
            for _ in range(self.group_size):
                state=self.model.policy.prefill(torch.tensor([prefix],device=device));actions=[];old=[]
                for i in range(self.max_new_tokens):
                    logits=state['logits'][0,-1].float();token=int(torch.multinomial(logits.softmax(-1),1))
                    actions.append(token);old.append(float(logits.log_softmax(-1)[token]))
                    if token==2 or i+1==self.max_new_tokens:break
                    state=self.model.policy.decode(torch.tensor([[token]],device=device),state['cache'])
                del state
                response=self.tokenizer.decode(actions);reward=self.reward.score(source_id,response)
                trajectories.append(dict(response_ids=actions,actual_cached_behavior_logps=old,own_response=response,
                    actual_execution_reward=reward,temperature=1.,full_vocabulary_sampling=True))
        self.model.train();self.model.policy.gradient_checkpointing=True;self.optimizer.zero_grad(set_to_none=True)
        logits=[];references=[];cache_differences=[]
        with torch.autocast(device_type=device.type,enabled=False):
            for t in trajectories:
                ids=torch.tensor([prefix+t['response_ids']],device=device)
                logits.append(self.model(ids)['logits'][0,len(prefix)-1:-1])
                with torch.no_grad():references.append(self.model.base(ids)['logits'][0,len(prefix)-1:-1])
                actual=torch.tensor(t['actual_cached_behavior_logps'],device=device)
                full=logits[-1].detach().float().log_softmax(-1).gather(1,torch.tensor(t['response_ids'],device=device)[:,None]).squeeze(1)
                cache_differences.append(float((full-actual).abs().max()))
                if not torch.allclose(full,actual,atol=1e-4,rtol=1e-5):
                    raise ValueError('Actual FP32 cached behavior must match full causal action probabilities before any update')
            rewards=[t['actual_execution_reward']['reward'] for t in trajectories]
            objective,metric=group_objective(logits,references,[t['response_ids'] for t in trajectories],
                [t['actual_cached_behavior_logps'] for t in trajectories],rewards,KL_weight=self.KL_weight)
            fallback=0.
            if metric['verified_public_target_fallback_required']:
                b=self.encoder.batch([dict(packed_code_index=position)],'response',device)
                fallback=self.model(**b)['loss'];objective=objective+self.fallback_weight*fallback
            objective.backward(inputs=self.parameters)
        if any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in self.parameters):
            raise ValueError('Actual actor-only execution-RL/public-fallback gradients must be finite')
        norm=float(torch.nn.utils.clip_grad_norm_(self.parameters,1.,error_if_nonfinite=True));self.optimizer.step();self.optimizer.zero_grad(set_to_none=True)
        return dict(original_train_position=position,source_id=source_id,complete_prompt_ids=prefix,trajectories=trajectories,
            **metric,public_fallback_CE=float(fallback.detach()) if isinstance(fallback,torch.Tensor) else 0.,
            actual_cached_full_action_logp_max_absolute_differences=cache_differences,
            fallback_target_sha256=row['unchanged_retained_target_sha256'],public_fallback_weight=self.fallback_weight,
            gradient_norm_before_clip=norm,actual_counts=self.model.counts(),genuine_human_preference_RLHF=False,
            same_counted_dense_backbone_is_frozen_reference=True,external_teacher_or_extra_reward_encoder=False,goal_achieved=False)
