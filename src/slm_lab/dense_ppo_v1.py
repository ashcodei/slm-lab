"""Own-response, token-weighted PPO on the counted dense actor and critic.

Sampling uses the unmodified full-vocabulary policy at temperature1. A response
budget ends the explicitly finite-horizon text episode. A scoring-only EOS is
added for the reward encoder when necessary; it is never a fabricated action.
This kernel implements mechanics, not evidence of useful policy alignment.
"""
from contextlib import nullcontext
import hashlib
import json
import math

import torch
from torch.nn import functional as F

from .codenet_stage_v1 import EpochOrder
from .dense_human_preference_v1 import pair_information
from .dense_ppo_reward_v1 import FrozenGeneralReward,signature
from .objectives import generalized_advantages,normalize_rollout_advantages,ppo_loss
from .own_dense_actor_checkpoint_v1 import make_state,restore_state,selected
from .own_dense_actor_v1 import OwnDenseActor
from .whole_human_data_v2 import WholeHumanData
from .whole_human_sft_evidence_v1 import tensor_digest

FORMAT='own-dense-own-response-human-reward-PPO-actor-critic-v1'


def fingerprint(ids):
    return hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest()


def prompt_source(data,position):
    info=pair_information(data,position);a,labels=data.sequence(position,'chosen');b,_=data.sequence(position,'rejected')
    start=data.rows[position]['native']['chosen']['target_start'];other=data.rows[position]['native']['rejected']['target_start']
    if start!=other or a[:start]!=b[:other] or labels[:start]!=[-100]*start or start<1:
        raise ValueError('The complete original shared train-only prompt must reconstruct without changing its frame')
    return dict(original_train_position=position,id=info['id'],source_record_sha256=info['source_record_sha256'],
        whole_original_pair_fingerprint=info['whole_token_source_fingerprint'],prompt_ids=a[:start],
        prompt_tokens_sha256=fingerprint(a[:start]),original_human_ranked_responses_not_used_as_actions=True)


def action_outputs(model,ids,prompt_length,amp):
    with amp():out=model(ids)
    logits=out['logits'][0,prompt_length-1:-1].float();actions=ids[0,prompt_length:]
    logps=logits.log_softmax(-1).gather(1,actions[:,None]).squeeze(1)
    return logps,out['values'][0,prompt_length-1:-1].float()


def validate_rollouts(data,positions,setting,rows,reward_provenance,vocab,max_length):
    if len(rows)!=len(positions):raise ValueError('Every sampled source prompt and complete bounded response must be retained')
    reconstructed=[]
    for p,row in zip(positions,rows):
        wanted=prompt_source(data,p)
        if any(row.get(k)!=v for k,v in wanted.items()):raise ValueError('Every whole original prompt/source must reconstruct')
        actions=row['response_ids'];prompt=row['prompt_ids'];eos=bool(actions and actions[-1]==2)
        if (not actions or len(actions)>setting['max_new_tokens'] or any(type(t) is not int or not 0<=t<vocab for t in actions)
                or 2 in actions[:-1] or (not eos and len(actions)!=setting['max_new_tokens'])
                or len(prompt)+setting['max_new_tokens']+1>max_length
                or row['termination']!=('EOS' if eos else 'response_budget')
                or row['finite_horizon_terminal'] is not True or row['bootstrap_value']!=0.
                or row['reward_provenance']!=reward_provenance
                or row['scoring_ids']!=prompt+actions+([] if eos else [2])
                or row['scoring_only_EOS_added'] is not (not eos)
                or row['temperature']!=1. or row['full_vocabulary_sampling'] is not True):
            raise ValueError('Actual on-policy actions, explicit finite horizon and distinct scoring-only EOS required')
        tensors={k:torch.tensor(row[k],dtype=torch.float32) for k in ('old_logps','reference_logps','old_values','token_rewards','raw_advantages','returns','advantages')}
        if (any(t.shape!=(len(actions),) or not bool(torch.isfinite(t).all()) for t in tensors.values())
                or not math.isfinite(row['human_reward_score']) or bool((tensors['old_logps']>0).any())
                or bool((tensors['reference_logps']>0).any())):
            raise ValueError('Every actual finite response log probability, critic, reward and advantage must be retained')
        if not torch.equal(torch.tensor(row['actual_sampling_logps'],dtype=torch.float32),tensors['old_logps']):
            raise ValueError('PPO behavior probabilities must be the actual unmodified sampling probabilities')
        reward=-setting['KL_coefficient']*(tensors['old_logps']-tensors['reference_logps']);reward[-1]+=row['human_reward_score']
        advantage,returns=generalized_advantages(reward,tensors['old_values'],setting['gamma'],setting['GAE_lambda'])
        if (not torch.equal(reward,tensors['token_rewards']) or not torch.equal(advantage,tensors['raw_advantages'])
                or not torch.equal(returns,tensors['returns'])):
            raise ValueError('All sampled KL rewards, finite-horizon GAE and value targets must reconstruct exactly')
        reconstructed.append(dict(advantages=advantage))
    normalize_rollout_advantages(reconstructed)
    if any(not torch.equal(r['advantages'],torch.tensor(row['advantages'],dtype=torch.float32)) for r,row in zip(reconstructed,rows)):
        raise ValueError('Advantages must be normalized jointly across actual response tokens')


class DensePPOStage:
    def __init__(self,model,optimizer,data,reward,options,sources,device='cpu',amp=None):
        keys={'seed','steps','prompts_per_update','max_new_tokens','learning_rate','betas','weight_decay','gradient_clip',
            'KL_coefficient','gamma','GAE_lambda','clip','value_coefficient'}
        if (type(model) is not OwnDenseActor or type(data) is not WholeHumanData
                or (data.family,data.role,data.purpose)!=('preferences','train','optimization')
                or type(reward) is not FrozenGeneralReward or set(options)!=keys
                or any(type(options[k]) is not int or options[k]<1 for k in ('seed','steps','prompts_per_update','max_new_tokens'))
                or options['prompts_per_update']<2
                or any(not math.isfinite(options[k]) or options[k]<=0 for k in ('learning_rate','gradient_clip','value_coefficient'))
                or not 0<options['clip']<1 or not 0<=options['gamma']<=1 or not 0<=options['GAE_lambda']<=1
                or not math.isfinite(options['KL_coefficient']) or options['KL_coefficient']<0
                or not math.isfinite(options['weight_decay']) or options['weight_decay']<0
                or len(options['betas'])!=2 or any(not 0<=v<1 for v in options['betas'])):
            raise ValueError('Exact own dense policy/critic, original train-only prompts and explicit bounded PPO options required')
        named=selected(model,optimizer,('actor','critic'));group=optimizer.param_groups[0]
        if group['lr']!=options['learning_rate'] or tuple(group['betas'])!=tuple(options['betas']) or group['weight_decay']!=options['weight_decay']:
            raise ValueError('The actual ordinary actor/critic Adam must match the fixed declared settings')
        if model.config.vocab_size!=reward.encoder.config.vocab_size:
            raise ValueError('Policy and original reward must use the same complete own native vocabulary')
        ids={id(p) for _,p in named}
        if ids&{id(p) for p in reward.parameters()} or {id(p) for p in model.parameters()}&{id(p) for p in reward.parameters()}:
            raise ValueError('Training-only original reward must be separate from all changing policy and critic parameters')
        reward.check();self.model,self.optimizer,self.data,self.reward=model,optimizer,data,reward
        self.options,self.sources,self.device,self.amp=dict(options),dict(sources),torch.device(device),amp or nullcontext
        self.parameters=[p for _,p in named];self.fixed=[p for p in model.parameters() if id(p) not in ids]
        self.fixed_signature=signature(self.fixed);self.frozen_digest=tensor_digest(model.base.named_parameters())
        self.order=EpochOrder(len(data),options['seed']);self.completed=0;self.metrics=[]

    def positions(self,step):
        if type(step) is not int or not 0<=step<self.options['steps']:raise ValueError('Bounded actual PPO rollout update required')
        width=self.options['prompts_per_update'];return [self.order.at(step*width+i) for i in range(width)]

    def cursor(self,completed):
        draws=completed*self.options['prompts_per_update']
        return dict(completed_updates=completed,original_prompt_population=len(self.data),actual_prompt_draws=draws,
            complete_prompt_passes=draws//len(self.data),next_pass_prompts=draws%len(self.data),seed=self.options['seed'],world_size=1)

    @torch.no_grad()
    def collect(self,positions):
        self.model.eval();self.reward.check();rows=[]
        max_length=min(self.model.config.max_text_length,self.reward.encoder.config.max_text_length)
        # Validate all prompts before consuming any sampling RNG.
        sources=[prompt_source(self.data,p) for p in positions]
        if any(len(r['prompt_ids'])+self.options['max_new_tokens']+1>max_length for r in sources):
            raise ValueError('Whole original prompts and the full declared action/scoring budget must fit without truncation')
        for source in sources:
            prompt=source['prompt_ids'];ids=torch.tensor([prompt],device=self.device);actions=[];sampled=[]
            for _ in range(self.options['max_new_tokens']):
                with self.amp():logits=self.model(ids)['logits'][0,-1].float()
                probabilities=logits.softmax(-1);token=int(torch.multinomial(probabilities,1));sampled.append(float(logits.log_softmax(-1)[token]))
                actions.append(token);ids=torch.cat((ids,torch.tensor([[token]],device=self.device)),1)
                if token==2:break
            old_logps,old_values=action_outputs(self.model,ids,len(prompt),self.amp)
            if not torch.allclose(old_logps,torch.tensor(sampled,device=self.device),atol=1e-4,rtol=1e-5):
                raise ValueError('Full causal action scores must match the actual unmodified sampling distribution')
            old_logps=torch.tensor(sampled,dtype=torch.float32,device=self.device)
            reference,_=action_outputs(self.model.base,ids,len(prompt),self.amp)
            scoring=prompt+actions+([] if actions[-1]==2 else [2])
            score=float(self.reward.score(torch.tensor([scoring],device=self.device),self.amp))
            # Construct reward/GAE on CPU FP32, identically in training and
            # checkpoint readback, independent of CUDA reduction ordering.
            old_logps,old_values,reference=old_logps.cpu(),old_values.cpu(),reference.cpu()
            rewards=-self.options['KL_coefficient']*(old_logps-reference);rewards[-1]+=score
            advantage,returns=generalized_advantages(rewards,old_values,self.options['gamma'],self.options['GAE_lambda'])
            rows.append(dict(**source,response_ids=actions,actual_sampling_logps=sampled,old_logps=old_logps.tolist(),
                reference_logps=reference.tolist(),old_values=old_values.tolist(),token_rewards=rewards.tolist(),
                raw_advantages=advantage.tolist(),returns=returns.tolist(),advantages=advantage,
                human_reward_score=score,reward_provenance=self.reward.provenance,scoring_ids=scoring,
                scoring_only_EOS_added=actions[-1]!=2,termination='EOS' if actions[-1]==2 else 'response_budget',
                finite_horizon_terminal=True,bootstrap_value=0.,temperature=1.,full_vocabulary_sampling=True))
        normalize_rollout_advantages(rows)
        for row in rows:row['advantages']=row['advantages'].tolist()
        validate_rollouts(self.data,positions,self.options,rows,self.reward.provenance,self.model.config.vocab_size,max_length)
        return rows

    def update(self):
        positions=self.positions(self.completed)
        if signature(self.fixed)!=self.fixed_signature:raise ValueError('Every counted frozen base/head must remain fixed')
        self.optimizer.zero_grad(set_to_none=True);rows=self.collect(positions);total=sum(len(r['response_ids']) for r in rows)
        self.model.train();self.model.policy.gradient_checkpointing=True;losses=[]
        try:
            for row in rows:
                ids=torch.tensor([row['prompt_ids']+row['response_ids']],device=self.device)
                logps,values=action_outputs(self.model,ids,len(row['prompt_ids']),self.amp)
                old,advantage,old_value,returns=[torch.tensor(row[k],device=self.device) for k in ('old_logps','advantages','old_values','returns')]
                loss=ppo_loss(logps,old,advantage,values,old_value,returns,self.options['clip'],self.options['value_coefficient'])*len(logps)/total
                if not bool(torch.isfinite(loss)):raise ValueError('The actual token-weighted actor/critic PPO loss must be finite')
                loss.backward(inputs=self.parameters);losses.append(float(loss.detach()))
            if any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in self.parameters) or any(p.grad is not None for p in self.fixed):
                raise ValueError('Every actor/critic and no frozen base/head parameter needs finite gradients')
            self.reward.check();norm=float(torch.nn.utils.clip_grad_norm_(self.parameters,self.options['gradient_clip'],error_if_nonfinite=True))
            self.optimizer.step();self.optimizer.zero_grad(set_to_none=True)
        except BaseException:
            self.optimizer.zero_grad(set_to_none=True);raise
        record=dict(step=self.completed+1,source_positions=positions,rollouts=rows,actual_action_tokens=total,
            actual_complete_prompt_tokens=sum(len(r['prompt_ids']) for r in rows),token_weighted_PPO_loss=math.fsum(losses),
            gradient_norm_before_clip=norm,learning_rate=self.options['learning_rate'],
            policy_unique_parameters=self.model.counts()['actual_unique'],separate_training_only_reward=self.reward.counts(),
            reward_model_not_in_policy_export=True,own_sampled_responses_not_new_human_judgments=True)
        self.completed+=1;self.metrics.append(record);return record

    def state(self):
        self.reward.check()
        if signature(self.fixed)!=self.fixed_signature or tensor_digest(self.model.base.named_parameters())!=self.frozen_digest:
            raise ValueError('Every frozen dense/vision/word/auxiliary value must remain exact')
        return dict(format=FORMAT,core=make_state(self.model,self.optimizer,groups=('actor','critic'),sources=self.sources,
            completed=self.completed,planned=self.options['steps'],cursor=self.cursor(self.completed),metrics=self.metrics,device=self.device),
            options=self.options,source_population=len(self.data),reward_provenance=self.reward.provenance,
            reward_counts=self.reward.counts(),candidate_promoted=False,goal_achieved=False)

    def resume(self,state,*,allow_completed=False):
        core=state.get('core',{});completed=core.get('completed');max_length=min(self.model.config.max_text_length,self.reward.encoder.config.max_text_length)
        if (state.get('format')!=FORMAT or state.get('options')!=self.options or state.get('source_population')!=len(self.data)
                or state.get('reward_provenance')!=self.reward.provenance or state.get('reward_counts')!=self.reward.counts()
                or type(completed) is not int or not 0<completed<=self.options['steps']
                or len(core.get('metrics',[]))!=completed or state.get('candidate_promoted') is not False or state.get('goal_achieved') is not False):
            raise ValueError('Exact whole-source PPO/critic, reward basis and bounded update state required before mutation')
        for step,record in enumerate(core['metrics']):
            positions=self.positions(step);rows=record['rollouts']
            validate_rollouts(self.data,positions,self.options,rows,self.reward.provenance,self.model.config.vocab_size,max_length)
            if (record['step']!=step+1 or record['source_positions']!=positions or record['actual_action_tokens']!=sum(len(r['response_ids']) for r in rows)
                    or record['actual_complete_prompt_tokens']!=sum(len(r['prompt_ids']) for r in rows)
                    or record['learning_rate']!=self.options['learning_rate'] or not math.isfinite(record['token_weighted_PPO_loss'])
                    or not math.isfinite(record['gradient_norm_before_clip']) or record['gradient_norm_before_clip']<0
                    or record['policy_unique_parameters']!=self.model.counts()['actual_unique']
                    or record['separate_training_only_reward']!=self.reward.counts()
                    or record['reward_model_not_in_policy_export'] is not True or record['own_sampled_responses_not_new_human_judgments'] is not True):
                raise ValueError('Every original on-policy trajectory and actual actor/critic metric must match')
        self.reward.check()
        self.completed,self.metrics=restore_state(self.model,self.optimizer,core,groups=('actor','critic'),expected_sources=self.sources,
            planned=self.options['steps'],expected_cursor=self.cursor(completed),device=self.device,allow_completed=allow_completed)
