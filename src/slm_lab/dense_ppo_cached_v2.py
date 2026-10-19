"""FP32 cached own-response PPO, with measured full-causal probability drift.

The saved behavior probabilities come from the actual cached sampler. A full
causal pass provides old critic values and a tightly bounded numerical check;
it never replaces behavior probabilities with a different distribution.
"""
from contextlib import nullcontext
import torch

from .dense_ppo_v1 import DensePPOStage,FORMAT as ORIGINAL_FORMAT,action_outputs,prompt_source,validate_rollouts
from .objectives import generalized_advantages,normalize_rollout_advantages

FORMAT='own-dense-FP32-cached-response-human-reward-PPO-v2'
SAMPLER=dict(kind='one_prefill_then_one_token_decode',precision='float32',temperature=1.,
    probability_atol=1e-4,probability_rtol=1e-5,behavior_logps_from_actual_cache=True)


def validate_cached(rows):
    for row in rows:
        actual=torch.tensor(row['actual_sampling_logps'],dtype=torch.float32)
        full=torch.tensor(row['full_causal_logps'],dtype=torch.float32)
        delta=float((actual-full).abs().max()) if actual.numel() and actual.shape==full.shape else float('inf')
        if (row.get('sampler')!=SAMPLER or row.get('actual_prefill_calls')!=1
                or row.get('actual_decode_calls')!=len(row['response_ids'])-1
                or row.get('max_absolute_cache_full_logp_difference')!=delta
                or not torch.allclose(actual,full,atol=SAMPLER['probability_atol'],rtol=SAMPLER['probability_rtol'])):
            raise ValueError('Actual FP32 cache calls and tightly matched full-causal behavior probabilities required')


class DenseCachedPPOStage(DensePPOStage):
    def __init__(self,model,optimizer,data,reward,options,sources,device='cpu',amp=None):
        if amp not in (None,nullcontext) or any(p.dtype!=torch.float32 for p in [*model.parameters(),*reward.parameters()]):
            raise ValueError('This cached PPO computation is explicitly FP32, including its training-only reward')
        super().__init__(model,optimizer,data,reward,options,sources,device,None)

    @torch.no_grad()
    def collect(self,positions):
        with torch.autocast(device_type=self.device.type,enabled=False):return self._collect(positions)

    def _collect(self,positions):
        self.model.eval();self.reward.check();rows=[]
        max_length=min(self.model.config.max_text_length,self.reward.encoder.config.max_text_length)
        sources=[prompt_source(self.data,p) for p in positions]
        if any(len(r['prompt_ids'])+self.options['max_new_tokens']+1>max_length for r in sources):
            raise ValueError('Whole original prompts and complete action/scoring budget must fit without truncation')
        for source in sources:
            prompt=source['prompt_ids'];ids=torch.tensor([prompt],device=self.device);actions=[];sampled=[]
            state=self.model.policy.prefill(ids);decode_calls=0
            for step in range(self.options['max_new_tokens']):
                logits=state['logits'][0,-1].float();token=int(torch.multinomial(logits.softmax(-1),1))
                sampled.append(float(logits.log_softmax(-1)[token]));actions.append(token)
                if token==2 or step+1==self.options['max_new_tokens']:break
                state=self.model.policy.decode(torch.tensor([[token]],device=self.device),state['cache']);decode_calls+=1
            del state
            ids=torch.tensor([prompt+actions],device=self.device)
            full,old_values=action_outputs(self.model,ids,len(prompt),self.amp)
            reference,_=action_outputs(self.model.base,ids,len(prompt),self.amp)
            scoring=prompt+actions+([] if actions[-1]==2 else [2])
            score=float(self.reward.score(torch.tensor([scoring],device=self.device)))
            old=torch.tensor(sampled,dtype=torch.float32);reference=reference.cpu();old_values=old_values.cpu();full=full.cpu()
            rewards=-self.options['KL_coefficient']*(old-reference);rewards[-1]+=score
            advantages,returns=generalized_advantages(rewards,old_values,self.options['gamma'],self.options['GAE_lambda'])
            rows.append(dict(**source,response_ids=actions,actual_sampling_logps=sampled,old_logps=sampled,
                reference_logps=reference.tolist(),old_values=old_values.tolist(),token_rewards=rewards.tolist(),
                raw_advantages=advantages.tolist(),returns=returns.tolist(),advantages=advantages,
                human_reward_score=score,reward_provenance=self.reward.provenance,scoring_ids=scoring,
                scoring_only_EOS_added=actions[-1]!=2,termination='EOS' if actions[-1]==2 else 'response_budget',
                finite_horizon_terminal=True,bootstrap_value=0.,temperature=1.,full_vocabulary_sampling=True,
                sampler=SAMPLER,actual_prefill_calls=1,actual_decode_calls=decode_calls,full_causal_logps=full.tolist(),
                max_absolute_cache_full_logp_difference=float((old-full).abs().max())))
        normalize_rollout_advantages(rows)
        for row in rows:row['advantages']=row['advantages'].tolist()
        validate_rollouts(self.data,positions,self.options,rows,self.reward.provenance,self.model.config.vocab_size,max_length)
        validate_cached(rows);return rows

    def update(self):
        with torch.autocast(device_type=self.device.type,enabled=False):return super().update()

    def state(self):return {**super().state(),'format':FORMAT,'sampler':dict(SAMPLER)}

    def resume(self,state,*,allow_completed=False):
        if state.get('format')!=FORMAT or state.get('sampler')!=SAMPLER:
            raise ValueError('The exact FP32 cached PPO sampler and recovery format required before mutation')
        for record in state.get('core',{}).get('metrics',[]):validate_cached(record['rollouts'])
        return super().resume({**state,'format':ORIGINAL_FORMAT},allow_completed=allow_completed)
