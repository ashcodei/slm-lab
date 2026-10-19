"""Source-bound Adam/RNG boundaries around admitted reasoning kernels.

Capacity mode may use a declared small own-neural graph and explicit short CPU
horizons. Training mode requires the complete original source schedule and
actual existing984Mbase/999Mactor graph. A production caller still owns full-size
capacity, resource admission, persistence and independent native assessment.
"""
import copy
import hashlib
import json
import math

import torch

from .execution_group_controlled_v2 import advantages
from .frozen_own_actor_reference_v1 import FrozenOwnActorReference,signature
from .own_dense_actor_checkpoint_v1 import make_state,restore_state,selected
from .public_reasoning_group_gradients_v1 import FORMAT as KERNEL,warmup_gradients,group_gradients
from .public_reasoning_train_data_v1 import PublicReasoningTrainData

FORMAT='own-public-reasoning-whole-source-selected-Adam-RNG-stage-v1'
PROFILES=('whole_answer_warmup','anchor_CE','anchor_RL')


def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


class PublicReasoningStage:
    def __init__(self,model,optimizer,data,entries,profile,options,sources,*,mode='train',cache=None):
        if (type(data) is not PublicReasoningTrainData or profile not in PROFILES or mode not in ('train','capacity')
                or not isinstance(entries,list) or not entries or optimizer.state
                or options.get('precision') not in ('CPU_FP32','CPU_BF16_cached')
                or not sources or any(not isinstance(v,str) or len(v)!=64 or any(c not in '0123456789abcdef' for c in v) for v in sources.values())):
            raise ValueError('Exact admitted data, fresh Adam, source batches, profile/precision and SHA bindings required')
        self.named=selected(model,optimizer,('actor',));self.parameters=[p for _,p in self.named]
        if any(type(options.get(key)) not in (float,int) or not math.isfinite(options[key]) or options[key]<=0 for key in ['learning_rate','gradient_clip']):
            raise ValueError('Positive finite original learning rate and gradient clip required')
        if options['learning_rate']!=optimizer.param_groups[0]['lr'] or next(model.parameters()).device.type!='cpu':
            raise ValueError('This version binds the original CPU Adam learning rate and CPU graph')
        if profile!='whole_answer_warmup' and (options['precision']!='CPU_FP32' or cache is not None or any(len(entry)!=1 for entry in entries)):
            raise ValueError('Verifier groups require one whole source and original CPU FP32')
        if options['precision']=='CPU_BF16_cached' and cache is None:raise ValueError('The declared training-only cast cache must actually be attached')
        if mode=='train':
            expected=data.complete_schedule(epochs=options['epochs'],width=options['width'],seed=options['seed'])
            if (entries!=expected or model.base.counts()['total_unique']!=984399690 or model.counts()['actual_unique']!=999706442
                    or options.get('CPU_mechanics_horizon') is not None):
                raise ValueError('Production requires the complete source epochs, actual full-size graph and full native action horizons')
        data.budget(entries)
        self.model,self.optimizer,self.data=model,optimizer,data
        self.entries,self.profile,self.options,self.sources,self.mode=copy.deepcopy(entries),profile,copy.deepcopy(options),dict(sources),mode
        self.cache=cache;identities={id(p) for p in self.parameters};self.fixed=[p for p in model.parameters() if id(p) not in identities]
        self.fixed_signature=signature(self.fixed)
        self.reference=FrozenOwnActorReference(model,sources=sources) if profile!='whole_answer_warmup' else None
        self.reference_binding=None if self.reference is None else self.reference.binding()
        self.schedule_identity=data.identity(entries)
        self.expected=[[data.information(position) for position in entry] for entry in entries]
        self.initial_binding=copy.deepcopy(self.binding());self.completed=0;self.metrics=[];self.failed=False

    def binding(self):
        return dict(format=FORMAT,profile=self.profile,mode=self.mode,options=self.options,sources=self.sources,
            schedule_identity_sha256=self.schedule_identity,complete_source_batches_sha256=digest(self.entries),reference=self.reference_binding)

    def intact(self):
        if self.binding()!=self.initial_binding or signature(self.fixed)!=self.fixed_signature:
            raise ValueError('Original options/sources/schedule/reference and fixed neural values must remain unchanged')
        if self.reference is not None:self.reference.assert_bound(self.model)
        if self.cache is not None:self.cache.assert_current()

    def validate_anchor(self,row,expected):
        target_count=sum(value['response_EOS_targets'] for value in expected)
        if (row.get('format')!=KERNEL or row.get('profile')!='whole_answer_warmup'
                or row.get('complete_sources')!=expected or row.get('response_EOS_targets')!=target_count
                or row.get('all_whole_question_answer_and_true_EOS_masks_preserved') is not True
                or row.get('neural_rollouts')!=0 or row.get('externally_generated_teacher_or_human_preference') is not False
                or any(type(row.get(key)) not in (float,int) or not math.isfinite(row[key]) or row[key]<0 for key in ['original_CE_sum','original_token_mean_NLL'])
                or row['original_token_mean_NLL']!=row['original_CE_sum']/target_count):
            raise ValueError('Every original complete public answer target and finite CE aggregate must reconstruct')

    def validate(self,rows,*,start=0):
        if (not isinstance(rows,list) or type(start) is not int or start<0 or not rows
                or start+len(rows)>len(self.entries)):
            raise ValueError('Exact nonempty original complete-update range required')
        for index,row in enumerate(rows,start=start):
            if (row.get('step')!=index+1 or row.get('learning_rate')!=self.options['learning_rate']
                    or type(row.get('gradient_norm_before_clip')) not in (float,int)
                    or not math.isfinite(row['gradient_norm_before_clip']) or row['gradient_norm_before_clip']<0):
                raise ValueError('Every actual source update, learning rate and finite gradient norm required')
            expected=self.expected[index]
            if self.profile=='whole_answer_warmup':self.validate_anchor(row,expected);continue
            info=expected[0];position=info['position'];ids,labels=self.data.sequence(position);prefix=ids[:info['original_prompt_positions']]
            horizon=self.options.get('CPU_mechanics_horizon') or info['full_training_action_budget']
            if (row.get('format')!=KERNEL or row.get('profile')!=self.profile or row.get('complete_source')!=info
                    or row.get('complete_prompt_ids')!=prefix or row.get('group_size')!=self.options['group_size']
                    or row.get('KL_weight')!=self.options['KL_weight'] or row.get('anchor_weight')!=self.options['anchor_weight']
                    or row.get('cache_absolute_bound')!=self.options['cache_absolute_bound']
                    or row.get('CPU_mechanics_horizon')!=self.options.get('CPU_mechanics_horizon')
                    or row.get('own_reference_binding')!=self.reference_binding
                    or row.get('resident_inventory')!=self.reference.resident_inventory(self.model)
                    or row.get('original_reference_used_regardless_of_rewards') is not True
                    or row.get('native_model_outputs_and_sampled_tokens_unchanged') is not True
                    or row.get('human_feedback') is not False or row.get('external_neural_teacher') is not False):
                raise ValueError('Exact original public task, whole generation budget and own-reference objective required')
            trajectories=row.get('trajectories')
            if not isinstance(trajectories,list) or len(trajectories)!=self.options['group_size']:raise ValueError('Every actual fresh own-model sample must be retained')
            rewards=[];actions=0
            for sample,item in enumerate(trajectories):
                emitted=item.get('response_ids');score=self.data.score(position,emitted,horizon=horizon)
                cached=item.get('actual_cached_behavior_logps');full=item.get('actual_full_behavior_logps')
                if (item.get('local_sample')!=sample or item.get('actual_verifier')!=score or item.get('temperature')!=1.
                        or item.get('full_vocabulary_sampling') is not True or not isinstance(cached,list) or not isinstance(full,list)
                        or len(cached)!=len(emitted) or len(full)!=len(emitted)
                        or any(type(value) not in (float,int) or not math.isfinite(value) or value>0 for value in cached+full)):
                    raise ValueError('Every actual original action, strict reward and behavior probability must reconstruct')
                difference=max(abs(a-b) for a,b in zip(cached,full))
                if difference!=item.get('cached_full_max_absolute_difference') or difference>self.options['cache_absolute_bound']:
                    raise ValueError('Actual cached/full behavior must remain within the original bound')
                rewards.append(score['reward']);actions+=len(emitted)
            if (row.get('global_rewards')!=rewards or row.get('global_advantages')!=advantages(rewards)
                    or row.get('actual_sampled_action_tokens')!=actions
                    or row.get('identical_reward_group_has_zero_policy_advantages')!=(len(set(rewards))==1)
                    or any(type(row.get(key)) not in (float,int) or not math.isfinite(row[key]) for key in ['policy_surrogate','reference_KL'])):
                raise ValueError('Every complete reward group, advantage and actual action denominator must reconcile')
            self.validate_anchor(row.get('anchor',{}),expected)

    def update(self):
        if self.failed or self.completed>=len(self.entries):raise ValueError('Only a healthy remaining complete source update may run')
        self.intact();self.optimizer.zero_grad(set_to_none=True)
        try:
            if self.profile=='whole_answer_warmup':
                amp=lambda:torch.autocast('cpu',dtype=torch.bfloat16) if self.options['precision']=='CPU_BF16_cached' else torch.autocast('cpu',enabled=False)
                row=warmup_gradients(self.model,self.data,self.entries[self.completed],amp=amp)
            else:
                row=group_gradients(self.model,self.data,self.entries[self.completed][0],self.reference,profile=self.profile,
                    **{key:self.options[key] for key in ['group_size','KL_weight','anchor_weight','cache_absolute_bound']},
                    CPU_mechanics_horizon=self.options.get('CPU_mechanics_horizon'))
            norm=float(torch.nn.utils.clip_grad_norm_(self.parameters,self.options['gradient_clip'],error_if_nonfinite=True))
            row.update(step=self.completed+1,learning_rate=self.options['learning_rate'],gradient_norm_before_clip=norm)
            self.validate([row],start=self.completed);self.intact()
            self.optimizer.step();self.optimizer.zero_grad(set_to_none=True);self.intact()
        except BaseException:
            self.failed=True;self.optimizer.zero_grad(set_to_none=True);raise
        self.completed+=1;self.metrics.append(row)
        return row

    def cursor(self,completed):
        return dict(completed_updates=completed,planned_updates=len(self.entries),stage_binding_sha256=digest(self.initial_binding),
            actual_complete_source_batches=self.entries[:completed])

    def state(self):
        if self.failed or any(p.grad is not None for p in self.model.parameters()):raise ValueError('Only a healthy complete optimizer boundary can be saved')
        self.intact();self.validate(self.metrics)
        core=make_state(self.model,self.optimizer,groups=('actor',),sources=self.sources,completed=self.completed,
            planned=len(self.entries),cursor=self.cursor(self.completed),metrics=self.metrics,device='cpu')
        return dict(format=FORMAT,stage_binding=copy.deepcopy(self.initial_binding),core=core,candidate_promoted=False,goal_achieved=False)

    def resume(self,payload,*,allow_completed=False):
        self.intact()
        if (self.failed or payload.get('format')!=FORMAT or payload.get('stage_binding')!=self.initial_binding
                or payload.get('candidate_promoted') is not False or payload.get('goal_achieved') is not False):
            raise ValueError('Exact healthy original stage/options/source/reference binding required before recovery')
        core=payload.get('core',{});completed=core.get('completed')
        if type(completed) is not int or not 0<completed<=len(self.entries):raise ValueError('Exact actual complete source cursor required')
        self.validate(core.get('metrics'))
        count,metrics=restore_state(self.model,self.optimizer,core,groups=('actor',),expected_sources=self.sources,
            planned=len(self.entries),expected_cursor=self.cursor(completed),device='cpu',allow_completed=allow_completed)
        self.completed,self.metrics=count,metrics
        self.intact()
