"""One complete public human-conversation epoch on own language extensions.

Independent world-one learning-rate arms use identical original source order.
Every complete response/EOS token has equal weight within an optimizer update.
Original/code/vision/grounding/reward/critic/word weights remain outside Adam.
"""
from contextlib import nullcontext
import math

import torch

from .codenet_checkpoint_v1 import atomic_checkpoint
from .codenet_runner_v1 import FixedReductionModel
from .codenet_stage_v1 import EpochOrder
from .own_aligned_assistant_v2 import OwnAlignedAssistantV2
from .own_alignment_checkpoint_v1 import make_state, restore_state, parameters
from .training import supervised_gradients

FORMAT = 'whole-original-human-dialogue-SFT-resume-v1'


def make_optimizer(model, options):
    return torch.optim.AdamW(model.extension_parameters('language'), lr=options['learning_rate'],
        betas=tuple(options['betas']), weight_decay=options['weight_decay'], foreach=False)


class _LanguageRestoreView:
    """Use the complete validator while never copying unused extension tensors."""
    def __init__(self, model):
        self.model=model

    def __getattr__(self, name):
        return getattr(self.model,name)

    def validate_export(self, payload, *, expected_sources):
        self.model.validate_export(payload,expected_sources=expected_sources)
        live=self.model.parts.state_dict()
        if any(not torch.equal(t.detach().cpu(),payload['extensions'][name].detach().cpu())
               for name,t in live.items() if not name.startswith(('language_linears.','language_norms.'))):
            raise ValueError('Every unused vision/reward/critic/word extension must remain fixed before resume')

    def import_export(self, payload, *, expected_sources):
        self.validate_export(payload,expected_sources=expected_sources)
        with torch.no_grad():
            for name,parameter in self.model.parts.named_parameters():
                if name.startswith(('language_linears.','language_norms.')):
                    parameter.copy_(payload['extensions'][name])


class WholeHumanSFTStage:
    def __init__(self, model, optimizer, data, options, sources, device, amp=None):
        if type(model) is not OwnAlignedAssistantV2 or (data.family,data.role,data.purpose)!=('dialogue','train','optimization'):
            raise ValueError('Exact own v2 assistant and admitted original human dialogue optimization role required')
        expected={'seed','steps','microbatch','accumulation','learning_rate','betas','weight_decay','gradient_clip'}
        if set(options)!=expected or any(type(options[k]) is not int or options[k]<1 for k in ('seed','steps','microbatch','accumulation')):
            raise ValueError('Exact positive complete-source fixed SFT options required')
        if (not len(data) or options['steps']!=math.ceil(len(data)/(options['microbatch']*options['accumulation']))
                or any(type(options[k]) not in (float,int) or not math.isfinite(options[k]) or options[k]<=0
                       for k in ('learning_rate','weight_decay','gradient_clip'))
                or len(options['betas'])!=2 or any(not 0<=v<1 for v in options['betas'])):
            raise ValueError('One complete actual source pass, explicit rounding and finite Adam options required')
        self.model,self.optimizer,self.data=model,optimizer,data
        self.options,self.sources=dict(options),dict(sources)
        self.device=torch.device(device);self.amp=amp or nullcontext
        self.parameters=parameters(model,optimizer,('language',))
        group=optimizer.param_groups[0]
        if (group['lr']!=options['learning_rate'] or tuple(group['betas'])!=tuple(options['betas']) or group['weight_decay']!=options['weight_decay']):
            raise ValueError('Exact predeclared ordinary language-only AdamW hyperparameters required')
        selected={id(p) for p in self.parameters}
        self.fixed=[p for p in model.parameters() if id(p) not in selected]
        self.order=EpochOrder(len(data),options['seed']);self.completed=0;self.metrics=[]
        self.network=FixedReductionModel(model.aligned)

    def positions(self, step):
        if type(step) is not int or not 0<=step<self.options['steps']:
            raise ValueError('Exact bounded original source update required')
        return self.order.rank_groups(step,0,1,self.options['microbatch'],self.options['accumulation'])

    def cursor(self, completed):
        width=self.options['microbatch']*self.options['accumulation'];draws=completed*width
        return dict(source_population=len(self.data),seed=self.options['seed'],completed_draws=draws,
            complete_passes=draws//len(self.data),next_pass_remainder=draws%len(self.data),
            completed_updates=completed,world_size=1,microbatch=self.options['microbatch'],accumulation=self.options['accumulation'])

    def update(self):
        groups=self.positions(self.completed)
        positions=[p for group in groups for p in group]
        context=sum(self.data.rows[p]['native']['response']['length'] for p in positions)
        expected=sum(self.data.rows[p]['native']['response']['length']-self.data.rows[p]['native']['response']['target_start'] for p in positions)
        self.model.assert_word_contract();self.model.aligned.train()
        self.optimizer.zero_grad(set_to_none=True)
        try:
            loss,count=supervised_gradients(self.network,self.data,groups,'response',self.device,self.amp,world=1)
            if (expected!=count or any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in self.parameters)
                    or any(p.grad is not None for p in self.fixed)):
                raise ValueError('Every and only new own language tensor must receive finite source gradients')
            norm=float(torch.nn.utils.clip_grad_norm_(self.parameters,self.options['gradient_clip'],error_if_nonfinite=True))
            self.optimizer.step();self.optimizer.zero_grad(set_to_none=True)
        except BaseException:
            self.optimizer.zero_grad(set_to_none=True);raise
        self.completed+=1
        record=dict(step=self.completed,token_mean_nll=float(loss.detach()),response_EOS_targets=count,
            complete_context_tokens=context,source_positions=positions,learning_rate=self.options['learning_rate'],gradient_norm_before_clip=norm)
        self.metrics.append(record);return record

    def state(self):
        core=make_state(self.model,self.optimizer,groups=('language',),sources=self.sources,
            completed_steps=self.completed,planned_steps=self.options['steps'],sampler_state=self.cursor(self.completed),
            metrics=self.metrics,device=self.device)
        core['model']['extension_training_not_established']=False
        return dict(format=FORMAT,core=core,options=self.options,source_population=len(self.data),
            actual_human_SFT_updates=self.completed,candidate_promoted=False,goal_achieved=False)

    def save(self, path, maximum_bytes, minimum_free_bytes):
        atomic_checkpoint(path,self.state(),maximum_bytes,minimum_free_bytes)

    def resume(self, state, *, allow_completed=False):
        if (state.get('format')!=FORMAT or state.get('options')!=self.options or state.get('source_population')!=len(self.data)
                or state.get('candidate_promoted') is not False or state.get('goal_achieved') is not False
                or state.get('actual_human_SFT_updates')!=state.get('core',{}).get('steps')):
            raise ValueError('Exact whole-human SFT source/options/update contract required before mutation')
        core=state['core'];steps=core.get('steps')
        if type(steps) is not int or not 0<steps<=self.options['steps'] or len(core.get('metrics',[]))!=steps:
            raise ValueError('Complete original update budget and recorded source metrics required')
        for step,metric in enumerate(core['metrics']):
            positions=[p for group in self.positions(step) for p in group]
            expected_targets=sum(self.data.rows[p]['native']['response']['length']-self.data.rows[p]['native']['response']['target_start'] for p in positions)
            expected_context=sum(self.data.rows[p]['native']['response']['length'] for p in positions)
            if (metric.get('step')!=step+1 or metric.get('source_positions')!=positions or metric.get('response_EOS_targets')!=expected_targets
                    or metric.get('complete_context_tokens')!=expected_context or metric.get('learning_rate')!=self.options['learning_rate']
                    or any(type(metric.get(k)) not in (float,int) or not math.isfinite(metric[k])
                           for k in ('token_mean_nll','gradient_norm_before_clip'))):
                raise ValueError('Every saved original source draw and full-token loss normalizer must reconstruct')
        completed,metrics=restore_state(_LanguageRestoreView(self.model),self.optimizer,core,groups=('language',),expected_sources=self.sources,
            planned_steps=self.options['steps'],expected_sampler_state=self.cursor(steps),device=self.device,allow_completed=allow_completed)
        self.completed,self.metrics=completed,metrics
