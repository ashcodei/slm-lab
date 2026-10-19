"""One canonical joint code/human/visual optimizer boundary, no job admission.

Source ownership, the entire fixed curriculum, bounded durable history and
full-size GPU capacity/controller gates belong to an owned training driver.
This integration primitive verifies replicas once, guards metadata thereafter,
reduces each complete task before mixing, and performs exactly one AdamW call.
"""
from contextlib import nullcontext
import copy
import math

import torch

from .code_priority_gradient_blend_v1 import combine
from .joint_code_task_gradients_v1 import gradients as code_gradients
from .joint_batched_human_preferences_v1 import gradients as human_gradients
from .joint_visual_task_gradients_v1 import native_gradients,word_gradients,photo_gradients
from .parallel_function_repair_stage_v1 import value_digest
from .shared_joint_checkpoint_v2 import FixedGuard,agree,control,gather,selected
from .shared_joint_gradient_reduce_v2 import reduce
from .whole_human_sft_evidence_v1 import tensor_digest

OPTIONS={'maximum_examples','maximum_padded_tokens','gradient_bucket_values','gradient_clip','human_weight','visual_weight',
    'CTC_weight','native_word_CE_weight','matching_weight','temperature','correction_weight','DPO_beta','chosen_CE_weight','project_conflicts','precision'}


class JointReplicaGuardV1:
    def __init__(self,view,optimizer,*,control_group=None):
        self.view,self.optimizer,self.group=view,optimizer,control_group
        _,world=control(control_group);error=identity=None
        try:
            selected(view,optimizer);self.fixed=FixedGuard(view,optimizer)
            if any(p.grad is not None for p in view.parameters()):raise ValueError('A complete fresh or independently restored optimizer boundary is required')
            identity=value_digest(dict(counts=view.actor.counts(),metadata=self.fixed.metadata,
                all_actual_parameters_sha256=tensor_digest(view.named_parameters()),Adam_sha256=value_digest(optimizer.state_dict())))
            self.signature=self.current()
        except Exception as exc:error=type(exc).__name__+': '+str(exc)
        agree(error,identity,control_group,world)

    def current(self):
        return ([(n,id(p),p._version,p.data_ptr(),p.requires_grad,str(p.dtype),str(p.device)) for n,p in self.view.named_parameters()],
            [(n,tuple((k,id(v),v._version,v.data_ptr()) for k,v in self.optimizer.state.get(p,{}).items())) for n,p in self.view.named_selected()],
            copy.deepcopy([{k:v for k,v in group.items() if k!='params'} for group in self.optimizer.param_groups]))

    def check(self):
        selected(self.view,self.optimizer);self.fixed.check(self.view,self.optimizer)
        if self.current()!=self.signature or any(p.grad is not None for p in self.view.parameters()):
            raise ValueError('Every own model/optimizer pointer, clock, metadata and clear-gradient boundary must remain guarded')

    def accept_own_update(self):
        self.fixed.check(self.view,self.optimizer);self.signature=self.current()


def update(view,reference,optimizer,guard,*,code=None,human=None,visual=None,options,source_entry_sha256,control_group=None,gradient_group=None):
    rank,world=control(control_group);error=identity=None
    try:
        if (type(guard) is not JointReplicaGuardV1 or guard.view is not view or guard.optimizer is not optimizer
                or guard.group is not control_group or not any(part is not None for part in [code,human,visual])
                or set(options)!=OPTIONS or options['precision'] not in ['fp32','bf16'] or type(options['project_conflicts']) is not bool
                or any(type(options[k]) is not int or options[k]<1 for k in ['maximum_examples','maximum_padded_tokens','gradient_bucket_values'])
                or any(type(options[k]) not in [int,float] or not math.isfinite(options[k]) or options[k]<=0
                    for k in OPTIONS-{'maximum_examples','maximum_padded_tokens','gradient_bucket_values','project_conflicts','precision'})
                or type(source_entry_sha256) is not str or len(source_entry_sha256)!=64):
            raise ValueError('One guarded joint optimizer, actual sources and finite complete fixed settings are required')
        guard.check();device=next(view.parameters()).device
        identity=value_digest(dict(source_entry_sha256=source_entry_sha256,options=options,
            code=None if code is None else dict(identity=code['data'].identity_sha256,positions=code['positions'],profile=code['profile']),
            human=None if human is None else dict(positions=human['positions'],DPO_weight=human['DPO_weight']),
            visual=None if visual is None else dict(kind=visual['kind'],source=visual['source'])))
    except Exception as exc:error=type(exc).__name__+': '+str(exc)
    agree(error,identity,control_group,world)
    amp=(lambda:torch.autocast(device.type,dtype=torch.bfloat16,cache_enabled=False)) if options['precision']=='bf16' else nullcontext
    bounds=dict(maximum_examples=options['maximum_examples'],maximum_padded_tokens=options['maximum_padded_tokens'],amp=amp)
    empty={n:None for n,_ in view.named_selected()};primary=empty.copy();tasks={};transports={};local_losses={}
    def summed(values,name,denominators):
        reduced,receipt=reduce(values,named_parameters=view.named_selected(),task_sha256=value_digest(dict(entry=source_entry_sha256,task=name,source=identity)),
            global_denominators=denominators,bucket_values=options['gradient_bucket_values'],control_group=control_group,gradient_group=gradient_group)
        transports[name]=receipt;return reduced
    try:
        if code is not None:
            values,record=code_gradients(view,code['data'],code['positions'],profile=code['profile'],group=control_group,**bounds)
            primary=summed(values,'code',{'positive_weighted_targets':record['original_global_weighted_targets']})
            tasks['code']=record;local_losses['code']=record['local_weighted_CE_sum']/record['original_global_weighted_targets']
        side=empty.copy()
        if human is not None:
            # Local human kernels have no collectives; coordinate a failure
            # before any peer enters its subsequent task gradient reduction.
            error=None
            try:
                values,record=human_gradients(view,reference if human['DPO_weight'] else None,human['data'],human['positions'],DPO_weight=human['DPO_weight'],
                    beta=options['DPO_beta'],chosen_CE_weight=options['chosen_CE_weight'],rank=rank,world=world,device=device,**bounds)
            except Exception as exc:error=type(exc).__name__+': '+str(exc)
            agree(error,None,control_group,world)
            agree(None,value_digest(dict(complete_human_source_sha256=record['complete_global_human_source_sha256'],
                chosen_targets=record['original_global_chosen_response_EOS_denominator'],pairs=record['original_global_pair_denominator'])),control_group,world)
            reduced=summed(values,'human',{'whole_human_pairs':record['original_global_pair_denominator'],
                'chosen_response_EOS_targets':record['original_global_chosen_response_EOS_denominator']})
            for name,value in reduced.items():
                if value is not None:side[name]=value*options['human_weight']
            tasks['human']=record;local_losses['human']=record['local_global_normalized_objective']
        if visual is not None:
            source=visual['source'];kind=visual['kind'];denominators={'native_response_EOS_targets':sum(source['response_EOS_targets'])}
            if kind=='photo':
                values,record=photo_gradients(view,visual['native'],visual['matching'],source,matching_weight=options['matching_weight'],temperature=options['temperature'],group=control_group,**bounds)
                denominators['image_pairs']=len(source['positions'])
                local=record['local_native_CE_sum']/denominators['native_response_EOS_targets']
            elif kind=='word':
                values,record=word_gradients(view,visual['native'],visual['word'],source,CTC_weight=options['CTC_weight'],native_CE_weight=options['native_word_CE_weight'],group=control_group,**bounds)
                denominators['whole_words']=len(source['positions'])
                local=options['native_word_CE_weight']*record['local_native_CE_sum']/denominators['native_response_EOS_targets']+options['CTC_weight']*record['local_byte_normalized_CTC_sum']/len(source['positions'])
            elif kind=='native':
                values,record=native_gradients(view,visual['native'],source,negatives=visual['negatives'],correction_weight=options['correction_weight'],group=control_group,**bounds)
                n=sum(n>0 for n in source['negative_context_lengths'])
                if n:denominators['corrective_pairs']=n
                local=record['local_CE_sum']/denominators['native_response_EOS_targets']+(options['correction_weight']*record['local_correction_ranking_sum']/n if n else 0.)
            else:raise ValueError('Only the actual declared original visual family may be dispatched')
            reduced=summed(values,'visual',denominators)
            for name,value in reduced.items():
                if value is not None:side[name]=options['visual_weight']*value if side[name] is None else side[name]+options['visual_weight']*value
            tasks['visual']=record;local_losses['visual']=local
        gradients,interaction=combine(primary,side,primary_names=view.task_names('text'),secondary_weight=1.,project_conflicts=options['project_conflicts'])
        rows=gather(dict(tasks=tasks,local_losses=local_losses),control_group,world)
        losses={name:math.fsum(row['local_losses'][name] for row in rows) for name in local_losses}
        if visual is not None and visual['kind']=='photo':
            scores=[row['tasks']['visual']['complete_global_matching']['weighted_loss'] for row in rows]
            if len(set(scores))!=1:raise ValueError('The same complete global matching objective must be retained by every replica')
            losses['visual']+=scores[0]
        active=[n for n,g in gradients.items() if g is not None]
        guard.check()
        for name,parameter in view.named_selected():parameter.grad=gradients[name]
        norm=float(torch.nn.utils.clip_grad_norm_([p for _,p in view.named_selected()],options['gradient_clip'],error_if_nonfinite=True))
        common=value_digest(dict(source_entry_sha256=source_entry_sha256,losses=losses,gradient_interaction=interaction,active=active,gradient_norm_before_clip=norm))
        agree(None,common,control_group,world)
        optimizer.step();optimizer.zero_grad(set_to_none=True);guard.accept_own_update()
        return dict(source_entry_sha256=source_entry_sha256,global_normalized_task_losses=losses,actual_active_parameter_names=active,
            gradient_norm_before_clip=norm,gradient_interaction=interaction,actual_all_rank_task_records=rows,actual_task_transports=transports,
            one_actual_AdamW_call=True,global_task_reductions_before_projection=True,production_job_admission_or_model_quality_not_proved=True)
    except BaseException:
        optimizer.zero_grad(set_to_none=True)
        # No automatic retry is allowed after a possibly partial optimizer
        # failure. The owned controller must retain the actual failure state
        # and restore a separately committed complete generation.
        raise
