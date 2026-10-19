"""Deliver the complete original joint sources to the common Adam boundary.

Translate whole source budgets into the checkpoint protocol without losing
corrective negatives or conflating native EOS and CTC byte normalizers. This
bridge grants no ownership, durable-history or full-size GPU job admission.
"""
from dataclasses import asdict
import math

from .complete_joint_training_data_v1 import CompleteJointTrainingDataV1
from .joint_objective_step_v1 import update
from .parallel_function_repair_stage_v1 import value_digest
from .shared_joint_checkpoint_v2 import agree,control,sha

FORMAT='own-complete-original-joint-source-to-objective-bridge-v1'


def checkpoint_entry(metadata,*,source_identity_sha256):
    """Map one unchanged admitted source header to every original objective."""
    if (not sha(source_identity_sha256) or type(metadata) is not dict or set(metadata)!={'step','source_entry_sha256','components'}
            or type(metadata['step']) is not int or metadata['step']<1 or not sha(metadata['source_entry_sha256'])
            or type(metadata['components']) is not dict or not metadata['components']
            or set(metadata['components'])-{'code','human','visual'}):
        raise ValueError('One complete original paced source header and admitted data identity are required')
    objectives=[]
    def append(task,row,context,targets,denominators,bytes_=0):
        objectives.append(dict(task=task,source_family=row['family'],source_role='train',source_sha256=source_identity_sha256,
            original_positions=list(row['original_positions']),complete_context_tokens=context,response_EOS_targets=targets,
            word_UTF8_bytes=bytes_,normalization_denominators=denominators))
    for name in ['code','human','visual']:
        if name not in metadata['components']:continue
        part=metadata['components'][name];row=part['entry'];reference=part['reference']
        if (reference['stream']!=name or reference['source_step']!=row['source_step']
                or not sha(reference['original_entry_sha256']) or type(row['original_positions']) is not list
                or not row['original_positions'] or len(set(row['original_positions']))!=len(row['original_positions'])
                or any(type(p) is not int or p<0 for p in row['original_positions'])
                or type(row['complete_context_tokens']) is not int or type(row['response_EOS_targets']) is not int
                or not 0<row['response_EOS_targets']<row['complete_context_tokens']):
            raise ValueError('Every complete source position/context/response and paced reference must remain explicit')
        context,targets=row['complete_context_tokens'],row['response_EOS_targets']
        if name=='code':
            mass=row['original_global_positive_weighted_targets']
            if type(mass) not in [int,float] or not math.isfinite(mass) or mass<=0:raise ValueError('The real positive weighted code-token mass cannot be rounded or dropped')
            append('text',row,context,targets,dict(original_positive_weighted_targets=mass))
        elif name=='human':
            if (row['chosen_context_tokens']+row['rejected_context_tokens']!=context
                    or row['chosen_response_EOS_targets']+row['rejected_response_EOS_targets']!=targets
                    or not 0<row['chosen_response_EOS_targets']<row['chosen_context_tokens']
                    or not 0<row['rejected_response_EOS_targets']<row['rejected_context_tokens']):
                raise ValueError('Both whole human responses and independent pair/chosen-token denominators must remain')
            append('text',row,context,targets,dict(human_pairs=len(row['original_positions']),chosen_response_EOS=row['chosen_response_EOS_targets']))
        elif row['family']=='words':
            if row['complete_original_negative_response_EOS_targets'] or row['complete_original_correction_indices'] or row['whole_word_UTF8_bytes']<1:
                raise ValueError('Whole original word CTC and native CE cannot acquire corrective-image targets')
            append('native_image',row,context,targets,dict(native_response_EOS=targets))
            append('word_CTC',row,0,0,dict(whole_words=len(row['original_positions'])),row['whole_word_UTF8_bytes'])
        elif row['family']=='images:additional_photo_captions':
            if row['complete_original_negative_response_EOS_targets'] or row['complete_original_correction_indices'] or row['whole_word_UTF8_bytes']:
                raise ValueError('Original paired-photo matching cannot acquire a different image objective')
            append('image_matching',row,context,targets,dict(native_response_EOS=targets,image_pairs=len(row['original_positions'])))
        else:
            negative=row['complete_original_negative_response_EOS_targets'];pairs=len(row['complete_original_correction_indices'])
            if (type(negative) is not int or not 0<=negative<targets or bool(negative)!=bool(pairs) or row['whole_word_UTF8_bytes']):
                raise ValueError('Every original corrective pair and negative EOS target must remain in the source budget')
            denominators=dict(positive_native_EOS=targets-negative)
            if pairs:denominators.update(corrective_pairs=pairs,negative_EOS=negative)
            append('native_image',row,context,targets,denominators)
    return dict(step=metadata['step'],objectives=objectives)


def complete_entries(source):
    if type(source) is not CompleteJointTrainingDataV1 or len(source)!=5362:
        raise ValueError('The actual complete5362step source delivery is required')
    source.guard.check(full=True)
    entries=[checkpoint_entry(source.metadata(step),source_identity_sha256=source.identity_sha256) for step in range(len(source))]
    source.guard.check(full=True)
    return entries


def train_step(source,step,view,reference,optimizer,guard,*,options,control_group=None,gradient_group=None):
    """One original full step; ownership/checkpoint/history remain external."""
    rank,world=control(control_group);error=identity=None
    try:
        if (type(source) is not CompleteJointTrainingDataV1 or len(source)!=5362
                or asdict(source.config)!=asdict(view.actor.config)
                or view.actor.base.heads['word'].expansion!=5):
            raise ValueError('Every original native context,448px image and140frame word protocol must fit the actual joint model')
        source.guard.check();metadata=source.metadata(step)
        entry=checkpoint_entry(metadata,source_identity_sha256=source.identity_sha256)
        parts=metadata['components'];device=next(view.parameters()).device
        code=dict(data=source.code,positions=parts['code']['entry']['original_positions'],profile='balanced_sources') if 'code' in parts else None
        human=dict(data=source.human,positions=parts['human']['entry']['original_positions'],DPO_weight=1) if 'human' in parts else None
        visual=source.prepare_visual(parts['visual'],rank=rank,world=world,device=device) if 'visual' in parts else None
        identity=value_digest(dict(format=FORMAT,data_identity=source.identity_sha256,source_metadata=metadata,checkpoint_entry=entry))
    except Exception as exc:error=type(exc).__name__+': '+str(exc)
    agree(error,identity,control_group,world)
    record=update(view,reference,optimizer,guard,code=code,human=human,visual=visual,options=options,
        source_entry_sha256=value_digest(entry),control_group=control_group,gradient_group=gradient_group)
    rows=record['actual_all_rank_task_records'];losses=[]
    for name in ['code','human','visual']:
        if name not in parts:continue
        if name=='visual' and visual['kind']=='word':
            losses.extend([options['native_word_CE_weight']*math.fsum(r['tasks']['visual']['local_native_CE_sum'] for r in rows)/sum(visual['source']['response_EOS_targets']),
                options['CTC_weight']*math.fsum(r['tasks']['visual']['local_byte_normalized_CTC_sum'] for r in rows)/len(visual['source']['positions'])])
        else:losses.append(record['global_normalized_task_losses'][name])
    metric=dict(step=step+1,entry_sha256=value_digest(entry),active_parameter_names=record['actual_active_parameter_names'],normalized_task_losses=losses,
        gradient_norm_before_clip=record['gradient_norm_before_clip'],globally_reduced_task_gradients_before_combining=True,
        gradient_interaction=dict(complete_original_paced_entry_sha256=metadata['source_entry_sha256'],source_delivery_identity_sha256=source.identity_sha256,
            actual_blend=record['gradient_interaction'],actual_task_transports=record['actual_task_transports'],actual_all_rank_tasks=rows))
    return entry,metric
