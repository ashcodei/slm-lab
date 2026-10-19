"""Complete original joint source delivery with durable pre-Adam metrics."""
from dataclasses import asdict
import math

from .complete_joint_training_data_v1 import CompleteJointTrainingDataV1
from .complete_joint_objective_bridge_v1 import checkpoint_entry,complete_entries
from .joint_objective_step_v2 import update
from .parallel_function_repair_stage_v1 import value_digest
from .shared_joint_checkpoint_v2 import agree,control

FORMAT='own-complete-original-joint-source-pre-Adam-bridge-v2'


def train_step(source,step,view,reference,optimizer,guard,*,options,control_group=None,gradient_group=None,before_optimizer=None):
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
    def metric_for(record):
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
        return metric
    def publish(record):
        if before_optimizer is not None:before_optimizer(metric_for(record))
    record=update(view,reference,optimizer,guard,code=code,human=human,visual=visual,options=options,
        source_entry_sha256=value_digest(entry),control_group=control_group,gradient_group=gradient_group,before_optimizer=publish)
    return entry,metric_for(record)
