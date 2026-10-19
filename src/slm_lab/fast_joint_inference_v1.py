"""Native inference for the one counted scratch-origin joint dense student."""
import json
from pathlib import Path

import torch

from .corpus import file_hash
from .codenet_checkpoint_v1 import FileBindings
from .dense_foundation_inference_v1 import fresh as dense_fresh,DenseNativeView

FORMAT='own-fast-joint-full-weight-native-model-v1'


def fresh(plan,device='cpu'):
    return dense_fresh(plan['initial_dense_settings'],device)


def sources(plan,digest):
    return dict(joint_run_plan_sha256=digest,initializer_sha256=plan['initial_dense_settings']['initializer_sha256'])


def export(model,plan,plan_sha256,arm,completed):
    if model.counts()!=plan['actual_counts'] or not 0<=completed<=plan['planned_steps']:
        raise ValueError('Counted complete student and exact bounded actual update cursor required')
    return dict(format=FORMAT,model=model.export(sources(plan,plan_sha256)),run_plan_sha256=plan_sha256,
        arm=arm,completed_updates=completed,whole_source_epochs_completed=int(completed==plan['planned_steps']),
        candidate_promoted=False,goal_achieved=False)


def load(checkpoint,run_plan,arm,*,expected_checkpoint_sha256,expected_plan_sha256,device='cpu'):
    checkpoint,run_plan=map(Path,(checkpoint,run_plan));guard=FileBindings({str(checkpoint):expected_checkpoint_sha256,str(run_plan):expected_plan_sha256})
    plan=json.loads(run_plan.read_text());raw=torch.load(checkpoint,map_location='cpu',weights_only=True,mmap=True)
    if (raw.get('format')!=FORMAT or raw.get('arm')!=arm or raw.get('run_plan_sha256')!=expected_plan_sha256
            or type(raw.get('completed_updates')) is not int or not 0<=raw['completed_updates']<=plan['planned_steps']
            or raw.get('whole_source_epochs_completed')!=int(raw['completed_updates']==plan['planned_steps'])
            or raw.get('candidate_promoted') is not False or raw.get('goal_achieved') is not False):
        raise ValueError('Exact joint export identity, source plan and actual training cursor required')
    model,metadata=fresh(plan,'cpu');model.import_export(raw['model'],expected_sources=sources(plan,expected_plan_sha256));guard.check(full=True)
    return model.eval().to(device),dict(metadata,joint_full_weight_training=True,actual_completed_updates=raw['completed_updates'],
        training_options=dict(image_size=448),tokenizer_sha256=plan['tokenizer_sha256'])
