"""A completed production DPO policy plus new source-bound PPO actor/critic.

The original production fused base is reused. No reward encoder is loaded or
stored by this inference path, and a learned separate critic is permitted.
"""
import json
from pathlib import Path
import torch

from .codenet_checkpoint_v1 import FileBindings
from .whole_human_sft_evidence_v1 import tensor_digest

FORMAT='own-dense-cached-human-PPO-trained-policy-critic-v1'
INITIAL_FORMAT='own-dense-cached-human-PPO-initial-production-DPO-policy-v1'


def fresh(plan,device='cpu'):
    from .dense_preference_inference_v1 import load as load_initial
    inputs=plan['initial_policy']
    source_plan=json.loads(Path(inputs['plan']).read_text())
    if inputs['arm']=='initial' or source_plan.get('arms',{}).get(inputs['arm'],{}).get('DPO_weight')!=1:
        raise ValueError('A completed production DPO source policy is required')
    model,metadata=load_initial(inputs['checkpoint'],inputs['plan'],inputs['arm'],expected_checkpoint_sha256=inputs['checkpoint_sha256'],
        expected_plan_sha256=inputs['plan_sha256'],device='cpu')
    if (model.counts()!=plan['actual_counts'] or tensor_digest(model.base.named_parameters())!=plan['frozen_dense_sha256']
            or tensor_digest(model.parts.named_parameters())!=plan['initial_parts_sha256']
            or model.parts['critic'].weight.count_nonzero()):
        raise ValueError('Actual identical complete production policy, frozen fused base and separate initial zero critic required')
    return model.eval().to(device),dict(actual_counts=model.counts(),tokenizer_sha256=metadata['tokenizer_sha256'],
        training_options=metadata['training_options'],initial_completed_human_preference_policy=metadata['human_preference_policy'],
        separate_training_only_reward_not_loaded=True)


def options(plan,arm):
    if arm!='initial' and arm not in plan['arms']:raise ValueError('Exact predeclared own-response PPO or KL-only arm required')
    profile=dict(learning_rate=0.) if arm=='initial' else plan['arms'][arm]
    return {**plan['training_options'],'learning_rate':profile['learning_rate']}


def sources(plan,plan_sha256):
    return dict(run_plan_sha256=plan_sha256,initial_production_policy_sha256=plan['initial_policy']['checkpoint_sha256'],
        training_only_reward_checkpoint_sha256=plan['reward_inputs']['checkpoint_sha256'],human_admission_sha256=plan['human_admission_sha256'])


def export(model,plan,plan_sha256,arm,completed):
    initial=arm=='initial';setting=options(plan,arm)
    if completed!=(0 if initial else setting['steps']) or model.counts()!=plan['actual_counts']:
        raise ValueError('Every complete bounded PPO update and actual deployed parameter required')
    return dict(format=INITIAL_FORMAT if initial else FORMAT,policy=model.export(sources(plan,plan_sha256)),arm=arm,
        run_plan_sha256=plan_sha256,actual_own_response_PPO_updates=completed,training_options=setting,
        terminal_human_reward_scale=None if initial else plan['arms'][arm]['human_reward_scale'],
        source_exposures={} if initial else plan['source_exposures'][arm],separate_training_reward_not_in_policy=True,
        candidate_promoted=False,goal_achieved=False)


def load(checkpoint,run_plan,arm,*,expected_checkpoint_sha256,expected_plan_sha256,device='cpu'):
    checkpoint,run_plan=map(Path,(checkpoint,run_plan));guard=FileBindings({str(checkpoint):expected_checkpoint_sha256,str(run_plan):expected_plan_sha256})
    plan=json.loads(run_plan.read_text());raw=torch.load(checkpoint,map_location='cpu',weights_only=True,mmap=True)
    initial=arm=='initial';setting=options(plan,arm)
    keys={'format','policy','arm','run_plan_sha256','actual_own_response_PPO_updates','training_options','terminal_human_reward_scale',
        'source_exposures','separate_training_reward_not_in_policy','candidate_promoted','goal_achieved'}
    if (set(raw)!=keys or raw.get('format')!=(INITIAL_FORMAT if initial else FORMAT) or raw.get('arm')!=arm or raw.get('run_plan_sha256')!=expected_plan_sha256
            or raw.get('actual_own_response_PPO_updates')!=(0 if initial else setting['steps']) or raw.get('training_options')!=setting
            or raw.get('terminal_human_reward_scale')!=(None if initial else plan['arms'][arm]['human_reward_scale'])
            or raw.get('source_exposures')!=({} if initial else plan['source_exposures'][arm])
            or raw.get('separate_training_reward_not_in_policy') is not True
            or raw.get('candidate_promoted') is not False or raw.get('goal_achieved') is not False):
        raise ValueError('Exact complete bounded PPO/critic, profile and source exposure provenance required')
    model,metadata=fresh(plan,'cpu');payload=raw['policy'];model.validate_export(payload,expected_sources=sources(plan,expected_plan_sha256))
    if initial and tensor_digest(payload['parts'].items())!=plan['initial_parts_sha256']:
        raise ValueError('Every initial source policy value and zero critic must match')
    model.import_export(payload,expected_sources=sources(plan,expected_plan_sha256));guard.check(full=True)
    return model.eval().to(device),dict(metadata,own_response_PPO=dict(arm=arm,updates=raw['actual_own_response_PPO_updates'],
        terminal_human_reward_scale=raw['terminal_human_reward_scale']),useful_alignment_and_fresh_own_output_human_fidelity_not_established=True)
