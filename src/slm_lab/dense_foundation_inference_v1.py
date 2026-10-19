"""Exact initial/completed own dense foundation artifacts and native interface."""
import json
from pathlib import Path

import torch
from torch import nn

from .codenet_checkpoint_v1 import FileBindings
from .corpus import file_hash
from .model import ModelConfig
from .own_dense_growth_v1 import OwnDenseAssistant,new_backbone
from .whole_human_sft_evidence_v1 import tensor_digest

INITIAL_FORMAT='own-full-dense-foundation-initial-v1'
FORMAT='own-full-dense-foundation-trained-v1'


def fresh(plan,device='cpu'):
    path=Path(plan['initializer']);guard=FileBindings({str(path):plan['initializer_sha256']})
    payload=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
    model=OwnDenseAssistant(new_backbone(ModelConfig(**payload['config']),948531),**payload['settings']).eval()
    model.import_export(payload,expected_sources=plan['initializer_sources'])
    if model.counts()!=plan['actual_counts']:raise ValueError('Actual complete source-bound dense/head budget required')
    guard.check(full=True)
    return model.to(device),dict(tokenizer_sha256=plan['tokenizer_sha256'],training_options=dict(image_size=448),
        actual_counts=model.counts(),own_dense_foundation=True,external_pretrained_teacher_loaded=False)


def own_reference(plan,device='cpu'):
    from .whole_human_sft_inference_v1 import fresh as old_factory
    from .point_grounding_evaluation_v1 import state_digest
    prior,_=old_factory(plan['own_reference_model_plan'],'cpu');reference=prior.original;del prior
    if state_digest(reference)!=plan['own_reference_state_sha256']:raise ValueError('Exact frozen original own general/code reference required')
    return reference.eval().to(device)


def sources(plan,plan_sha256):
    return dict(run_plan_sha256=plan_sha256,initializer_sha256=plan['initializer_sha256'],
        education_admission_sha256=plan['education_admission_sha256'],code_admission_sha256=plan['code_admission_sha256'],
        human_admission_sha256=plan['human_admission_sha256'],program_admission_sha256=plan['program_admission_sha256'])


def options(plan,arm):
    if arm!='initial' and arm not in plan['arms']:raise ValueError('Exact predeclared dense foundation arm required')
    spec=dict(learning_rate=0.,reference_KL_alpha=0.) if arm=='initial' else plan['arms'][arm]
    return {**plan['training_options'],'learning_rate':spec['learning_rate'],'reference_KL_alpha':spec['reference_KL_alpha']}


def export(model,plan,plan_sha256,arm,completed):
    initial=arm=='initial';setting=options(plan,arm)
    if completed!=(0 if initial else setting['steps']) or model.counts()!=plan['actual_counts']:
        raise ValueError('Exact full original dense foundation update budget required')
    return dict(format=INITIAL_FORMAT if initial else FORMAT,model=model.export(sources(plan,plan_sha256)),arm=arm,
        run_plan_sha256=plan_sha256,actual_dense_foundation_updates=completed,training_options=setting,
        source_exposure_budget={} if initial else plan['source_exposure_budget'],candidate_promoted=False,goal_achieved=False)


def load(checkpoint,run_plan,arm,*,expected_checkpoint_sha256,expected_plan_sha256,device='cpu'):
    checkpoint,run_plan=map(Path,(checkpoint,run_plan));guard=FileBindings({str(checkpoint):expected_checkpoint_sha256,str(run_plan):expected_plan_sha256})
    plan=json.loads(run_plan.read_text());raw=torch.load(checkpoint,map_location='cpu',weights_only=True,mmap=True)
    initial=arm=='initial';setting=options(plan,arm)
    if (raw.get('format')!=(INITIAL_FORMAT if initial else FORMAT) or raw.get('arm')!=arm or raw.get('run_plan_sha256')!=expected_plan_sha256
            or raw.get('actual_dense_foundation_updates')!=(0 if initial else setting['steps']) or raw.get('training_options')!=setting
            or raw.get('source_exposure_budget')!=({} if initial else plan['source_exposure_budget'])
            or raw.get('candidate_promoted') is not False or raw.get('goal_achieved') is not False):
        raise ValueError('Exact source/arm/complete update exposure provenance required')
    model,metadata=fresh(plan,'cpu');model.validate_export(raw['model'],expected_sources=sources(plan,expected_plan_sha256))
    language={name for name,_ in model.language_named_parameters()};live=model.network.state_dict();saved=raw['model']
    if (tensor_digest(model.named_parameters())!=plan['initial_model_digest']
            or any(not torch.equal(t,saved['network'][name]) for name,t in live.items() if name not in language and name!='lm_head.weight')
            or any(not torch.equal(t,saved['heads'][name]) for name,t in model.heads.state_dict().items())):
        raise ValueError('Every initial source/vision/critic/auxiliary value must match before import')
    if initial and any(not torch.equal(t,saved['network'][name]) for name,t in live.items()):
        raise ValueError('Exact initial complete dense language weights required')
    model.import_export(raw['model'],expected_sources=sources(plan,expected_plan_sha256));guard.check(full=True)
    return model.eval().to(device),dict(metadata,full_dense_foundation=dict(arm=arm,updates=raw['actual_dense_foundation_updates']),
        final_multimodal_and_genuine_human_RLHF_quality_not_established=True)


class DenseNativeView(nn.Module):
    """Expose native evaluator APIs while counting all actually loaded heads."""
    def __init__(self,assistant):
        super().__init__();self.assistant=assistant;self.config=assistant.config

    def forward(self,*args,**kwargs):return self.assistant(*args,**kwargs)
    def generate(self,*args,**kwargs):return self.assistant.generate(*args,**kwargs)
    def prefill(self,*args,**kwargs):return self.assistant.network.prefill(*args,**kwargs)
    def decode(self,*args,**kwargs):return self.assistant.network.decode(*args,**kwargs)
