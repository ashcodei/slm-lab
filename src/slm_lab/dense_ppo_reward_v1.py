"""Training-only human reward on its original, separately counted encoder.

The changed dense policy is never used as the reward feature extractor.
Only the original general backbone and the completed reward head are retained;
the old code/grounding/alignment extensions are released after strict loading.
"""
import json
from pathlib import Path

import torch
from torch import nn

from .codenet_checkpoint_v1 import FileBindings
from .model import TinyMultimodalLM
from .point_grounding_evaluation_v1 import state_digest
from .whole_human_sft_evidence_v1 import tensor_digest


def signature(parameters):
    return [(id(p),p._version,p.data_ptr(),p.requires_grad) for p in parameters]


class FrozenGeneralReward(nn.Module):
    def __init__(self,encoder,head,provenance):
        super().__init__()
        if (type(encoder) is not TinyMultimodalLM or type(head) is not nn.Sequential
                or encoder.training or head.training
                or state_digest(encoder)!=provenance['encoder_sha256']
                or tensor_digest(head.named_parameters())!=provenance['head_sha256']):
            raise ValueError('Exact fixed original general encoder and human-trained head feature basis required')
        self.encoder,self.head=encoder,head
        self.provenance=dict(provenance)
        self.eval();self.fixed_signature=signature(self.parameters())

    def train(self,mode=True):
        if mode:raise ValueError('The separately counted training-only reward model stays in evaluation mode')
        return super().train(False)

    def counts(self):
        return dict(original_general_encoder=sum(p.numel() for p in self.encoder.parameters()),
            human_reward_head=sum(p.numel() for p in self.head.parameters()),
            actual_training_only_total=sum(p.numel() for p in self.parameters()),
            included_in_deployed_policy=False)

    def check(self):
        if self.training or self.encoder.training or self.head.training or signature(self.parameters())!=self.fixed_signature:
            raise ValueError('Every reward encoder/head value, mode and parameter metadata must stay fixed')
        if any(p.grad is not None for p in self.parameters()):
            raise ValueError('The training-only human reward receives no optimizer gradients')

    @torch.no_grad()
    def score(self,ids,amp=None):
        from contextlib import nullcontext
        self.check()
        if (not isinstance(ids,torch.Tensor) or ids.dtype!=torch.long or ids.ndim!=2 or ids.shape[0]!=1
                or not 2<=ids.shape[1]<=self.encoder.config.max_text_length or int(ids[0,-1])!=2
                or bool((ids<0).any()) or bool((ids>=self.encoder.config.vocab_size).any())
                or ids.device!=next(self.encoder.parameters()).device):
            raise ValueError('One whole native prompt/response with an explicit scoring EOS and no padding required')
        with (amp or nullcontext)():
            hidden=self.encoder(ids,return_hidden=True,return_logits=False)['hidden'][:,-1].float()
        with torch.autocast(device_type=hidden.device.type,enabled=False):value=self.head(hidden).reshape(())
        if not bool(torch.isfinite(value)):raise ValueError('The actual human reward score must be finite')
        return value.detach()


def load(inputs,device='cpu'):
    """Strictly load a completed head before extracting its original encoder."""
    from .whole_human_sft_inference_v1 import fresh
    from .whole_human_reward_inference_v1 import attach
    files={str(Path(inputs[k])):inputs[k+'_sha256'] for k in ('checkpoint','plan','feature_admission')}
    guard=FileBindings(files);plan=json.loads(Path(inputs['plan']).read_text())
    model,_=fresh(plan['initial_model_plan'],'cpu')
    metadata=attach(model,inputs['checkpoint'],inputs['plan'],inputs['feature_admission'],inputs['arm'],
        expected_checkpoint_sha256=inputs['checkpoint_sha256'],expected_plan_sha256=inputs['plan_sha256'],
        expected_feature_admission_sha256=inputs['feature_admission_sha256'])
    provenance=dict(encoder_sha256=plan['frozen_encoder_sha256'],head_sha256=tensor_digest(model.parts['reward'].named_parameters()),
        reward_checkpoint_sha256=inputs['checkpoint_sha256'],reward_plan_sha256=inputs['plan_sha256'],
        feature_admission_sha256=inputs['feature_admission_sha256'],arm=inputs['arm'],
        actual_human_reward_updates=metadata['actual_human_reward_updates'],original_training_pairs=plan['source_population'],
        feature_route=metadata['reward_feature_route'],training_only=True,
        human_rankings_on_other_model_responses_not_own_output_judgments=True)
    encoder,head=model.original.language.base,model.parts['reward'];del model
    # This move changes only the private training-only encoder, before its
    # immutable runtime signature is established.
    result=FrozenGeneralReward(encoder.to(device).eval(),head.to(device).eval(),provenance)
    guard.check(full=True)
    return result
