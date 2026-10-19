"""One standalone native backbone with its unfused policy and frozen reward.

Keep policy adapters rather than merging them into the reward feature basis.
The complete native, adapter/norm policy, separate policy critic and existing
reward head are counted once. No ancestor checkpoint, optimizer, data feature
cache or second encoder is needed after loading the model/tokenizer package.
"""
import json
from pathlib import Path

import torch
from tokenizers import Tokenizer

from .code_residual_checkpoint_v1 import _cpu, _digest
from .corpus import file_hash
from .deterministic_positions_v1 import PROTOCOL, install, patch_grid, vision_forward
from .model import ModelConfig
from .native_human_reward_v1 import FrozenSharedNativeReward, signature, sources_ok
from .own_dense_actor_v1 import OwnDenseActor
from .own_dense_growth_v1 import OwnDenseAssistant, new_backbone
from .whole_human_sft_evidence_v1 import tensor_digest

FORMAT='own-standalone-unfused-policy-shared-native-human-reward-v1'
MAXIMUM_BYTES=5*1024**3


def inventory(actor,reward,*,allow_mechanics):
    if type(actor) is not OwnDenseActor or type(reward) is not FrozenSharedNativeReward or reward.base is not actor.base:
        raise ValueError('Exactly one own native must be shared by the actual policy and trained reward')
    actor.assert_shared();reward.check();actual=sum(p.numel() for p in actor.parameters())
    union={id(p):p for module in [actor,reward] for p in module.parameters()}
    full=actor.base.counts()['total_unique']==984399690 and actual==999706442 and actor.settings['rank']==16
    small=allow_mechanics is True and actual<10**7 and next(actor.parameters()).device.type=='cpu'
    if (not full and not small or actual!=actor.counts()['actual_unique'] or sum(p.numel() for p in union.values())!=actual
            or actual>=10**9 or actor.training or any(m.training for m in actor.modules())
            or any(p.dtype!=torch.float32 or p.grad is not None for p in actor.parameters())
            or any(m._forward_hooks or m._forward_pre_hooks or m._backward_hooks for m in actor.modules())
            or getattr(actor.base.patch_grid,'__func__',None) is not patch_grid
            or getattr(actor.base.network.vision.forward,'__func__',None) is not vision_forward):
        raise ValueError('One complete eval-mode FP32 current native/policy/reward and the original position arithmetic are required')
    return dict(**actor.counts(),reward=reward.counts(),shared_backbone=True,
        additional_reward_encoder_values=0,full_current_model=full,mechanics_only=not full)


class StandalonePolicyReward(torch.nn.Module):
    def __init__(self,actor,reward,*,allow_mechanics=False):
        super().__init__();self.actor,self.reward=actor,reward
        self.eval();self.actual_counts=inventory(actor,reward,allow_mechanics=allow_mechanics)
        self.fixed_signature=signature(actor.parameters())

    @property
    def config(self):return self.actor.config

    @property
    def policy(self):return self.actor.policy

    @property
    def base(self):return self.actor.base

    def train(self,mode=True):
        if mode:raise ValueError('The standalone policy/reward delivery is an immutable inference graph')
        return super().train(False)

    def counts(self):return _cpu(self.actual_counts)

    def check(self):
        self.actor.assert_shared();self.reward.check()
        if (self.training or any(m.training for m in self.modules())
                or signature(self.actor.parameters())!=self.fixed_signature
                or sum(p.numel() for p in self.parameters())!=self.actual_counts['actual_unique']):
            raise ValueError('Every complete standalone policy/reward value and the one-backbone inventory must remain fixed')

    @torch.no_grad()
    def forward(self,*args,**kwargs):self.check();return self.actor(*args,**kwargs)

    @torch.no_grad()
    def generate(self,*args,**kwargs):self.check();return self.actor.generate(*args,**kwargs)

    @torch.no_grad()
    def preference_score(self,ids):self.check();return self.reward.score(ids)

    @torch.no_grad()
    def word_logits(self,pixels):self.check();return self.actor.base.word_logits(pixels)


def export(actor,reward,*,sources,provenance,completed_updates,tokenizer_sha256,allow_mechanics=False):
    actual=inventory(actor,reward,allow_mechanics=allow_mechanics)
    if (any(p.device.type!='cpu' for p in actor.parameters()) or not isinstance(sources,dict) or not sources
            or any(not isinstance(k,str) or not _digest(v) for k,v in sources.items())
            or any(sources.get(k)!=v for k,v in reward.provenance['sources'].items())
            or not _digest(tokenizer_sha256) or sources.get('tokenizer_sha256')!=tokenizer_sha256
            or type(completed_updates) is not int or completed_updates<0 or not isinstance(provenance,dict)
            or len(json.dumps(provenance,allow_nan=False).encode())>256*1024):
        raise ValueError('Complete CPU policy/reward, exact own sources/vocabulary and bounded explicit provenance are required')
    before=signature(actor.parameters());rng=torch.get_rng_state().clone()
    policy_sha=tensor_digest(actor.named_parameters());reward_sha=tensor_digest(reward.named_parameters())
    result=dict(format=FORMAT,complete_base=actor.base.export(sources),policy_adapter=actor.export(sources),
        reward_provenance=_cpu(reward.provenance),sources=dict(sources),actual_counts=actual,
        policy_parameters_sha256=policy_sha,reward_parameters_sha256=reward_sha,
        tokenizer_sha256=tokenizer_sha256,position_interpolation=PROTOCOL,completed_updates=completed_updates,
        provenance=_cpu(provenance),old_base_actor_optimizer_or_training_features_needed_for_inference=False,
        policy_factors_not_merged_into_frozen_reward_encoder=True,
        actual_own_output_human_alignment_or_model_quality_proved=False,candidate_promoted=False,goal_achieved=False)
    if (signature(actor.parameters())!=before or tensor_digest(actor.named_parameters())!=policy_sha
            or tensor_digest(reward.named_parameters())!=reward_sha or not torch.equal(rng,torch.get_rng_state())):
        raise ValueError('Standalone export must preserve every source value, parameter metadata and CPU generator')
    return result


def inspect_package(path,*,expected_checkpoint_sha256=None,expected_sources=None):
    path=Path(path).expanduser().resolve()
    if not path.is_file() or path.stat().st_size>MAXIMUM_BYTES:raise ValueError('One bounded complete standalone policy/reward checkpoint is required')
    digest=file_hash(path)
    if expected_checkpoint_sha256 is not None and digest!=expected_checkpoint_sha256:raise ValueError('The complete standalone checkpoint checksum differs')
    raw=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
    sources=raw.get('sources');counts=raw.get('actual_counts',{});base=raw.get('complete_base',{});actor=raw.get('policy_adapter',{})
    reward=raw.get('reward_provenance',{})
    if (raw.get('format')!=FORMAT or raw.get('position_interpolation')!=PROTOCOL
            or raw.get('old_base_actor_optimizer_or_training_features_needed_for_inference') is not False
            or raw.get('policy_factors_not_merged_into_frozen_reward_encoder') is not True
            or any(raw.get(k) is not False for k in ['candidate_promoted','goal_achieved','actual_own_output_human_alignment_or_model_quality_proved'])
            or not isinstance(sources,dict) or not sources or any(not isinstance(k,str) or not _digest(v) for k,v in sources.items())
            or expected_sources is not None and sources!=expected_sources
            or base.get('sources')!=sources or actor.get('sources')!=sources
            or not sources_ok(reward.get('sources'))
            or any(sources.get(k)!=v for k,v in reward['sources'].items())
            or reward.get('frozen_native_encoder_sha256')!=sources.get('frozen_native_encoder_sha256')
            or type(raw.get('completed_updates')) is not int or raw['completed_updates']<0
            or type(counts.get('actual_unique')) is not int or not 0<counts['actual_unique']<10**9
            or counts.get('additional_reward_encoder_values')!=0 or counts.get('shared_backbone') is not True
            or actor.get('actual_counts',{}).get('actual_unique')!=counts['actual_unique']
            or base.get('actual_counts',{}).get('total_unique')!=counts.get('frozen_dense_and_all_heads')
            or not _digest(raw.get('policy_parameters_sha256')) or not _digest(raw.get('reward_parameters_sha256'))):
        raise ValueError('The exact complete own policy/reward source, one-backbone count and research artifact flags are required')
    token_path=path.parent/'tokenizer.json';token_sha=file_hash(token_path)
    if token_sha!=raw.get('tokenizer_sha256') or sources.get('tokenizer_sha256')!=token_sha:
        raise ValueError('The exact packaged native tokenizer bytes must match before parsing')
    tokenizer=Tokenizer.from_file(str(token_path))
    if (tokenizer.get_vocab_size()!=base.get('config',{}).get('vocab_size')
            or [tokenizer.token_to_id(t) for t in ['<pad>','<bos>','<eos>']]!=[0,1,2]
            or file_hash(path)!=digest):
        raise ValueError('Every original model/tokenizer byte and the complete native vocabulary must remain exact')
    return dict(checkpoint=str(path),checkpoint_sha256=digest,tokenizer_sha256=token_sha,sources=dict(sources),
        actual_counts=_cpu(counts),completed_updates=raw['completed_updates'],inspect_only=True,
        registered_values_and_neural_equivalence_not_proved_by_inspection=True,
        source_map_verified_against_external_input=expected_sources is not None,
        checkpoint_verified_against_external_input=expected_checkpoint_sha256 is not None)


def load(path,*,expected_checkpoint_sha256,expected_sources,device='cpu',allow_mechanics=False):
    package=inspect_package(path,expected_checkpoint_sha256=expected_checkpoint_sha256,expected_sources=expected_sources)
    if package['actual_counts']['mechanics_only'] and not allow_mechanics:
        raise ValueError('Small CPU controls require explicit admission and cannot impersonate the complete current model')
    raw=torch.load(package['checkpoint'],map_location='cpu',weights_only=True,mmap=True);base_payload=raw['complete_base']
    device=torch.device(device)
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(999117)
        base=install(OwnDenseAssistant(new_backbone(ModelConfig(**base_payload['config']),999117),**base_payload['settings'])).eval()
        base.import_export(base_payload,expected_sources=expected_sources)
        actor=OwnDenseActor(base,**raw['policy_adapter']['settings']).eval()
        actor.import_export(raw['policy_adapter'],expected_sources=expected_sources)
    if (tensor_digest(actor.named_parameters())!=raw['policy_parameters_sha256']
            or any(p.grad is not None or p.dtype!=torch.float32 for p in actor.parameters())):
        raise ValueError('Every complete source policy/critic/native parameter must reconstruct exactly in FP32')
    actor.to(device).eval()
    reward=FrozenSharedNativeReward(base,raw['reward_provenance'],allow_mechanics=allow_mechanics)
    if tensor_digest(reward.named_parameters())!=raw['reward_parameters_sha256']:
        raise ValueError('Every actual frozen shared encoder/reward value must match its independent source digest')
    bundle=StandalonePolicyReward(actor,reward,allow_mechanics=allow_mechanics)
    if (bundle.counts()!=package['actual_counts'] or file_hash(package['checkpoint'])!=package['checkpoint_sha256']
            or file_hash(Path(package['checkpoint']).parent/'tokenizer.json')!=package['tokenizer_sha256']):
        raise ValueError('All actual counted shared values and complete package bytes must remain exact during load')
    return bundle,dict(package,inspect_only=False,actual_registered_parameters=sum(p.numel() for p in bundle.parameters()),
        old_base_actor_optimizer_or_training_features_loaded=False,
        trained_reward_feature_basis_preserved=True,policy_factors_not_merged=True,
        actual_own_output_human_alignment_or_model_quality_proved=False)
