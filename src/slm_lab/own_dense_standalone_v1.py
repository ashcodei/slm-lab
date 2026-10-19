"""A complete own dense student and own tokenizer, without training ancestors.

This format does not invent an actor-fold operation for a plain dense source.
All vision/word/auxiliary/critic weights and one tied vocabulary are included.
"""
from pathlib import Path

import torch
from tokenizers import Tokenizer

from .corpus import file_hash
from .deterministic_positions_v1 import PROTOCOL,install
from .model import ModelConfig
from .own_dense_growth_v1 import OwnDenseAssistant,new_backbone

FORMAT='own-complete-dense-standalone-student-v1'


def export(model,*,sources,provenance,completed_updates,tokenizer_sha256):
    if (type(model) is not OwnDenseAssistant or type(completed_updates) is not int or completed_updates<0
            or model.counts()['total_unique']>=10**9 or not isinstance(provenance,dict)
            or not isinstance(tokenizer_sha256,str) or len(tokenizer_sha256)!=64):
        raise ValueError('Exact own under1B complete dense student, source recipe, own tokenizer and explicit cursor required')
    return dict(format=FORMAT,complete_dense=model.export(sources),provenance=dict(provenance),
        completed_updates=completed_updates,actual_counts=model.counts(),tokenizer_sha256=tokenizer_sha256,
        position_interpolation=PROTOCOL,old_checkpoint_optimizer_or_training_reference_needed_for_inference=False,
        candidate_promoted=False,goal_achieved=False)


def load(path,*,expected_checkpoint_sha256,expected_sources,seed=975117):
    path=Path(path)
    if file_hash(path)!=expected_checkpoint_sha256:raise ValueError('Exact complete own standalone bytes required')
    raw=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
    if (raw.get('format')!=FORMAT or raw.get('position_interpolation')!=PROTOCOL
            or raw.get('old_checkpoint_optimizer_or_training_reference_needed_for_inference') is not False
            or raw.get('candidate_promoted') is not False or raw.get('goal_achieved') is not False
            or type(raw.get('completed_updates')) is not int or raw['completed_updates']<0
            or not isinstance(raw.get('provenance'),dict)):
        raise ValueError('Complete own standalone identity and cursor required')
    token_path=path.parent/'tokenizer.json'
    if file_hash(token_path)!=raw.get('tokenizer_sha256'):raise ValueError('Exact packaged own tokenizer required')
    tokenizer=Tokenizer.from_file(str(token_path));payload=raw['complete_dense']
    if (tokenizer.get_vocab_size()!=payload['config']['vocab_size']
            or [tokenizer.token_to_id(s) for s in ['<pad>','<bos>','<eos>']]!=[0,1,2]):
        raise ValueError('One exact native vocabulary and boundary convention required')
    # Loading must not advance a caller's generator, including auxiliary-head
    # initialization before those complete own weights are imported.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        model=install(OwnDenseAssistant(new_backbone(ModelConfig(**payload['config']),seed),**payload['settings'])).eval()
        model.import_export(payload,expected_sources=expected_sources)
    if (model.counts()!=raw['actual_counts'] or model.counts()['total_unique']>=10**9
            or model.network.embedding.weight is not model.network.lm_head.weight
            or file_hash(path)!=expected_checkpoint_sha256 or file_hash(token_path)!=raw['tokenizer_sha256']):
        raise ValueError('Every actual complete value/count, single vocabulary and source byte must remain exact')
    return model,dict(actual_counts=model.counts(),completed_updates=raw['completed_updates'],
        tokenizer_sha256=raw['tokenizer_sha256'],training_options=dict(image_size=model.config.max_image_size),provenance=raw['provenance'],
        no_old_base_actor_optimizer_or_training_reference_loaded=True,model_quality_not_proven_by_loading=True)
