"""Exact policy/reward semantics and independent self-contained loading."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from slm_lab.corpus import file_hash
from slm_lab.deterministic_positions_v1 import install
from slm_lab.native_human_reward_v1 import FrozenSharedNativeReward
from slm_lab.own_dense_actor_fold_v1 import fold_actor
from slm_lab.own_policy_reward_standalone_v1 import StandalonePolicyReward,export,inspect_package,load
from slm_lab.whole_human_sft_evidence_v1 import tensor_digest
from test_native_shared_ppo_v1 import stage
from test_own_aligned_assistant_v1 import original_signature


def package(root):
    root.mkdir(exist_ok=True);learner=stage(root,name='training-history');install(learner.model.base)
    torch.manual_seed(999971)
    for _ in range(3):learner.update()
    learner.model.eval()
    vocab={'<pad>':0,'<bos>':1,'<eos>':2,**{f't{i}':i for i in range(3,learner.model.config.vocab_size)}}
    tokenizer=Tokenizer(WordLevel(vocab=vocab,unk_token='<pad>'));tokenizer.pre_tokenizer=Whitespace()
    folder=root/'portable';folder.mkdir();token_path=folder/'tokenizer.json';tokenizer.save(str(token_path))
    sources={**learner.sources,'tokenizer_sha256':file_hash(token_path)}
    before=original_signature(learner.model);rng=torch.get_rng_state().clone()
    payload=export(learner.model,learner.reward,sources=sources,provenance=dict(small_CPU_PPO_control=True),
        completed_updates=3,tokenizer_sha256=file_hash(token_path),allow_mechanics=True)
    assert before==original_signature(learner.model) and torch.equal(rng,torch.get_rng_state())
    path=folder/'model.pt';torch.save(payload,path)
    return learner,sources,path


def cold(path,sources):
    return load(path,expected_checkpoint_sha256=file_hash(path),expected_sources=sources,allow_mechanics=True)


def test_unfused_delivery_keeps_trained_reward_feature_basis_and_every_policy_image_word_output(tmp_path):
    learner,sources,path=package(tmp_path);loaded,metadata=cold(path,sources)
    size=learner.model.config.max_image_size
    ids=torch.tensor([[1,4,7,2]]);pixels=torch.linspace(-1,1,3*size*size).reshape(1,3,size,size)
    word=torch.linspace(-1,1,3*8*32).reshape(1,3,8,32)
    assert loaded.actor.base is loaded.reward.base and loaded.base.network.embedding.weight is loaded.base.network.lm_head.weight
    assert sum(p.numel() for p in loaded.parameters())==learner.model.counts()['actual_unique']
    assert loaded.counts()['additional_reward_encoder_values']==0 and loaded.counts()['mechanics_only']
    assert tensor_digest(loaded.actor.named_parameters())==tensor_digest(learner.model.named_parameters())
    with torch.no_grad():
        for image in [None,pixels]:
            expected=learner.model(ids,image);actual=loaded(ids,image)
            assert torch.equal(expected['logits'],actual['logits']) and torch.equal(expected['values'],actual['values'])
        assert torch.equal(learner.model.base.word_logits(word),loaded.word_logits(word))
        assert torch.equal(learner.reward.score(ids),loaded.preference_score(ids))
    folded,_=fold_actor(learner.model,sources=sources,working_budget_bytes=512*1024**2)
    with torch.no_grad():assert not torch.equal(learner.reward.score(ids),folded.preference_score(ids).reshape(()))
    with pytest.raises(ValueError,match='actually completed'):FrozenSharedNativeReward(folded,learner.reward.provenance,allow_mechanics=True)
    assert metadata['trained_reward_feature_basis_preserved'] and not metadata['old_base_actor_optimizer_or_training_features_loaded']


def test_exact_independent_cached_actions_and_private_caller_RNG_preservation(tmp_path):
    learner,sources,path=package(tmp_path);rng=torch.get_rng_state().clone();loaded,_=cold(path,sources)
    assert torch.equal(rng,torch.get_rng_state())
    ids=torch.tensor([[1,4,7]])
    with torch.no_grad():
        a=learner.model.policy.prefill(ids);b=loaded.policy.prefill(ids)
        assert torch.equal(a['logits'],b['logits'])
        for token in [8,9,4]:
            a=learner.model.policy.decode(torch.tensor([[token]]),a['cache'])
            b=loaded.policy.decode(torch.tensor([[token]]),b['cache'])
            assert torch.equal(a['logits'],b['logits'])
        expected=learner.model.generate(ids,max_new_tokens=8,eos_token_id=2,use_cache=True)
        assert torch.equal(expected,loaded.generate(ids,max_new_tokens=8,eos_token_id=2,use_cache=True))
    assert torch.equal(rng,torch.get_rng_state()) and not torch.cuda.is_initialized()


def test_corrupt_policy_native_reward_sources_and_vocab_reject_without_touching_source_or_RNG(tmp_path):
    learner,sources,path=package(tmp_path);original=torch.load(path,map_location='cpu',weights_only=True)
    before=original_signature(learner.model);rng=torch.get_rng_state().clone()
    modifications=[lambda x:x['policy_adapter']['parts']['critic.weight'].fill_(float('nan')),
        lambda x:x['complete_base']['network']['norm.weight'].fill_(float('nan')),
        lambda x:x['policy_adapter']['parts'].pop(next(iter(x['policy_adapter']['parts']))),
        lambda x:x['reward_provenance'].update(trained_head_sha256='0'*64),
        lambda x:x['reward_provenance']['sources'].update(frozen_native_encoder_sha256='0'*64),
        lambda x:x.update(policy_parameters_sha256='0'*64),
        lambda x:x['sources'].update(human_reward_bundle_sha256='0'*64)]
    for change in modifications:
        changed=copy.deepcopy(original);change(changed);torch.save(changed,path)
        with pytest.raises(ValueError):cold(path,sources)
        assert before==original_signature(learner.model) and torch.equal(rng,torch.get_rng_state())
    torch.save(original,path);(path.parent/'tokenizer.json').write_text('{}')
    with pytest.raises(ValueError,match='tokenizer'):cold(path,sources)
    assert before==original_signature(learner.model) and torch.equal(rng,torch.get_rng_state())


def test_second_reward_encoder_training_or_mutation_cannot_enter_read_only_delivery(tmp_path):
    learner,sources,path=package(tmp_path)
    with pytest.raises(ValueError,match='Small CPU'):load(path,expected_checkpoint_sha256=file_hash(path),expected_sources=sources)
    separate=copy.deepcopy(learner.model.base).eval()
    reward=FrozenSharedNativeReward(separate,learner.reward.provenance,allow_mechanics=True)
    with pytest.raises(ValueError,match='Exactly one'):StandalonePolicyReward(learner.model,reward,allow_mechanics=True)
    loaded,_=cold(path,sources)
    with pytest.raises(ValueError,match='immutable'):loaded.train()
    with torch.no_grad():loaded.actor.parts['critic'].weight.add_(.01)
    with pytest.raises(ValueError,match='remain fixed'):loaded(torch.tensor([[1,4,7]]))


def test_real_separate_process_queries_and_scores_only_packaged_model_and_tokenizer(tmp_path):
    learner,sources,path=package(tmp_path)
    before=original_signature(learner.model)
    command=[sys.executable,'-m','slm_lab.own_policy_reward_query_v1','--checkpoint',str(path),
        '--prompt','t4','--raw-prompt','--threads','1','--allow-small-cpu-control']
    env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'}
    out=subprocess.check_output(command+['--score-response','t7'],env=env,text=True)
    result=json.loads(out)
    assert result['shared_human_reward_score']==float(learner.reward.score(torch.tensor([[1,4,7,2]])))
    assert result['actual_registered_parameters']==learner.model.counts()['actual_unique']
    assert result['complete_scoring_token_ids']==[1,4,7,2]
    assert result['learned_human_ranking_score_not_a_program_correctness_certificate']
    actual=json.loads(subprocess.check_output(command+['--max-new-tokens','4','--json'],env=env,text=True))
    assert 1<=actual['actual_native_actions']<=4 and actual['complete_prompt_token_ids']==[1,4]
    assert actual['trained_reward_feature_basis_preserved'] and not actual['model_quality_proved']
    assert before==original_signature(learner.model)


def test_inspection_never_constructs_neural_graph_and_does_not_claim_tensor_equivalence(tmp_path,monkeypatch):
    import slm_lab.own_policy_reward_standalone_v1 as implementation
    _,sources,path=package(tmp_path)
    def forbidden(*a,**k):pytest.fail('Metadata inspection must not initialize a neural graph')
    monkeypatch.setattr(implementation,'new_backbone',forbidden)
    result=inspect_package(path,expected_checkpoint_sha256=file_hash(path),expected_sources=sources)
    assert result['inspect_only'] and result['registered_values_and_neural_equivalence_not_proved_by_inspection']
