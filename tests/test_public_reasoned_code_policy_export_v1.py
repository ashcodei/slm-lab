"""Real small PPO-origin critic/reward survive actual whole-response SFT."""
import copy

import pytest
import torch

from slm_lab.corpus import file_hash
from slm_lab.own_policy_reward_standalone_v1 import load
from slm_lab.parallel_function_repair_stage_v1 import value_digest
from slm_lab.parallel_public_reasoned_code_SFT_stage_v1 import ParallelPublicReasonedCodeSFTStage
from slm_lab.public_reasoned_code_policy_export_v1 import export
from slm_lab.public_reasoned_code_train_data_v1 import PublicReasonedCodeSFTData
from slm_lab.public_reasoned_code_training_controller_v1 import TRAINING
from slm_lab.whole_human_sft_evidence_v1 import tensor_digest
from test_own_policy_reward_standalone_v1 import package
from test_parallel_public_reasoned_code_SFT_stage_v1 import OPTIONS
from test_public_reasoned_code_train_data_v1 import packet


def learned(root,profile):
    _,old_sources,initial=package(root)
    bundle,_=load(initial,expected_checkpoint_sha256=file_hash(initial),expected_sources=old_sources,allow_mechanics=True)
    packet(root/'corpus')
    data=PublicReasonedCodeSFTData(root/'corpus')
    actor=bundle.actor
    sources={**old_sources,'public_code_manifest_sha256':data.manifest_sha256,
        'public_code_admission_sha256':data.admission_sha256,
        'initial_policy_critic_sha256':tensor_digest(actor.parts['critic'].named_parameters())}
    assert bool((actor.parts['critic'].weight!=0).any())
    before=dict(base=tensor_digest(actor.base.named_parameters()),critic=actor.parts['critic'].weight.detach().clone(),
        actor=tensor_digest(actor.parts.named_parameters()),reward=bundle.reward.score(torch.tensor([[1,4,7,2]])).detach().clone())
    adam=torch.optim.AdamW([p for _,p in actor.named_selected(('actor',))],lr=OPTIONS['learning_rate'],weight_decay=0.,foreach=False)
    entries=data.complete_schedule(epochs=2,width=2,seed=OPTIONS['seed'])
    runner=ParallelPublicReasonedCodeSFTStage(actor,adam,data,entries,profile,OPTIONS,sources,mode='mechanics')
    while runner.completed<len(entries):runner.update()
    state,_=runner.snapshot()
    actor.eval()
    receipt=dict(format=TRAINING,complete=True,mode='mechanics',profile=profile,world=1,
        actual_complete_updates=runner.completed,actual_research_actor_parameters=actor.counts()['actual_unique'],
        actual_native_parameters=actor.base.counts()['total_unique'],every_original_complete_source_and_CE_target_retained=True,
        original_complete_training_budget=data.budget(entries),actual_final_actor_sha256=tensor_digest(actor.parts.named_parameters()),
        actual_final_Adam_sha256=value_digest(adam.state_dict()),production_updates=0,candidate_promoted=False,goal_achieved=False)
    return runner,bundle.reward,state,receipt,sources,initial,before


@pytest.mark.parametrize('profile',['plain_tokens','eos_weight4'])
def test_actual_SFT_export_and_separate_cold_keep_all_policy_reward_multimodal_and_nonzero_critic_values(tmp_path,profile):
    runner,reward,state,receipt,sources,initial,before=learned(tmp_path,profile)
    actor=runner.model
    assert tensor_digest(actor.base.named_parameters())==before['base']
    assert tensor_digest(actor.parts.named_parameters())!=before['actor']
    assert torch.equal(actor.parts['critic'].weight,before['critic'])
    assert torch.equal(reward.score(torch.tensor([[1,4,7,2]])),before['reward'])
    prior=tensor_digest(actor.named_parameters());rng=torch.get_rng_state().clone()
    payload,proof=export(actor,reward,state,receipt,data=runner.data,sources=sources,
        tokenizer_sha256=sources['tokenizer_sha256'],allow_mechanics=True)
    assert tensor_digest(actor.named_parameters())==prior and torch.equal(rng,torch.get_rng_state())
    assert proof['inherited_policy_critic_and_shared_reward_preserved'] and not proof['whole_production_code_source_schedule_completed']
    folder=tmp_path/'new-package';folder.mkdir()
    (folder/'tokenizer.json').write_bytes((initial.parent/'tokenizer.json').read_bytes())
    path=folder/'model.pt';torch.save(payload,path)
    cold,meta=load(path,expected_checkpoint_sha256=file_hash(path),expected_sources=sources,allow_mechanics=True)
    assert tensor_digest(cold.actor.named_parameters())==prior
    assert cold.actor.base is cold.reward.base and cold.counts()['additional_reward_encoder_values']==0
    assert meta['trained_reward_feature_basis_preserved']
    assert not ({p.data_ptr() for p in actor.parameters()}&{p.data_ptr() for p in cold.parameters()})
    ids=torch.tensor([[1,4,7,2]])
    side=actor.config.max_image_size
    pixels=torch.linspace(-1,1,3*side*side).reshape(1,3,side,side)
    words=torch.linspace(-1,1,3*8*32).reshape(1,3,8,32)
    with torch.no_grad():
        for image in [None,pixels]:
            a,b=actor(ids,image),cold(ids,image)
            assert torch.equal(a['logits'],b['logits']) and torch.equal(a['values'],b['values'])
        assert torch.equal(actor.base.word_logits(words),cold.word_logits(words))
        assert torch.equal(reward.score(ids),cold.preference_score(ids))
        assert torch.equal(actor.generate(ids[:,:-1],max_new_tokens=4,eos_token_id=2,use_cache=True),
            cold.generate(ids[:,:-1],max_new_tokens=4,eos_token_id=2,use_cache=True))
    assert torch.equal(rng,torch.get_rng_state()) and not torch.cuda.is_initialized()


def test_partial_budget_changed_Adam_critic_source_and_fake_production_reject_without_export_mutation(tmp_path):
    runner,reward,state,receipt,sources,_,_=learned(tmp_path,'eos_weight4')
    prior=tensor_digest(runner.model.named_parameters());rng=torch.get_rng_state().clone()
    for corruption in ['partial','budget','Adam','critic_binding','production_claim','missing_opt_in']:
        actual=copy.deepcopy(receipt);bound=dict(sources)
        if corruption=='partial':actual['actual_complete_updates']-=1
        if corruption=='budget':actual['original_complete_training_budget']['response_EOS_targets']-=1
        if corruption=='Adam':actual['actual_final_Adam_sha256']='0'*64
        if corruption=='critic_binding':bound['initial_policy_critic_sha256']='0'*64
        if corruption=='production_claim':actual['mode']='train'
        with pytest.raises(ValueError):export(runner.model,reward,state,actual,data=runner.data,sources=bound,
            tokenizer_sha256=sources['tokenizer_sha256'],allow_mechanics=corruption!='missing_opt_in')
        assert tensor_digest(runner.model.named_parameters())==prior and torch.equal(rng,torch.get_rng_state())
