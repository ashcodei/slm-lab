import copy

import pytest
import torch

from slm_lab.dense_ppo_cached_v2 import DenseCachedPPOStage,validate_cached
from slm_lab.dense_ppo_v1 import action_outputs
from test_dense_ppo_v1 import stage as old_stage
from test_own_alignment_checkpoint_v1 import assert_nested_equal
from test_own_aligned_assistant_v1 import original_signature


def stage():
    old=old_stage();options={**old.options,'max_new_tokens':8}
    return DenseCachedPPOStage(old.model,old.optimizer,old.data,old.reward,options,old.sources)


def test_actual_cached_sampling_uses_one_prefill_then_single_token_decodes_and_matches_full_causal_probabilities(monkeypatch):
    learner=stage();calls=[];prefill=learner.model.policy.prefill;decode=learner.model.policy.decode
    def start(ids):calls.append(('prefill',ids.shape[1]));return prefill(ids)
    def step(ids,cache):calls.append(('decode',ids.shape[1]));return decode(ids,cache)
    monkeypatch.setattr(learner.model.policy,'prefill',start);monkeypatch.setattr(learner.model.policy,'decode',step)
    torch.manual_seed(952131);rows=learner.collect(learner.positions(0));validate_cached(rows)
    assert sum(k=='prefill' for k,_ in calls)==2
    assert sum(k=='decode' for k,_ in calls)==sum(len(r['response_ids'])-1 for r in rows)
    assert all(n==1 for k,n in calls if k=='decode')
    for row in rows:
        assert row['old_logps']==row['actual_sampling_logps']
        with torch.no_grad():
            ids=torch.tensor([row['prompt_ids']+row['response_ids']]);full,values=action_outputs(learner.model,ids,len(row['prompt_ids']),learner.amp)
        assert full.tolist()==row['full_causal_logps'] and values.tolist()==row['old_values']


def test_cached_actor_critic_next_actions_Adam_rng_recovery_and_bad_sampler_reject_before_mutation():
    learner=stage();fixed=original_signature(learner.model.base);reward=original_signature(learner.reward)
    torch.manual_seed(952231);learner.update();saved=learner.state();pristine=copy.deepcopy(saved)
    expected=learner.update();weights=learner.model.export(learner.sources);adam=copy.deepcopy(learner.optimizer.state_dict());rng=torch.get_rng_state().clone()
    learner.resume(saved);assert learner.update()==expected
    assert_nested_equal(weights,learner.model.export(learner.sources));assert_nested_equal(adam,learner.optimizer.state_dict());assert_nested_equal(saved,pristine)
    assert torch.equal(rng,torch.get_rng_state()) and fixed==original_signature(learner.model.base) and reward==original_signature(learner.reward)
    for mutate in [lambda s:s['sampler'].update(precision='bf16'),
        lambda s:s['core']['metrics'][0]['rollouts'][0].update(actual_decode_calls=999),
        lambda s:s['core']['metrics'][0]['rollouts'][0]['full_causal_logps'].__setitem__(0,-999.)]:
        bad=copy.deepcopy(saved);mutate(bad);before=original_signature(learner.model)
        with pytest.raises(ValueError):learner.resume(bad)
        assert before==original_signature(learner.model);assert_nested_equal(weights,learner.model.export(learner.sources))
        assert_nested_equal(adam,learner.optimizer.state_dict());assert torch.equal(rng,torch.get_rng_state())


def test_scoring_EOS_does_not_add_a_decode_call_or_action_and_stale_cache_rejects(monkeypatch):
    learner=stage();monkeypatch.setattr(torch,'multinomial',lambda p,n:torch.tensor([7]))
    rows=learner.collect(learner.positions(0))
    assert all(r['response_ids']==[7]*8 and r['actual_decode_calls']==7 and r['scoring_only_EOS_added'] for r in rows)
    assert all(r['scoring_ids']==r['prompt_ids']+[7]*8+[2] for r in rows)
    learner.model.eval();cache=learner.model.policy.prefill(torch.tensor([[1,4]]))['cache'];learner.update()
    learner.model.eval()
    with pytest.raises(ValueError,match='unchanged'):learner.model.policy.decode(torch.tensor([[7]]),cache)


def test_precision_contract_rejects_autocast_configuration_and_sampler_disables_ambient_autocast():
    old=old_stage()
    with pytest.raises(ValueError,match='FP32'):
        DenseCachedPPOStage(old.model,old.optimizer,old.data,old.reward,old.options,old.sources,amp=lambda:torch.autocast('cpu',dtype=torch.bfloat16))
    learner=stage();torch.manual_seed(952331);rng=torch.get_rng_state().clone();expected=learner.collect(learner.positions(0))
    torch.set_rng_state(rng)
    with torch.autocast('cpu',dtype=torch.bfloat16):actual=learner.collect(learner.positions(0))
    assert expected==actual
