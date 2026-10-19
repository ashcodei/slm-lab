"""Real small shared-native PPO, full trajectories and independent oracles."""
import copy

import pytest
import torch

from slm_lab.codenet_checkpoint_v1 import capture_rng
from slm_lab.dense_ppo_v1 import action_outputs
from slm_lab.indexed_native_history_v2 import read_failure
from slm_lab.native_human_reward_v1 import FrozenSharedNativeReward, NativeRewardHeadStage
from slm_lab.native_shared_ppo_history_v1 import read_prefix
from slm_lab.native_shared_ppo_v1 import NativeSharedPPOStage, OPTIONS
from slm_lab.own_dense_actor_v1 import OwnDenseActor
from slm_lab.parallel_function_repair_stage_v1 import value_digest
from slm_lab.whole_human_sft_evidence_v1 import tensor_digest
from test_native_human_reward_v1 import setup
from test_own_aligned_assistant_v1 import original_signature
from test_own_alignment_checkpoint_v1 import assert_nested_equal

BUDGETS = dict(maximum_records=3, maximum_compact_record_bytes=256*1024,
    maximum_expanded_record_bytes=256*1024, maximum_logical_bytes=16*1024**2,
    maximum_physical_bytes=64*1024**2, minimum_free_bytes=0)


def stage(tmp_path, name='history', start=True):
    base, data, _, features, setting, opt = setup()
    head = NativeRewardHeadStage(base, opt, features, setting)
    for _ in range(setting['steps']): head.update()
    reward = FrozenSharedNativeReward(base, head.finish(), allow_mechanics=True)
    model = OwnDenseActor(base, rank=2).eval()
    options = {**OPTIONS, 'steps': 3, 'max_new_tokens': 8, 'learning_rate': .001}
    optimizer = torch.optim.AdamW([p for _, p in model.named_selected(('actor', 'critic'))],
        lr=options['learning_rate'], betas=tuple(options['betas']), weight_decay=options['weight_decay'], foreach=False)
    sources = {**reward.provenance['sources'], 'human_reward_bundle_sha256': 'd'*64, 'tokenizer_sha256': 'e'*64}
    learner = NativeSharedPPOStage(model, optimizer, data, reward, options, sources, allow_mechanics=True)
    if start: learner.start_history(tmp_path/name, BUDGETS)
    return learner


def test_complete_unrepeated_epoch_real_tail_shared_values_and_constant_checkpoint_tip(tmp_path):
    learner = stage(tmp_path); base = original_signature(learner.model.base)
    digest = tensor_digest(learner.model.base.named_parameters())
    head_score = learner.reward.score(torch.tensor([[1, 4, 7, 2]]))
    initial = tensor_digest(learner.model.parts.named_parameters()); rows = []
    for _ in range(3): rows.append(learner.update())
    assert [len(r['source_positions']) for r in rows] == [2, 2, 1]
    assert sorted(p for r in rows for p in r['source_positions']) == list(range(5))
    assert not hasattr(learner, 'metrics') and not hasattr(learner, 'last_record')
    state = learner.state(); assert 'metrics' not in state and 'rollouts' not in state['history']
    result = learner.finish()
    assert result['complete'] and result['actual_updates'] == 3 and result['original_train_prompts'] == 5
    assert not result['full_original_PPO_recipe_complete'] and not result['useful_own_output_alignment_proved']
    assert result['additional_reward_encoder_parameters'] == 0
    assert learner.model.base is learner.reward.base
    union = {id(p): p for m in [learner.model, learner.reward] for p in m.parameters()}
    assert sum(p.numel() for p in union.values()) == learner.model.counts()['actual_unique']
    assert learner.model.parts['critic'].weight.count_nonzero() > 0
    assert initial != tensor_digest(learner.model.parts.named_parameters())
    assert base == original_signature(learner.model.base) and digest == tensor_digest(learner.model.base.named_parameters())
    assert torch.equal(head_score, learner.reward.score(torch.tensor([[1, 4, 7, 2]])))
    assert all(r['old_logps'] == r['actual_sampling_logps'] for row in rows for r in row['rollouts'])


def test_real_shared_actor_critic_update_matches_independent_joint_PPO_Adam_oracle(tmp_path):
    learner = stage(tmp_path); oracle = copy.deepcopy(learner.model)
    parameters = [p for _, p in oracle.named_selected(('actor', 'critic'))]
    optimizer = torch.optim.AdamW(parameters, lr=.001, betas=(.9, .95), weight_decay=.01, foreach=False)
    torch.manual_seed(999271); rng = torch.get_rng_state().clone()
    rows = learner.collect(learner.positions(0)); logps, values = [], []
    for row in rows:
        ids = torch.tensor([row['prompt_ids']+row['response_ids']])
        lp, value = action_outputs(oracle, ids, len(row['prompt_ids']), learner.amp)
        logps.append(lp); values.append(value)
    logps, values = torch.cat(logps), torch.cat(values)
    old, advantage, old_value, returns = [torch.tensor([v for row in rows for v in row[k]])
        for k in ['old_logps', 'advantages', 'old_values', 'returns']]
    ratio = torch.exp(logps-old); clipped = torch.clamp(ratio, .8, 1.2)
    value_clipped = old_value+torch.clamp(values-old_value, -.2, .2)
    loss = -torch.minimum(ratio*advantage, clipped*advantage).mean()
    loss += .5*torch.maximum((values-returns)**2, (value_clipped-returns)**2).mean()/2
    loss.backward(inputs=parameters); torch.nn.utils.clip_grad_norm_(parameters, 1.); optimizer.step()
    torch.set_rng_state(rng); actual = learner.update()
    assert actual['rollouts'] == rows
    assert actual['token_weighted_PPO_loss'] == pytest.approx(float(loss.detach()), abs=2e-7, rel=2e-6)
    for (_, a), (_, b) in zip(learner.model.named_selected(('actor', 'critic')), oracle.named_selected(('actor', 'critic'))):
        assert torch.allclose(a, b, atol=3e-7, rtol=3e-5)
    for a, b in zip(learner.optimizer.state.values(), optimizer.state.values()):
        assert torch.allclose(a['exp_avg'], b['exp_avg'], atol=3e-7, rtol=3e-5)
        assert torch.allclose(a['exp_avg_sq'], b['exp_avg_sq'], atol=3e-7, rtol=3e-5)


def test_next_actions_Adam_all_private_RNG_and_full_prefix_resume_without_replacing_old_future(tmp_path):
    learner = stage(tmp_path); torch.manual_seed(999371)
    learner.update(); saved = learner.state(); pristine = copy.deepcopy(saved)
    expected = learner.update(); learned = learner.model.export(learner.sources)
    adam = copy.deepcopy(learner.optimizer.state_dict()); rng = value_digest(capture_rng('cpu'))
    old_files = {p.name: p.read_bytes() for p in (tmp_path/'history').iterdir() if p.is_file() and 'reserved' not in p.name}
    result = learner.resume(saved)
    assert result['complete_updates'] == 1 and result['original_prompt_draws'] == 2
    with pytest.raises(ValueError, match='writer'): learner.update()
    learner.start_history(tmp_path/'recovery', BUDGETS)
    assert learner.update() == expected
    assert_nested_equal(learned, learner.model.export(learner.sources)); assert_nested_equal(adam, learner.optimizer.state_dict())
    assert_nested_equal(saved, pristine); assert value_digest(capture_rng('cpu')) == rng
    for name, raw in old_files.items(): assert (tmp_path/'history'/name).read_bytes() == raw
    learner.update(); result = learner.finish(); assert result['actual_updates'] == 3
    current = learner.state()
    inspected = read_prefix(current['history'], binding=learner.binding, budgets=BUDGETS, validate=learner.validate_record)
    assert inspected['segments'] == 2 and inspected['complete_updates'] == 3 and inspected['original_prompt_draws'] == 5


def test_corrupt_neural_RNG_source_and_history_tips_reject_before_any_live_mutation(tmp_path):
    learner = stage(tmp_path); learner.update(); saved = learner.state()
    weights = learner.model.export(learner.sources); adam = copy.deepcopy(learner.optimizer.state_dict())
    rng = value_digest(capture_rng('cpu')); before = original_signature(learner.model)
    changes = [lambda s: s['binding']['sources'].update(human_reward_bundle_sha256='0'*64),
        lambda s: s['optimizer']['state'][0]['exp_avg'].fill_(float('nan')),
        lambda s: s['optimizer']['state'][0]['exp_avg_sq'].fill_(-1.),
        lambda s: s['policy']['parts']['critic.weight'].fill_(float('nan')),
        lambda s: s['rng'].update(torch=torch.zeros(4, dtype=torch.uint8)),
        lambda s: s['history'].update(last_receipt_sha256='0'*64),
        lambda s: s['history'].update(segment_records=2),
        lambda s: s['cursor'].update(actual_prompt_draws=3)]
    for change in changes:
        corrupt = copy.deepcopy(saved); change(corrupt)
        with pytest.raises(ValueError): learner.resume(corrupt)
        assert before == original_signature(learner.model)
        assert_nested_equal(weights, learner.model.export(learner.sources)); assert_nested_equal(adam, learner.optimizer.state_dict())
        assert value_digest(capture_rng('cpu')) == rng


def test_corrupt_actual_trajectory_bytes_reject_before_policy_restore(tmp_path):
    learner = stage(tmp_path); learner.update(); saved = learner.state(); learner.update()
    path = tmp_path/'history'/'record-000000000001.json.gz'
    payload = bytearray(path.read_bytes()); payload[len(payload)//2] ^= 1; path.write_bytes(payload)
    before = original_signature(learner.model); weights = learner.model.export(learner.sources)
    adam = copy.deepcopy(learner.optimizer.state_dict()); rng = value_digest(capture_rng('cpu'))
    with pytest.raises(ValueError, match='observation byte'): learner.resume(saved)
    assert before == original_signature(learner.model)
    assert_nested_equal(weights, learner.model.export(learner.sources)); assert_nested_equal(adam, learner.optimizer.state_dict())
    assert value_digest(capture_rng('cpu')) == rng


def test_reservation_failure_precedes_sampling_and_independent_post_Adam_failure_is_not_a_checkpoint(tmp_path, monkeypatch):
    learner = stage(tmp_path, start=False)
    budgets = {**BUDGETS, 'maximum_physical_bytes': 32*1024}
    learner.start_history(tmp_path/'too-small', budgets)
    before = original_signature(learner.model); rng = value_digest(capture_rng('cpu'))
    monkeypatch.setattr(learner, 'collect', lambda *a: pytest.fail('No sampling before actual full-record capacity'))
    with pytest.raises(ValueError, match='fit before sampling'): learner.update()
    assert before == original_signature(learner.model) and not learner.optimizer.state
    assert value_digest(capture_rng('cpu')) == rng
    learner = stage(tmp_path, name='failed-Adam')
    def failed(): raise RuntimeError('injected actual Adam boundary failure')
    monkeypatch.setattr(learner.optimizer, 'step', failed)
    with pytest.raises(RuntimeError, match='injected actual Adam'): learner.update()
    assert learner.failed and learner.completed == 0
    with pytest.raises(ValueError): learner.state()
    receipt = learner.history.writer.failure_artifact
    payload = read_failure(tmp_path/'failed-Adam', receipt, maximum_bytes=256*1024)
    observation = payload['original_observation']
    assert not payload['successful_optimizer_commit_certified']
    assert observation['completed_policy_update_not_certified'] and observation['rollouts']
    assert not observation['optimizer_step_returned']
    assert (tmp_path/'failed-Adam'/'record-000000000001.json.gz').exists()


def test_train_role_full_scope_and_whole_action_budget_cannot_be_reduced_to_admit_partial_run(tmp_path):
    learner = stage(tmp_path, start=False)
    args = (learner.model, learner.optimizer, learner.data, learner.reward)
    with pytest.raises(ValueError, match='all85786train'): NativeSharedPPOStage(*args, learner.options, learner.sources)
    learner.data.role = 'valid'
    with pytest.raises(ValueError, match='train-only'): NativeSharedPPOStage(*args, learner.options, learner.sources, allow_mechanics=True)
    learner.data.role = 'train'
    with pytest.raises(ValueError, match='unrepeated'): NativeSharedPPOStage(*args, {**learner.options, 'steps': 2}, learner.sources, allow_mechanics=True)
    with pytest.raises(ValueError, match='without truncation'):
        NativeSharedPPOStage(*args, {**learner.options, 'max_new_tokens': learner.model.config.max_text_length}, learner.sources, allow_mechanics=True)


def test_action_EOS_and_scoring_only_EOS_are_distinct_and_every_saved_GAE_is_checked(tmp_path, monkeypatch):
    learner = stage(tmp_path); original = torch.multinomial
    monkeypatch.setattr(torch, 'multinomial', lambda probabilities, _: torch.tensor([2]))
    rows = learner.collect(learner.positions(0))
    assert all(r['response_ids'] == [2] and r['termination'] == 'EOS' and not r['scoring_only_EOS_added'] for r in rows)
    monkeypatch.setattr(torch, 'multinomial', lambda probabilities, _: torch.tensor([7]))
    rows = learner.collect(learner.positions(0))
    assert all(len(r['response_ids']) == 8 and r['termination'] == 'response_budget'
        and r['scoring_only_EOS_added'] and r['scoring_ids'][-1] == 2 for r in rows)
    # This controlled sampler chooses an action but the collector still saves
    # the real policy probability of that action, never a fabricated EOS action.
    monkeypatch.setattr(torch, 'multinomial', original)
    actual = learner.update(); corrupt = copy.deepcopy(actual)
    corrupt['rollouts'][0]['returns'][0] += .25
    with pytest.raises(ValueError, match='GAE'): learner.validate_record(corrupt)


def test_independent_cold_actor_and_serialized_private_state_reproduce_next_update(tmp_path):
    learner = stage(tmp_path); torch.manual_seed(999471); learner.update()
    saved = learner.state(); path = tmp_path/'private.pt'; torch.save(saved, path)
    expected = learner.update(); weights = learner.model.export(learner.sources)
    adam = copy.deepcopy(learner.optimizer.state_dict()); rng = value_digest(capture_rng('cpu'))
    # Reconstruct the actor and ordinary optimizer independently around the
    # exact same fixed native/reward. No trained actor object is reused.
    model = OwnDenseActor(learner.model.base, rank=2).eval()
    optimizer = torch.optim.AdamW([p for _, p in model.named_selected(('actor', 'critic'))],
        lr=.001, betas=(.9, .95), weight_decay=.01, foreach=False)
    cold = NativeSharedPPOStage(model, optimizer, learner.data, learner.reward,
        learner.options, learner.sources, allow_mechanics=True)
    restored = torch.load(path, map_location='cpu', weights_only=True)
    cold.resume(restored); cold.start_history(tmp_path/'cold', BUDGETS)
    assert cold.update() == expected
    assert_nested_equal(weights, cold.model.export(cold.sources)); assert_nested_equal(adam, cold.optimizer.state_dict())
    assert value_digest(capture_rng('cpu')) == rng
    final = cold.state(); other = tmp_path/'private-later.pt'; torch.save(final, other)
    assert abs(other.stat().st_size-path.stat().st_size) < 16*1024
    assert len(str(final['history'])) < 2000
    assert_nested_equal(saved, restored)


def test_failure_after_real_Adam_preserves_full_actions_but_never_certifies_checkpoint(tmp_path, monkeypatch):
    learner = stage(tmp_path)
    def fail(_): raise RuntimeError('injected failure after the real optimizer returned')
    monkeypatch.setattr(learner.history, 'acknowledge', fail)
    with pytest.raises(RuntimeError, match='real optimizer'): learner.update()
    assert learner.completed == 0 and learner.failed
    assert learner.optimizer.state and all(float(v['step']) == 1 for v in learner.optimizer.state.values())
    payload = read_failure(tmp_path/'history', learner.history.writer.failure_artifact, maximum_bytes=256*1024)
    assert payload['original_observation']['optimizer_step_returned']
    assert payload['original_observation']['rollouts']
    assert not payload['successful_optimizer_commit_certified']
    with pytest.raises(ValueError): learner.state()
