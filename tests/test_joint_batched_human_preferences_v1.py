import copy
import hashlib
import json

import pytest
import torch
from torch.nn import functional as F

from slm_lab.code_priority_gradient_blend_v1 import combine
from slm_lab.codenet_checkpoint_v1 import capture_rng
from slm_lab.corpus import file_hash
from slm_lab.frozen_own_text_reference_v1 import FrozenOwnTextReferenceV1
from slm_lab.joint_batched_human_preferences_v1 import gradients, plan
from slm_lab.parallel_function_repair_stage_v1 import value_digest
from slm_lab.shared_joint_checkpoint_v1 import FixedGuard
from slm_lab.whole_human_data_v2 import WholeHumanData
from slm_lab.whole_human_sft_evidence_v1 import tensor_digest
from test_dense_human_preference_v1 import pairs
from test_shared_joint_actor_vision_v1 import setup, gradients as task_gradients


def fixture(path):
    view, ids, pixels, words = setup(path)
    data = pairs()
    data.batch = WholeHumanData.batch.__get__(data, WholeHumanData)
    reference = FrozenOwnTextReferenceV1(view.actor.base, sources={'synthetic_source': 'a' * 64},
        tokenizer_sha256=file_hash(path / 'tokenizer.json'))
    return view, reference, data, ids, pixels, words


def oracle_loss(view, data, DPO_weight, coefficient=.2):
    objective = 0.
    for position in [0, 1]:
        values = {}; reference = {}
        for field in ['chosen', 'rejected'] if DPO_weight else ['chosen']:
            batch = data.batch([position], field, 'cpu'); target = batch['labels'][:, 1:]; mask = target != -100
            output = view.text(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])['logits'][:, :-1].float().log_softmax(-1)
            values[field] = output[mask].gather(1, target[mask, None]).sum()
            if DPO_weight:
                with torch.no_grad(): output = view.actor.base(batch['input_ids'], attention_mask=batch['attention_mask'])['logits'][:, :-1].float().log_softmax(-1)
                reference[field] = output[mask].gather(1, target[mask, None]).sum()
        objective = objective - coefficient * values['chosen'] / 5
        if DPO_weight:
            margin = (values['chosen'] - values['rejected']) - (reference['chosen'] - reference['rejected'])
            objective = objective - F.logsigmoid(.1 * margin) / 2
    return objective


@pytest.mark.parametrize('DPO_weight', [0, 1])
def test_batched_whole_unequal_responses_match_independent_scalar_loss_and_gradients(tmp_path, DPO_weight):
    view, reference, data, *_ = fixture(tmp_path); oracle = copy.deepcopy(view)
    before = tensor_digest(view.named_parameters()), value_digest(capture_rng('cpu'))
    actual, record = gradients(view, reference if DPO_weight else None, data, [0, 1], DPO_weight=DPO_weight,
        chosen_CE_weight=.2, maximum_examples=2, maximum_padded_tokens=22)
    loss = oracle_loss(oracle, data, DPO_weight)
    wanted = task_gradients(oracle, loss)
    for name, value in actual.items():
        if wanted[name] is None: assert value is None
        else: assert torch.allclose(value, wanted[name], atol=3e-7, rtol=3e-5), name
    assert record['local_global_normalized_objective'] == pytest.approx(float(loss.detach()), rel=2e-6)
    assert len(record['actual_microbatch_positions']) == 1 and len(record['actual_microbatch_positions'][0]) == 2
    assert record['original_global_chosen_response_EOS_denominator'] == 5 and record['original_global_pair_denominator'] == 2
    assert record['actual_policy_model_calls'] == 1 + DPO_weight
    assert record['actual_reference_model_calls'] == 2 * DPO_weight
    assert record['reference_queried'] is bool(DPO_weight)
    assert before == (tensor_digest(view.named_parameters()), value_digest(capture_rng('cpu')))
    assert all(p.grad is None for p in view.parameters()) and not torch.cuda.is_initialized()


@pytest.mark.parametrize('DPO_weight', [0, 1])
def test_four_unequal_and_empty_replica_tasks_sum_to_the_full_batch_gradient(tmp_path, DPO_weight):
    view, reference, data, *_ = fixture(tmp_path)
    whole, _ = gradients(view, reference if DPO_weight else None, data, [0, 1], DPO_weight=DPO_weight, chosen_CE_weight=.2)
    shards = []
    for rank in range(4):
        values, report = gradients(view, reference if DPO_weight else None, data, [0, 1], DPO_weight=DPO_weight,
            chosen_CE_weight=.2, rank=rank, world=4)
        shards.append(values)
        if rank > 1:
            assert report['actual_policy_model_calls'] == report['actual_reference_model_calls'] == 0
            assert all(v is None for v in values.values())
    for name, value in whole.items():
        present = [shard[name] for shard in shards if shard[name] is not None]
        if value is None: assert not present
        else: assert torch.allclose(sum(present), value, atol=3e-7, rtol=3e-5)


@pytest.mark.parametrize('case', ['held', 'machine', 'duplicate', 'changed_prompt', 'pair_too_large', 'unused_reference', 'different_reference'])
def test_invalid_source_or_reference_rejects_before_any_model_or_RNG_mutation(tmp_path, case):
    view, reference, data, *_ = fixture(tmp_path); positions = [0, 1]; weight = 1; maximum = 22
    if case == 'held': data.role = 'valid'
    elif case == 'machine': data.rows[0]['feedback_source'] = 'machine'
    elif case == 'duplicate': positions = [0, 0]
    elif case == 'changed_prompt':
        original = data.sequence
        def changed(p, field):
            ids, labels = original(p, field); ids = list(ids)
            if field == 'rejected': ids[1] = 15
            return ids, labels
        data.sequence = changed
    elif case == 'pair_too_large': maximum = 8
    elif case == 'unused_reference': weight = 0
    else:
        copied = copy.deepcopy(view.actor.base)
        reference = FrozenOwnTextReferenceV1(copied, sources={'synthetic': 'b' * 64}, tokenizer_sha256='a' * 64)
    before = tensor_digest(view.named_parameters()), value_digest(capture_rng('cpu'))
    with pytest.raises(ValueError):
        gradients(view, reference, data, positions, DPO_weight=weight, maximum_padded_tokens=maximum)
    assert before == (tensor_digest(view.named_parameters()), value_digest(capture_rng('cpu')))
    assert all(p.grad is None for p in view.parameters())


def test_actual_joint_code_human_DPO_and_photo_update_matches_independent_complete_objective(tmp_path):
    view, reference, data, ids, pixels, _ = fixture(tmp_path); oracle = copy.deepcopy(view)
    optimizer = torch.optim.AdamW([p for _, p in view.named_selected()], lr=.001, weight_decay=.01, foreach=False)
    fixed = FixedGuard(view, optimizer); original_reference = reference(ids)
    human, record = gradients(view, reference, data, [0, 1], DPO_weight=1, chosen_CE_weight=.2,
        maximum_examples=2, maximum_padded_tokens=22)
    code = task_gradients(view, view.text(input_ids=ids, labels=ids)['loss'])
    photo = task_gradients(view, view.native_image(input_ids=ids, labels=ids, pixel_values=pixels)['loss'])
    side = {name: None if human[name] is None and photo[name] is None else
        (0 if human[name] is None else .2 * human[name]) + (0 if photo[name] is None else .3 * photo[name]) for name in code}
    blended, _ = combine(code, side, primary_names=view.task_names('text'), secondary_weight=1., project_conflicts=False)
    expected_loss = oracle.text(input_ids=ids, labels=ids)['loss'] + .2 * oracle_loss(oracle, data, 1)
    expected_loss = expected_loss + .3 * oracle.native_image(input_ids=ids, labels=ids, pixel_values=pixels)['loss']
    expected = task_gradients(oracle, expected_loss)
    for name, value in blended.items():
        if value is None: assert expected[name] is None
        else: assert torch.allclose(value, expected[name], atol=3e-7, rtol=3e-5), name
    for name, parameter in view.named_selected(): parameter.grad = blended[name]
    optimizer.step(); optimizer.zero_grad(set_to_none=True)
    fixed.check(view, optimizer, full=True); reference.assert_bound(full=True)
    assert torch.equal(reference(ids), original_reference)
    with torch.no_grad(): assert not torch.equal(view.text(input_ids=ids)['logits'], original_reference)
    assert all(p.grad is None for p in view.parameters()) and not torch.cuda.is_initialized()
    (tmp_path / 'actual-small-joint-code-human-DPO-photo-update.json').write_text(json.dumps(dict(
        actual_joint_AdamW_updates=1, actual_human_pairs=2, chosen_response_EOS_targets=5, rejected_response_EOS_targets=6,
        actual_batched_human_policy_calls=record['actual_policy_model_calls'], actual_batched_reference_calls=record['actual_reference_model_calls'],
        independent_complete_joint_gradient_oracle_passed=True, frozen_text_reference_logits_exact_after_joint_update=True,
        frozen_language_critic_other_heads_values_and_metadata_unchanged=True,
        synthetic_controls_only=True, full_size_GPUcapacity_or_speed_or_quality_gain=False), indent=2))
