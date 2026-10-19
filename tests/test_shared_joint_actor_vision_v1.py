"""Small own neural joint code/image/OCR gradients and sparse Adam recovery.

Synthetic fixtures do not establish real source admission, full-GPU capacity,
coding/visual generalization, or compatibility with earlier actor-only states.
"""
import copy
import json

import pytest
import torch
from torch.nn import functional as F

from slm_lab.code_priority_gradient_blend_v1 import combine
from slm_lab.corpus import file_hash
from slm_lab.frozen_own_text_reference_v1 import FrozenOwnTextReferenceV1
from slm_lab.own_dense_actor_v1 import OwnDenseActor
from slm_lab.own_modal_endpoint_fusion_v1 import fuse
from slm_lab.parallel_function_repair_stage_v1 import value_digest
from slm_lab.shared_joint_actor_vision_v1 import SharedJointActorVisionV1
from slm_lab.whole_human_sft_evidence_v1 import tensor_digest
from test_own_modal_endpoint_fusion_v1 import sources


def setup(tmp_path):
    text, visual, identities, ids, pixels, words = sources(tmp_path)
    dense, _ = fuse(text, visual, text_identity=identities[0], visual_identity=identities[1], working_budget_bytes=256 * 1024**2)
    actor = OwnDenseActor(dense, rank=2, seed=991401).eval()
    view = SharedJointActorVisionV1(actor).eval()
    return view, ids, pixels, words


def gradients(view, loss, *, retain=False):
    named = view.named_selected()
    grads = torch.autograd.grad(loss, [p for _, p in named], allow_unused=True, retain_graph=retain)
    return {n: g for (n, _), g in zip(named, grads)}


def test_actual_joint_plain_gradients_match_independent_total_loss_and_no_parameters_are_added(tmp_path):
    view, ids, pixels, _ = setup(tmp_path)
    assert view.actor.counts()['actual_unique'] == sum(p.numel() for p in view.parameters())
    assert {id(p) for p in view.parameters()} == {id(p) for p in view.actor.parameters()}
    primary = view.text(input_ids=ids, labels=ids)['loss']
    secondary = view.image_matching(dict(input_ids=ids, labels=ids, pixel_values=pixels), ids,
        torch.ones_like(ids, dtype=torch.bool), torch.eye(2, dtype=torch.bool))['loss']
    p, s = gradients(view, primary, retain=True), gradients(view, secondary, retain=True)
    mixed, _ = combine(p, s, primary_names=view.task_names('text'), secondary_weight=.3, project_conflicts=False)
    expected = gradients(view, primary + .3 * secondary)
    for name in mixed:
        if expected[name] is None: assert mixed[name] is None
        else: assert torch.allclose(mixed[name], expected[name], atol=2e-6, rtol=2e-5)
    assert all(p.grad is None for p in view.parameters())
    assert not torch.cuda.is_initialized()


def test_actual_code_image_then_code_CTC_shared_Adam_sparse_clocks_and_next_update_recovery(tmp_path):
    view, ids, pixels, words = setup(tmp_path)
    names = view.named_selected(); chosen = {id(p) for _, p in names}
    fixed = {n: (p.detach().clone(), p._version, p.data_ptr(), p.requires_grad) for n, p in view.named_parameters() if id(p) not in chosen}
    optimizer = torch.optim.AdamW([p for _, p in names], lr=.001, weight_decay=.01, foreach=False)
    old_actor_only = OwnDenseActor(copy.deepcopy(view.actor.base), rank=2, seed=991401).eval()
    def step(current, adam, kind):
        code = current.text(input_ids=ids, labels=ids)['loss']
        if kind == 'image':
            other = current.image_matching(dict(input_ids=ids, labels=ids, pixel_values=pixels), ids,
                torch.ones_like(ids, dtype=torch.bool), torch.eye(2, dtype=torch.bool))['loss']
        else:
            logits = current.word_logits(words)
            other = F.ctc_loss(logits.log_softmax(-1).transpose(0, 1), torch.tensor([97, 98]),
                torch.full((2,), logits.shape[1], dtype=torch.long), torch.ones(2, dtype=torch.long), blank=256)
        p, s = gradients(current, code), gradients(current, other)
        active = set(current.task_names('image_matching' if kind == 'image' else 'word_CTC'))
        assert {n for n, g in s.items() if g is not None} == active
        mixed, proof = combine(p, s, primary_names=current.task_names('text'), secondary_weight=.3, project_conflicts=True)
        for name, parameter in current.named_selected(): parameter.grad = mixed[name]
        torch.nn.utils.clip_grad_norm_([p for _, p in current.named_selected()], 1.)
        adam.step(); adam.zero_grad(set_to_none=True)
        return proof
    first = step(view, optimizer, 'image')
    with pytest.raises(ValueError, match='frozen dense features'):
        old_actor_only.import_export(view.actor.export({'synthetic_joint_control': 'a' * 64}),
            expected_sources={'synthetic_joint_control': 'a' * 64})
    assert any(not torch.equal(view.get_parameter(n), original[0]) for n, original in fixed.items()) is False
    saved_values = {n: p.detach().clone() for n, p in view.named_selected()}
    saved_adam = copy.deepcopy(optimizer.state_dict()); rng = torch.get_rng_state().clone()
    second = step(view, optimizer, 'word')
    by_name = dict(view.named_selected())
    assert float(optimizer.state[by_name['vision.project.weight']]['step']) == 1
    assert all(float(optimizer.state[p]['step']) == 1 for n, p in names if n.startswith(('word.', 'pointing.')))
    assert all(float(optimizer.state[p]['step']) == 2 for n, p in names if n.startswith('actor.'))
    for n, (value, version, pointer, flag) in fixed.items():
        parameter = view.get_parameter(n)
        assert torch.equal(parameter, value) and (parameter._version, parameter.data_ptr(), parameter.requires_grad) == (version, pointer, flag)
    # Restore only selected values/Adam into a separate graph. Frozen dense
    # language/critic/auxiliary tensor metadata is never rewritten.
    resumed = copy.deepcopy(view)
    with torch.no_grad():
        for n, p in resumed.named_selected(): p.copy_(saved_values[n])
    other_adam = torch.optim.AdamW([p for _, p in resumed.named_selected()], lr=.001, weight_decay=.01, foreach=False)
    other_adam.load_state_dict(saved_adam); torch.set_rng_state(rng)
    repeated = step(resumed, other_adam, 'word')
    assert second == repeated
    assert tensor_digest(view.named_parameters()) == tensor_digest(resumed.named_parameters())
    assert value_digest(optimizer.state_dict()) == value_digest(other_adam.state_dict())
    assert first['actual_primary_gradient_tensors'] > 0 and second['actual_secondary_gradient_tensors'] > 0
    assert not second['actual_projection_applied']
    assert all(p.grad is None for p in view.parameters()) and not torch.cuda.is_initialized()
    (tmp_path / 'actual-small-joint-code-image-word-Adam-recovery.json').write_text(json.dumps(dict(
        actual_small_parameter_counts=view.actor.counts(), added_neural_parameters_in_joint_view=0,
        source_fixture_training_calls=2, actual_joint_physical_AdamW_calls=3, actual_joint_logical_updates=2,
        actual_code_image_gradient_interaction=first, actual_code_word_gradient_interaction=second,
        exact_next_joint_parameters_Adam_and_gradient_interaction_recovery=True,
        original_frozen_language_critic_other_heads_values_metadata_unchanged=True,
        actor_steps=2, vision_projection_steps=1, pointing_steps=1, word_steps=1,
        incompatible_earlier_actor_only_state_rejected=True,
        durable_joint_state_and_full_source_4GPU_training_still_required=True,
        synthetic_controls_only=True, full_size_GPUcapacity_or_quality_gain=False), indent=2, allow_nan=False))


def test_actual_shared_text_reference_survives_joint_vision_word_actor_updates_without_extra_weights(tmp_path):
    view, ids, pixels, words = setup(tmp_path)
    source = tmp_path / 'text-actual-small-source.pt'
    reference = FrozenOwnTextReferenceV1(view.actor.base, sources={str(source): file_hash(source)},
        tokenizer_sha256=file_hash(tmp_path / 'tokenizer.json'))
    before = reference(ids)
    assert not before.requires_grad
    assert {id(p) for p in reference.parameters()} <= {id(p) for p in view.parameters()}
    named = view.named_selected()
    optimizer = torch.optim.AdamW([p for _, p in named], lr=.001, weight_decay=.01, foreach=False)
    image = view.native_image(input_ids=ids, labels=ids, pixel_values=pixels)['loss']
    code = view.text(input_ids=ids, labels=ids)['loss']
    logits = view.word_logits(words)
    words_loss = F.ctc_loss(logits.log_softmax(-1).transpose(0, 1), torch.tensor([97, 98]),
        torch.full((2,), logits.shape[1], dtype=torch.long), torch.ones(2, dtype=torch.long), blank=256)
    (code + .3 * image + .1 * words_loss).backward(inputs=[p for _, p in named])
    optimizer.step(); optimizer.zero_grad(set_to_none=True)
    reference.assert_bound(full=True)
    assert torch.equal(before, reference(ids))
    with torch.no_grad(): assert not torch.equal(before, view.text(input_ids=ids)['logits'])
    with pytest.raises(ValueError, match='text only'): reference(ids, pixel_values=pixels)
    with pytest.raises(ValueError, match='training mode'): reference.train()
    with torch.no_grad(): view.actor.base.network.norm.weight.add_(.0001)
    with pytest.raises(ValueError, match='text-reference'): reference(ids)
    (tmp_path / 'actual-small-shared-frozen-text-reference.json').write_text(json.dumps(dict(
        shared_training_reference_adds_no_parameters=True,
        actual_joint_code_image_CTC_AdamW_updates=1, exact_complete_text_reference_logits_after_joint_update=True,
        actual_actor_logits_changed=True, image_reference_calls_and_changed_language_rejected=True,
        full_source_DPO_or_RLHF_not_admitted=True, synthetic_controls_only=True,
        full_size_GPUcapacity_or_quality_gain=False), indent=2))
