import pytest
import torch

from slm_lab.codenet_checkpoint_v1 import capture_rng
from slm_lab.joint_native_fold_v1 import fold
from slm_lab.parallel_function_repair_stage_v1 import value_digest
from slm_lab.whole_human_sft_evidence_v1 import tensor_digest
from test_joint_task_gradients_v1 import view, fixture, word_fixture


def test_nonzero_native_critic_and_current_joint_visual_word_values_survive_language_fold_without_source_mutation():
    current = view(); actor = current.actor
    with torch.no_grad():
        actor.base.network.value_head.weight.copy_(torch.linspace(-.25, .25, actor.config.hidden_size).reshape(1, -1))
        for part in actor.parts['linears']:
            part.up.add_(.001)
        next(actor.base.network.vision.parameters()).add_(.02)
        next(actor.base.heads['word'].parameters()).add_(.03)
    before = tensor_digest(actor.named_parameters()); rng = value_digest(capture_rng('cpu'))
    batch, _ = fixture(1); _, word, _ = word_fixture()
    with torch.no_grad():
        expected_logits = actor(batch['input_ids'], attention_mask=batch['attention_mask'], pixel_values=batch['pixel_values'])['logits']
        expected_word = actor.base.word_logits(word['word_pixels'])
    native, receipt = fold(actor, sources={'explicit_own_small_joint_control': 'a' * 64}, working_budget_bytes=256 * 1024**2)
    with torch.no_grad():
        actual_logits = native(batch['input_ids'], attention_mask=batch['attention_mask'], pixel_values=batch['pixel_values'])['logits']
        torch.testing.assert_close(actual_logits, expected_logits, atol=1e-6, rtol=1e-5)
        assert torch.equal(native.word_logits(word['word_pixels']), expected_word)
    assert torch.equal(native.network.value_head.weight, actor.base.network.value_head.weight)
    assert bool(native.network.value_head.weight.count_nonzero())
    assert receipt['original_native_critic_preserved'] and receipt['unused_zero_research_critic_excluded']
    assert receipt['current_joint_visual_word_and_auxiliary_values_preserved']
    assert receipt['policy_critic_folded_into_existing_dense_value_head'] is False
    assert native.counts() == actor.base.counts()
    assert native.network.embedding.weight is native.network.lm_head.weight
    assert not ({p.data_ptr() for p in native.parameters()} & {p.data_ptr() for p in actor.parameters()})
    assert tensor_digest(actor.named_parameters()) == before and value_digest(capture_rng('cpu')) == rng
    assert not torch.cuda.is_initialized()


def test_trained_research_critic_cannot_be_silently_discarded_by_joint_export():
    actor = view().actor
    with torch.no_grad():
        actor.parts['critic'].weight.fill_(.25)
    before = tensor_digest(actor.named_parameters())
    with pytest.raises(ValueError, match='unused zero research critic'):
        fold(actor, sources={'explicit_own_small_joint_control': 'a' * 64}, working_budget_bytes=256 * 1024**2)
    assert tensor_digest(actor.named_parameters()) == before
