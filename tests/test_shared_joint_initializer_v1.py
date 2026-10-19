import copy

import pytest
import torch

from slm_lab.codenet_checkpoint_v1 import capture_rng
from slm_lab.parallel_function_repair_stage_v1 import value_digest
from slm_lab.shared_joint_initializer_v1 import assemble, VISUAL_PREFIXES
from slm_lab.whole_human_sft_evidence_v1 import tensor_digest
from test_joint_task_gradients_v1 import view, fixture, word_fixture


def test_actual_own_small_text_and_word_outputs_source_values_and_private_RNG_survive_hybrid_assembly():
    text = view().actor.base; visual = copy.deepcopy(text)
    with torch.no_grad():
        visual.network.embedding.weight.add_(.03)
        visual.network.value_head.weight.fill_(.75)
        text.network.value_head.weight.fill_(.25)
        for name, parameter in visual.named_parameters():
            if name.startswith(VISUAL_PREFIXES):
                parameter.add_(.001)
    before = (tensor_digest(text.named_parameters()), tensor_digest(visual.named_parameters()), value_digest(capture_rng('cpu')))
    batch, _ = fixture(1); _, word, _ = word_fixture()
    with torch.no_grad():
        expected_text = text(batch['input_ids'], attention_mask=batch['attention_mask'])['logits']
        expected_word = visual.word_logits(word['word_pixels'])
    result, proof = assemble(text, visual, sources={'fixed_own_text': 'a' * 64, 'fixed_own_visual': 'b' * 64},
        working_budget_bytes=256 * 1024**2, allow_mechanics=True)
    with torch.no_grad():
        assert torch.equal(result(batch['input_ids'], attention_mask=batch['attention_mask'])['logits'], expected_text)
        assert torch.equal(result.word_logits(word['word_pixels']), expected_word)
    assert torch.equal(result.network.value_head.weight, text.network.value_head.weight)
    assert result.counts() == text.counts() and proof['new_neural_values'] == 0
    assert proof['every_selected_source_value_exact'] and proof['independent_result_storage']
    assert before == (tensor_digest(text.named_parameters()), tensor_digest(visual.named_parameters()), value_digest(capture_rng('cpu')))


def test_full_size_gate_does_not_admit_a_small_assembly_as_a_completed_production_initializer():
    text = view().actor.base; visual = copy.deepcopy(text)
    with pytest.raises(ValueError, match='explicit small mechanics'):
        assemble(text, visual, sources={'fixed_own_text': 'a' * 64}, working_budget_bytes=256 * 1024**2)


def test_mismatched_native_head_protocol_is_rejected_without_source_mutation():
    text = view().actor.base; visual = copy.deepcopy(text)
    visual.settings['word_expansion'] = 5
    before = tensor_digest(text.named_parameters())
    with pytest.raises(ValueError, match='compatible'):
        assemble(text, visual, sources={'fixed_own_text': 'a' * 64}, working_budget_bytes=256 * 1024**2, allow_mechanics=True)
    assert tensor_digest(text.named_parameters()) == before
