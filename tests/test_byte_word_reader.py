import pytest
import torch

from slm_lab.byte_word_reader import ByteWordReader, collapse, labels, loss


def test_byte_ctc_keeps_repeated_letters_and_unicode_without_external_tokenizer():
    path = []
    for value in labels('book café 字'): path.extend([value, value, 0])
    assert collapse(path) == ('book café 字', True)
    assert collapse([256]) == ('�', False)
    with pytest.raises(ValueError): labels('a' * 65)


def test_word_reader_backpropagates_finite_transcript_loss_to_own_scratch_parameters():
    model = ByteWordReader(); images = torch.randn(2, 3, 32, 512)
    logits = model(images)
    assert logits.shape == (2, 128, 257)
    value = loss(logits, ['book', 'café']); value.backward()
    assert torch.isfinite(value) and model.features[0].weight.grad is not None
    assert torch.isfinite(model.sequence.weight_ih_l0.grad).all()
    assert model.output.weight.grad.abs().sum() > 0


def test_impossible_ctc_alignment_is_not_silently_discarded():
    with pytest.raises(ValueError, match='more CTC frames'):
        loss(torch.randn(1, 2, 257), ['book'])
