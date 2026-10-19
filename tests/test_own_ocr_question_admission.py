import copy
import pytest

from slm_lab.own_ocr_question_admission import unchanged_variant


def expected():
    return {'id': 'original:own-context', 'source_question_id': 'original', 'group_id': 'original-parent',
            'prompt': 'Original question and own predicted words', 'response': 'Original human answer',
            'image': 'original-photo.png', 'stored_image_sha256': 'source-bytes', 'target_tokens': 4}


def test_exact_reconstructed_question_variant_is_admitted():
    row = expected(); unchanged_variant(copy.deepcopy(row), row)


@pytest.mark.parametrize('field,value', [('prompt', 'question with answer hints'), ('response', 'model generated answer'),
                                       ('image', 'word-crop.png'), ('source_question_id', 'other'), ('target_tokens', 0)])
def test_mutated_prompt_answer_image_identity_or_target_receipt_is_rejected(field, value):
    original = expected(); changed = copy.deepcopy(original); changed[field] = value
    with pytest.raises(ValueError): unchanged_variant(changed, original)
