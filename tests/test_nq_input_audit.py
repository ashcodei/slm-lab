from pathlib import Path

import pytest
from tokenizers import Tokenizer

from slm_lab.model import ModelConfig
from slm_lab.nq_input_audit import validate_row
from slm_lab.nq_open_data import REVISION, example
from slm_lab.training import Encoder

ROOT = Path(__file__).resolve().parents[1]


def case(question='What is the capital of Iceland?'):
    original = {'question': question, 'answer': ['Reykjavík', 'Reykjavik']}
    tokenizer = Tokenizer.from_file(str(ROOT / 'data/tokenizer-corpus/tokenizer.json'))
    encoder = Encoder(tokenizer, ModelConfig.load(ROOT / 'configs/model.json'), ROOT, 448)
    row = example(original, 'dev', 7)
    row['target_tokens'] = len(tokenizer.encode(original['answer'][0]).ids) + 1
    return original, row, encoder


def test_actual_own_encoder_keeps_complete_question_and_unicode_answer_with_response_only_labels():
    original, row, encoder = case()
    assert validate_row(row, original, 'dev', 7, encoder, REVISION) == row['target_tokens']
    ids, labels = encoder.sequence(row, 'response')
    assert all(value == -100 for value in labels[:-row['target_tokens']])
    assert labels[-row['target_tokens']:] == encoder.tokens('Reykjavík') + [encoder.eos]
    assert ids[0] == encoder.bos and ids[-1] == encoder.eos


def test_missing_original_answer_alias_is_rejected_even_when_training_target_is_identical():
    original, row, encoder = case(); row['reference_answers'] = ['Reykjavík']
    with pytest.raises(ValueError): validate_row(row, original, 'dev', 7, encoder, REVISION)


def test_packed_tokens_cannot_override_original_question_and_answer_encoder_path():
    original, row, encoder = case(); row['tokens'] = [encoder.bos, encoder.eos]
    with pytest.raises(ValueError): validate_row(row, original, 'dev', 7, encoder, REVISION)


def test_truncated_question_is_rejected_even_with_complete_answer_and_valid_source_metadata():
    original, row, encoder = case('word ' * 5000)
    with pytest.raises(ValueError): validate_row(row, original, 'dev', 7, encoder, REVISION)
