from types import SimpleNamespace
import pytest

from slm_lab.own_ocr_training_data import compact_predictions, variants


class Bytes:
    def encode(self, text): return SimpleNamespace(ids=list(text.encode()))


def test_compact_fields_preserve_actual_words_without_answer_or_class_hints():
    prediction = {'text': 'QATAR', 'box': [.12345678, .2, .456789, .5], 'score': .987654, 'valid_utf8': True,
                  'expected': 'SECRET', 'image_classes': ['SECRET']}
    result = compact_predictions([prediction])[0]
    assert result['text'] == 'QATAR' and result['box'] == [.1235, .2, .4568, .5] and result['score'] == .9877
    assert set(result) == {'text', 'box', 'score', 'valid_utf8'}
    with pytest.raises(ValueError): compact_predictions([{**prediction, 'score': float('inf')}])


def test_original_human_answer_is_retained_only_as_training_target():
    row = {'group_id': 'image', 'image': 'photo.png', 'stored_image_sha256': 'bytes', 'model_pixel_sha256': 'p224',
           'model_pixel_sha256_448': 'p448', 'source_image_id': 'id', 'source_url': 'url', 'author': 'author',
           'author_profile_url': 'profile', 'license': 'CC-BY-2.0', 'license_url': 'license',
           'id': 'question', 'prompt': 'What word?\nAnswer briefly.', 'response': 'ANSWER_SECRET', 'split': 'train',
           'original_textvqa_question': {'question': 'What word?'}, 'question_source_sha256': 'source', 'annotation_license': 'CC-BY-4.0'}
    plain, own = variants(row, [{'text': 'PREDICTED', 'box': [.1, .1, .5, .5], 'score': .9, 'valid_utf8': True}], Bytes(), SimpleNamespace(max_text_length=4096))
    assert plain['response'] == own['response'] == 'ANSWER_SECRET'
    assert 'ANSWER_SECRET' not in plain['prompt'] and 'ANSWER_SECRET' not in own['prompt']
    assert own['source_question_id'] == plain['source_question_id'] == 'question'
    assert plain['image'] == own['image'] == row['image'] and own['own_predicted_context_supplied'] is True
    assert 'PREDICTED' in own['prompt'] and 'PREDICTED' not in plain['prompt']


def test_tiny_boxes_do_not_collapse_into_invalid_context_rectangles():
    rectangle = [.10001, .1, .10002, .2]
    result = compact_predictions([{'text': 'I', 'box': rectangle, 'score': .9, 'valid_utf8': True}])[0]
    assert result['box'] == rectangle
