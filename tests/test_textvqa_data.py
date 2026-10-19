import pytest
from slm_lab.textvqa_data import agreed_target, same_photo
from slm_lab.training import Encoder


def test_only_actual_strong_original_answer_consensus_supplies_a_target():
    assert agreed_target({'answers': ['nokia'] * 6 + ['Nokia'] * 4}) == ('nokia', 6)
    assert agreed_target({'answers': ['nokia'] * 5 + ['Nokia'] * 5}) is None
    assert agreed_target({'answers': ['unanswerable'] * 10}) is None
    with pytest.raises(ValueError): agreed_target({'answers': ['nokia'] * 9})


def test_photo_identity_requires_matching_nonempty_original_and_stored_flickr_ids():
    original = {'image_id': 'image-a', 'flickr_original_url': 'https://farm1.staticflickr.com/2/123_secret_o.jpg',
                'flickr_300k_url': 'https://farm2.staticflickr.com/3/123_secret_z.jpg'}
    stored = {'group_id': 'openimages:image-a', 'source_url': 'https://www.flickr.com/photos/author/123',
              'original_url': 'https://farm1.staticflickr.com/2/123_secret_o.jpg'}
    assert same_photo(original, stored)
    assert not same_photo(original, {**stored, 'group_id': 'openimages:image-b'})
    assert not same_photo(original, {**stored, 'original_url': 'https://farm1.staticflickr.com/2/456_secret_o.jpg'})
    assert not same_photo({**original, 'flickr_original_url': '', 'flickr_300k_url': ''},
                          {'group_id': 'openimages:image-a'})


def test_native_question_input_does_not_expose_source_answers_classes_or_ocr_hints():
    encoder = object.__new__(Encoder)
    row = {'prompt': 'What brand is printed on the phone?\nAnswer briefly.',
           'image': 'full-photo.jpg', 'task': 'text_reading_question', 'response': 'secret answer',
           'original_textvqa_question': {'answers': ['secret answer'] * 10, 'image_classes': ['secret class']},
           'ocr_tokens': ['secret OCR hint']}
    actual = encoder.prompt(row)
    assert actual == 'User: What brand is printed on the phone?\nAnswer briefly.\nAssistant:'
    assert 'secret' not in actual
