import copy
import pytest

from slm_lab.textvqa_input_audit import validate_photo_binding, validate_question


def records():
    original = {'question_id': 1, 'image_id': 'a', 'question': 'What brand?', 'set_name': 'train',
                'answers': ['nokia'] * 7 + ['other'] * 3,
                'flickr_original_url': 'https://farm1.staticflickr.com/2/123_secret_o.jpg'}
    photo = {'group_id': 'openimages:a', 'image': 'photos/a.png', 'task': 'caption',
             'source_url': 'https://www.flickr.com/photos/author/123', 'license': 'CC-BY-2.0',
             'author': 'Author', 'author_profile_url': 'https://www.flickr.com/people/author/',
             'stored_image_sha256': 'full-photo', 'model_pixel_sha256': '224', 'model_pixel_sha256_448': '448'}
    from pathlib import Path
    row = {**photo, 'id': 'textvqa:train:1', 'image': str((Path('/tmp/source') / photo['image']).resolve()),
           'task': 'text_reading_question', 'prompt': 'What brand?\nAnswer briefly.', 'response': 'nokia',
           'original_textvqa_question': copy.deepcopy(original), 'identical_original_answer_votes': 7}
    return original, photo, row


@pytest.mark.parametrize('change', ['answer', 'hint', 'source_answers', 'group', 'consensus'])
def test_original_input_audit_rejects_changed_targets_prompts_answers_images_or_votes(change):
    original, _, row = records()
    validate_question(row, original, 'train')
    if change == 'answer': row['response'] = 'other'
    elif change == 'hint': row['prompt'] += '\nOCR hint: nokia'
    elif change == 'source_answers': row['original_textvqa_question']['answers'][0] = 'fabricated'
    elif change == 'group': row['group_id'] = 'openimages:another'
    else: row['identical_original_answer_votes'] = 10
    with pytest.raises(ValueError): validate_question(row, original, 'train')


@pytest.mark.parametrize('change', ['crop', 'path', 'credit', 'pixels'])
def test_source_photo_audit_rejects_crops_or_relabelled_photos(change):
    _, photo, row = records()
    validate_photo_binding(row, photo, '/tmp/source/train.jsonl')
    if change == 'crop': photo['task'] = 'word_ocr'
    elif change == 'path': row['image'] = '/tmp/source/photos/other.png'
    elif change == 'credit': row['author'] = 'Invented author'
    else: row['model_pixel_sha256_448'] = 'wrong pixels'
    with pytest.raises(ValueError): validate_photo_binding(row, photo, '/tmp/source/train.jsonl')
