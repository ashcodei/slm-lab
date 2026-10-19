import copy
import gzip
import json

from PIL import Image
import pytest
from tokenizers import Tokenizer, models, pre_tokenizers

from slm_lab.corpus import file_hash
from slm_lab.diverse_images import pixel_hash
from slm_lab.model import ModelConfig
from slm_lab.training import Encoder
from slm_lab.vqa_attribution import attribution
from slm_lab.vqa_pair_admission import admit, reconstruct_question, unchanged, verified_photo


def original_pair():
    samples = []
    for image_id, answer in [(1, 'red'), (2, 'blue')]:
        source = {'id': image_id, 'license': 4, 'flickr_url': 'https://www.flickr.com/photos/example/' + str(image_id)}
        qid = image_id * 10
        samples.append({'question': {'image_id': image_id, 'question_id': qid, 'question': 'What color is this?'},
                        'image': source, 'answer': answer, 'exact_human_votes': 10,
                        'annotation': {'image_id': image_id, 'question_id': qid, 'answer_type': 'other',
                                       'answers': [{'answer': answer, 'answer_id': i + 1} for i in range(10)]}})
    return {'id': 'original-pair', 'source_subset': 'train2014', 'first': samples[0], 'second': samples[1]}


def encoder(tmp_path, context=128):
    vocabulary = ['<unk>', '<bos>', '<eos>', '<pad>', 'User', ':', 'what', 'color', 'is', 'this', '?',
                  'Answer', 'briefly', '.', 'Assistant', 'red', 'blue']
    tokenizer = Tokenizer(models.WordLevel({word: i for i, word in enumerate(vocabulary)}, unk_token='<unk>'))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    return Encoder(tokenizer, ModelConfig(vocab_size=len(vocabulary), max_text_length=context), tmp_path, 448)


def record(pair):
    return {'source_subset': pair['source_subset'], 'coco_image_id': 1, 'source_flickr_photo_id': '1',
            'original_COCO_image_metadata': copy.deepcopy(pair['first']['image']), 'image': 'original.png',
            'group_id': 'flickr:1'}


def test_unfinalized_live_journal_cannot_be_used_for_training(tmp_path):
    data = tmp_path / 'data/vqa-real-pairs-v1'; data.mkdir(parents=True)
    (data / 'images.collected.jsonl').write_text('a still collecting journal')
    with pytest.raises(ValueError, match='Finalized original real-photo pair manifest required'):
        admit(tmp_path)


def test_reconstruction_uses_the_complete_original_human_target_and_response_only_mask(tmp_path):
    pair = original_pair(); enc = encoder(tmp_path)
    row = reconstruct_question(pair, 'first', record(pair), enc)
    assert row['prompt'] == 'what color is this?\nAnswer briefly.'
    assert row['response'] == 'red' and row['exact_human_agreement'] == 10
    assert row['original_human_annotation'] == pair['first']['annotation']
    inputs, labels = enc.sequence(row, 'response')
    assert inputs[0] == enc.bos and inputs[-1] == enc.eos
    assert row['context_tokens'] == len(inputs)
    assert row['target_tokens'] == sum(value != -100 for value in labels[1:]) == 2
    assert labels[:-(row['target_tokens'])] == [-100] * (len(inputs) - row['target_tokens'])


@pytest.mark.parametrize('change', ['subset', 'photo_id', 'photo_metadata', 'question_photo', 'annotation_photo',
                                  'annotation_question', 'flickr_alias', 'license', 'votes', 'answer_ids', 'target'])
def test_original_question_answer_and_photo_join_rejects_identity_or_consensus_changes(tmp_path, change):
    pair = original_pair(); photo = record(pair); sample = pair['first']
    if change == 'subset': photo['source_subset'] = 'val2014'
    elif change == 'photo_id': photo['coco_image_id'] = 99
    elif change == 'photo_metadata': photo['original_COCO_image_metadata']['id'] = 99
    elif change == 'question_photo': sample['question']['image_id'] = 99
    elif change == 'annotation_photo': sample['annotation']['image_id'] = 99
    elif change == 'annotation_question': sample['annotation']['question_id'] = 99
    elif change == 'flickr_alias': photo['source_flickr_photo_id'] = '99'
    elif change == 'license': sample['image']['license'] = 1; photo['original_COCO_image_metadata']['license'] = 1
    elif change == 'votes': sample['exact_human_votes'] = 9
    elif change == 'answer_ids': sample['annotation']['answers'][0]['answer_id'] = 2
    else: sample['answer'] = 'green'
    with pytest.raises(ValueError): reconstruct_question(pair, 'first', photo, encoder(tmp_path))


@pytest.mark.parametrize('field', ['prompt', 'response', 'original_human_annotation', 'pair_ids', 'context_tokens', 'target_tokens'])
def test_rehashed_prepared_rows_still_require_exact_source_reconstruction(tmp_path, field):
    pair = original_pair(); row = reconstruct_question(pair, 'first', record(pair), encoder(tmp_path))
    changed = copy.deepcopy(row); changed[field] = 'replaced field'
    with pytest.raises(ValueError, match='source reconstruction'): unchanged(changed, row)


@pytest.mark.parametrize('field', ['question', 'answer'])
def test_full_questions_and_answers_cannot_silently_clip(tmp_path, field):
    pair = original_pair()
    if field == 'question':
        for side in ('first', 'second'): pair[side]['question']['question'] = 'what ' * 40
    else:
        pair['first']['answer'] = 'red ' * 40
        for answer in pair['first']['annotation']['answers']: answer['answer'] = pair['first']['answer']
    with pytest.raises(ValueError, match='clipped or mislabeled'):
        reconstruct_question(pair, 'first', record(pair), encoder(tmp_path, 32))


def stored_photo(tmp_path):
    declaration = {'@type': 'ImageObject', 'acquireLicensePage': 'https://www.flickr.com/photos/example/1',
                   'license': 'https://creativecommons.org/licenses/by/2.0/',
                   'author': {'name': 'Original photographer', 'url': 'https://www.flickr.com/photos/example'},
                   'contentUrl': 'https://live.staticflickr.com/1/1_signature_b.jpg'}
    raw = ('<script type="application/ld+json">' + json.dumps(declaration) + '</script>').encode()
    with gzip.open(tmp_path / 'original.html.gz', 'wb') as stream: stream.write(raw)
    image = Image.new('RGB', (53, 31), 'red'); image.save(tmp_path / 'original.png')
    return {**attribution(raw, '1', declaration['acquireLicensePage']), 'group_id': 'flickr:1',
            'image': 'original.png', 'stored_image_sha256': file_hash(tmp_path / 'original.png'),
            'attribution_page': 'original.html.gz', 'attribution_page_sha256': file_hash(tmp_path / 'original.html.gz'),
            'model_pixel_sha256': pixel_hash(image, 224), 'model_pixel_sha256_448': pixel_hash(image, 448)}


@pytest.mark.parametrize('change', ['image_bytes', 'html_bytes', 'creator', 'license', 'group', 'pixels', 'escape'])
def test_actual_photo_bytes_creator_declaration_and_model_pixels_are_rechecked(tmp_path, change):
    photo = stored_photo(tmp_path); verified_photo(tmp_path, photo)
    if change == 'image_bytes': Image.new('RGB', (53, 31), 'blue').save(tmp_path / 'original.png')
    elif change == 'html_bytes': (tmp_path / 'original.html.gz').write_bytes(b'replaced declaration')
    elif change == 'creator': photo['author'] = 'Changed author'
    elif change == 'license': photo['license'] = 'public domain'
    elif change == 'group': photo['group_id'] = 'flickr:99'
    elif change == 'pixels': photo['model_pixel_sha256_448'] = 'wrong input pixels'
    else: photo['image'] = '../outside.png'
    with pytest.raises(ValueError): verified_photo(tmp_path, photo)
