import copy
from types import SimpleNamespace

import pytest
from tokenizers import Tokenizer, models, pre_tokenizers

from slm_lab.object_evaluation import summarize
from slm_lab.own_context_transfer_evaluation import compare_reports, conditioned_prompt
from slm_lab.own_ocr_training_data import variants


def test_evaluation_uses_training_context_budget_and_never_serializes_answer_metadata():
    tokenizer = Tokenizer(models.WordLevel({'[UNK]': 0}, unk_token='[UNK]'))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    config = SimpleNamespace(max_text_length=4096)
    prediction = {'text': 'visible word', 'box': [.123456, .234567, .654321, .876543],
                  'score': .987654, 'valid_utf8': True, 'answer': 'LEAKED ANSWER'}
    row = {key: 'source' for key in ['group_id', 'image', 'stored_image_sha256', 'model_pixel_sha256',
                                   'model_pixel_sha256_448', 'source_image_id', 'source_url', 'author',
                                   'author_profile_url', 'license', 'license_url', 'question_source_sha256', 'annotation_license']}
    row.update(id='q', split='valid', response='FIRST HUMAN ANSWER', prompt='What is visible?\nAnswer briefly.',
               original_textvqa_question={'question': 'What is visible?'})
    context = conditioned_prompt('What is visible?', [prediction], tokenizer, config)
    assert context['prompt'] == variants(row, [prediction], tokenizer, config)[1]['prompt']
    changed = {**row, 'response': 'DIFFERENT HUMAN ANSWER'}
    assert context['prompt'] == variants(changed, [prediction], tokenizer, config)[1]['prompt']
    assert 'LEAKED ANSWER' not in context['prompt'] and row['response'] not in context['prompt']
    assert context['generation_tokens'] == 256
    assert context['selected_context'][0]['box'] == [.1235, .2346, .6543, .8765]


def report():
    rows = []
    for index in range(2):
        row = {'id': str(index), 'group_id': str(index), 'expected': 'yes', 'task': 'own_context_reading_question',
               'prompt_sha256': 'prompt', 'input_pixels_sha256': 'proper', 'wrong_pixels_sha256': 'wrong',
               'wrong_image_id': str(1 - index), 'target_tokens': 2, 'original_answers_sha256': 'answers',
               'own_OCR_input_rgb_sha256_768': 'proper-rgb', 'wrong_OCR_rgb_sha256_768': 'wrong-rgb'}
        for control in ['proper', 'blank', 'wrong']:
            row.update({control + '_generated': 'yes', control + '_exact': True, control + '_nll': .5,
                        control + '_prompt_sha256': control, control + '_target_tokens': 2,
                        control + '_selected_context': [{'text': control}]})
        rows.append(row)
    result = {key: 'same' for key in ['evaluation_provenance', 'data_sha256', 'manifest_sha256', 'tokenizer_sha256',
                                    'inference_precision', 'inference_device_type', 'protocol', 'human_score_protocol',
                                    'own_context_manifest_sha256', 'detector_sha256', 'reader_sha256']}
    result.update(checkpoint_sha256='model', image_size=448, max_new_tokens=32, max_examples=128,
                  context_headroom_tokens=256, maximum_context_words=64, image_size_own_reader=768,
                  results=rows, overall=summarize(rows))
    return result


def test_retention_rejects_losing_a_previous_correct_answer_even_if_net_score_is_repaired():
    before = report()
    before['results'][1]['proper_generated'] = 'no'; before['results'][1]['proper_exact'] = False
    before['overall'] = summarize(before['results'])
    after = copy.deepcopy(before); after['checkpoint_sha256'] = 'candidate'
    after['results'][0]['proper_generated'] = 'no'; after['results'][0]['proper_exact'] = False
    after['results'][1]['proper_generated'] = 'yes'; after['results'][1]['proper_exact'] = True
    after['overall'] = summarize(after['results'])
    guard = compare_reports(before, after)
    assert not guard['passed'] and guard['previously_correct_questions_lost'] == ['0']


@pytest.mark.parametrize('change', ['wrong_prompt', 'wrong_text', 'component', 'false_correct_flag', 'aggregate'])
def test_unmatched_controls_components_or_forged_scores_are_rejected(change):
    before = report(); after = copy.deepcopy(before)
    if change == 'wrong_prompt': after['results'][0]['wrong_prompt_sha256'] = 'proper hint'
    if change == 'wrong_text': after['results'][0]['wrong_selected_context'] = [{'text': 'proper hint'}]
    if change == 'component': after['reader_sha256'] = 'other model'
    if change == 'false_correct_flag': after['results'][0]['proper_generated'] = 'no'
    if change == 'aggregate': after['overall']['correct_image_exact_accuracy'] = 0
    with pytest.raises(ValueError): compare_reports(before, after)


def test_matched_unchanged_capabilities_pass():
    before = report(); after = copy.deepcopy(before); after['checkpoint_sha256'] = 'candidate'
    assert compare_reports(before, after)['passed']
