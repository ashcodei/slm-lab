import copy
import hashlib
import json

import pytest

from slm_lab.corpus import file_hash
from slm_lab.object_evaluation import summarize
from slm_lab.vqa_pair_evaluation import paired_accuracy, select
from slm_lab.vqa_pair_guard import PROTOCOL, compare, enforce, validate_coverage


def refresh(report):
    rows = report['results']; report['overall'] = summarize(rows); report['paired'] = paired_accuracy(rows)
    report['mean_human_agreement'] = {name: sum(row[name + '_human_agreement'] for row in rows) / len(rows)
                                    for name in ('proper', 'blank', 'wrong')}


def reports():
    rows = []
    for index in range(4):
        pair = index // 2; prompt = 'User: question ' + str(pair) + '?\nAnswer briefly.\nAssistant:'
        row = {'id': 'q' + str(index), 'pair_id': 'pair' + str(pair), 'group_id': 'photo' + str(index),
               'expected': 'red' if index % 2 == 0 else 'blue', 'task': 'other',
               'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(), 'input_pixels_sha256': 'rgb' + str(index),
               'wrong_pixels_sha256': 'rgb' + str(index ^ 1), 'wrong_image_id': 'q' + str(index ^ 1), 'target_tokens': 2}
        for name in ('proper', 'blank', 'wrong'):
            correct = name == 'proper' and index < 2
            row.update({name + '_generated': row['expected'] if correct else 'green', name + '_exact': correct,
                        name + '_nll': .5 if correct else 1.5, name + '_human_agreement': 1. if correct else 0.})
        rows.append(row)
    before = {key: 'fixed' for key in ('evaluation_provenance', 'data_sha256', 'pairs_sha256', 'manifest_sha256',
                                     'tokenizer_sha256', 'inference_precision', 'inference_device_type', 'human_score_protocol')}
    before.update(checkpoint_sha256='base', protocol=PROTOCOL, image_size=448, max_examples=256, max_new_tokens=16, results=rows)
    refresh(before)
    after = copy.deepcopy(before); after['checkpoint_sha256'] = 'candidate'
    result = {'eligible_for_selection': True, 'base_checkpoint_sha256': 'base', 'candidate_checkpoint_sha256': 'candidate',
              'checks': {'code': {'passed': True}}, 'failed_checks': []}
    return result, before, after


def test_original_selected_pairs_may_be_fewer_than_128_without_inventing_coverage():
    result, before, after = reports(); coverage = validate_coverage(before)
    assert coverage == {'selected_pairs': 2, 'selected_unique_photos': 4, 'maximum_pairs': 128}
    assert enforce(result, before, after)['eligible_for_selection']


def test_equal_net_accuracy_and_positive_preference_cannot_hide_lost_original_answers():
    result, before, after = reports()
    for index in range(4):
        correct = index >= 2; row = after['results'][index]
        row.update(proper_generated=row['expected'] if correct else 'green', proper_exact=correct,
                   proper_nll=.5 if correct else 1.5, proper_human_agreement=1. if correct else 0.)
    refresh(after)
    assert before['overall']['correct_image_exact_accuracy'] == after['overall']['correct_image_exact_accuracy']
    guarded = enforce(result, before, after)
    assert not guarded['eligible_for_selection'] and guarded['failed_checks'] == ['paired_photo_retention']
    assert guarded['paired_photo_guard']['previously_correct_questions_lost'] == ['q0', 'q1']
    assert result['eligible_for_selection'] and not result['failed_checks']


def test_paired_photo_retention_preserves_existing_code_and_reading_failure():
    result, before, after = reports(); result['eligible_for_selection'] = False; result['failed_checks'] = ['code']
    guarded = enforce(result, before, after)
    assert not guarded['eligible_for_selection'] and guarded['failed_checks'] == ['code']
    assert guarded['checks']['code'] == result['checks']['code']


@pytest.mark.parametrize('change', ['policy', 'subset', 'repeat_photo', 'budget', 'protocol', 'pair', 'wrong_id',
                                  'wrong_pixels', 'blank', 'target', 'prompt', 'score', 'aggregate', 'nan_nll'])
def test_partial_cases_changed_controls_and_inconsistent_scores_cannot_select_a_policy(change):
    result, before, after = reports(); row = after['results'][0]
    if change == 'policy': result['candidate_checkpoint_sha256'] = 'other branch'
    elif change == 'subset': after['results'] = after['results'][2:]; refresh(after)
    elif change == 'repeat_photo': row['group_id'] = after['results'][1]['group_id']
    elif change == 'budget': after['max_new_tokens'] = 8
    elif change == 'protocol': after['protocol'] = 'caption loss only'
    elif change == 'pair': row['pair_id'] = 'missing counterpart'
    elif change == 'wrong_id': row['wrong_image_id'] = 'q3'
    elif change == 'wrong_pixels': row['wrong_pixels_sha256'] = 'rgb3'
    elif change == 'blank': row['blank_generated'] = 'different blank answer'
    elif change == 'target': row['expected'] = 'green'
    elif change == 'prompt': row['prompt_sha256'] = 'new prompt'
    elif change == 'score': row['proper_exact'] = False
    elif change == 'aggregate': after['overall']['blank_original_target_match'] = 1.
    else: row['proper_nll'] = float('nan')
    with pytest.raises(ValueError): enforce(result, before, after)


def finalized_inputs(tmp_path, report):
    rows = []; pairs = []
    for index in range(4):
        source = report['results'][index]
        rows.append({'id': source['id'], 'group_id': source['group_id'], 'response': source['expected'],
                     'target_tokens': 2, 'prompt': 'question ' + str(index // 2) + '?\nAnswer briefly.',
                     'original_human_annotation': {'answer_type': 'other', 'answers': [
                         {'answer_id': i + 1, 'answer': source['expected']} for i in range(10)]}})
        if index % 2 == 0:
            pairs.append({'id': source['pair_id'], 'question_ids': ['q' + str(index), 'q' + str(index + 1)],
                          'canonical_question': 'question ' + str(index // 2) + '?'})
    for filename, contents in [('valid.jsonl', rows), ('pairs.valid.jsonl', pairs)]:
        (tmp_path / filename).write_text(''.join(json.dumps(row) + '\n' for row in contents))
    manifest = {'files_sha256': {name: file_hash(tmp_path / name) for name in ('valid.jsonl', 'pairs.valid.jsonl')},
                'tokenizer_sha256': report['tokenizer_sha256']}
    (tmp_path / 'manifest.json').write_text(json.dumps(manifest))
    report.update(manifest_sha256=file_hash(tmp_path / 'manifest.json'), data_sha256=manifest['files_sha256']['valid.jsonl'],
                  pairs_sha256=manifest['files_sha256']['pairs.valid.jsonl'])
    chosen, _ = select(pairs, rows, 128); by_id = {row['id']: row for row in report['results']}
    report['results'] = [by_id[qid] for pair in chosen for qid in pair['question_ids']]; refresh(report)


def test_finalized_selector_prevents_rehashed_but_partial_pair_reports(tmp_path):
    _, before, _ = reports(); finalized_inputs(tmp_path, before)
    validate_coverage(before, tmp_path)
    partial = copy.deepcopy(before); partial['results'] = partial['results'][:2]; refresh(partial)
    with pytest.raises(ValueError, match='omits or changes finalized selected cases'): validate_coverage(partial, tmp_path)


def test_each_reported_human_score_is_recomputed_from_original_annotations(tmp_path):
    _, before, _ = reports(); finalized_inputs(tmp_path, before)
    before['results'][0]['proper_human_agreement'] = .5; refresh(before)
    with pytest.raises(ValueError, match='original human answers'): validate_coverage(before, tmp_path)


def test_matching_control_protocol_is_required_even_with_increased_pair_accuracy():
    _, before, after = reports(); after['pairs_sha256'] = 'different publisher pairs'
    with pytest.raises(ValueError): compare(before, after)
