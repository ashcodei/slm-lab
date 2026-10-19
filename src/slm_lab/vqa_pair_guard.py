"""Require the finalized paired-photo selection and retain its native answers."""
from collections import defaultdict
import copy
import hashlib
import json
import math
from pathlib import Path

from .corpus import file_hash
from .diverse_images import jsonl
from .object_evaluation import summarize
from .vqa_pair_evaluation import compare_reports, human_agreement, select, text

PROTOCOL = 'Answer-type round-robin, unique photos across pairs; identical complete canonical model question, distinct original consensus targets; greedy448px/16tokens; actual/blank/paired counterpart controls.'


def validate_coverage(report, data=None):
    rows = report['results']
    if (report['protocol'] != PROTOCOL or report['max_examples'] != 256
            or report['max_new_tokens'] != 16 or report['image_size'] != 448
            or not rows or len(rows) > 256 or len({row['group_id'] for row in rows}) != len(rows)):
        raise ValueError('Paired-photo retention requires the complete declared unique-photo protocol')
    if compare_reports(report, report)['passed'] is not True:
        raise ValueError('Paired-photo case/aggregate evidence is inconsistent')
    grouped = defaultdict(list)
    for row in rows:
        if type(row['target_tokens']) is not int or row['target_tokens'] < 1:
            raise ValueError('Paired-photo target token counts differ')
        grouped[row['pair_id']].append(row)
        for name in ('proper', 'blank', 'wrong'):
            if (type(row[name + '_exact']) is not bool
                    or row[name + '_exact'] != (text(row[name + '_generated']) == text(row['expected']))
                    or not math.isfinite(row[name + '_nll']) or row[name + '_nll'] < 0
                    or not 0 <= row[name + '_human_agreement'] <= 1):
                raise ValueError('Paired-photo native output/score evidence differs')
    for pair in grouped.values():
        first, second = pair
        if (first['prompt_sha256'] != second['prompt_sha256'] or text(first['expected']) == text(second['expected'])
                or first['blank_generated'] != second['blank_generated']):
            raise ValueError('Paired-photo identical-question/blank/contrasting-target controls differ')
        for row, other in ((first, second), (second, first)):
            if (row['wrong_image_id'] != other['id'] or row['wrong_pixels_sha256'] != other['input_pixels_sha256']):
                raise ValueError('Paired-photo wrong-image control is not the actual counterpart')
    if report['overall'] != summarize(rows): raise ValueError('Paired-photo control aggregate differs')
    means = {name: sum(row[name + '_human_agreement'] for row in rows) / len(rows) for name in ('proper', 'blank', 'wrong')}
    if report['mean_human_agreement'] != means: raise ValueError('Paired-photo human-score aggregate differs')
    if data is not None:
        data = Path(data); manifest = json.loads((data / 'manifest.json').read_text())
        for field, filename in [('data_sha256', 'valid.jsonl'), ('pairs_sha256', 'pairs.valid.jsonl')]:
            if report[field] != file_hash(data / filename) or report[field] != manifest['files_sha256'][filename]:
                raise ValueError('Paired-photo report differs from finalized validation inputs')
        if (report['manifest_sha256'] != file_hash(data / 'manifest.json')
                or report['tokenizer_sha256'] != manifest['tokenizer_sha256']):
            raise ValueError('Paired-photo manifest or own tokenizer changed')
        chosen, by_id = select(list(jsonl(data / 'pairs.valid.jsonl')), list(jsonl(data / 'valid.jsonl')), 128)
        expected = [(by_id[qid]['id'], pair['id'], by_id[qid]['group_id'], by_id[qid]['response'],
                     by_id[qid]['original_human_annotation']['answer_type'], by_id[qid]['target_tokens'],
                     hashlib.sha256(('User: ' + by_id[qid]['prompt'] + '\nAssistant:').encode()).hexdigest())
                    for pair in chosen for qid in pair['question_ids']]
        actual = [(row['id'], row['pair_id'], row['group_id'], row['expected'], row['task'],
                   row['target_tokens'], row['prompt_sha256']) for row in rows]
        if actual != expected: raise ValueError('Paired-photo report omits or changes finalized selected cases')
        for row in rows:
            answers = by_id[row['id']]['original_human_annotation']['answers']
            for name in ('proper', 'blank', 'wrong'):
                if row[name + '_human_agreement'] != human_agreement(row[name + '_generated'], answers):
                    raise ValueError('Paired-photo report does not use its original human answers')
    return {'selected_pairs': len(grouped), 'selected_unique_photos': len(rows), 'maximum_pairs': 128}


def compare(before, after, data=None):
    validate_coverage(before, data); validate_coverage(after, data)
    return compare_reports(before, after)


def enforce(result, before, after, data=None):
    guard = compare(before, after, data)
    if (guard['base_checkpoint_sha256'] != result['base_checkpoint_sha256']
            or guard['candidate_checkpoint_sha256'] != result['candidate_checkpoint_sha256']):
        raise ValueError('Paired-photo and existing selection compare different policies')
    merged = copy.deepcopy(result); merged['paired_photo_guard'] = guard
    merged['checks']['paired_photo_retention'] = {'passed': guard['passed'],
        'before_accuracy': guard['before_exact_accuracy'], 'after_accuracy': guard['after_exact_accuracy'],
        'before_pairs': guard['before_pairs'], 'after_pairs': guard['after_pairs'],
        'previously_correct_questions_lost': guard['previously_correct_questions_lost']}
    if not guard['passed']: merged['failed_checks'].append('paired_photo_retention')
    merged['eligible_for_selection'] = result['eligible_for_selection'] and guard['passed']
    return merged
