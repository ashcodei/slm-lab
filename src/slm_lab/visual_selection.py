"""Conservative experimental checkpoint selection; not a capability certificate."""
import math


def compare(baseline, candidate):
    names = ['ocr', 'vision', 'knowledge', 'code', 'code_instruction']
    protocols = {
        'ocr': ['data_sha256', 'image_size', 'max_new_tokens', 'protocol'],
        'vision': ['data_sha256', 'image_size', 'images'],
        'knowledge': ['questions_sha256', 'protocol'],
        'code': ['suite_sha256', 'task_count', 'generation_budget', 'decoding', 'prompt_mode'],
        'code_instruction': ['suite_sha256', 'task_count', 'generation_budget', 'decoding', 'prompt_mode'],
    }
    for collection in [baseline, candidate]:
        digests = {collection[name]['checkpoint_sha256'] for name in names}
        if len(digests) != 1:
            raise ValueError('Comparison mixes reports from different checkpoints')
    for name, keys in protocols.items():
        for key in keys:
            if baseline[name].get(key) != candidate[name].get(key):
                raise ValueError(f'Incomparable {name} protocol: {key}')
    for name, key, identity in [
        ('ocr', 'results', lambda row: row['id']),
        ('vision', 'samples', lambda row: row['id']),
        ('knowledge', 'predictions', lambda row: row['id']),
        ('code', 'results', lambda row: (row['language'], row['task'])),
        ('code_instruction', 'results', lambda row: (row['language'], row['task'])),
    ]:
        first = [identity(row) for row in baseline[name][key]]
        second = [identity(row) for row in candidate[name][key]]
        if not first or first != second or len(set(first)) != len(first):
            raise ValueError(f'Incomparable or duplicate {name} evaluation cases')
    for name, key in [('ocr', 'by_task'), ('knowledge', 'by_domain')]:
        if set(baseline[name][key]) != set(candidate[name][key]):
            raise ValueError(f'Incomparable {name} task/domain coverage')

    checks = {}

    def check(name, before, after, minimum=None, maximum=None):
        before, after = float(before), float(after)
        if not all(math.isfinite(value) for value in [before, after]):
            raise ValueError('Non-finite selection metric: ' + name)
        delta = after - before
        passed = ((minimum is None or delta >= minimum - 1e-12)
                  and (maximum is None or delta <= maximum + 1e-12))
        checks[name] = {'baseline': before, 'candidate': after, 'delta': delta, 'passed': passed}

    old_word = baseline['ocr']['by_task']['word_ocr']
    new_word = candidate['ocr']['by_task']['word_ocr']
    check('word_exact_match_gain', old_word['exact_match_accuracy'], new_word['exact_match_accuracy'], minimum=.05)
    check('word_character_error_reduction', old_word['character_error_rate'], new_word['character_error_rate'], maximum=-.10)
    for task, old in baseline['ocr']['by_task'].items():
        new = candidate['ocr']['by_task'][task]
        if old['examples'] != new['examples']:
            raise ValueError('Incomparable OCR task counts')
        check(task + '_character_error_guard', old['character_error_rate'], new['character_error_rate'], maximum=.05)
    check('caption_nll_guard', baseline['vision']['proper_nll'], candidate['vision']['proper_nll'], maximum=.10)
    old_gain = baseline['vision']['mismatch_nll_gain']
    new_gain = candidate['vision']['mismatch_nll_gain']
    check('caption_image_dependence_guard', old_gain, new_gain, minimum=-.2 * max(0, old_gain))
    for metric in ['generated_accuracy', 'likelihood_accuracy']:
        check('knowledge_' + metric, baseline['knowledge']['overall'][metric],
              candidate['knowledge']['overall'][metric], minimum=-.02)
    for name in ['code', 'code_instruction']:
        for metric in ['functional_pass_at_1', 'syntax_pass_rate', 'efficiency_stress_pass_rate']:
            check(name + '_' + metric, baseline[name][metric], candidate[name][metric], minimum=0)
    return {'promote_candidate': all(item['passed'] for item in checks.values()),
            'checks': checks, 'failed_checks': [name for name, item in checks.items() if not item['passed']],
            'scope': 'Tuning-validation heuristic for one lettering experiment; independent final tests remain required.',
            'goal_achieved': False}
