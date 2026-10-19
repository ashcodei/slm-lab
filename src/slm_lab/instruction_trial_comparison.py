"""Compare complete instruction-trial evidence without promoting a policy."""
import math

from .alignment_selection import CASE_FIELDS, PROTOCOL_FIELDS
from .code_tasks import TASKS

COMMON = ['evaluation_provenance', 'inference_device_type', 'inference_precision', 'tokenizer_sha256']


def same(before, after, fields):
    for field in fields:
        if field not in before or field not in after or before[field] != after[field]:
            raise ValueError('Unmatched assessment field: ' + field)


def rate(value, count, total, name):
    if (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
            or not math.isclose(value, count / total, rel_tol=1e-10, abs_tol=1e-10)):
        raise ValueError('Reported score differs from actual cases: ' + name)


def code_summary(report):
    rows = report['results']; total = len(rows)
    if not total or report['task_count'] != total:
        raise ValueError('Missing or inconsistent complete code population')
    if any(row['efficiency_stress_passed'] and not row['candidate'].get('functional_passed') for row in rows):
        raise ValueError('Efficiency pass without functional correctness')
    counts = {'syntax': sum(bool(row['candidate'].get('syntax_valid')) for row in rows),
              'functional': sum(bool(row['candidate'].get('functional_passed')) for row in rows),
              'efficiency': sum(bool(row['efficiency_stress_passed']) for row in rows)}
    for key, field in [('syntax', 'syntax_pass_rate'), ('functional', 'functional_pass_at_1'),
                       ('efficiency', 'efficiency_stress_pass_rate')]:
        rate(report[field], counts[key], total, field)
    languages = {row.get('language', 'C++') for row in rows}; groups = {}
    for language in sorted(languages):
        items = [row for row in rows if row.get('language', 'C++') == language]
        groups[language] = {'tasks': len(items), 'syntax_passed': sum(bool(row['candidate'].get('syntax_valid')) for row in items),
                           'functional_passed': sum(bool(row['candidate'].get('functional_passed')) for row in items),
                           'efficiency_passed': sum(bool(row['efficiency_stress_passed']) for row in items)}
    if 'by_language' in report and report['by_language'] != groups:
        raise ValueError('Reported language score differs from actual code cases')
    return {'tasks': total, **counts, 'by_language': groups,
            'syntax_passed_with_empty_extracted_code': sum(bool(row['candidate'].get('syntax_valid')) and not row['code'].strip() for row in rows)}


def compare_code(before, after, family, expected_count):
    same(before, after, COMMON + PROTOCOL_FIELDS[family])
    key, identity = CASE_FIELDS[family]
    identity = identity + ['intended_scaling']
    first = [tuple(row[field] for field in identity) for row in before[key]]
    second = [tuple(row[field] for field in identity) for row in after[key]]
    if len(first) != expected_count or first != second or len(set(first)) != len(first):
        raise ValueError('Changed, duplicate or incomplete code task identities')
    if family == 'cpp':
        same(before['toolchain'], after['toolchain'], ['version', 'compiler_sha256', 'archive_sha256'])
    old, new = code_summary(before), code_summary(after); changes = {}
    label = lambda row: row.get('language', 'C++') + ':' + row['task']
    for metric, predicate in [('syntax', lambda row: bool(row['candidate'].get('syntax_valid'))),
                              ('functional', lambda row: bool(row['candidate'].get('functional_passed'))),
                              ('efficiency', lambda row: bool(row['efficiency_stress_passed']))]:
        changes[metric] = {'gained': [label(b) for a, b in zip(before[key], after[key]) if not predicate(a) and predicate(b)],
                           'lost': [label(a) for a, b in zip(before[key], after[key]) if predicate(a) and not predicate(b)]}
    return {'before': old, 'after': new, 'case_changes': changes}


def knowledge_summary(report):
    rows = report['predictions']; total = len(rows)
    if total != 559 or report['skipped'] or report['overall']['questions'] != total:
        raise ValueError('Complete559-question assessment required')
    for row in rows:
        if (row['generated_correct'] != (row['generated_index'] == row['answer_index'])
                or row['likelihood_correct'] != (row['likelihood_index'] == row['answer_index'])):
            raise ValueError('Knowledge outcome flags differ from selected answers')
    generated = sum(row['generated_correct'] for row in rows); likelihood = sum(row['likelihood_correct'] for row in rows)
    rate(report['overall']['generated_accuracy'], generated, total, 'generated_accuracy')
    rate(report['overall']['likelihood_accuracy'], likelihood, total, 'likelihood_accuracy')
    rate(report['overall']['format_compliance'], sum(row['generated_index'] is not None for row in rows), total, 'format_compliance')
    domains = {row['domain'] for row in rows}
    if domains != set(report['by_domain']):
        raise ValueError('Knowledge domain population differs')
    for domain in domains:
        items = [row for row in rows if row['domain'] == domain]; group = report['by_domain'][domain]
        if group['questions'] != len(items):
            raise ValueError('Knowledge domain question count differs')
        rate(group['generated_accuracy'], sum(row['generated_correct'] for row in items), len(items), 'domain generated accuracy')
        rate(group['likelihood_accuracy'], sum(row['likelihood_correct'] for row in items), len(items), 'domain likelihood accuracy')
    strict = 0; strict_correct = 0
    for row in rows:
        answer = row['generated_response'].strip()
        options = round(1 / row['chance_accuracy'])
        if len(answer) == 1 and 'A' <= answer <= chr(64 + options):
            strict += 1; strict_correct += ord(answer) - 65 == row['answer_index']
    return {'questions': total, 'generated_correct': generated, 'likelihood_correct': likelihood,
            'overall': report['overall'], 'by_domain': report['by_domain'],
            'derived_strict_letter_only_responses': strict, 'derived_strict_letter_only_correct': strict_correct,
            'strict_metric_scope': 'Separate post-hoc one-letter-only check; original benchmark scores and parser remain unchanged.'}


def compare_reports(before, after):
    families = ['code', 'code_instruction', 'cpp', 'knowledge', 'module_loss']
    for collection in [before, after]:
        if set(collection) != set(families) or len({collection[name]['checkpoint_sha256'] for name in families}) != 1:
            raise ValueError('Complete single-policy trial report collection required')
        for family in families:
            same(collection['code'], collection[family], COMMON)
    output = {}
    for family in ['code', 'code_instruction', 'cpp']:
        count = 12 if family == 'cpp' else 24
        output[family] = compare_code(before[family], after[family], family, count)
        expected = {task[0] for task in TASKS}
        if family == 'cpp':
            if {row['task'] for row in before[family]['results']} != expected:
                raise ValueError('Original complete C++ task population required')
        elif {(row['language'], row['task']) for row in before[family]['results']} != {
                (language, task) for language in ['Python', 'JavaScript'] for task in expected}:
            raise ValueError('Original complete Python/JavaScript tasks required')
    old, new = before['knowledge'], after['knowledge']
    same(old, new, COMMON + PROTOCOL_FIELDS['knowledge'] + ['contamination_audit_sha256'])
    fields = ['id', 'domain', 'answer_index', 'context_tokens', 'chance_accuracy']
    identities = [[tuple(row[field] for field in fields) for row in report['predictions']] for report in [old, new]]
    if identities[0] != identities[1] or len(set(identities[0])) != 559:
        raise ValueError('Changed or duplicate full knowledge population')
    output['knowledge'] = {'before': knowledge_summary(old), 'after': knowledge_summary(new),
        'generated_correct_lost': [a['id'] for a, b in zip(old['predictions'], new['predictions']) if a['generated_correct'] and not b['generated_correct']],
        'generated_correct_gained': [b['id'] for a, b in zip(old['predictions'], new['predictions']) if not a['generated_correct'] and b['generated_correct']],
        'likelihood_correct_lost': [a['id'] for a, b in zip(old['predictions'], new['predictions']) if a['likelihood_correct'] and not b['likelihood_correct']],
        'likelihood_correct_gained': [b['id'] for a, b in zip(old['predictions'], new['predictions']) if not a['likelihood_correct'] and b['likelihood_correct']]}
    old, new = before['module_loss'], after['module_loss']
    same(old, new, COMMON + ['task', 'data_sha256', 'examples', 'available_examples', 'sample_indices', 'target_tokens', 'bf16'])
    if old['examples'] != 181 or old['available_examples'] != 181 or old['truncated_sequences'] or new['truncated_sequences']:
        raise ValueError('Complete unclipped181-module loss required')
    groups = {name for name in old['by_group'] if name.startswith('language:')}
    if len(groups) != 6 or groups != {name for name in new['by_group'] if name.startswith('language:')}:
        raise ValueError('Complete six-language module response loss required')
    for report in [old, new]:
        tokens = sum(report['by_group'][name]['target_tokens'] for name in groups)
        if tokens != report['target_tokens'] or sum(report['by_group'][name]['examples'] for name in groups) != 181:
            raise ValueError('Module language counts/target budgets differ')
        mean = sum(report['by_group'][name]['nll'] * report['by_group'][name]['target_tokens'] for name in groups) / tokens
        if not math.isfinite(report['nll']) or not math.isclose(report['nll'], mean, rel_tol=1e-8, abs_tol=1e-8):
            raise ValueError('Module aggregate loss differs from actual language groups')
    for name in groups:
        same(old['by_group'][name], new['by_group'][name], ['examples', 'target_tokens'])
    output['module_response_loss'] = {'before_nll': old['nll'], 'after_nll': new['nll'], 'target_tokens': old['target_tokens'],
        'examples': 181, 'relative_nll_change': (new['nll'] - old['nll']) / old['nll'],
        'by_language': {name: {'before_nll': old['by_group'][name]['nll'], 'after_nll': new['by_group'][name]['nll']} for name in sorted(groups)},
        'scope': 'Teacher-forced full response prediction, separate from generated correctness; all inputs are text-only.'}
    return {'base_checkpoint_sha256': before['code']['checkpoint_sha256'], 'candidate_checkpoint_sha256': after['code']['checkpoint_sha256'],
            'assessments': output, 'automatic_policy_promotion': False, 'goal_achieved': False,
            'limitations': ['Reused development tasks, not independent final or frontier-model comparison.',
                           'No new image behavior or human-feedback preference result measured by this trial.',
                           'Syntax, teacher-forced loss and strict format checks do not certify correct algorithms or broad knowledge.']}
