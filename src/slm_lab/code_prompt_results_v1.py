"""Recompute source diagnostics and unchanged code/knowledge primary counts."""
import math

from .knowledge_evaluation import letter


def close(actual, expected):
    if not isinstance(actual, (int, float)) or not math.isfinite(actual) or not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError('Reported aggregate does not match original individual scores')


def source_metrics(report, selected):
    result = {}
    for role in ['train', 'validation']:
        value = report[role]; rows = value['rows']; original = selected[role]
        if len(rows) != len(original) or value['examples'] != len(original) or value['correctness_metric'] is not False:
            raise ValueError('Complete declared source association diagnostics required')
        for row, source in zip(rows, original):
            if (row['id'] != source['id'] or row['role'] != role or row['language'] != source['language']
                    or row['targets'] != source['target_tokens'] or not math.isfinite(row['matched'])
                    or (row['mismatched'] is not None and not math.isfinite(row['mismatched']))):
                raise ValueError('Exact original source order/target counts and finite full-code scores required')
        n = sum(r['targets'] for r in rows)
        paired = [r for r in rows if r['mismatched'] is not None]
        scores = dict(matched_token_mean_NLL=sum(r['targets'] * r['matched'] for r in rows) / n,
            matching_gap_per_token=sum(r['targets'] * (r['mismatched'] - r['matched']) for r in paired)
                / sum(r['targets'] for r in paired),
            source_association_accuracy=sum(r['matched'] < r['mismatched'] for r in paired) / len(paired))
        if value['target_tokens'] != n:
            raise ValueError('Every original response target must be included')
        for k, v in scores.items():close(value[k], v)
        result[role] = dict(examples=len(rows), target_tokens=n, **scores, correctness_metric=False)
    return result


def code_metrics(report, baseline, expected_checkpoint_sha256):
    if (report['checkpoint_sha256'] != expected_checkpoint_sha256 or report['task_count'] != 24
            or len(report['results']) != 24 or report['prompt_mode'] != baseline['prompt_mode']
            or report['generation_budget'] != baseline['generation_budget']
            or report['decoding'] != baseline['decoding'] or report['suite_sha256'] != baseline['suite_sha256']
            or report['evaluation_provenance'] != baseline['evaluation_provenance']):
        raise ValueError('Exact complete local24-task protocol and runtime baseline required')
    counts = dict(syntax=0, functional=0, efficiency=0)
    correct = []
    for row, before in zip(report['results'], baseline['results']):
        key = (row['language'], row['task'])
        if key != (before['language'], before['task']) or row['specification'] != before['specification']:
            raise ValueError('Same complete original code tasks/order required')
        counts['syntax'] += bool(row['candidate'].get('syntax_valid', False))
        counts['functional'] += bool(row['candidate'].get('functional_passed', False))
        counts['efficiency'] += bool(row['efficiency_stress_passed'])
        if row['candidate'].get('functional_passed'):correct.append(list(key))
    for field, key in [('syntax_pass_rate','syntax'),('functional_pass_at_1','functional'),('efficiency_stress_pass_rate','efficiency')]:
        close(report[field],counts[key]/24)
    return dict(tasks=24, **counts, functional_correct_tasks=correct)


def knowledge_metrics(report, baseline, questions, expected_checkpoint_sha256):
    fields = ['questions_sha256', 'generation_budget', 'contamination_audit_sha256', 'protocol',
        'inference_device_type', 'inference_precision', 'tokenizer_sha256', 'evaluation_provenance']
    if (report['checkpoint_sha256'] != expected_checkpoint_sha256 or any(report[k]!=baseline[k] for k in fields)
            or report['skipped'] or len(report['predictions']) != 559 or len(questions) != 559):
        raise ValueError('Exact complete559-question unchanged development protocol required')
    groups = {};correct = []; choice_correct = []
    for row, before, source in zip(report['predictions'], baseline['predictions'], questions):
        if (row['id'] != source['id'] or row['id'] != before['id'] or row['domain'] != source['domain']
                or row['answer_index'] != int(source['answer_index']) or row['context_tokens'] != before['context_tokens']
                or len(row['candidate_scores']) != len(source['options'])
                or any(not math.isfinite(v) for v in row['candidate_scores'])):
            raise ValueError('Same original questions/gold/order/context and finite option scores required')
        gen = letter(row['generated_response'],len(source['options']))
        choice = max(range(len(source['options'])),key=lambda i:row['candidate_scores'][i])
        generated_right = gen == row['answer_index']; choice_right = choice == row['answer_index']
        if (row['generated_index'] != gen or row['likelihood_index'] != choice
                or row['generated_correct'] != generated_right or row['likelihood_correct'] != choice_right):
            raise ValueError('Primary original generated and choice-score outcomes must recompute')
        counts = groups.setdefault(row['domain'],dict(questions=0,generated=0,likelihood=0))
        counts['questions']+=1;counts['generated']+=generated_right;counts['likelihood']+=choice_right
        if generated_right:correct.append(row['id'])
        if choice_right:choice_correct.append(row['id'])
    for name, counts in groups.items():
        domain = report['by_domain'][name]
        if domain['questions'] != counts['questions']:raise ValueError('Every original domain must be counted')
        close(domain['generated_accuracy'],counts['generated']/counts['questions'])
        close(domain['likelihood_accuracy'],counts['likelihood']/counts['questions'])
    close(report['overall']['generated_accuracy'],len(correct)/559)
    close(report['overall']['likelihood_accuracy'],len(choice_correct)/559)
    return dict(questions=559, generated=len(correct), likelihood=len(choice_correct),
        generated_correct_ids=correct, likelihood_correct_ids=choice_correct, by_domain=groups,
        independent_final_benchmark_certified=False)
