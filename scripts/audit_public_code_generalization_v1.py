"""Check matched public-code scores and diagnose callable-contract failures.

Alternative-name execution is diagnostic only and never changes the primary score.
It is attempted only when the requested definition is absent and exactly one
top-level function exists. The same generated code and original cases are used.
"""
import ast
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path

from slm_lab.code_quality import execute
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json, jsonl
from slm_lab.evaluation_provenance import capture

ROOT = Path(__file__).resolve().parents[1]


def main():
    folder = ROOT / 'reports/public-code-generalization-v1'
    output = folder / 'comparison-independent-check-v1.json'
    if output.exists():
        raise ValueError('Preserve the earlier independent assessment')
    status = json.loads((folder / 'status.json').read_text())
    if status['status'] != 'public-code policy comparison completed; broader capability unverified':
        raise ValueError('The entire matched assessment must finish first')
    data = ROOT / 'data/public-code-assessment-v1/questions.jsonl'
    questions = list(jsonl(data))
    expected_ids = [(r['id'], r['task_id']) for r in questions]
    if len(expected_ids) != 175 or len(set(expected_ids)) != 175:
        raise ValueError('Original complete supported population required')
    overlap_path = folder / 'instruction-source-overlap-audit.json'
    overlap = json.loads(overlap_path.read_text())
    for name, digest in overlap['bound_files_sha256'].items():
        if file_hash(ROOT / name) != digest:
            raise ValueError('Source-overlap audit inputs changed')
    flagged = {r['benchmark_id'] for r in overlap['function_identifier_normalized_reference_overlaps']}
    comparison = json.loads((folder / 'comparison.json').read_text())
    reports = {label: json.loads((folder / (label + '.json')).read_text())
               for label in ['original_instruction', 'binding_rehearsal']}
    first, second = reports.values()
    fields = ['data_sha256', 'official_sanitized_test_problems', 'supported_problems',
              'excluded_problems', 'protocol', 'generation_budget', 'evaluation_provenance',
              'tokenizer_sha256', 'inference_device_type', 'inference_precision']
    if any(first[k] != second[k] for k in fields):
        raise ValueError('Before/after protocols differ')
    if first['evaluation_provenance'] != capture() or first['data_sha256'] != file_hash(data):
        raise ValueError('Actual evaluation runtime/data differ')
    summaries = {}
    records = {}
    for index, (label, report) in enumerate(reports.items()):
        if [(r['id'], r['task_id']) for r in report['results']] != expected_ids:
            raise ValueError('Report case identities/order differ from all original questions')
        correct = sum(bool(r['execution'].get('functional_passed')) for r in report['results'])
        syntax = sum(bool(r['execution'].get('syntax_valid')) for r in report['results'])
        aggregate = {'problems': 175, 'syntax_correct': syntax, 'functional_correct': correct}
        if comparison[['before', 'after'][index]] != aggregate:
            raise ValueError('Comparison totals differ from actual cases')
        if report['pass_at_1_on_supported_subset'] != correct / 175 or report['syntax_rate_on_supported_subset'] != syntax / 175:
            raise ValueError('Published rates differ from actual cases')
        expected_hash = comparison[['base_checkpoint_sha256', 'candidate_checkpoint_sha256'][index]]
        if report['checkpoint_sha256'] != expected_hash:
            raise ValueError('Policy identity differs from comparison')
        classes = Counter()
        detail = []
        for question, row in zip(questions, report['results']):
            execution = row['execution']
            if execution.get('total_cases') != len(question['cases']):
                raise ValueError('Published case count changed')
            if execution.get('functional_passed') and not (
                    execution.get('syntax_valid') and execution.get('passed_cases') == len(question['cases'])):
                raise ValueError('Functional pass lacks complete published-case success')
            try:
                tree = ast.parse(row['code'])
                definitions = [n.name for n in tree.body if isinstance(n, ast.FunctionDef)]
            except SyntaxError:
                definitions = []
            present = question['function'] in definitions
            if execution.get('functional_passed'):
                category = 'published cases passed'
            elif not execution.get('syntax_valid'):
                category = 'syntax rejected'
            elif not present:
                category = 'requested function definition absent'
            elif execution.get('error') == 'Hidden case mismatch':
                category = 'wrong published values'
            else:
                category = execution.get('error', 'unknown').split(':', 1)[0]
            classes[category] += 1
            alternate = None
            if execution.get('syntax_valid') and not present and len(definitions) == 1:
                alternate = execute({'language': 'Python', 'code': row['code'],
                    'function': definitions[0], 'cases': question['cases'], 'stress': []})
            detail.append({'id': row['id'], 'task_id': row['task_id'],
                'primary_correct': bool(execution.get('functional_passed')),
                'requested_function': question['function'], 'requested_definition_present': present,
                'failure_category': category,
                'flagged_training_reference_overlap': row['id'] in flagged,
                'unique_alternative_name': definitions[0] if alternate is not None else None,
                'alternative_execution': alternate})
        summaries[label] = {**aggregate, 'outcome_categories': dict(classes),
            'flagged_training_reference_cases': len(flagged),
            'flagged_training_reference_correct': sum(r['primary_correct'] for r in detail if r['flagged_training_reference_overlap']),
            'remaining_cases': 175 - len(flagged),
            'remaining_correct': sum(r['primary_correct'] for r in detail if not r['flagged_training_reference_overlap']),
            'unique_alternative_executions': sum(r['alternative_execution'] is not None for r in detail),
            'unique_alternative_passes': sum(bool((r['alternative_execution'] or {}).get('functional_passed')) for r in detail)}
        records[label] = detail
    gained = [b['id'] for a, b in zip(first['results'], second['results'])
              if not a['execution'].get('functional_passed') and b['execution'].get('functional_passed')]
    lost = [a['id'] for a, b in zip(first['results'], second['results'])
            if a['execution'].get('functional_passed') and not b['execution'].get('functional_passed')]
    if gained != comparison['functional_correct_gained'] or lost != comparison['functional_correct_lost']:
        raise ValueError('Gained/lost identities differ from actual outcomes')
    atomic_json(output, {'checked_at': datetime.now(timezone.utc).isoformat(),
        'report_sha256': {name: file_hash(folder / name) for name in
            ['original_instruction.json', 'binding_rehearsal.json', 'comparison.json', overlap_path.name]},
        'audit_source_sha256': file_hash(Path(__file__)), 'summary': summaries, 'records': records,
        'functional_correct_gained': gained, 'functional_correct_lost': lost,
        'primary_score_changed': False, 'new_model_training': False, 'goal_achieved': False,
        'scope': 'Fixed175-case development comparison. Alternative-name executions classify interface failures and never increase the primary score. The remaining173 cases are not certified free of other overlap or pretraining contamination; no efficiency or frontier claim.'})
    print(json.dumps(summaries, indent=2))


if __name__ == '__main__':
    main()
