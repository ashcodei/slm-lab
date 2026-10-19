"""Recompute all three completed fixed assessments; never generate or update."""
import ast
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile

from assess_code_residual_lr_pilot_v1 import compare_fixed_results
from slm_lab.codenet_checkpoint_v1 import FileBindings
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / 'reports/code-objective-exclusive-assessment-v1'


def read(path):
    return json.loads(path.read_text())


def main():
    output = REPORTS / 'independent-complete-readback.json'
    if output.exists():
        raise ValueError('Preserve the earlier completed readback')
    plan = read(REPORTS / 'plan.json')
    guard = FileBindings(plan['source_files_sha256'])
    guard.check(full=True)
    parent = read(REPORTS / 'results.json')
    baseline = read(ROOT / 'reports/language-completed-knowledge-baseline-v1/knowledge.json')['predictions']
    subjects = []
    bound = dict(plan['source_files_sha256'])
    for spec, parent_result in zip(plan['subjects'], parent['subjects'], strict=True):
        name = spec['name']
        folder = REPORTS / name
        stored = read(folder / 'matched-comparison.json')
        assert stored == parent_result and stored['subject'] == name
        assert stored['training_checkpoint_sha256'] == spec['checkpoint_sha256']
        code_report = read(folder / 'code/results.json')
        qa_report = read(folder / 'knowledge.json')
        assert code_report['checkpoint_sha256'] == qa_report['checkpoint_sha256'] == spec['checkpoint_sha256']
        with tempfile.TemporaryDirectory(prefix='slm-full-objectives-readback-') as temporary:
            recomputed = compare_fixed_results(folder / 'code', folder / 'knowledge.json', Path(temporary) / 'comparison.json')
        # The legacy comparator's descriptive scope is specific to its original
        # pilot. Compare all actual data/scores/bindings, then use this run's scope.
        for key, value in recomputed.items():
            if key not in {'completed_at', 'scope'}:
                assert stored[key] == value, (name, key)
        paired = Counter()
        distribution = Counter()
        likelihood_distribution = Counter()
        for before, after in zip(baseline, qa_report['predictions'], strict=True):
            assert before['id'] == after['id']
            paired[f"{int(before['generated_correct'])}->{int(after['generated_correct'])}"] += 1
            distribution[str(after['generated_index'])] += 1
            likelihood_distribution[str(after['likelihood_index'])] += 1
        python_ast = Counter()
        for line in (folder / 'code/generations.jsonl').open():
            generation = json.loads(line)
            if generation['language'] != 'Python':
                continue
            try:
                tree = ast.parse(generation['program_text'])
            except SyntaxError:
                python_ast['syntax_error'] += 1
                continue
            trivial = all(isinstance(node, (ast.Import, ast.ImportFrom, ast.Pass)) or
                          isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and
                          isinstance(node.value.value, str) for node in tree.body)
            python_ast['empty_imports_pass_or_strings_only' if trivial else 'other_statements'] += 1
        subjects.append(dict(subject=name, checkpoint_sha256=spec['checkpoint_sha256'],
            code=recomputed['candidate_code'], knowledge=recomputed['candidate_code_route_raw_knowledge'],
            paired_generated_correct_transitions=dict(paired), parsed_generated_choice_distribution=dict(distribution),
            likelihood_choice_distribution=dict(likelihood_distribution), raw_python_AST=dict(python_ast),
            nominal_uniform_multiple_choice_expected_correct=sum(p['chance_accuracy'] for p in qa_report['predictions']),
            scope=stored['scope']))
        bound.update(recomputed['artifacts_sha256'])
        bound[str(folder / 'matched-comparison.json')] = file_hash(folder / 'matched-comparison.json')
    guard.check(full=True)
    for name in ['plan.json', 'results.json', 'status.json']:
        bound[str(REPORTS / name)] = file_hash(REPORTS / name)
    bound[str(Path(__file__))] = file_hash(Path(__file__))
    atomic_json(output, dict(completed_at=datetime.now(timezone.utc).isoformat(),
        subjects=subjects, all84_original_code_records_and1677_QA_records_recomputed=True,
        source_and_artifact_files_sha256=bound, optimizer_updates=0, program_repair_or_execution_rerun=False,
        candidate_promoted=False, goal_achieved=False,
        limitation='Same original development questions and earlier-source exposure qualifications. '
            'Stored native execution evidence is rescored without rerunning generated programs. '
            'Python AST categories describe raw text only and do not prove functional correctness. '
            'Observed knowledge changes on these reused questions are not independent broad generalization.'))
    for subject in subjects:
        print(json.dumps(subject), flush=True)


if __name__ == '__main__':
    main()
