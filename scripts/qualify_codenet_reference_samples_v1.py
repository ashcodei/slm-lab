"""Preserve raw diagnostics and independently check original nonliteral rules."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from slm_lab.codenet_sample_oracles_v1 import SUPPORTED, check_sample
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json

ROOT = Path(__file__).resolve().parents[1]


def main():
    folder = ROOT / 'reports/codenet-standalone-reference-pilot-v2'
    destination = folder / 'source-qualified-samples-v1.json'
    if destination.exists():
        raise ValueError('Preserve earlier sample qualifications')
    summary_path, plan_path = folder / 'summary.json', folder / 'plan.json'
    records_path, selection_path = folder / 'records.jsonl', folder / 'selection.jsonl'
    summary, plan = json.loads(summary_path.read_text()), json.loads(plan_path.read_text())
    for path, key in [(records_path, 'records_sha256'), (selection_path, 'selection_sha256'), (plan_path, 'plan_sha256')]:
        if file_hash(path) != summary[key]:
            raise ValueError('Completed reference assessment differs')
    prior_selection = ROOT / 'reports/codenet-standalone-reference-pilot-v1/selection.jsonl'
    if file_hash(prior_selection) != summary['selection_sha256']:
        raise ValueError('Original predeclared pilot selection differs')
    restored_bindings = {}
    for name, digest in plan['input_files_sha256'].items():
        path = Path(name)
        if file_hash(path) != digest:
            # The completed pilot used the preserved runner before a subsequent
            # reporting-only fix for input pipes closed early by a target.
            if path != ROOT / 'src/slm_lab/standalone_program_v1.py':
                raise ValueError('Unexpected post-assessment bound-source change')
            snapshot = ROOT / 'reports/source-snapshots/codenet-standalone-v2-standalone_program_v1.py'
            if file_hash(snapshot) != digest:
                raise ValueError('Completed pilot runtime snapshot differs')
            restored_bindings[name] = {'original_snapshot': str(snapshot),
                'pilot_sha256': digest, 'current_sha256': file_hash(path),
                'later_change': 'Record actual successfully written input bytes when a target closes stdin early; output judgement is unchanged.'}
    records, selection = list(map(json.loads, records_path.open())), list(map(json.loads, selection_path.open()))
    if len(records) != len(selection) or len(records) != 64:
        raise ValueError('Completed reference population differs')
    quotas = Counter((row['language'], row['split']) for row in records)
    for language in ['Python', 'C++']:
        for role, expected in [('train', 16), ('validation', 8), ('test', 8)]:
            if quotas[(language, role)] != expected:
                raise ValueError('Predeclared language/role quota differs')
    for row, selected in zip(records, selection):
        for key in ['archive_member', 'problem_id', 'group_id', 'language', 'split', 'original_source_body_sha256']:
            if row[key] != selected[key]:
                raise ValueError('Executed original reference identity differs')
    desc_path = ROOT / 'reports/codenet-description-format-v1/records.jsonl'
    descriptions = {row['problem_id']: row for row in map(json.loads, desc_path.open())}
    body = json.loads((ROOT / 'reports/codenet-candidate-bodies-v1/bodies.json').read_text())
    inspection = Path(body['candidate_cache']).parent / 'inspection-v1'
    qualifications, final = [], Counter()
    for row in records:
        outcome = row['qualification']
        if outcome == 'sample_output_or_oracle_review_needed' and row['problem_id'] in SUPPORTED:
            desc = descriptions[row['problem_id']]
            html = inspection / desc['original_archive_member'].split('/', 1)[1]
            if file_hash(html) != desc['original_html_sha256'] or len(row['samples']) != len(desc['samples']):
                raise ValueError('Original HTML/sample identity differs')
            cases = []
            for observed, original in zip(row['samples'], desc['samples']):
                if observed['status'] != 'completed' or hashlib.sha256(observed['stdout_prefix'].encode()).hexdigest() != observed['stdout_sha256']:
                    raise ValueError('Complete original observed output required for qualification')
                cases.append(check_sample(row['problem_id'], original['input'], original['output'], observed['stdout_prefix']))
            qualifications.append({'archive_member': row['archive_member'], 'problem_id': row['problem_id'],
                'original_html_sha256': desc['original_html_sha256'], 'sample_results': cases,
                'all_recognized_samples_pass_source_rules': all(case['passed'] for case in cases)})
            if all(case['passed'] for case in cases):
                outcome = 'recognized_samples_agree_under_original_nonliteral_rule'
        final[outcome] += 1
    source_paths = [Path(__file__), ROOT / 'src/slm_lab/codenet_sample_oracles_v1.py',
        ROOT / 'tests/test_codenet_sample_oracles_v1.py', ROOT / 'reports/codenet-sample-oracles-v1-focused-tests-v2.log',
        records_path, selection_path, summary_path, desc_path]
    atomic_json(destination, {
        'checked_at': datetime.now(timezone.utc).isoformat(), 'references_checked': len(records),
        'same_original_predeclared_selection_verified': True, 'language_role_quotas_verified': True,
        'bound_source_hashes_and_original_html_verified': True,
        'source_files_sha256': {str(path): file_hash(path) for path in source_paths},
        'completed_pilot_historical_runtime_bindings': restored_bindings,
        'raw_qualification_counts': dict(Counter(row['qualification'] for row in records)),
        'source_qualified_counts': dict(final), 'nonliteral_problem_qualifications': qualifications,
        'source_rule_focused_tests_passed': 6, 'raw_reference_records_changed': False,
        'dataset_admitted_to_training': False, 'new_model_training_started': False,
        'goal_achieved': False,
        'scope': 'Original recognised samples only. Four source-specific checks qualify nonunique score/sugar-water outputs and stated probability/route-time error tolerances. Reference programs were executed; model outputs, hidden cases, efficiency and human preferences were not assessed.'})
    print(json.dumps(dict(final), indent=2))


if __name__ == '__main__':
    main()
