"""Assess every predeclared held reference on published, authored and stress cases."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

from slm_lab.algorithm_program_v1 import PROFILE, assess_program
from slm_lab.codenet_checkpoint_v1 import FileBindings
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.standalone_gnu_v1 import gnu_toolchain
from slm_lab.standalone_numpy_v1 import numpy_runtime_paths

ROOT = Path(__file__).resolve().parents[1]


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('Reference assessment must not use the live GPUs')
    folder = ROOT / 'reports/codenet-algorithm-reference-readiness-v1'
    if folder.exists():
        raise ValueError('Preserve every earlier selected-reference result')
    maximum = 32 * 1024**2
    if shutil.disk_usage(ROOT).free < 26 * 1024**3 + maximum:
        raise ValueError('Reserve the complete small reference report above persistent writers')
    folder.mkdir()

    def status(message, **fields):
        atomic_json(folder / 'status.json', {'status': message, 'updated_at': datetime.now(timezone.utc).isoformat(),
            'model_evaluated': False, 'new_model_training_started': False, 'goal_achieved': False, **fields})

    try:
        status('verifying fixed held assessment and sandbox runtime files')
        data = ROOT / 'data/codenet-algorithm-assessment-v1'
        manifest = json.loads((data / 'manifest.json').read_text())
        selection_path = ROOT / 'reports/codenet-algorithm-assessment-v1-selection.json'
        selection = json.loads(selection_path.read_text())
        if file_hash(selection_path) != manifest['selection_sha256'] or not manifest['test_only'] or manifest['training_or_preference_use']:
            raise ValueError('Exact preselected held-only assessment required')
        tools = gnu_toolchain()
        files = [Path(__file__), ROOT / 'src/slm_lab/algorithm_program_v1.py',
            ROOT / 'src/slm_lab/codenet_algorithm_assessment_v1.py', ROOT / 'src/slm_lab/codenet_sample_oracles_v1.py',
            ROOT / 'src/slm_lab/codenet_checkpoint_v1.py', ROOT / 'src/slm_lab/standalone_gnu_v1.py',
            ROOT / 'src/slm_lab/standalone_numpy_v1.py', ROOT / 'src/slm_lab/standalone_program_v1.py',
            ROOT / 'src/slm_lab/native_code.py', ROOT / 'scripts/native_sandbox.py',
            ROOT / 'scripts/python_program_sandbox_v1.py', ROOT / '.tools/gnu-toolchain-v1.json',
            Path(tools['file_manifest']), data / 'manifest.json', selection_path]
        bound = {**selection['source_files_sha256'], **{str(p): file_hash(p) for p in files}}
        for name, digest in manifest['files_sha256'].items():
            bound[str(data / name)] = digest
        for name, digest in selection['source_files_sha256'].items():
            if file_hash(Path(name)) != digest:
                raise ValueError('Original fixed selection source changed')
        guard = FileBindings(bound)
        runtime_files = sorted({p.resolve() for directory in numpy_runtime_paths() for p in directory.rglob('*') if p.is_file()})
        runtime_guard = FileBindings({str(p): file_hash(p) for p in runtime_files})
        refs = list(map(json.loads, (data / 'references.jsonl').open()))
        cases = defaultdict(list)
        for case in map(json.loads, (data / 'cases.jsonl').open()):
            if case['split'] != 'test' or case['use_for_training_or_preferences']:
                raise ValueError('Every test case must remain outside training/preferences')
            cases[case['problem_id']].append(case)
        expected_refs = {r['original_archive_member'] for task in selection['tasks'] for rows in task['references'].values() for r in rows}
        if {r['original_archive_member'] for r in refs} != expected_refs or len(refs) != len(expected_refs):
            raise ValueError('Original predeclared reference set differs')
        atomic_json(folder / 'plan.json', {'recorded_at': datetime.now(timezone.utc).isoformat(),
            'source_files_sha256': bound, 'runtime_profile': PROFILE,
            'numpy_runtime_files_sha256': runtime_guard.files and {str(p): digest for p,digest in runtime_guard.files.items()},
            'predeclared_references': len(refs), 'tasks': manifest['tasks'],
            'language_questions': manifest['language_questions'], 'case_counts': manifest['case_counts'],
            'maximum_workers': 1, 'maximum_report_bytes': maximum,
            'scope': 'Original fixed test-reference behavior only, no model evaluation/training, no outcome-based removal.',
            'model_evaluated': False, 'new_model_training_started': False, 'goal_achieved': False})
        qualifications, supported, records = Counter(), defaultdict(list), []
        with (folder / 'records.jsonl').open('x') as output:
            for row in refs:
                guard.check()
                if shutil.disk_usage(tempfile.gettempdir()).free < 12 * 1024**3 + 256 * 1024**2:
                    raise ValueError('Preserve temporary compiler/runtime reserve')
                if row['split'] != 'test' or hashlib.sha256(row['response'].encode()).hexdigest() != row['original_source_body_sha256']:
                    raise ValueError('Original unchanged held reference required')
                record = {key: row[key] for key in ['id','source_problem_id','group_id','split','language','original_language',
                    'original_archive_member','original_source_body_sha256']}
                record.update(assess_program(row['response'].encode(), row['language'], cases[row['source_problem_id']], tools))
                records.append(record)
                output.write(json.dumps(record) + '\n')
                output.flush()
                qualifications[record['qualification']] += 1
                if record['qualification'] == 'all_selected_cases_agree_under_fixed_profile':
                    supported[(row['source_problem_id'],row['language'])].append(row['id'])
                if sum(p.stat().st_size for p in folder.iterdir()) > maximum or shutil.disk_usage(ROOT).free < 512 * 1024**2:
                    raise ValueError('Reference report reservation exceeded')
                status('checking all predeclared held references with one bounded CPU worker',
                    checked_references=len(records), expected_references=len(refs), qualifications=dict(qualifications),
                    task_language_interfaces_with_compatible_reference=len(supported))
                print('checked_held_reference',len(records),row['source_problem_id'],row['language'],record['qualification'],flush=True)
        guard.check(full=True)
        runtime_guard.check(full=True)
        gnu_toolchain()  # Recheck every verified compiler/runtime byte after the pass.
        summary = {'completed_at': datetime.now(timezone.utc).isoformat(),
            'assessment_manifest_sha256': file_hash(data / 'manifest.json'), 'checked_references':len(records),
            'all_predeclared_references_retained':len(records)==len(expected_refs),
            'qualification_counts':dict(qualifications),
            'source_rule_coverage':{p:{language:supported[(p,language)] for language in ['C++','Python']} for p in manifest['task_ids']},
            'task_language_interfaces_with_compatible_reference':len(supported),
            'expected_task_language_interfaces':manifest['language_questions'],
            'no_separate_stress_case_for_tasks':[p for p,c in manifest['cases_by_task'].items() if not c.get('stress')],
            'source_reference_coverage_complete':len(supported)==manifest['language_questions'],
            'model_evaluated':False,'new_model_training_started':False,'goal_achieved':False,
            'scope':'Original-source/local authored-case compatibility and bounded-runtime qualifications. Not a model score, universal judge, original hidden-case assessment, asymptotic efficiency proof, human preference or pretraining-independence certificate.'}
        atomic_json(folder/'summary.json',summary)
        status('predeclared held-reference assessment completed; model baselines remain pending',
            checked_references=len(records),qualifications=dict(qualifications),
            task_language_interfaces_with_compatible_reference=len(supported))
        print(json.dumps(summary),flush=True)
    except Exception as error:
        status('held-reference assessment stopped; preserve exact results and inspect',error=str(error))
        raise


if __name__ == '__main__':
    main()
