"""Prepare the preselected sealed-test algorithm assessment without model access."""
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil

from tokenizers import Tokenizer

from slm_lab.codenet_algorithm_assessment_v1 import PROBLEMS, authored_inputs, check_output, expected_output
from slm_lab.codenet_checkpoint_v1 import FileBindings
from slm_lab.codenet_token_pack_v1 import CompleteCodePack
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.model import ModelConfig
from slm_lab.training import Encoder

ROOT = Path(__file__).resolve().parents[1]


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('Prepare evaluation data on CPU without touching live GPUs')
    output, reports = ROOT / 'data/codenet-algorithm-assessment-v1', ROOT / 'reports/codenet-algorithm-assessment-v1'
    if output.exists() or reports.exists():
        raise ValueError('Preserve any earlier algorithm assessment preparation')
    maximum = 64 * 1024**2
    if shutil.disk_usage(ROOT).free < 26 * 1024**3 + maximum:
        raise ValueError('Reserve the complete assessment above live persistent writers')
    output.mkdir()
    reports.mkdir()

    def status(message, **fields):
        atomic_json(reports / 'status.json', {'status': message, 'updated_at': datetime.now(timezone.utc).isoformat(),
            'model_evaluated': False, 'new_model_training_started': False, 'goal_achieved': False, **fields})

    try:
        selection_path = ROOT / 'reports/codenet-algorithm-assessment-v1-selection.json'
        selection = json.loads(selection_path.read_text())
        assert tuple(task['problem_id'] for task in selection['tasks']) == PROBLEMS
        pack_folder = ROOT / 'data/codenet-durable-pack-v1'
        pack = CompleteCodePack(pack_folder, 'test', purpose='audit')
        desc_path = ROOT / 'reports/codenet-description-format-v1/records.jsonl'
        roles_path = ROOT / 'data/codenet-internal-splits-v1/problem-roles.json'
        paths = [Path(__file__), ROOT / 'src/slm_lab/codenet_algorithm_assessment_v1.py',
            ROOT / 'src/slm_lab/codenet_sample_oracles_v1.py', ROOT / 'src/slm_lab/codenet_checkpoint_v1.py',
            ROOT / 'src/slm_lab/codenet_token_pack_v1.py', ROOT / 'src/slm_lab/training.py',
            selection_path, desc_path, roles_path, pack_folder / 'manifest.json', pack_folder / 'tokenizer.json',
            pack_folder / 'model-config.json', pack_folder / 'original-rows.jsonl.gz']
        bound = {**selection['source_files_sha256'], **{str(p): file_hash(p) for p in paths}}
        guard = FileBindings(bound)
        atomic_json(reports / 'plan.json', {'recorded_at': datetime.now(timezone.utc).isoformat(),
            'source_files_sha256': bound, 'maximum_persistent_bytes': maximum,
            'predeclared_selection_sha256': file_hash(selection_path), 'test_only': True,
            'model_evaluated': False, 'new_model_training_started': False, 'goal_achieved': False})
        status('preparing fixed held-task cases, source roles and original references')
        descriptions = {r['problem_id']: r for r in map(json.loads, desc_path.open()) if r['problem_id'] in PROBLEMS}
        roles = json.loads(roles_path.read_text())
        selected = {r['original_archive_member']: r for task in selection['tasks'] for refs in task['references'].values() for r in refs}
        references = {}
        with gzip.open(pack_folder / 'original-rows.jsonl.gz', 'rt') as source:
            for line in source:
                row = json.loads(line)
                member = row['original_archive_member']
                if member not in selected:
                    continue
                if member in references or row['split'] != 'test' or any(row[key] != value for key, value in selected[member].items()):
                    raise ValueError('Original sealed reference identity differs')
                if hashlib.sha256(row['response'].encode()).hexdigest() != row['original_source_body_sha256']:
                    raise ValueError('Original accepted reference bytes differ')
                references[member] = row
        if set(references) != set(selected):
            raise ValueError('Predeclared reference population missing')
        with (output / 'references.jsonl').open('x') as target:
            for member in sorted(references):
                target.write(json.dumps(references[member], ensure_ascii=False) + '\n')
        tokenizer = Tokenizer.from_file(str(pack_folder / 'tokenizer.json'))
        encoder = Encoder(tokenizer, ModelConfig.load(pack_folder / 'model-config.json'), output, image_size=448)
        counts, cases_count, prompt_counts = Counter(), {}, {}
        with (output / 'cases.jsonl').open('x') as cases, (output / 'questions.jsonl').open('x') as questions:
            for task in selection['tasks']:
                p = task['problem_id']
                desc = descriptions[p]
                if (roles[p] != {'group_id': task['group_id'], 'split': 'test'} or task['split'] != 'test'
                    or desc['original_html_sha256'] != task['original_html_sha256']
                    or desc['requires_media_review'] or not desc['recognized_samples_usable']):
                    raise ValueError('Original complete held problem/role differs')
                source_cases = [{'kind': 'published_sample', 'input': pair['input'], 'source_output': pair['output'],
                                 'sample_number': pair['number']} for pair in desc['samples']]
                inputs = source_cases + authored_inputs(p, selection['authored_case_seed'])
                cases_count[p] = Counter()
                for number, case in enumerate(inputs):
                    expected = expected_output(p, case['input'])
                    if not check_output(p, case['input'], expected, case.get('source_output', expected)):
                        raise ValueError('Original published output differs from source-qualified checker: ' + p)
                    if len(case['input'].encode()) > 4 * 1024**2 or len(expected.encode()) > 1024**2:
                        raise ValueError('Bounded case input/output capacity exceeded')
                    record = {**case, 'id': f'{p}:{number:03d}', 'problem_id': p, 'group_id': task['group_id'],
                        'split': 'test', 'expected_output': expected,
                        'input_sha256': hashlib.sha256(case['input'].encode()).hexdigest(),
                        'output_policy': 'source-qualified numeric tolerance' if p == 'p00719' else
                            'source-qualified property validator' if p in {'p03601','p03746'} else 'integer token sequence',
                        'use_for_training_or_preferences': False}
                    cases.write(json.dumps(record, ensure_ascii=False) + '\n')
                    counts[case['kind']] += 1
                    cases_count[p][case['kind']] += 1
                for language, runtime in [('C++', 'GNU C++17'), ('Python', 'Python3.11')]:
                    prompt = (f'Write a complete {language} program.\nTarget language/runtime: {runtime}.\n'
                        'Use standard input and output as the problem specifies. Return only the complete program.\n\n' + desc['text'])
                    row = {'id': f'held-codenet:{p}:{language}', 'problem_id': p, 'group_id': task['group_id'],
                        'split': 'test', 'language': language, 'target_runtime': runtime, 'prompt': prompt,
                        'original_html_sha256': desc['original_html_sha256'], 'source': 'IBM Project CodeNet original problem',
                        'license': 'CDLA-Permissive-2.0', 'use_for_training_or_preferences': False}
                    prompt_tokens = 1 + len(encoder.tokens(encoder.prompt(row)))
                    if prompt_tokens + 2048 > encoder.config.max_text_length:
                        raise ValueError('Whole held prompt plus declared generation window must fit without clipping')
                    row['complete_prompt_tokens'] = prompt_tokens
                    prompt_counts[row['id']] = prompt_tokens
                    questions.write(json.dumps(row, ensure_ascii=False) + '\n')
                cases.flush()
                questions.flush()
                guard.check()
                if sum(p.stat().st_size for p in output.iterdir()) > maximum or shutil.disk_usage(ROOT).free < 512 * 1024**2:
                    raise ValueError('Assessment reservation exceeded')
                print('prepared_held_algorithm', p, dict(cases_count[p]), flush=True)
        shutil.copyfile(pack_folder / 'DATASET-LICENSE.txt', output / 'DATASET-LICENSE.txt')
        files = ['references.jsonl', 'cases.jsonl', 'questions.jsonl', 'DATASET-LICENSE.txt']
        guard.check(full=True)
        manifest = {'format': 'slm-codenet-held-algorithm-assessment-v1',
            'completed_at': datetime.now(timezone.utc).isoformat(), 'selection_sha256': file_hash(selection_path),
            'plan_sha256': file_hash(reports / 'plan.json'), 'original_dataset_manifest_sha256': file_hash(pack_folder / 'manifest.json'),
            'task_ids': list(PROBLEMS), 'tasks': len(PROBLEMS), 'independent_sealed_components': selection['distinct_sealed_components'],
            'language_questions': len(prompt_counts), 'original_reference_sources': len(references),
            'case_counts': dict(counts), 'cases_by_task': {p: dict(c) for p,c in cases_count.items()},
            'whole_prompt_tokens': prompt_counts, 'generation_budget': 2048,
            'tokenizer_sha256': pack.manifest['files_sha256']['tokenizer.json'],
            'files_sha256': {name: file_hash(output / name) for name in files},
            'stored_bytes': sum((output / name).stat().st_size for name in files),
            'test_only': True, 'training_or_preference_use': False,
            'source_reference_execution_pending': True, 'model_evaluated': False,
            'new_model_training_started': False, 'goal_achieved': False,
            'limitations': ['Targeted hand-selected14-task assessment, not random coverage of every language/domain.',
                '13 original sealed components; p00097 and p00439 remain linked under the earlier source grouping.',
                'Fresh authored inputs are not the original judges hidden tests or proof of algorithmic correctness/complexity.',
                'Source isolation applies to the new CodeNet roles; earlier corpora and incomplete clone/semantic coverage remain unverified.',
                'Stress runtime thresholds are fixed local resource checks, not cross-language speed rankings or human preference labels.']}
        if manifest['stored_bytes'] > maximum:
            raise ValueError('Final assessment data budget exceeded')
        atomic_json(output / 'manifest.json', manifest)
        status('fixed held algorithm questions/cases prepared; original-reference execution and model baselines pending',
               tasks=14, language_questions=28, case_counts=dict(counts), references=len(references),
               stored_bytes=manifest['stored_bytes'])
        print(json.dumps(manifest), flush=True)
    except Exception as error:
        status('algorithm assessment preparation stopped; preserve partial evidence', error=str(error))
        raise


if __name__ == '__main__':
    main()
