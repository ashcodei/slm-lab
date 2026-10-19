"""Encode complete original APPS human programs; preparation, not admission.

All publisher5000test problems stay outside this candidate. Internal train/
validation roles are grouped by original problem/URL/normalized question.
Every omitted original solution is retained in an explicit review index.
"""
import ast
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

from tokenizers import Tokenizer

from slm_lab.codenet_token_pack_v1 import CodePackWriter
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'data/public-apps-original-inventory-v1'
PUBLISHER = ROOT / 'data/public-apps-publisher-source-v1'
OUTPUT = ROOT / 'data/public-apps-complete-pack-v2'
OWN = ROOT / 'checkpoints/extended-curriculum-v1/joint'


def main():
    if OUTPUT.exists(): raise ValueError('Preserve earlier candidate; no implicit restart or overwrite')
    inventory = json.loads((SOURCE / 'inventory.json').read_text())
    for name, digest in inventory['files_sha256'].items():
        if file_hash(SOURCE / name) != digest: raise ValueError('Exact complete original source inventory required')
    if (inventory['publisher_problem_counts'] != {'train': 5000, 'test': 5000}
            or any(inventory['publisher_train_test_overlap'].values())
            or not inventory['independent_second_archive_pass_reconciles_every_member_and_extracted_train_byte']):
        raise ValueError('Complete independently reconciled original roles with no original URL/question train/test overlap required')
    if shutil.disk_usage(ROOT).free < 30 * 1024**3: raise ValueError('Fund bounded candidate above original26GiB reserve')
    tokenizer_path = OWN / 'tokenizer.json'
    if file_hash(tokenizer_path) != '0072bb6de61da87bd9ba5f4e6739b594cc3e60c62c75a59d25d499629a70088f':
        raise ValueError('Exact existing own scratch-trained BPE required')
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    problems = json.loads((SOURCE / 'train-problems.json').read_text())
    # A connected component cannot cross roles, including duplicate URLs or
    # duplicate full normalized questions under different original IDs.
    parent = {i: i for i in problems}
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    seen = {}
    for i in sorted(problems):
        e = problems[i]
        for key in [('question', e['normalized_question_sha256']), ('url', e['metadata'].get('url'))]:
            if not key[1]: continue
            if key in seen:
                a, b = find(i), find(seen[key]); parent[max(a, b)] = min(a, b)
            else: seen[key] = i
    assignment = {i: ('validation' if int(hashlib.sha256(('APPS-original-role-v1:' + find(i)).encode()).hexdigest()[:8], 16) % 10 == 0 else 'train') for i in problems}
    OUTPUT.mkdir()
    shutil.copy2(tokenizer_path, OUTPUT / 'tokenizer.json')
    shutil.copy2(ROOT / 'configs/model.json', OUTPUT / 'model-config.json')
    shutil.copy2(PUBLISHER / 'publisher-LICENSE', OUTPUT / 'DATASET-LICENSE.txt')
    shutil.copy2(PUBLISHER / 'publisher-README.md', OUTPUT / 'publisher-README.md')
    writer = CodePackWriter(OUTPUT, maximum_bytes=2 * 1024**3, minimum_free_gib=26)
    counts = Counter(); rows_by_role = Counter(); targets = Counter(); fits_by_problem = Counter()
    original_programs = 0; longest = 0; maximum_prompt = 0; role_groups = {'train': set(), 'validation': set()}
    with (OUTPUT / 'original-review-index.jsonl').open('x') as reviews:
        for number, i in enumerate(sorted(problems)):
            entry = problems[i]; folder = SOURCE / 'original-train' / i
            for name, info in entry['files'].items():
                if file_hash(folder / name) != info['sha256']: raise ValueError('Every original train question/program/test/metadata byte must match')
            question = (folder / 'question.txt').read_bytes().decode('utf-8')
            starter = (folder / 'starter_code.py').read_bytes().decode('utf-8') if (folder / 'starter_code.py').exists() else None
            tests = json.loads((folder / 'input_output.json').read_bytes(), parse_int=str) if (folder / 'input_output.json').exists() else None
            fn_name = tests.get('fn_name') if isinstance(tests, dict) else None
            interface = ('Use the original required function or class interface.' if starter or fn_name else 'Read standard input and write standard output as specified.')
            prompt = 'Write a Python solution to the following problem. ' + interface + ' Return only the complete code.\n\n' + question
            if starter: prompt += '\n\nOriginal starter code:\n' + starter
            prefix = [1] + tokenizer.encode('User: ' + prompt + '\nAssistant:').ids
            maximum_prompt = max(maximum_prompt, len(prefix))
            solutions = json.loads((folder / 'solutions.json').read_bytes())
            if not isinstance(solutions, list) or not all(isinstance(s, str) for s in solutions):
                raise ValueError('Every original publisher solution must be a string in its original list')
            role = assignment[i]; role_groups[role].add(find(i))
            for j, program in enumerate(solutions):
                original_programs += 1
                identity = f'apps:publisher-train:{i}:solution:{j}'
                body = tokenizer.encode(program).ids + [2]
                length = len(prefix) + len(body); longest = max(longest, length)
                reasons = []
                if not program.strip(): reasons.append('empty_original_program')
                if length > 4096: reasons.append('complete_original_context_exceeds4096')
                if any(t in program or t in prompt for t in ['<pad>', '<bos>', '<eos>']): reasons.append('literal_reserved_token_marker_requires_review')
                try: ast.parse(program)
                except (SyntaxError, ValueError, RecursionError): reasons.append('not_valid_Python3_AST_unmodified')
                if reasons:
                    for reason in reasons: counts[reason] += 1
                    reviews.write(json.dumps(dict(id=identity, publisher_role='train', candidate_role=role,
                        original_problem=i, original_solution_index=j, complete_context_tokens=length,
                        response_targets=len(body), original_solution_sha256=hashlib.sha256(program.encode()).hexdigest(),
                        original_solutions_file_sha256=entry['files']['solutions.json']['sha256'],
                        reasons=reasons, original_bytes_preserved=True, trained=False), separators=(',', ':')) + '\n')
                    counts['reviewed_original_programs'] += 1
                    continue
                row = dict(id=identity, source_problem_id=i, group_id='apps-original-group:' + find(i), split=role,
                    publisher_split='train', prompt=prompt, response=program, language='Python',
                    source='Original APPS public publisher archive; repository revision ' + inventory['original_publisher_revision'],
                    source_url=entry['metadata'].get('url'), difficulty=entry['metadata']['difficulty'],
                    license='Publisher repository MIT declaration retained; original per-source provenance retained; no additional rights assertion.',
                    annotation_origin='original_publisher_human_solution_not_independently_executed',
                    original_solution_index=j, original_source_body_sha256=hashlib.sha256(program.encode()).hexdigest(),
                    original_solutions_file_sha256=entry['files']['solutions.json']['sha256'],
                    original_question_sha256=entry['files']['question.txt']['sha256'],
                    original_tests_sha256=entry['files'].get('input_output.json', {}).get('sha256'),
                    original_test_file_present=tests is not None, original_fn_name=fn_name,
                    dataset_admitted_to_training=False, execution_verified=False, human_preferences=False)
                ids = prefix + body; labels = [-100] * len(prefix) + body
                writer.add(row, ids, labels)
                rows_by_role[role] += 1; targets[role] += len(body); fits_by_problem[i] += 1
            if (number + 1) % 250 == 0:
                print('original_APPS_problems_encoded', number + 1, 'programs', original_programs, 'fit', sum(rows_by_role.values()), flush=True)
    if role_groups['train'] & role_groups['validation']: raise ValueError('Original problem component crossed candidate roles')
    manifest = writer.finish(dict(source_dataset='APPS', source_inventory=str(SOURCE / 'inventory.json'),
        source_inventory_sha256=file_hash(SOURCE / 'inventory.json'), original_archive_sha256=inventory['original_archive_sha256'],
        original_publisher_revision=inventory['original_publisher_revision'], publisher_train_problems=5000,
        publisher_test_problems_ever_encoded=0, source_group_assignment='SHA256 APPS-original-role-v1: plus URL/question-connected component; modulo10==0 validation',
        current_benchmark_overlap_screen_pending=True, source_execution_verification_pending=True,
        original_review_index_sha256=file_hash(OUTPUT / 'original-review-index.jsonl')))
    if original_programs != sum(rows_by_role.values()) + counts['reviewed_original_programs']:
        raise ValueError('Every original human solution must be either complete fit candidate or preserved explicit review')
    report = dict(completed_at=datetime.now(timezone.utc).isoformat(), original_publisher_train_problems=5000,
        original_human_programs=original_programs, complete_fit_programs_by_role=dict(rows_by_role),
        complete_targets_by_role=dict(targets), original_problem_counts_by_role=dict(Counter(assignment.values())),
        complete_fit_source_problems=len(fits_by_problem), problems_without_fit=[i for i in problems if not fits_by_problem[i]],
        review_counts=dict(counts), maximum_complete_original_context_tokens=longest, maximum_original_prompt_tokens=maximum_prompt,
        publisher_test_programs_encoded=0, original_problem_components_crossing_roles=0, original_programs_clipped=0,
        actual_own_BPE_sha256=file_hash(OUTPUT / 'tokenizer.json'), candidate_manifest_sha256=file_hash(OUTPUT / 'manifest.json'),
        source_inventory_sha256=file_hash(SOURCE / 'inventory.json'), original_review_index_sha256=file_hash(OUTPUT / 'original-review-index.jsonl'),
        original_missing_test_files=195, only_AST_syntax_screened_not_execution_verified=True,
        tests_parsed_only_to_read_original_fn_name=True, raw_numeric_test_bytes_unchanged=True,
        parser_integer_representation='strings in metadata-only parsing; no test cases encoded or executed',
        stored_complete_pack_bytes=manifest['stored_bytes'], training_admitted=False, model_updates=0, goal_achieved=False,
        remaining_before_training=['Screen original candidate train programs/prompts/groups against current held sources and explicit internal validation duplicates.',
            'Verify original publisher cases under a controlled execution interface; retain missing195test-file and runtime qualifications.',
            'Bind admitted exact original own tokenizer/source artifacts to a separate new training plan; do not alter running architecture experiment data.'])
    atomic_json(OUTPUT / 'candidate-preparation.json', report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__': main()
