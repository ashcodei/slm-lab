"""Fixed64original publisher-train questions for own-policy execution feedback; no held data."""
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

from .apps_published_cases_v2 import contract
from .corpus import file_hash

SEED = 'own-onpolicy-code-64-source-v1'


def prepare(root, tokenizer):
    root = Path(root)
    pack = root / 'data/APPS-verified-generation-v1'
    original = root / 'data/public-apps-original-inventory-v1/original-train'
    admission = json.loads((pack / 'admission.json').read_text())
    if (admission.get('format') != 'own-complete-human-code-verified-generation-admission-v1'
            or admission.get('admitted_to_training') is not True or admission.get('original_train_groups_only') is not True
            or admission.get('all_original_questions_programs_tokens_masks_and_raw_case_results_reconciled') is not True
            or file_hash(pack / 'manifest.json') != admission['dataset_manifest_sha256']):
        raise ValueError('Original verified human generation admission required')
    manifest = json.loads((pack / 'manifest.json').read_text())
    for name in ['original-rows.jsonl.gz', 'tokenizer.json']:
        if file_hash(pack / name) != manifest['files_sha256'][name]:
            raise ValueError('Exact admitted original source records and own tokenizer required')
    with gzip.open(pack / 'original-rows.jsonl.gz', 'rt') as stream:
        rows = list(map(json.loads, stream))
    if len(rows) != admission['complete_examples'] or len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Every original verified training record exactly once required')
    eligible = {'stdin': [], 'function': []}
    for row in rows:
        if row.get('split') != 'train' or row.get('publisher_split') != 'train':
            raise ValueError('Every calibration source must be publisher train before accessing its response')
        prefix = [1, *tokenizer.encode('User: ' + row['prompt'] + '\nAssistant:').ids]
        body = [*tokenizer.encode(row['response']).ids, 2]
        if (len(body) != row['target_tokens'] or hashlib.sha256(np.asarray(prefix + body, dtype='<u2').tobytes()).hexdigest()
                != row['complete_input_ids_sha256']):
            raise ValueError('Whole original verified prompt, response and own-token sequence must reconstruct')
        if len(body) > 512 or len(prefix) + 512 > 4096:
            continue
        mode = 'function' if row['original_fn_name'] is not None else 'stdin'
        eligible[mode].append((hashlib.sha256((SEED + row['source_problem_id']).encode()).hexdigest(), row, prefix))
    chosen = []
    for mode in ['stdin', 'function']:
        candidates = sorted(eligible[mode], key=lambda item: (item[0], item[1]['id']))
        seen = set()
        for _, row, prefix in candidates:
            if row['source_problem_id'] in seen:
                continue
            seen.add(row['source_problem_id'])
            folder = original / row['source_problem_id']
            paths = [folder / name for name in ['question.txt', 'solutions.json', 'input_output.json']]
            if (folder / 'starter_code.py').exists():
                paths.append(folder / 'starter_code.py')
            solution = json.loads((folder / 'solutions.json').read_text())[row['original_solution_index']]
            cases = contract((folder / 'input_output.json').read_bytes())
            if (solution != row['response'] or file_hash(folder / 'question.txt') != row['original_question_sha256']
                    or file_hash(folder / 'solutions.json') != row['original_solutions_file_sha256']
                    or cases['original_tests_sha256'] != row['original_tests_sha256'] or cases['mode'] != mode
                    or cases['total_cases'] != row['complete_original_cases']):
                raise ValueError('Exact unchanged original human target and all original finite cases required')
            chosen.append(dict(id=row['id'], original_problem=row['source_problem_id'], mode=mode,
                split='train', publisher_split='train', prefix_ids=prefix, prompt=row['prompt'],
                original_human_program=row['response'], target_tokens=row['target_tokens'],
                complete_input_ids_sha256=row['complete_input_ids_sha256'], cases=cases,
                original_files_sha256={str(path): file_hash(path) for path in paths},
                original_reference_execution_already_verified=True, held_cases_supplied_to_model=False))
            if len(seen) == 32:
                break
    if Counter(r['mode'] for r in chosen) != {'stdin': 32, 'function': 32} or len({r['original_problem'] for r in chosen}) != 64:
        raise ValueError('Exactly64distinct train problems:32stdin/32function required')
    return chosen
