"""Independently read actual original APPS/code tokens and review accounting.

Exact normalized duplicate screens are limited to named retained held sources;
they do not certify semantic independence or published-case correctness.
"""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import ast
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

from slm_lab.code_tasks import TASKS
from slm_lab.codenet_token_pack_v1 import CompleteCodePack
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json

ROOT = Path(__file__).resolve().parents[1]
PACK = ROOT / 'data/public-apps-complete-pack-v2'
SOURCE = ROOT / 'data/public-apps-original-inventory-v1'


def digest(code):
    # ast.unparse recurses on long legal expression chains. Hash the complete
    # AST iteratively instead, retaining identifiers/literals/operator fields
    # and removing only comments, formatting and standalone docstrings.
    tree = ast.parse(code)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
                node.body.pop(0)
                if not node.body: node.body.append(ast.Pass())
    result = hashlib.sha256(); pending = [tree]
    while pending:
        value = pending.pop()
        if isinstance(value, ast.AST):
            token = ('AST:' + type(value).__name__).encode()
            fields = list(ast.iter_fields(value))
            pending.extend(reversed([item for key, item in fields for item in [key, item]]))
        elif isinstance(value, list):
            token = ('LIST:' + str(len(value))).encode(); pending.extend(reversed(value))
        else: token = (type(value).__name__ + ':' + repr(value)).encode()
        result.update(len(token).to_bytes(8, 'little')); result.update(token)
    return result.hexdigest()


def main():
    target = PACK / 'independent-source-audit.json'
    if target.exists(): raise ValueError('Preserve earlier complete independent source audit')
    prepared = json.loads((PACK / 'candidate-preparation.json').read_text())
    if file_hash(PACK / 'manifest.json') != prepared['candidate_manifest_sha256']:
        raise ValueError('Exact completed candidate required')
    pack = CompleteCodePack(PACK, 'train', purpose='audit')
    held_pack = CompleteCodePack(PACK, 'validation', purpose='audit')
    tokenizer = Tokenizer.from_file(str(PACK / 'tokenizer.json'))
    originals = {}; questions = {}; metadata = json.loads((SOURCE / 'train-problems.json').read_text())
    expected = set(); all_source_files = []
    for i, entry in metadata.items():
        folder = SOURCE / 'original-train' / i
        for name, info in entry['files'].items():
            if file_hash(folder / name) != info['sha256']: raise ValueError('Every original publisher train file must match')
        values = json.loads((folder / 'solutions.json').read_bytes())
        originals[i] = values; questions[i] = (folder / 'question.txt').read_bytes().decode('utf-8')
        expected.update(f'apps:publisher-train:{i}:solution:{j}' for j in range(len(values)))
    held_hashes = defaultdict(set)
    mbpp = ROOT / 'data/verified-code-source-cache-v1/sanitized-mbpp.json'
    for row in json.loads(mbpp.read_text()):
        if 11 <= row['task_id'] <= 510:
            for code in [row['code'], '\n'.join(row.get('test_imports', [])) + '\n' + row['code']]:
                held_hashes[digest(code)].add('sanitized-MBPP-publisher-test')
    for task in TASKS: held_hashes[digest(task[4])].add('local-Python12-development-suite')
    codenet = ROOT / 'data/codenet-durable-pack-v2/original-rows.jsonl.gz'
    with gzip.open(codenet, 'rt') as stream:
        for row in map(json.loads, stream):
            if row['split'] != 'train' and 'python' in row['language'].lower():
                try: held_hashes[digest(row['response'])].add('CodeNet-' + row['split'])
                except (SyntaxError, ValueError, RecursionError): pass
    ids = set(); count = Counter(); tokens = Counter(); observed_groups = defaultdict(set)
    observed_urls = defaultdict(set); observed_questions = defaultdict(set); code_rows = defaultdict(list); held_matches = []
    with gzip.open(PACK / 'original-rows.jsonl.gz', 'rt') as stream:
        for position, row in enumerate(map(json.loads, stream)):
            i = row['source_problem_id']; j = row['original_solution_index']; role = row['split']
            original = originals[i][j]
            if (row['id'] not in expected or row['id'] in ids or row['response'] != original
                    or row['publisher_split'] != 'train' or row['language'] != 'Python'
                    or row['original_source_body_sha256'] != hashlib.sha256(original.encode()).hexdigest()
                    or questions[i] not in row['prompt'] or row['token_pack_index'] != position):
                raise ValueError('Every complete candidate must match its unique original human source exactly')
            index = pack.index[position]; start = int(index['target_start']); offset = int(index['offset']); length = int(index['length'])
            prefix = [1] + tokenizer.encode('User: ' + row['prompt'] + '\nAssistant:').ids
            body = tokenizer.encode(original).ids + [2]
            actual = pack.ids[offset:offset + length]
            correct = np.asarray(prefix + body, dtype='<u2')
            if (start != len(prefix) or not np.array_equal(actual, correct) or len(body) != row['target_tokens']
                    or row['complete_input_ids_sha256'] != hashlib.sha256(correct.tobytes()).hexdigest()
                    or int(index['role']) != (0 if role == 'train' else 1) or length > 4096):
                raise ValueError('Every actual token, original whole prompt/program/EOS, target mask and sealed role must reconstruct')
            normalized = digest(original)
            if normalized in held_hashes:
                held_matches.append(dict(id=row['id'], role=role, held_sources=sorted(held_hashes[normalized])))
            code_rows[normalized].append((row['id'], role))
            ids.add(row['id']); count[role] += 1; tokens[role] += len(body)
            observed_groups[row['group_id']].add(role)
            url = row.get('source_url')
            if url: observed_urls[url].add(role)
            observed_questions[metadata[i]['normalized_question_sha256']].add(role)
            if len(ids) % 10000 == 0: print('complete_original_APPS_rows_reencoded', len(ids), flush=True)
    if count != Counter(prepared['complete_fit_programs_by_role']) or tokens != Counter(prepared['complete_targets_by_role']):
        raise ValueError('All complete original role populations/target counts required')
    reviewed = set()
    for row in map(json.loads, (PACK / 'original-review-index.jsonl').read_text().splitlines()):
        i = row['original_problem']; j = row['original_solution_index']; identity = row['id']
        if (identity not in expected or identity in ids or identity in reviewed or row['trained'] is not False
                or row['original_solution_sha256'] != hashlib.sha256(originals[i][j].encode()).hexdigest()):
            raise ValueError('Every reviewed original must remain unique, unmodified and outside candidate fit inputs')
        reviewed.add(identity)
    if ids | reviewed != expected or len(expected) != prepared['original_human_programs']:
        raise ValueError('Every original publisher human program must be accounted for exactly once')
    if any(len(roles) != 1 for groups in [observed_groups, observed_urls, observed_questions] for roles in groups.values()):
        raise ValueError('Original problem, URL or normalized question crossed internal roles')
    duplicates = [dict(normalized_code_sha256=h, train_ids=[i for i, r in values if r == 'train'],
        validation_ids=[i for i, r in values if r == 'validation']) for h, values in code_rows.items()
        if {r for _, r in values} == {'train', 'validation'}]
    atomic_json(PACK / 'exact-normalized-duplicate-review.json', dict(internal_train_validation=duplicates, named_held_source_matches=held_matches))
    bound = [Path(__file__).resolve(), PACK / 'manifest.json', PACK / 'candidate-preparation.json',
        PACK / 'original-review-index.jsonl', PACK / 'exact-normalized-duplicate-review.json',
        SOURCE / 'inventory.json', SOURCE / 'train-problems.json', mbpp, codenet,
        ROOT / 'src/slm_lab/code_tasks.py', ROOT / 'src/slm_lab/verified_code.py']
    result = dict(completed_at=datetime.now(timezone.utc).isoformat(), all_actual_complete_pack_tokens_masks_and_source_rows_reencoded=True,
        original_human_programs=len(expected), complete_programs_by_role=dict(count), complete_targets_by_role=dict(tokens),
        unique_reviewed_originals=len(reviewed), all_originals_accounted_for_exactly_once=True,
        original_problem_URL_question_components_crossing_roles=0, publisher_test_rows_encoded=0,
        exact_normalized_code_groups_crossing_internal_roles=len(duplicates),
        named_held_source_matching_rows=len(held_matches), named_held_source_matching_train_rows=sum(r['role'] == 'train' for r in held_matches),
        held_screen_scope=['All retained sanitized-MBPP test references in original IDs11..510, including currently unsupported references.',
            'All12local Python task reference programs.', 'All Python3-parseable original CodeNet validation/test programs.'],
        source_execution_verified=False, semantic_near_duplicate_independence_certified=False,
        exact_duplicate_normalization='Python AST fields, iterative length-prefixedSHA256, docstrings removed; identifiers and literals retained',
        source_admitted_to_training=False, model_updates=0, goal_achieved=False,
        files_sha256={str(p): file_hash(p) for p in bound},
        next_required='Retain exact duplicate/held-match reviews and create an explicit uncontaminated training view before admission; validate finite original publisher tests separately.')
    atomic_json(target, result); print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__': main()
