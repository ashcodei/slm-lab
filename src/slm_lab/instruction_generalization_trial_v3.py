"""Preserve complete public instruction targets and explicit language exposure."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import re
from tokenizers import Tokenizer

from .code_tasks import TASKS
from .corpus import file_hash
from .diverse_images import atomic_json, jsonl
from .exercise_probe import validate_splits
from .training import Encoder
from .wikipedia_corpus import protected_questions

LANGUAGES = ('Python', 'JavaScript', 'Go', 'Java', 'Rust', 'C++')


class TargetDoesNotFit(ValueError):
    pass


def complete_encoding(row, encoder):
    prompt = encoder.tokens(encoder.prompt(row)); response = encoder.tokens(row['response'])
    if len(response) > encoder.config.max_text_length // 2 - 1 or len(prompt) + len(response) + 2 > encoder.config.max_text_length:
        raise TargetDoesNotFit('Original complete prompt and response do not fit; never train a clipped target')
    ids, labels = encoder.sequence(row, 'response')
    if (ids != [encoder.bos] + prompt + response + [encoder.eos]
            or labels != [-100] * (len(prompt) + 1) + response + [encoder.eos]):
        raise ValueError('Actual complete response mask differs from the original source')
    return {'context_tokens': len(ids), 'target_tokens': len(response) + 1}


def order(size, seed):
    if size < 1: raise ValueError('A scheduled source pool is empty')
    result = list(range(size)); random.Random(seed).shuffle(result); return result


def schedule(pools, steps, seed):
    if type(steps) is not int or steps < 1: raise ValueError('Positive complete update budget required')
    required = [*LANGUAGES, 'verified_python', 'human_instruction']
    if set(pools) != set(required): raise ValueError('Every declared language and public instruction pool required')
    orders = {name: order(len(pools[name]), seed + i) for i, name in enumerate(required)}
    verified_groups = defaultdict(list)
    for index, row in enumerate(pools['verified_python']): verified_groups[row['group_id']].append(index)
    group_names = list(sorted(verified_groups)); random.Random(seed + 71).shuffle(group_names)
    # Visit each original algorithm before adding another name variant.
    orders['verified_python'] = [indices[variant] for variant in range(max(map(len, verified_groups.values())))
        for group in group_names if variant < len(indices := verified_groups[group])]
    positions = Counter(); groups = []
    for step in range(steps):
        names = [LANGUAGES[(2 * step) % 6], LANGUAGES[(2 * step + 1) % 6],
                 'verified_python', 'human_instruction']
        group = []
        for name in names:
            position = positions[name]; index = orders[name][position % len(orders[name])]
            group.append({'pool': name, 'index': index}); positions[name] += 1
        groups.append(group)
    return groups


def prepare(root, output, tokenizer, config, steps=1500, seed=88123):
    root, output = Path(root), Path(output)
    if output.exists(): raise ValueError('Preserve the original instruction trial source preparation')
    output.mkdir()
    exercises = root / 'data/exercism-support-context-v3'; verified = root / 'data/verified-code-curriculum-v1'
    human = root / 'data/alignment-curriculum-v1'
    exercise_manifest = json.loads((exercises / 'manifest.json').read_text())
    verified_manifest = json.loads((verified / 'manifest.json').read_text())
    human_manifest = json.loads((human / 'manifest.json').read_text())
    for directory, manifest in [(exercises, exercise_manifest), (human, human_manifest)]:
        for name in ['train.jsonl', 'valid.jsonl'] if directory == exercises else ['instruction.train.jsonl']:
            if file_hash(directory / name) != manifest['files_sha256'][name]: raise ValueError('Original admitted instruction source changed')
    if file_hash(verified / 'train.jsonl') != verified_manifest['train_sha256']:
        raise ValueError('Original published-example verified programs changed')
    token_hash = file_hash(tokenizer)
    if token_hash != verified_manifest['source_inputs']['tokenizer_sha256']:
        raise ValueError('Own tokenizer differs from original verified instruction source')
    exercise_train = list(jsonl(exercises / 'train.jsonl')); exercise_valid = list(jsonl(exercises / 'valid.jsonl'))
    validate_splits(exercise_train, exercise_valid)
    encoder = Encoder(Tokenizer.from_file(str(tokenizer)), config, exercises, 448)
    matcher = protected_questions(root / 'data/knowledge-eval-v1/questions.jsonl')
    normalize = lambda text: ' '.join(text.casefold().split())
    reserved_programs = [normalize(row['response']) for row in exercise_valid if len(normalize(row['response'])) >= 80]
    reserved_programs += [normalize(code) for task in TASKS for code in task[-2:] if len(normalize(code)) >= 80]
    reserved_specs = [normalize(task[2]) for task in TASKS]
    reserved_names = {re.sub(r'[^a-z0-9]', '', row['exercise'].casefold()) for row in exercise_valid}
    reserved_names |= {re.sub(r'[^a-z0-9]', '', task[0].casefold()) for task in TASKS}
    verified_rows = list(jsonl(verified / 'train.jsonl'))
    forbidden_groups = {row['group_id'] for row in verified_rows
        if row['variant_index'] == 0 and re.sub(r'[^a-z0-9]', '', row['function'].casefold()) in reserved_names}
    pools = {name: [] for name in [*LANGUAGES, 'verified_python', 'human_instruction']}
    excluded = Counter(); records = []
    sources = [(row['language'], row) for row in exercise_train]
    sources += [('verified_python', row) for row in verified_rows]
    sources += [('human_instruction', row) for row in jsonl(human / 'instruction.train.jsonl')
                if row.get('source') == 'HuggingFaceH4/no_robots']
    for pool, row in sources:
        if row.get('image'): raise ValueError('This trial requires original text-only source rows')
        text = normalize(row['prompt'] + '\n' + row['response'])
        if row.get('group_id') in forbidden_groups: excluded['reserved_original_function_group'] += 1; continue
        if list(matcher.iter(text)): excluded['reserved_exact_knowledge_question'] += 1; continue
        if any(program in text for program in reserved_programs): excluded['reserved_exact_program'] += 1; continue
        if any(spec in text for spec in reserved_specs): excluded['reserved_exact_local_specification'] += 1; continue
        try: counts = complete_encoding(row, encoder)
        except TargetDoesNotFit: excluded['complete_original_target_does_not_fit'] += 1; continue
        pools[pool].append(row)
        records.append({'pool': pool, 'id': row['id'], 'group_id': row.get('group_id'),
                        'original_row_sha256': hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest(), **counts})
    if encoder.truncated: raise ValueError('Accepted original training target was clipped')
    allocation = schedule(pools, steps, seed)
    count_lookup = {(row['pool'], row['id']): row['target_tokens'] for row in records}
    exposures = []; targets = 0; language_counts = Counter()
    for step, group in enumerate(allocation):
        for pick in group:
            row = pools[pick['pool']][pick['index']]; count = count_lookup[pick['pool'], row['id']]
            exposures.append({'update': step + 1, **pick, 'id': row['id'], 'group_id': row.get('group_id'),
                              'language': row.get('language', 'human_instruction'), 'target_tokens': count})
            targets += count; language_counts[pick['pool']] += 1
    for name, rows in pools.items():
        with (output / (name + '.jsonl')).open('w') as stream:
            for row in rows: stream.write(json.dumps(row, ensure_ascii=False) + '\n')
    with (output / 'exposures.jsonl').open('w') as stream:
        for row in exposures: stream.write(json.dumps(row) + '\n')
    input_files = [exercises / 'manifest.json', exercises / 'train.jsonl', exercises / 'valid.jsonl',
        verified / 'manifest.json', verified / 'train.jsonl', human / 'manifest.json', human / 'instruction.train.jsonl',
        root / 'data/knowledge-eval-v1/questions.jsonl', tokenizer]
    receipt = {'source_inputs_sha256': {str(p.resolve()): file_hash(p) for p in input_files},
        'tokenizer_sha256': token_hash, 'steps': steps, 'microbatch_size': 1, 'gradient_accumulation': 4,
        'source_pool_examples': {name: len(rows) for name, rows in pools.items()},
        'sample_exposures': len(exposures), 'source_pool_exposures': dict(language_counts),
        'planned_actual_response_targets': targets, 'maximum_complete_context': max(row['context_tokens'] for row in records),
        'excluded_original_rows': dict(excluded), 'accepted_source_generation_rows_unchanged': True,
        'all_accepted_targets_and_masks_complete': True, 'source_records': records,
        'files_sha256': {p.name: file_hash(p) for p in output.glob('*.jsonl')},
        'scope': 'Six-language publisher training modules50%, published-example verified Python25%, original human demonstrations25% by example exposure. The source generation supplies12 original supporting-file contexts, relocates five static imports in one publisher target, and excludes three inspected JavaScript rows; other targets remain original. Native validation modules and known exact names/programs/specifications excluded. Source checks do not prove all training program correctness; semantic/prior exposure remains unaudited. No external model teacher.',
        'goal_achieved': False, 'trained': False}
    atomic_json(output / 'manifest.json', receipt)
    return pools, allocation, receipt
