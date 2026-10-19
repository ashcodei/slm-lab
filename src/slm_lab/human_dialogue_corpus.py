"""Reviewed human-only OpenAssistant dialogues, preserving official held-out trees."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path

import ahocorasick
import pyarrow.parquet as pq
from tokenizers import Tokenizer

from .corpus import file_hash
from .diverse_images import atomic_json, jsonl
from .model import ModelConfig
from .training import Encoder

REVISION = '179dd21fc55192153d94adb0e0ce8f69e222bf75'
TREE_FILE = '2023-11-05_oasst2_ready.trees.jsonl.gz'
TREE_SHA256 = '7a886a16ccfc1173c4f00a6897523e3c95b2785a86ee44a18a98f4f2807ee29b'
VALID_FILE = 'validation-00000-of-00001-1deeef95c3248fe0.parquet'
VALID_SHA256 = 'fd46c63babc0756909a957ee40975bcf5610d58c5d1ab5ff2c57c0f57be1f273'


def normalized(text):
    return ' '.join(text.casefold().split())


def message_issue(node):
    if node.get('synthetic') is not False or node.get('model_name'):
        return 'synthetic/model/unknown authorship'
    if node.get('deleted') is not False or node.get('review_result') is not True:
        return 'deleted/unreviewed/rejected message'
    if node.get('role') not in {'assistant', 'prompter'} or not isinstance(node.get('text'), str) or not node['text'].strip():
        return 'invalid role or empty text'
    if len(node['text']) > 100000:
        return 'oversized message'
    labels = node.get('labels') or {}
    for name in ['spam', 'pii', 'lang_mismatch', 'fails_task']:
        label = labels.get(name, {})
        if label.get('count', 0) > 0 and label.get('value', 0) > 0:
            return 'positive human exclusion label: ' + name
    if node['role'] == 'assistant':
        quality = labels.get('quality', {})
        if quality.get('count', 0) < 1 or quality.get('value', 0) < .5:
            return 'missing/low human quality rating'
    return None


def examples(tree):
    """Keep best reviewed human reply paths; a bad ancestor invalidates its descendants."""
    rejected = []
    seen = set()

    def walk(node, history, expected_role):
        identifier = node.get('message_id')
        if not isinstance(identifier, str) or identifier in seen:
            raise ValueError('Duplicate/missing message ID in a conversation tree')
        seen.add(identifier)
        if len(history) > 32:
            rejected.append({'message_id': identifier, 'reason': 'conversation depth limit'}); return
        issue = message_issue(node)
        if node.get('role') != expected_role:
            issue = 'non-alternating message roles'
        if issue:
            rejected.append({'message_id': identifier, 'reason': issue}); return
        if node['role'] == 'assistant':
            if not history or history[-1]['role'] != 'prompter':
                raise ValueError('Assistant target has no user request')
            prompt = '\n'.join(('User: ' if row['role'] == 'prompter' else 'Assistant: ') + row['text'] for row in history) + '\nAssistant:'
            yield {'id': 'oasst2-' + identifier, 'group_id': 'oasst2:' + tree['message_tree_id'],
                   'prompt': prompt, 'response': node['text'], 'language': node.get('lang'),
                   'conversation_turns': len(history), 'source': 'OpenAssistant/oasst2 original human conversation export',
                   'source_url': 'https://huggingface.co/datasets/OpenAssistant/oasst2/tree/' + REVISION,
                   'source_revision': REVISION, 'source_message_id': identifier,
                   'source_message_tree_id': tree['message_tree_id'], 'license': 'Apache-2.0',
                   'response_origin': 'Platform-marked human message: synthetic=false and no model_name',
                   'human_review_count': node.get('review_count'), 'human_quality': node['labels']['quality'],
                   'human_rank': node.get('rank'), 'synthetic': False,
                   'source_text_sha256': hashlib.sha256(node['text'].encode()).hexdigest()}
        history = history + [node]
        replies = node.get('replies', [])
        if node['role'] == 'prompter':
            approved = [child for child in replies if child.get('role') == 'assistant' and message_issue(child) is None]
            selected = [child for child in approved if child.get('rank') == 0
                        or (child.get('rank') is None and len(approved) == 1)]
            for child in replies:
                if child not in selected:
                    rejected.append({'message_id': child.get('message_id'),
                                     'reason': message_issue(child) or 'not a top-ranked/single reviewed reply'})
            replies = selected
        for child in replies:
            yield from walk(child, history, 'assistant' if expected_role == 'prompter' else 'prompter')

    if tree.get('tree_state') != 'ready_for_export' or tree['prompt'].get('message_id') != tree['message_tree_id']:
        raise ValueError('Expected original ready-for-export rooted message trees')
    return list(walk(tree['prompt'], [], 'prompter')), rejected


def prepare(source, output, tokenizer, questions, prior_validation, math_questions, public_code_questions):
    source, output = map(Path, [source, output]); output.mkdir(parents=True, exist_ok=True)
    if file_hash(source / TREE_FILE) != TREE_SHA256 or file_hash(source / VALID_FILE) != VALID_SHA256:
        raise ValueError('OpenAssistant publisher LFS hashes differ')
    readme = (source / 'README.md').read_text()
    if 'license: apache-2.0' not in readme:
        raise ValueError('Publisher data license declaration missing')
    protection = []
    protected_prompts = set()
    for path in prior_validation:
        protection.append({'path': str(Path(path).resolve()), 'sha256': file_hash(path)})
        for row in jsonl(path):
            text = row.get('prompt', '')
            if text: protected_prompts.add(normalized(text))
    matcher = ahocorasick.Automaton()
    for path in [questions, math_questions, public_code_questions]:
        protection.append({'path': str(Path(path).resolve()), 'sha256': file_hash(path)})
        for row in jsonl(path):
            text = row.get('question') or row.get('text') or row.get('prompt')
            if text: matcher.add_word(normalized(text), row.get('id', 'reserved-assessment'))
    matcher.make_automaton()
    table = pq.read_table(source / VALID_FILE, columns=['message_tree_id'])
    validation_ids = set(table.column('message_tree_id').to_pylist())
    if None in validation_ids:
        raise ValueError('Official validation table has unknown tree IDs')
    trees = []
    total_uncompressed = 0
    with gzip.open(source / TREE_FILE, 'rt') as stream:
        for line in stream:
            total_uncompressed += len(line.encode())
            if total_uncompressed > 500000000 or len(line) > 10000000:
                raise ValueError('Conversation export exceeds decoding budget')
            trees.append(json.loads(line))
    ids = {tree['message_tree_id'] for tree in trees}
    if len(ids) != len(trees) or not validation_ids <= ids:
        raise ValueError('Official validation roots do not match the frozen tree export')
    official_prompts = {normalized(tree['prompt']['text']) for tree in trees if tree['message_tree_id'] in validation_ids}
    config = {'version': 1, 'source_revision': REVISION, 'tree_sha256': TREE_SHA256,
              'official_validation_sha256': VALID_SHA256, 'readme_sha256': file_hash(source / 'README.md'),
              'tokenizer_sha256': file_hash(tokenizer), 'protected_sources': protection,
              'human_only': True, 'reviewed_only': True, 'minimum_quality': .5,
              'best_rank_or_single_reviewed_reply': True, 'no_truncation': True,
              'implementation_sha256': file_hash(__file__)}
    path = output / 'manifest.json'
    if path.exists():
        manifest = json.loads(path.read_text())
        if manifest['configuration'] != config:
            raise ValueError('Use a new human dialogue corpus directory')
        for name, digest in manifest['files_sha256'].items():
            if file_hash(output / name) != digest: raise ValueError('Frozen human dialogue corpus changed')
        return manifest
    encoder = Encoder(Tokenizer.from_file(str(tokenizer)), ModelConfig(), output, 448)
    rows = {'train': [], 'valid': []}; rejected = []; duplicate_answers = set()
    # Official validation takes precedence when source conversations repeat.
    trees.sort(key=lambda tree: (tree['message_tree_id'] not in validation_ids, tree['message_tree_id']))
    for tree in trees:
        identifier = tree['message_tree_id']; root_prompt = normalized(tree['prompt']['text'])
        split = 'valid' if identifier in validation_ids else 'train'
        if split == 'train' and (root_prompt in official_prompts or root_prompt in protected_prompts):
            rejected.append({'tree_id': identifier, 'reason': 'protected official/prior validation prompt'}); continue
        selected, issues = examples(tree)
        rejected.extend({'tree_id': identifier, **issue} for issue in issues)
        if any(list(matcher.iter(normalized(row['prompt'] + ' ' + row['response']))) for row in selected):
            rejected.append({'tree_id': identifier, 'reason': 'reserved knowledge/math/public-code assessment text'}); continue
        for row in selected:
            key = (normalized(row['prompt']), normalized(row['response']))
            if key in duplicate_answers:
                rejected.append({'tree_id': identifier, 'message_id': row['source_message_id'], 'reason': 'duplicate prompt/response'}); continue
            previous = encoder.truncated
            _, labels = encoder.sequence(row, 'response')
            if encoder.truncated != previous:
                rejected.append({'tree_id': identifier, 'message_id': row['source_message_id'], 'reason': 'would truncate conversation/response'}); continue
            duplicate_answers.add(key); rows[split].append({**row, 'split': split,
                                  'target_tokens': sum(value != -100 for value in labels[1:])})
    # Protect complete conversation prefixes, not just source root IDs.
    valid_prompts = {normalized(row['prompt']) for row in rows['valid']}
    rows['train'] = [row for row in rows['train'] if normalized(row['prompt']) not in valid_prompts]
    if {row['group_id'] for row in rows['train']} & {row['group_id'] for row in rows['valid']}:
        raise ValueError('Human conversation trees overlap')
    for split, records in rows.items():
        (output / (split + '.jsonl')).write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in records))
    (output / 'excluded.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rejected))
    (output / 'publisher-README.md').write_text(readme)
    manifest = {'configuration': config, 'source_trees': len(trees), 'official_validation_trees': len(validation_ids),
                'splits': {split: {'examples': len(records), 'trees': len({row['group_id'] for row in records}),
                                   'target_tokens': sum(row['target_tokens'] for row in records),
                                   'languages': dict(Counter(row['language'] for row in records))} for split, records in rows.items()},
                'excluded': dict(Counter(row['reason'] for row in rejected)), 'tree_overlap': 0, 'prompt_overlap': 0,
                'external_model_weights_used': False, 'synthetic_responses_admitted': 0,
                'files_sha256': {name: file_hash(output / name) for name in ['train.jsonl', 'valid.jsonl', 'excluded.jsonl', 'publisher-README.md']},
                'limitations': ['Publisher flags/reviews are the authorship evidence; volunteer claims and factual accuracy are not independently verified.',
                                'Human quality ratings are not code execution/arithmetic/medical correctness proofs.',
                                'Filtered branches, rare languages and overlength conversations limit coverage.',
                                'Exact normalized text protection does not exclude semantic benchmark overlap or certify all earlier pretraining.']}
    atomic_json(path, manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', default='data/openassistant-source-v1'); parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = prepare(args.source, args.output, Path('data/tokenizer-corpus/tokenizer.json'),
                     Path('data/knowledge-eval-v1/questions.jsonl'),
                     [Path('data/alignment-curriculum-v1/instruction.valid.jsonl'), Path('data/human-preference-curriculum-v1/preferences.valid.jsonl')],
                     Path('data/math-reasoning-v2/test.jsonl'), Path('data/public-code-assessment-v1/questions.jsonl'))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
