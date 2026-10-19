"""Pinned original GSM8K solutions, with bounded arithmetic annotation checks."""
import argparse
import ast
from collections import Counter
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import re
import unicodedata

from .corpus import file_hash
from .diverse_images import asset, atomic_json, jsonl

REVISION = '3101c7d5072418e28b9008a6636bde82a006892c'
REPOSITORY = 'https://github.com/openai/grade-school-math'
SOURCE_FILES = {
    'train.jsonl': ('grade_school_math/data/train.jsonl', '17f347dc51477c50d4efb83959dbb7c56297aba886e5544ee2aaed3024813465'),
    'test.jsonl': ('grade_school_math/data/test.jsonl', '3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14'),
    'LICENSE': ('LICENSE', '86bbb73e855821d7c401912fd4bf82e34313e6e3b6fd6f909f2b6cc9e209a53b'),
    'README.md': ('README.md', '27e1676d572f2289210170ff6dbb8c2ef7d9f6950c0f08c978ea8ee6b39dca29'),
}
NUMBER = re.compile(r'[+-]?(?:(?:[0-9]{1,3}(?:,[0-9]{3})+|[0-9]+)(?:\.[0-9]*)?|\.[0-9]+)\Z')
CALCULATION = re.compile(r'<<([^<>]+)>>')


def normalized_question(text):
    return re.sub(r'\W+', ' ', unicodedata.normalize('NFKC', text).casefold()).strip()


def numeric(text):
    text = text.strip()
    if len(text) > 64 or not NUMBER.fullmatch(text):
        raise ValueError('Expected a bounded decimal number with optional grouped commas')
    value = Fraction(text.replace(',', ''))
    if max(value.numerator.bit_length(), value.denominator.bit_length()) > 256:
        raise ValueError('Numeric value exceeds limit')
    return value


def final_answer(text):
    """Only the last complete #### line counts; prose/expressions are not evaluated."""
    lines = re.findall(r'^\s*####[ \t]*([^\r\n]*)$', text, re.M)
    if not lines:
        return None
    try:
        return numeric(lines[-1])
    except ValueError:
        return None


def arithmetic(expression):
    """Interpret a small numeric AST as rational arithmetic, never Python eval."""
    if len(expression) > 192:
        raise ValueError('Arithmetic expression exceeds limit')
    try:
        tree = ast.parse(expression.strip(), mode='eval')
    except (SyntaxError, RecursionError) as error:
        raise ValueError('Invalid arithmetic syntax') from error
    if sum(1 for _ in ast.walk(tree)) > 80:
        raise ValueError('Arithmetic AST exceeds limit')
    expression = expression.strip()

    def visit(node, depth=0):
        if depth > 16:
            raise ValueError('Arithmetic nesting exceeds limit')
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            value = numeric(ast.get_source_segment(expression, node))
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = visit(node.operand, depth+1)
            if isinstance(node.op, ast.USub):
                value = -value
        elif isinstance(node, ast.BinOp):
            left, right = visit(node.left, depth+1), visit(node.right, depth+1)
            if isinstance(node.op, ast.Add): value = left + right
            elif isinstance(node.op, ast.Sub): value = left - right
            elif isinstance(node.op, ast.Mult): value = left * right
            elif isinstance(node.op, ast.Div): value = left / right
            elif isinstance(node.op, ast.FloorDiv): value = Fraction(left // right)
            elif isinstance(node.op, ast.Mod): value = left % right
            elif isinstance(node.op, ast.Pow):
                if right.denominator != 1 or not 0 <= right <= 12:
                    raise ValueError('Power exceeds limit')
                if max(left.numerator.bit_length(), left.denominator.bit_length()) * int(right) > 256:
                    raise ValueError('Power result exceeds limit')
                value = left ** int(right)
            else:
                raise ValueError('Unsupported arithmetic operator')
        else:
            raise ValueError('Only numeric constants and bounded arithmetic operators are allowed')
        if max(value.numerator.bit_length(), value.denominator.bit_length()) > 256:
            raise ValueError('Arithmetic result exceeds limit')
        return value

    try:
        return visit(tree.body)
    except (ZeroDivisionError, OverflowError) as error:
        raise ValueError('Undefined arithmetic') from error


def checked_calculations(answer):
    annotations = CALCULATION.findall(answer)
    if not annotations:
        raise ValueError('No published arithmetic annotations')
    for annotation in annotations:
        if annotation.count('=') != 1:
            raise ValueError('Unsupported arithmetic annotation')
        expression, expected = annotation.split('=')
        expected = expected.strip()
        target = numeric(expected)
        # Published finite decimal results may round rational divisions. Integer
        # results require exact equality; decimals allow half their displayed unit.
        decimals = len(expected.rsplit('.', 1)[1]) if '.' in expected else 0
        tolerance = Fraction(1, 2*10**decimals) if decimals else Fraction(0)
        if abs(arithmetic(expression) - target) > tolerance:
            raise ValueError('Published arithmetic annotation does not match its result')
    return len(annotations)


def prompt(question):
    return 'Solve the following math problem. Show your work, then finish with a separate line "#### NUMBER" containing only the final numeric answer.\n\n' + question


def assemble(train, test, reserved=()):
    protected = {normalized_question(row['question']) for row in test}
    protected.update(normalized_question(text) for text in reserved)
    splits = {'train': [], 'valid': [], 'test': []}; excluded = []; seen = set()
    for index, row in enumerate(test):
        gold = final_answer(row['answer'])
        if gold is None:
            raise ValueError('Official test label is not a supported numeric answer')
        splits['test'].append({'id': f'gsm8k:test:{index}', 'question': row['question'],
                              'answer_numerator': gold.numerator, 'answer_denominator': gold.denominator,
                              'source_row': index+1, 'source_split': 'test'})
    for index, row in enumerate(train):
        group = normalized_question(row['question'])
        try:
            if group in protected: raise ValueError('Question matches protected test/reserved question')
            if group in seen: raise ValueError('Duplicate normalized training question')
            if final_answer(row['answer']) is None: raise ValueError('Unsupported final numeric label')
            checked = checked_calculations(row['answer'])
            digest = hashlib.sha256(group.encode()).hexdigest()
            split = 'valid' if int(digest[:8], 16) % 20 == 0 else 'train'
            splits[split].append({'id': f'gsm8k:train:{index}', 'group_id': 'gsm8k:'+digest,
                                 'prompt': prompt(row['question']),
                                 'response': CALCULATION.sub('', row['answer']),
                                 'source_row': index+1, 'source_split': 'train',
                                 'source_example_sha256': hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest(),
                                 'published_calculations_checked': checked,
                                 'domain': 'math_reasoning', 'source': 'GSM8K original human solutions',
                                 'annotation_origin': 'Original human-written solution; calculator annotations removed.',
                                 'feedback_source': 'human', 'license': 'MIT'})
            seen.add(group)
        except ValueError as error:
            excluded.append({'source_row': index+1, 'reason': str(error)})
    return splits, excluded


def prepare(cache, output, questions=None):
    cache, output = Path(cache), Path(output); output.mkdir(parents=True, exist_ok=True)
    sources = []
    for name, (path, digest) in SOURCE_FILES.items():
        url = f'https://raw.githubusercontent.com/openai/grade-school-math/{REVISION}/{path}'
        downloaded, record = asset(cache, name, url, 20_000_000)
        if file_hash(downloaded) != digest:
            raise ValueError('Pinned original math source hash mismatch: '+name)
        sources.append({'path': name, **record})
    config = {'revision': REVISION, 'reserved_questions_sha256': file_hash(questions) if questions else None,
              'split_rule': 'Original test protected; normalized training-question SHA256 modulo 20 reserves one bucket for validation.',
              'preparation_version': 2}
    manifest_path = output/'manifest.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest['configuration'] != config: raise ValueError('Use a new directory for a different math curriculum')
        for name, digest in manifest['files_sha256'].items():
            if file_hash(output/name) != digest: raise ValueError('Frozen math curriculum changed')
        return manifest
    train, test = list(jsonl(cache/'train.jsonl')), list(jsonl(cache/'test.jsonl'))
    if len(train) != 7473 or len(test) != 1319: raise ValueError('Original dataset row counts changed')
    reserved = [r['question'] for r in jsonl(questions)] if questions else []
    splits, excluded = assemble(train, test, reserved)
    for split, rows in splits.items():
        for row in rows:
            row['source_url'] = next(s['url'] for s in sources if s['path'] == ('test.jsonl' if split == 'test' else 'train.jsonl'))
            row['source_revision'] = REVISION
            row['license'] = 'MIT'; row['license_file'] = str((output/'LICENSE').resolve())
        path = output/(split+'.jsonl'); temporary = path.with_suffix('.tmp')
        temporary.write_text(''.join(json.dumps(row, ensure_ascii=True)+'\n' for row in rows)); temporary.replace(path)
    (output/'excluded.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in excluded))
    (output/'LICENSE').write_bytes((cache/'LICENSE').read_bytes())
    manifest = {'configuration': config, 'repository': REPOSITORY, 'sources': sources,
                'official_train_examples': len(train), 'official_test_examples': len(test),
                'splits': {name: {'examples': len(rows)} for name, rows in splits.items()},
                'excluded_training_examples': len(excluded),
                'exclusion_reasons': dict(Counter(row['reason'] for row in excluded)),
                'published_calculations_checked': sum(row['published_calculations_checked'] for name in ['train', 'valid'] for row in splits[name]),
                'files_sha256': {name: file_hash(output/name) for name in ['train.jsonl', 'valid.jsonl', 'test.jsonl', 'excluded.jsonl', 'LICENSE']},
                'limitations': ['Arithmetic annotations are checked; whole solution correctness is not independently proven.',
                               'Original test questions are excluded from this supplement; prior complete pretraining corpora are not contamination-certified.',
                               'Elementary English word problems do not cover advanced mathematics or other science domains.',
                               'Uses original human solutions, not Socratic/model-generated variants or pretrained weights.']}
    atomic_json(manifest_path, manifest); return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', required=True); parser.add_argument('--output', required=True)
    parser.add_argument('--reserved-questions'); args = parser.parse_args()
    print(json.dumps(prepare(args.cache, args.output, args.reserved_questions), indent=2))


if __name__ == '__main__': main()
