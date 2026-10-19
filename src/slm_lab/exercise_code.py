"""Pinned original Exercism tasks, interfaces, reference code and test artifacts."""
import argparse
import ast
from collections import Counter, defaultdict
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import zipfile

from tokenizers import Tokenizer
from .code_quality import execute, instruction_code
from .code_tasks import TASKS
from .corpus import file_hash
from .diverse_images import atomic_json, jsonl
from .model import ModelConfig
from .training import Encoder
from .verified_code import literal
from .wikipedia_corpus import protected_questions

ROOT = Path(__file__).resolve().parents[2]
LANGUAGES = {'python': 'Python', 'javascript': 'JavaScript', 'go': 'Go',
             'java': 'Java', 'rust': 'Rust', 'cpp': 'C++'}


def fingerprint_functions(code):
    """Exact AST structure after local identifier/docstring/type-hint normalization."""
    try: tree = ast.parse(code)
    except SyntaxError: return set()
    result = set()
    for source in tree.body:
        if not isinstance(source, ast.FunctionDef): continue
        node = copy.deepcopy(source)
        if node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant) and isinstance(node.body[0].value.value, str):
            node.body.pop(0)
        bound = {node.name: '__entry'}
        for item in ast.walk(node):
            name = item.arg if isinstance(item, ast.arg) else item.id if isinstance(item, ast.Name) and isinstance(item.ctx, ast.Store) else None
            if name is not None and name not in bound: bound[name] = '__v' + str(len(bound))
        class Normalize(ast.NodeTransformer):
            def visit_FunctionDef(self, item):
                item.name = bound.get(item.name, item.name); item.returns = None; item.type_comment = None
                return self.generic_visit(item)
            def visit_arg(self, item):
                item.arg = bound[item.arg]; item.annotation = None; item.type_comment = None
                return item
            def visit_Name(self, item):
                item.id = bound.get(item.id, item.id); return item
        normalized = ast.dump(Normalize().visit(node), include_attributes=False)
        result.add(hashlib.sha256(normalized.encode()).hexdigest())
    return result


def literal_tests(text):
    """Read literal function-call assertions without importing the published test."""
    groups = defaultdict(list)
    try: tree = ast.parse(text)
    except SyntaxError: return {}
    for node in ast.walk(tree):
        if (not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute)
                or not isinstance(node.func.value, ast.Name) or node.func.value.id != 'self'):
            continue
        mode = node.func.attr
        if mode not in {'assertEqual', 'assertTrue', 'assertFalse'} or len(node.args) != (2 if mode == 'assertEqual' else 1): continue
        call = node.args[0]
        if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name) or call.keywords: continue
        try:
            arguments = tuple(ast.literal_eval(item) for item in call.args)
            expected = ast.literal_eval(node.args[1]) if mode == 'assertEqual' else mode == 'assertTrue'
            case = {'args_literal': literal(arguments), 'expected_literal': literal(expected)}
        except (ValueError, TypeError, SyntaxError): continue
        if case not in groups[call.func.id]: groups[call.func.id].append(case)
    return dict(groups)


def exercise_split(slug):
    return 'valid' if int(hashlib.sha256(('exercism:' + slug).encode()).hexdigest()[:8], 16) % 10 < 2 else 'train'


def target_for(example, solutions):
    exact = [name for name in solutions if PurePosixPath(name).name == PurePosixPath(example).name]
    if len(exact) == 1: return exact[0]
    same_type = [name for name in solutions if PurePosixPath(name).suffix == PurePosixPath(example).suffix]
    if len(same_type) != 1: raise ValueError('Ambiguous reference-to-solution file mapping')
    return same_type[0]


class Repository:
    def __init__(self, folder):
        self.record = json.loads((folder / 'source.json').read_text())
        if file_hash(folder / 'source.zip') != self.record['archive_sha256'] or file_hash(folder / 'LICENSE') != self.record['license_sha256']:
            raise ValueError('Pinned source archive or license changed')
        self.archive = zipfile.ZipFile(folder / 'source.zip')
        self.prefix = self.archive.namelist()[0].split('/')[0] + '/'
        self.names = {name[len(self.prefix):] for name in self.archive.namelist() if name.startswith(self.prefix)}

    def read(self, path):
        value = PurePosixPath(path)
        if value.is_absolute() or '..' in value.parts: raise ValueError('Unsafe source path')
        if path not in self.names: raise ValueError('Published source file missing: ' + path)
        item = self.archive.getinfo(self.prefix + path)
        if item.file_size > 1_000_000: raise ValueError('Source file exceeds bounded read size')
        return self.archive.read(item).decode('utf-8')

    def evidence(self, path, text):
        return {'path': path, 'sha256': hashlib.sha256(text.encode()).hexdigest(),
                'url': f'https://github.com/{self.record["repository"]}/blob/{self.record["revision"]}/{path}'}


def prepare(output, cache, tokenizer):
    output, cache = Path(output), Path(cache)
    if output.exists(): raise ValueError('Use a fresh exercise corpus; do not replace frozen data')
    output.mkdir(parents=True); (output / 'tests').mkdir()
    questions = ROOT / 'data/knowledge-eval-v1/questions.jsonl'
    matcher = protected_questions(questions)
    protected = set(); known_train = set()
    for task in TASKS: protected.update(fingerprint_functions(task[4]))
    for row in jsonl(ROOT / 'data/verified-code-source-cache-v1/mbpp.jsonl'):
        if row['task_id'] <= 600: protected.update(fingerprint_functions(row['code']))
    for row in jsonl(ROOT / 'data/verified-code-curriculum-v1/train.jsonl'):
        if row['variant_index'] == 0: known_train.update(fingerprint_functions(instruction_code(row['response'])))
    specs = Repository(cache / 'problem-specifications')
    spec_record = specs.record
    config = ModelConfig(**json.loads((ROOT / 'configs/model.json').read_text()))
    tokenizer = Path(tokenizer)
    encoder = Encoder(Tokenizer.from_file(str(tokenizer)), config, output)
    proposals = []; blocked = {}; excluded = Counter(); sources = [spec_record]
    for repo, language in LANGUAGES.items():
        source = Repository(cache / repo); sources.append(source.record)
        metadata_paths = sorted(name for name in source.names if re.fullmatch(r'exercises/(practice|concept)/[a-z0-9-]+/\.meta/config\.json', name))
        for metadata_path in metadata_paths:
            slug = metadata_path.split('/')[2]; base = metadata_path[:-len('.meta/config.json')]
            try:
                meta_text = source.read(metadata_path); meta = json.loads(meta_text)
                files = meta['files']; examples = files.get('example', files.get('exemplar', [])); solutions = files['solution']
                if not examples: raise ValueError('No publisher reference implementation')
                doc_parts = []; evidence = [source.evidence(metadata_path, meta_text)]
                for suffix in ['introduction.md', 'introduction.append.md', 'instructions.md', 'instructions.append.md']:
                    path = base + '.docs/' + suffix
                    if path in source.names:
                        text = source.read(path); doc_parts.append(text); evidence.append(source.evidence(path, text))
                if base + '.docs/instructions.md' not in source.names: raise ValueError('No original task instructions')
                docs = '\n\n'.join(doc_parts); code_parts = []; implementation = []; targets = []
                for path in examples:
                    text = source.read(base + path); target = target_for(path, solutions)
                    if target in targets: raise ValueError('Multiple alternative references for one target')
                    targets.append(target); implementation.append(text)
                    code_parts.append(f'File: {target}\n```{language.lower() if repo != "cpp" else "cpp"}\n{text.rstrip()}\n```')
                    evidence.append(source.evidence(base + path, text))
                    if repo == 'python':
                        ast.parse(text)
                        signatures = fingerprint_functions(text)
                        if signatures & protected: blocked[slug] = 'exact structural alias of a reserved Python reference'
                        elif signatures & known_train and exercise_split(slug) == 'valid':
                            blocked[slug] = 'known previous verified-program training alias in prospective validation'
                if any(re.search(r'\b(?:todo|unimplemented)!\s*\(|raise\s+NotImplementedError', text) for text in implementation):
                    raise ValueError('Reference contains an unimplemented placeholder')
                stubs = []
                for path in solutions:
                    text = source.read(base + path)
                    stubs.append(f'File: {path}\n```\n{text.rstrip()}\n```'); evidence.append(source.evidence(base + path, text))
                tests = []; verification = []
                for path in files.get('test', []):
                    text = source.read(base + path); item = source.evidence(base + path, text)
                    item['contents'] = text; tests.append(item)
                    if repo == 'python' and len(implementation) == 1:
                        groups = literal_tests(text)
                        for function, cases in list(groups.items())[:8]:
                            result = execute({'language': 'Python', 'code': implementation[0], 'function': function, 'cases': cases, 'stress': []})
                            verification.append({'function': function, 'published_literal_cases': cases, 'result': result})
                canonical_path = f'exercises/{slug}/canonical-data.json'
                if canonical_path in specs.names:
                    canonical_text = specs.read(canonical_path); evidence.append({**specs.evidence(canonical_path, canonical_text), 'repository': spec_record['repository']})
                style = int(hashlib.sha256((repo + ':' + slug).encode()).hexdigest()[:8], 16) % 3
                prefix = [f'Implement the following {language} programming task.',
                          f'Write a complete {language} solution to this exercise.',
                          f'Complete the supplied {language} implementation to satisfy the requirements.'][style]
                prompt = prefix + '\n\n' + docs + '\n\nPublished starter interface:\n\n' + '\n\n'.join(stubs) + '\n\nReturn the implementations for the named solution files.'
                response = '\n\n'.join(code_parts)
                if list(matcher.iter(' '.join((prompt + '\n' + response).casefold().split()))):
                    blocked[slug] = 'reserved knowledge question text'
                row = {'id': f'exercism:{repo}:{slug}', 'group_id': 'exercism:' + slug,
                       'source': source.record['repository'], 'revision': source.record['revision'],
                       'source_archive_sha256': source.record['archive_sha256'], 'source_url': evidence[0]['url'],
                       'license': source.record['license'], 'license_url': source.record['license_url'],
                       'license_sha256': source.record['license_sha256'], 'authors': meta.get('authors', []),
                       'language': language, 'domain': 'software_engineering', 'task': 'public_code_exercise',
                       'exercise': slug, 'split': exercise_split(slug), 'prompt': prompt, 'response': response,
                       'source_files': evidence, 'reference_checks': verification,
                       'publisher_reference': True, 'authorship_independently_verified': False,
                       'external_model_teacher_used': False, 'test_execution_scope': 'Selected literal Python assertions in existing constrained worker; no whole repository tests or other-language execution.'}
                encoder.truncated = 0; ids, labels = encoder.sequence(row, 'response')
                if encoder.truncated: raise ValueError('Complete source interface/reference exceeds own model context or response budget')
                row['target_tokens'] = sum(value != -100 for value in labels[1:]); row['context_tokens'] = len(ids)
                if any(not item['result'].get('functional_passed') for item in verification):
                    raise ValueError('Extracted published Python cases fail or require unsupported worker features')
                proposals.append((row, tests))
            except (ValueError, SyntaxError, KeyError, UnicodeError) as error:
                excluded[str(error).split(':')[0][:140]] += 1
        source.archive.close()
        print('prepared exercise candidates', language, 'total', len(proposals), flush=True)
    specs.archive.close()
    rows = {'train': [], 'valid': []}; verification_count = 0
    for row, tests in proposals:
        if row['exercise'] in blocked:
            excluded[blocked[row['exercise']]] += 1; continue
        for item in tests:
            name = hashlib.sha256((row['id'] + ':' + item['path']).encode()).hexdigest() + '.txt'
            (output / 'tests' / name).write_text(item.pop('contents'))
            item['artifact'] = 'tests/' + name
        row['published_test_artifacts'] = tests
        verification_count += sum(len(item['published_literal_cases']) for item in row['reference_checks'])
        rows[row['split']].append(row)
    train_groups = {row['group_id'] for row in rows['train']}; valid_groups = {row['group_id'] for row in rows['valid']}
    if not train_groups or not valid_groups or train_groups & valid_groups: raise ValueError('Missing or overlapping exercise group splits')
    for split, values in rows.items():
        (output / (split + '.jsonl')).write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in values))
    manifest = {'created_at': datetime.now(timezone.utc).isoformat(), 'sources': sources,
                'tokenizer_sha256': file_hash(tokenizer), 'model_config_sha256': file_hash(ROOT / 'configs/model.json'),
                'implementation_sha256': file_hash(__file__), 'source_inventory_sha256': file_hash(cache / 'sources.json'),
                'splits': {split: {'examples': len(values), 'exercise_groups': len({r['group_id'] for r in values}),
                                  'languages': dict(Counter(r['language'] for r in values)), 'target_tokens': sum(r['target_tokens'] for r in values)}
                           for split, values in rows.items()},
                'cross_language_train_validation_group_overlap': 0, 'truncated_admitted_examples': 0,
                'selected_reference_literal_cases_passed': verification_count,
                'blocked_exercise_groups': blocked, 'excluded': dict(excluded),
                'files_sha256': {split + '.jsonl': file_hash(output / (split + '.jsonl')) for split in rows},
                'external_model_weights_used': False, 'external_model_teacher_used': False, 'goal_achieved': False,
                'limitations': ['Publisher examples/instructions are original source files; authorship and all per-file rights are not independently exhaustively audited.',
                                'Only selected literal Python assertions are executed; remaining interface/error cases, other languages and efficiency are unverified.',
                                'Groups are held out across these tracks and known verified-program aliases; all earlier pretraining and semantic overlap are not exhaustively audited.',
                                'Elementary/medium exercises do not establish repository engineering, all code, broad knowledge or high-end capability.',
                                'Prepared data are not trained weights or measured model improvement.']}
    atomic_json(output / 'manifest.json', manifest)
    atomic_json(output / 'progress.json', {'status': 'public task/interface/reference corpus prepared; not trained', 'splits': manifest['splits'], 'goal_achieved': False})
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--output', required=True)
    parser.add_argument('--cache', default='data/exercism-source-cache-v1')
    parser.add_argument('--tokenizer', default='checkpoints/human-feedback-pilot-v1/dpo/tokenizer.json')
    args = parser.parse_args()
    print(json.dumps(prepare(args.output, args.cache, args.tokenizer), indent=2), flush=True)


if __name__ == '__main__': main()
