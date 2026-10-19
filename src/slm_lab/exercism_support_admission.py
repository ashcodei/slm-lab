"""Reconstruct corrected training contexts from unchanged pinned publisher bytes."""
from collections import Counter
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path, PurePosixPath
import zipfile

from tokenizers import Tokenizer

from .corpus import file_hash
from .diverse_images import jsonl
from .exercise_probe import validate_splits
from .instruction_generalization_trial import complete_encoding
from .model import ModelConfig
from .training import Encoder

EXCLUDED = {('JavaScript', name) for name in ('crypto-square', 'parallel-letter-frequency', 'word-search')}


def admit(root):
    root = Path(root).resolve(); original = root / 'data/exercism-code-v1'; data = root / 'data/exercism-support-context-v3'
    manifest_path = data / 'manifest.json'; digest = file_hash(manifest_path); manifest = json.loads(manifest_path.read_text())
    original_manifest = json.loads((original / 'manifest.json').read_text())
    if (file_hash(original / 'manifest.json') != manifest['original_manifest_sha256']
            or file_hash(root / 'scripts/prepare_exercism_support_context_v3.py') != manifest['implementation_sha256']):
        raise ValueError('Original publisher corpus or context preparation differs')
    for name in ['train.jsonl', 'valid.jsonl']:
        if (file_hash(original / name) != original_manifest['files_sha256'][name]
                or file_hash(data / name) != manifest['files_sha256'][name]):
            raise ValueError('Original or corrected complete exercise rows changed')
    if file_hash(data / 'valid.jsonl') != original_manifest['files_sha256']['valid.jsonl']:
        raise ValueError('Held exercise population changed')
    originals = list(jsonl(original / 'train.jsonl')); rows = list(jsonl(data / 'train.jsonl'))
    valid = list(jsonl(original / 'valid.jsonl')); validate_splits(rows, valid)
    retained = [row for row in originals if (row['language'], row['exercise']) not in EXCLUDED]
    if len(rows) != len(retained) or len(rows) != manifest['train_examples']:
        raise ValueError('Corrected publisher population differs')
    tokenizer = root / 'data/tokenizer-corpus/tokenizer.json'
    if file_hash(tokenizer) != manifest['tokenizer_sha256']:
        raise ValueError('Corrected source own tokenizer differs')
    encoder = Encoder(Tokenizer.from_file(str(tokenizer)), ModelConfig.load(root / 'configs/model.json'), data, 448)
    helpers = {}; archives = {}; contexts = 0; relocations = 0; maximum = 0; targets = 0
    with ExitStack() as stack:
        for before, after in zip(retained, rows):
            expected = dict(before); language = before['language']; exercise = before['exercise']; support = []
            row_sha = hashlib.sha256(json.dumps(before, sort_keys=True).encode()).hexdigest()
            if language in ['Go', 'JavaScript']:
                key = 'go' if language == 'Go' else 'javascript'
                if key not in archives:
                    path = root / 'data/exercism-source-cache-v1' / key / 'source.zip'
                    if file_hash(path) != before['source_archive_sha256']:
                        raise ValueError('Pinned publisher support archive differs')
                    archives[key] = stack.enter_context(zipfile.ZipFile(path))
                archive = archives[key]; prefix = archive.namelist()[0].split('/')[0] + '/'
                item = next(item for item in before['source_files'] if item['path'].endswith('/.meta/config.json'))
                raw = archive.read(prefix + item['path'])
                if hashlib.sha256(raw).hexdigest() != item['sha256']:
                    raise ValueError('Publisher supporting-file declaration changed')
                config = json.loads(raw)['files']; base = item['path'].split('/.meta/')[0] + '/'
                context = 'The project also provides these fixed supporting files. Use their interfaces and return only the requested solution files.\n\n'
                for name in config.get('editor', []):
                    path = PurePosixPath(name)
                    if path.is_absolute() or '..' in path.parts or name in config.get('test', []):
                        raise ValueError('Unsafe or test-bearing declared support path')
                    if path.suffix != ('.go' if language == 'Go' else '.js'):
                        continue
                    raw = archive.read(prefix + base + name); sha = hashlib.sha256(raw).hexdigest()
                    stored = data / 'support-files' / key / exercise / name
                    if stored.read_bytes() != raw:
                        raise ValueError('Provided source differs from original publisher bytes')
                    helpers[str(stored)] = sha
                    support.append({'file_name': name, 'publisher_path': base + name, 'sha256': sha,
                        'bytes': len(raw), 'stored_path': str(stored.relative_to(data)),
                        'declared_role': 'Original publisher files.editor entry; provided supporting source'})
                    context += 'File: ' + name + '\n```' + ('go' if language == 'Go' else 'javascript') + '\n' + raw.decode() + '\n```\n\n'
                if support:
                    marker = 'Return the implementations for the named solution files.'
                    if before['prompt'].count(marker) != 1:
                        raise ValueError('Original instruction boundary differs')
                    expected.update(prompt=before['prompt'].replace(marker, context + marker),
                        id=before['id'] + ':support-context-v2', source_original_id=before['id'],
                        source_original_row_sha256=row_sha, provided_support_files=support)
                    contexts += 1
                    if language == 'JavaScript' and exercise == 'lens-person':
                        substitutions = []
                        for provided in support:
                            stem = provided['file_name'].removesuffix('.js'); old = "from '../" + stem + "'"; new = "from './" + stem + "'"
                            if expected['response'].count(old) != 1:
                                raise ValueError('Publisher import relocation differs')
                            expected['response'] = expected['response'].replace(old, new)
                            substitutions.append({'before': old, 'after': new})
                        expected['id'] += ':import-relocation-v3'
                        expected['reference_relocation'] = {'reason': 'Publisher proof file moves from.meta/ into the requested solution directory.',
                            'literal_import_substitutions': substitutions,
                            'original_response_sha256': hashlib.sha256(before['response'].encode()).hexdigest(),
                            'relocated_response_sha256': hashlib.sha256(expected['response'].encode()).hexdigest()}
                        relocations += 1
            if after != expected:
                raise ValueError('Training source differs beyond declared original context/import changes')
            count = complete_encoding(after, encoder); maximum = max(maximum, count['context_tokens']); targets += count['target_tokens']
    if (file_hash(manifest_path) != digest or contexts != 12 or relocations != 1
            or Counter(row['language'] for row in rows) != manifest['by_language'] or encoder.truncated):
        raise ValueError('Context source population or complete target budget differs')
    readiness_path = root / 'reports/exercism-support-context-v3-readiness.json'; readiness = json.loads(readiness_path.read_text())
    if (readiness['manifest_sha256'] != digest or readiness['train_sha256'] != file_hash(data / 'train.jsonl')
            or file_hash(root / readiness['lens_complete_suite_report']) != readiness['lens_complete_suite_report_sha256']):
        raise ValueError('Inspected context/native source evidence changed')
    return rows, {'passed': True, 'manifest_sha256': digest, 'train_sha256': file_hash(data / 'train.jsonl'),
        'original_manifest_sha256': manifest['original_manifest_sha256'], 'original_train_sha256': file_hash(original / 'train.jsonl'),
        'valid_sha256': original_manifest['files_sha256']['valid.jsonl'], 'examples': len(rows),
        'by_language': manifest['by_language'], 'complete_response_targets_per_pass': targets,
        'maximum_complete_context': maximum, 'exact_original_provided_source_files': helpers,
        'original_support_contexts': contexts, 'static_import_relocations': relocations,
        'all_other_source_fields_and_algorithm_bodies_unchanged': True, 'held_population_byte_identical': True,
        'readiness_sha256': file_hash(readiness_path), 'goal_achieved': False,
        'limitations': ['Source reconstruction and complete target masks do not prove general coding capability.',
                       'Native dependencies, mocking, file/process tests and explicitly skipped cases retain the recorded assessment limits.']}
