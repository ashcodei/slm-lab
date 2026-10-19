"""Supply pinned publisher editor files in training prompts; preserve targets."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import zipfile

from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json, jsonl
from slm_lab.exercise_probe import validate_splits

ROOT = Path(__file__).resolve().parents[1]
EXCLUDE = {'crypto-square': 'Original reference fails the published54-character ciphertext case.',
           'parallel-letter-frequency': 'CommonJS reference/ESM starter-test interoperability is not verified by the current native protocol.',
           'word-search': 'CommonJS reference/ESM starter-test interoperability is not verified by the current native protocol.'}


def main():
    original = ROOT / 'data/exercism-code-v1'; output = ROOT / 'data/exercism-support-context-v3'
    if output.exists():
        raise ValueError('Preserve an existing support-context generation')
    manifest = json.loads((original / 'manifest.json').read_text())
    for name in ['train.jsonl', 'valid.jsonl']:
        if file_hash(original / name) != manifest['files_sha256'][name]:
            raise ValueError('Original finalized exercise rows changed')
    output.mkdir(); rows = []; changes = []; excluded = []; archives = {}
    try:
        for original_row in jsonl(original / 'train.jsonl'):
            row = dict(original_row); language = row['language']; exercise = row['exercise']
            row_sha = hashlib.sha256(json.dumps(original_row, sort_keys=True).encode()).hexdigest()
            if language == 'JavaScript' and exercise in EXCLUDE:
                excluded.append({'id': row['id'], 'group_id': row['group_id'], 'original_row_sha256': row_sha,
                                 'reason': EXCLUDE[exercise]}); continue
            if language in ['Go', 'JavaScript']:
                key = 'go' if language == 'Go' else 'javascript'
                if key not in archives:
                    archive_path = ROOT / 'data/exercism-source-cache-v1' / key / 'source.zip'
                    if file_hash(archive_path) != row['source_archive_sha256']:
                        raise ValueError('Original publisher support archive differs')
                    archives[key] = zipfile.ZipFile(archive_path)
                archive = archives[key]; prefix = archive.namelist()[0].split('/')[0] + '/'
                item = next(item for item in row['source_files'] if item['path'].endswith('/.meta/config.json'))
                config_raw = archive.read(prefix + item['path'])
                if hashlib.sha256(config_raw).hexdigest() != item['sha256']:
                    raise ValueError('Original publisher file declaration differs')
                config = json.loads(config_raw)['files']; base = item['path'].split('/.meta/')[0] + '/'
                support = []
                for name in config.get('editor', []):
                    relative = PurePosixPath(name)
                    if relative.is_absolute() or '..' in relative.parts or name in config.get('test', []):
                        raise ValueError('Inspect an unsafe or test-bearing publisher editor file')
                    if relative.suffix != ('.go' if language == 'Go' else '.js'):
                        continue
                    raw = archive.read(prefix + base + name); text = raw.decode('utf-8'); digest = hashlib.sha256(raw).hexdigest()
                    if len(raw) > 200000 or '\x00' in text:
                        raise ValueError('Original complete support file exceeds source bound')
                    stored = output / 'support-files' / key / exercise / name
                    stored.parent.mkdir(parents=True, exist_ok=True); stored.write_bytes(raw)
                    support.append({'file_name': name, 'publisher_path': base + name, 'sha256': digest,
                                    'bytes': len(raw), 'stored_path': str(stored.relative_to(output)),
                                    'declared_role': 'Original publisher files.editor entry; provided supporting source'})
                if support:
                    marker = 'Return the implementations for the named solution files.'
                    if row['prompt'].count(marker) != 1:
                        raise ValueError('Original complete instruction boundary differs')
                    context = 'The project also provides these fixed supporting files. Use their interfaces and return only the requested solution files.\n\n'
                    for item in support:
                        context += 'File: ' + item['file_name'] + '\n```' + ('go' if language == 'Go' else 'javascript') + '\n'
                        context += (output / item['stored_path']).read_bytes().decode('utf-8') + '\n```\n\n'
                    row['prompt'] = row['prompt'].replace(marker, context + marker)
                    row['id'] += ':support-context-v2'; row['source_original_id'] = original_row['id']
                    row['source_original_row_sha256'] = row_sha; row['provided_support_files'] = support
                    if language == 'JavaScript' and exercise == 'lens-person':
                        substitutions = []
                        for provided in support:
                            stem = provided['file_name'].removesuffix('.js')
                            before = "from '../" + stem + "'"; after = "from './" + stem + "'"
                            if row['response'].count(before) != 1:
                                raise ValueError('Inspected publisher relative-import mapping differs')
                            row['response'] = row['response'].replace(before, after)
                            substitutions.append({'before': before, 'after': after})
                        row['id'] += ':import-relocation-v3'
                        row['reference_relocation'] = {'reason': 'Publisher proof file moves from.meta/ into the requested solution directory.',
                            'literal_import_substitutions': substitutions,
                            'original_response_sha256': hashlib.sha256(original_row['response'].encode()).hexdigest(),
                            'relocated_response_sha256': hashlib.sha256(row['response'].encode()).hexdigest()}
                    changes.append({'id': row['id'], 'original_id': original_row['id'], 'language': language,
                                    'exercise': exercise, 'original_row_sha256': row_sha, 'support_files': support,
                                    'original_response_unchanged': row['response'] == original_row['response']})
            rows.append(row)
        valid = list(jsonl(original / 'valid.jsonl')); validate_splits(rows, valid)
        (output / 'train.jsonl').write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))
        shutil.copy2(original / 'valid.jsonl', output / 'valid.jsonl')
        receipt = {'created_at': datetime.now(timezone.utc).isoformat(), 'original_manifest_sha256': file_hash(original / 'manifest.json'),
            'original_train_sha256': file_hash(original / 'train.jsonl'), 'original_valid_sha256': file_hash(original / 'valid.jsonl'),
            'tokenizer_sha256': manifest['tokenizer_sha256'], 'implementation_sha256': file_hash(__file__),
            'files_sha256': {name: file_hash(output / name) for name in ['train.jsonl', 'valid.jsonl']},
            'train_examples': len(rows), 'by_language': dict(Counter(row['language'] for row in rows)),
            'changed_contexts': changes, 'excluded_original_rows': excluded, 'all_retained_original_targets_unchanged': False,
            'target_transformations': 'Only five static relative imports in lens-person are relocated; all algorithm bodies and other targets remain original.',
            'validation_rows_byte_identical': file_hash(output / 'valid.jsonl') == manifest['files_sha256']['valid.jsonl'],
            'external_model_teacher_used': False, 'trained': False, 'goal_achieved': False,
            'limitations': ['This generation corrects specifically inspected Go/JavaScript provided-source context. Other-language support completeness is not certified.',
                           'Three JavaScript rows are excluded only from this instruction generation; original source corpora are preserved.',
                           'External dependency support and universal program correctness remain unverified. Complete native tests must assess the changed contexts.']}
        atomic_json(output / 'manifest.json', receipt)
        print('Publisher support context prepared', len(rows), 'rows;', len(changes), 'contexts augmented;', len(excluded), 'rows excluded', flush=True)
    finally:
        for archive in archives.values():
            archive.close()


if __name__ == '__main__':
    main()
