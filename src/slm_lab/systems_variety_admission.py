"""Reconstruct new OS/network/security inputs before any own-model training."""
from collections import Counter
import hashlib
import json
from pathlib import Path, PurePosixPath
import zipfile

from .corpus import file_hash
from .diverse_images import jsonl
from .systems_variety import CONFIG, LANGUAGES, benchmark_filter, effective_license


def audit_collection(root):
    root = Path(root).resolve(); data = (root / 'data/public-systems-variety-v1').resolve()
    path = data / 'manifest.json'; digest = file_hash(path); manifest = json.loads(path.read_text())
    configuration = json.loads(CONFIG.read_text())
    if (manifest['configuration'] != configuration or manifest['configuration_sha256'] != file_hash(CONFIG)
            or manifest['implementation_sha256'] != file_hash(root / 'src/slm_lab/systems_variety.py')
            or manifest['reserved_knowledge_sha256'] != file_hash(root / 'data/knowledge-eval-v1/questions.jsonl')
            or manifest['reserved_exercise_sha256'] != file_hash(root / 'data/exercism-code-v1/valid.jsonl')
            or manifest['previous_library_collection_sha256'] != file_hash(root / 'data/reviewed-systems-code-v1/all.jsonl')):
        raise ValueError('Original systems source/configuration/protected evidence changed')
    if [item['repo'] for item in manifest['sources']] != [item['repo'] for item in configuration['repositories']]:
        raise ValueError('Require all original configured OS/network/security sources')
    questions, code = benchmark_filter(); rows = []; groups = set(); hashes = set(); files = {}
    for spec, source in zip(configuration['repositories'], manifest['sources']):
        rows_path = data / source['jsonl']; archive_path = data / source['archive']
        license_path = data / 'licenses' / (source['repo'].replace('/', '--') + '.txt')
        for actual, expected in [(rows_path, source['jsonl_sha256']), (archive_path, source['archive_sha256']),
                                 (license_path, source['license_sha256'])]:
            if not actual.resolve().is_relative_to(data) or file_hash(actual) != expected:
                raise ValueError('Original public source path/row/archive/license bytes changed')
            files[str(actual)] = expected
        originals = list(jsonl(rows_path))
        if (len(originals) != source['documents'] or sum(row['source_file_bytes'] for row in originals) != source['source_bytes']
                or Counter(row['language'] for row in originals) != source['language_files']):
            raise ValueError('Original systems source counts/languages differ')
        with zipfile.ZipFile(archive_path) as archive:
            prefix = archive.namelist()[0].split('/')[0] + '/'
            if archive.read(prefix + source['license_path']) != license_path.read_bytes():
                raise ValueError('Original source archive and publisher license differ')
            for row in originals:
                relative = PurePosixPath(row['source_file'])
                if relative.is_absolute() or '..' in relative.parts: raise ValueError('Source file path escapes original archive')
                raw = archive.read(prefix + str(relative)); text = raw.decode('utf-8')
                sha = hashlib.sha256(raw).hexdigest(); language = LANGUAGES[relative.suffix.casefold()]
                expressions, scope = effective_license(text, spec['license'])
                if (row['text'] != 'Programming language: ' + language + '\nFile: ' + str(relative) + '\n\n' + text
                        or row['source_file_sha256'] != sha or row['source_file_bytes'] != len(raw)
                        or row['language'] != language or row['domain'] != spec['domain']
                        or row['source'] != 'GitHub/' + spec['repo'] or row['revision'] != source['revision']
                        or row['source_url'] != 'https://github.com/' + spec['repo'] + '/blob/' + source['revision'] + '/' + str(relative)
                        or row['group_id'] != 'github:' + spec['repo'] + ':' + str(relative)
                        or row['license_expressions'] != expressions or row['license_scope'] != scope
                        or row['repository_license_sha256'] != source['license_sha256']
                        or row['repository_license'] != spec['license'] or row['external_model_teacher_used'] is not False):
                    raise ValueError('Systems training row differs from complete original source reconstruction')
                if row['group_id'] in groups or sha in hashes: raise ValueError('Repeated new original source file/group')
                if (next(questions.iter(' '.join(text.casefold().split())), None) is not None
                        or next(code.iter(''.join(text.split())), None) is not None):
                    raise ValueError('Reserved exact benchmark question/reference entered new source data')
                groups.add(row['group_id']); hashes.add(sha); rows.append(row)
    if (len(rows) != manifest['source_files'] or Counter(row['domain'] for row in rows) != manifest['domain_files']
            or Counter(row['language'] for row in rows) != manifest['language_files']
            or file_hash(path) != digest or any(file_hash(filename) != sha for filename, sha in files.items())):
        raise ValueError('Original systems population or evidence changed during reconstruction')
    return rows, {'passed': True, 'source_manifest_sha256': digest, 'original_repositories': len(manifest['sources']),
        'original_files_reconstructed': len(rows), 'exact_reserved_code_or_knowledge_matches': 0,
        'domain_files': manifest['domain_files'], 'language_files': manifest['language_files'],
        'all_publisher_row_archive_license_bytes_bound': files, 'trained': False, 'goal_achieved': False}
