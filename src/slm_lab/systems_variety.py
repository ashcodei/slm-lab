"""Collect pinned public OS/network/security source as data, preserving old corpora."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import urllib.error
import urllib.request
import zipfile

import ahocorasick

from .corpus import file_hash
from .diverse_images import atomic_json, jsonl
from .wikipedia_corpus import protected_questions

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / 'configs/systems-variety-sources-v1.json'
LANGUAGES = {'.c': 'C', '.h': 'C/C++ header', '.cpp': 'C++', '.cc': 'C++', '.hpp': 'C++', '.rs': 'Rust',
    '.s': 'Assembly', '.asm': 'Assembly', '.py': 'Python', '.go': 'Go', '.java': 'Java', '.js': 'JavaScript',
    '.lua': 'Lua', '.sh': 'Shell', '.cmake': 'CMake', '.ld': 'Linker script', '.dts': 'Device tree',
    '.toml': 'TOML', '.md': 'Documentation', '.rst': 'Documentation'}
SKIP = {'vendor', 'vendors', 'third_party', 'third-party', 'thirdparty', 'external', 'node_modules',
        'test', 'tests', 'testing', 'bench', 'benches', 'benchmark', 'benchmarks', 'build', 'generated', '.git'}
ALLOWED = {'MIT', 'BSD-2-Clause', 'BSD-3-Clause', 'Apache-2.0', 'ISC', 'Zlib', 'BSL-1.0', 'CC0-1.0',
           'GPL-2.0', 'GPL-2.0-only', 'GPL-2.0-or-later', 'GPL-3.0', 'GPL-3.0-only', 'GPL-3.0-or-later',
           'LGPL-2.1', 'LGPL-2.1-only', 'LGPL-2.1-or-later', 'LGPL-3.0-only', 'LGPL-3.0-or-later',
           'AGPL-3.0', 'AGPL-3.0-only', 'AGPL-3.0-or-later',
           'Linux-syscall-note', 'AND', 'OR', 'WITH'}


def fetch(url, maximum):
    request = urllib.request.Request(url, headers={'User-Agent': 'slm-lab-public-source-research'})
    with urllib.request.urlopen(request, timeout=40) as source:
        declared = source.headers.get('Content-Length')
        if declared and int(declared) > maximum: raise ValueError('Public source exceeds download bound')
        raw = source.read(maximum + 1)
    if len(raw) > maximum: raise ValueError('Public source exceeds download bound')
    return raw


def license_matches(raw, family):
    text = ' '.join(raw.decode('utf-8').casefold().split())
    if family == 'MIT': return 'permission is hereby granted, free of charge' in text
    if family == 'BSD-3-Clause':
        return ('redistribution and use in source and binary forms' in text and 'all advertising materials' not in text
                and any(phrase in text for phrase in ('neither the name', 'names of its contributors',
                                                      'name of the author may not be used', 'names of the authors may not be used')))
    if family == 'Apache-2.0': return 'apache license' in text and 'version 2.0' in text
    if family == 'GPL-2.0': return 'gnu general public license' in text and 'version 2,' in text
    if family == 'LGPL-2.1': return 'gnu lesser general public license' in text and 'version 2.1,' in text
    if family == 'AGPL-3.0': return 'gnu affero general public license' in text and 'version 3,' in text
    if family == 'ISC': return 'permission to use, copy, modify, and/or distribute' in text
    return False


def effective_license(text, repository_license):
    values = re.findall(r'SPDX-License-Identifier:\s*([^\r\n]+)', text[:8000])
    expressions = [value.strip().removesuffix('*/').strip() for value in values]
    if expressions:
        for expression in expressions:
            terms = re.findall(r'[A-Za-z0-9][A-Za-z0-9.+-]*', expression)
            if not terms or not set(terms) <= ALLOWED: raise ValueError('Unsupported declared file license expression')
        return expressions, 'original file SPDX declarations'
    return [repository_license], 'recorded repository license declaration; no file SPDX override'


def benchmark_filter():
    question_path = ROOT / 'data/knowledge-eval-v1/questions.jsonl'; questions = protected_questions(question_path)
    code = ahocorasick.Automaton()
    for row in jsonl(ROOT / 'data/exercism-code-v1/valid.jsonl'):
        for body in re.findall(r'^File: [^\n]+\n```[^\n]*\n(.*?)^```', row['response'], re.M | re.S):
            text = ''.join(body.split())
            if len(text) >= 80: code.add_word(text, row['id'])
    from .code_tasks import TASKS
    from .cpp_tasks import reference
    for task in TASKS:
        for body in (task[4], reference(task[0])):
            text = ''.join(body.split())
            if len(text) >= 80: code.add_word(text, 'local:' + task[0])
    code.make_automaton()
    return questions, code


def collect(output, resume=False):
    output = Path(output).resolve()
    if output.exists() and not resume: raise ValueError('Existing systems collection requires inspection; never overwrite/restart it')
    if resume and (not output.exists() or (output / 'manifest.json').exists()):
        raise ValueError('Resume requires an inspected incomplete original source collection')
    configuration = json.loads(CONFIG.read_text()); config_digest = file_hash(CONFIG); implementation = file_hash(__file__)
    if not resume:
        output.mkdir(); (output / 'archives').mkdir(); (output / 'licenses').mkdir()
    questions, reserved_code = benchmark_filter(); previous = ROOT / 'data/reviewed-systems-code-v1/all.jsonl'
    previous_manifest = json.loads((previous.parent / 'manifest.json').read_text())
    if file_hash(previous) != previous_manifest['all_sha256']: raise ValueError('Prior library source changed')
    seen = {row['source_file_sha256'] for row in jsonl(previous) if row.get('source_file_sha256')}; previous_digest = file_hash(previous)
    sources = []; exclusions = Counter(); domains = Counter(); languages = Counter(); archive_bytes = 0
    resume_receipt = None
    if resume:
        path = output / 'partial-admission-before-resume-v1.json'; resume_receipt = json.loads(path.read_text())
        if (resume_receipt['previous_actor_verified_terminal'] is not True
                or resume_receipt['all_prior_rows_joined_to_exact_original_archives'] is not True
                or resume_receipt['configuration_sha256'] != config_digest
                or file_hash(resume_receipt['old_implementation_snapshot']) != resume_receipt['old_implementation_sha256']):
            raise ValueError('Original partial source/implementation inspection differs')
        expected_files = {item['jsonl'] for item in resume_receipt['sources']}
        if {path.name for path in output.glob('*.jsonl')} != expected_files:
            raise ValueError('Additional uninspected source files require a fresh partial-source audit')
        for item in resume_receipt['sources']:
            rows_path = output / item['jsonl']; license_path = output / 'licenses' / (item['repo'].replace('/', '--') + '.txt')
            if (file_hash(rows_path) != item['jsonl_sha256'] or file_hash(output / item['archive']) != item['archive_sha256']
                    or file_hash(license_path) != item['license_sha256']):
                raise ValueError('Original inspected partial source/license/archive bytes changed')
            rows = list(jsonl(rows_path))
            if len(rows) != item['documents']: raise ValueError('Original partial source counts differ')
            seen.update(row['source_file_sha256'] for row in rows)
            domains.update(row['domain'] for row in rows); languages.update(row['language'] for row in rows)
            archive_bytes += item['archive_bytes']; sources.append(item)
    completed = {item['repo'] for item in sources}
    def progress(state, **fields):
        atomic_json(output / 'progress.json', {'status': state, 'updated_at': datetime.now(timezone.utc).isoformat(),
            'goal_achieved': False, 'trained': False, **fields})
    progress('collecting original public OS/network/security source files', repositories=len(configuration['repositories']))
    try:
        for spec in configuration['repositories']:
            repo = spec['repo']
            if repo in completed: continue
            if not re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+', repo): raise ValueError('Invalid publisher repository')
            remote = subprocess.check_output(['git', 'ls-remote', 'https://github.com/' + repo + '.git', 'HEAD'], timeout=40, text=True)
            revision = remote.split()[0]
            if not re.fullmatch(r'[0-9a-f]{40}', revision): raise ValueError('Publisher revision identity differs')
            license_raw = None; license_path = None
            for candidate in spec['license_paths']:
                try: raw = fetch('https://raw.githubusercontent.com/' + repo + '/' + revision + '/' + candidate, 100_000)
                except urllib.error.HTTPError as error:
                    if error.code == 404: continue
                    raise
                if license_matches(raw, spec['license']): license_raw = raw; license_path = candidate; break
            if license_raw is None: raise ValueError('Original publisher license requires review: ' + repo)
            if shutil.disk_usage(output).free < 4 * 1024**3: raise ValueError('Systems source scratch-storage reserve reached')
            archive_raw = fetch('https://codeload.github.com/' + repo + '/zip/' + revision, configuration['maximum_archive_bytes'])
            archive_bytes += len(archive_raw)
            if archive_bytes > configuration['maximum_total_archive_bytes']: raise ValueError('Original source archive budget exceeded')
            name = repo.replace('/', '--'); archive_path = output / 'archives' / (name + '.zip'); archive_path.write_bytes(archive_raw)
            (output / 'licenses' / (name + '.txt')).write_bytes(license_raw)
            count = 0; stored = 0; rows_path = output / (name + '.jsonl'); per_languages = Counter()
            with zipfile.ZipFile(io.BytesIO(archive_raw)) as archive, rows_path.open('w') as target:
                prefix = archive.namelist()[0].split('/')[0] + '/'
                if archive.read(prefix + license_path) != license_raw: raise ValueError('License and original archive revision differ')
                # Stable source-hash ordering gives each project a bounded sample.
                members = sorted(archive.infolist(), key=lambda item: hashlib.sha256(item.filename.encode()).digest())
                for member in members:
                    path = PurePosixPath(member.filename); relative = PurePosixPath(*path.parts[1:]); extension = path.suffix.casefold()
                    if path.is_absolute() or '..' in path.parts: raise ValueError('Unsafe original repository archive path')
                    if member.is_dir() or extension not in LANGUAGES or member.file_size > configuration['maximum_file_bytes']: continue
                    if set(part.casefold() for part in relative.parts) & SKIP: exclusions['vendor/generated/test directories'] += 1; continue
                    raw = archive.read(member); digest = hashlib.sha256(raw).hexdigest()
                    if digest in seen: exclusions['exact source-file duplicate of old or new corpus'] += 1; continue
                    try: text = raw.decode('utf-8')
                    except UnicodeDecodeError: exclusions['non UTF8 source'] += 1; continue
                    if '\x00' in text or len(text.strip()) < 200: continue
                    if re.search(r'generated (?:file|code)|do not edit', text[:2000], re.I): exclusions['declared generated source'] += 1; continue
                    try: licenses, license_scope = effective_license(text, spec['license'])
                    except ValueError: exclusions['unknown file license expression'] += 1; continue
                    if next(questions.iter(' '.join(text.casefold().split())), None) is not None:
                        exclusions['exact reserved knowledge question'] += 1; continue
                    if next(reserved_code.iter(''.join(text.split())), None) is not None:
                        exclusions['exact reserved code reference'] += 1; continue
                    if stored + len(raw) > configuration['per_repository_source_bytes'] or count >= configuration['maximum_files_per_repository']: continue
                    seen.add(digest); stored += len(raw); count += 1; language = LANGUAGES[extension]
                    source_url = 'https://github.com/' + repo + '/blob/' + revision + '/' + str(relative)
                    row = {'id': 'public-code:' + name + ':' + digest, 'group_id': 'github:' + repo + ':' + str(relative),
                        'text': 'Programming language: ' + language + '\nFile: ' + str(relative) + '\n\n' + text,
                        'source': 'GitHub/' + repo, 'source_url': source_url, 'revision': revision, 'domain': spec['domain'],
                        'language': language, 'source_file': str(relative), 'source_file_sha256': digest,
                        'source_file_bytes': len(raw), 'license_expressions': licenses, 'license_scope': license_scope,
                        'repository_license': spec['license'], 'license_url': 'https://github.com/' + repo + '/blob/' + revision + '/' + license_path,
                        'repository_license_sha256': hashlib.sha256(license_raw).hexdigest(), 'external_model_teacher_used': False}
                    target.write(json.dumps(row, ensure_ascii=False) + '\n'); domains[spec['domain']] += 1; languages[language] += 1; per_languages[language] += 1
                if not count: raise ValueError('No eligible original source files from ' + repo)
            sources.append({'repo': repo, 'revision': revision, 'domain': spec['domain'], 'documents': count,
                'source_bytes': stored, 'language_files': dict(per_languages), 'jsonl': rows_path.name, 'jsonl_sha256': file_hash(rows_path),
                'archive': str(archive_path.relative_to(output)), 'archive_sha256': hashlib.sha256(archive_raw).hexdigest(),
                'archive_bytes': len(archive_raw), 'license': spec['license'], 'license_path': license_path,
                'license_sha256': hashlib.sha256(license_raw).hexdigest()})
            atomic_json(output / (name + '.source.json'), sources[-1])
            progress('collecting original public OS/network/security source files', completed_repositories=len(sources),
                     source_files=sum(item['documents'] for item in sources), archive_bytes=archive_bytes)
            print('Original systems source collected', repo, count, stored, flush=True)
        if file_hash(CONFIG) != config_digest or file_hash(__file__) != implementation or file_hash(previous) != previous_digest:
            raise ValueError('Systems variety configuration/implementation/prior sources changed during collection')
        result = {'created_at': datetime.now(timezone.utc).isoformat(), 'configuration': configuration,
            'configuration_sha256': config_digest, 'implementation_sha256': implementation, 'sources': sources,
            'source_files': sum(item['documents'] for item in sources), 'domain_files': dict(domains), 'language_files': dict(languages),
            'exclusions': dict(exclusions), 'previous_library_collection_sha256': previous_digest,
            'resume_original_source_receipt_sha256': file_hash(output / 'partial-admission-before-resume-v1.json') if resume else None,
            'exclusion_count_scope': 'Current resumed collection only; prior unreported counts are not reconstructed.' if resume else 'Entire collection.',
            'reserved_knowledge_sha256': file_hash(ROOT / 'data/knowledge-eval-v1/questions.jsonl'),
            'reserved_exercise_sha256': file_hash(ROOT / 'data/exercism-code-v1/valid.jsonl'),
            'external_model_teacher_used': False, 'downloaded_code_executed': False, 'training_admitted': False, 'trained': False,
            'goal_achieved': False, 'limitations': ['Bounded original repository samples, not all open source code.',
                'License metadata are recorded publisher/file declarations, not independent ownership certification.',
                'Exact reserved-question/reference and source-byte filters do not certify semantic or near-duplicate independence.',
                'Collection remains separate from the immutable current curriculum; complete tokenization/admission and scheduling are required before learning.',
                'Temporary scratch storage must remain available until the future source-bound training finishes.']}
        atomic_json(output / 'manifest.json', result); progress('original systems variety collected; not trained', source_files=result['source_files'])
        return result
    except Exception as error:
        progress('requires inspection; public systems variety collection stopped', error=str(error), completed_repositories=len(sources)); raise


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--output', required=True)
    parser.add_argument('--resume', action='store_true'); args = parser.parse_args()
    result = collect(args.output, args.resume); print('Original systems variety ready', result['source_files'], flush=True)


if __name__ == '__main__': main()
