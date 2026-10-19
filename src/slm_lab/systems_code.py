"""Recover explicitly reviewed publisher code sources without changing old data."""
import argparse
import base64
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import shutil
from urllib.error import HTTPError
import zipfile

from .corpus import file_hash
from .diverse_images import atomic_json, jsonl
from .public_sources import EXTENSIONS, SKIP, api, emit, fetch

ROOT = Path(__file__).resolve().parents[2]
RECOVERY = [('protocolbuffers/protobuf', 'BSD-3-Clause', 'software_engineering'),
            ('CLIUtils/CLI11', 'BSD-3-Clause', 'software_engineering'),
            ('boostorg/asio', 'BSL-1.0', 'networking'),
            ('taocpp/PEGTL', 'BSL-1.0', 'software_engineering'),
            ('simdjson/simdjson', 'Apache-2.0', 'software_engineering'),
            ('pybind/pybind11', 'BSD-3-Clause', 'software_engineering')]


def reviewed_identity(repo, raw, detected, expected, reviews):
    if detected == expected: return 'publisher API identification'
    receipt = reviews.get(repo)
    if (detected not in {None, 'NOASSERTION'} or receipt is None or receipt['expected_family'] != expected or
        hashlib.sha256(raw).hexdigest() != receipt['license_sha256']):
        raise ValueError('Unrecognized publisher license or bytes differ from explicit review: ' + repo)
    return 'exact publisher-license bytes matched recorded review'


def publisher_license(repo, commit, expected, reviews):
    try:
        info = api('https://api.github.com/repos/' + repo + '/license?ref=' + commit)
        raw = base64.b64decode(info['content']); detected = info['license'].get('spdx_id')
        method = reviewed_identity(repo, raw, detected, expected, reviews)
        return raw, {'github_license_id': detected, 'license_review_method': method,
                     'license_url': 'https://github.com/' + repo + '/blob/' + commit + '/' + info['path']}
    except HTTPError as error:
        if error.code != 404 or repo != 'boostorg/asio' or expected != 'BSL-1.0': raise
    notice_url = 'https://raw.githubusercontent.com/' + repo + '/' + commit + '/include/boost/asio.hpp'
    notice = fetch(notice_url, 100_000)
    if b'Distributed under the Boost Software License, Version 1.0.' not in notice:
        raise ValueError('Original Asio publisher license notice absent')
    canonical_url = 'https://www.boost.org/LICENSE_1_0.txt'; raw = fetch(canonical_url, 100_000)
    if not raw.startswith(b'Boost Software License - Version 1.0 - August 17th, 2003'):
        raise ValueError('Canonical Boost license text mismatch')
    return raw, {'github_license_id': None, 'license_review_method': 'original pinned publisher umbrella-header notice plus canonical Boost text',
                 'license_url': 'https://github.com/' + repo + '/blob/' + commit + '/include/boost/asio.hpp',
                 'canonical_license_url': canonical_url, 'publisher_license_notice_sha256': hashlib.sha256(notice).hexdigest()}


def recover(repo, expected, domain, output, reviews):
    info = api('https://api.github.com/repos/' + repo)
    commit = api('https://api.github.com/repos/' + repo + '/commits/' + info['default_branch'])['sha']
    license_bytes, license_evidence = publisher_license(repo, commit, expected, reviews)
    name = 'repo-' + repo.replace('/', '--')
    (output / 'licenses' / (name + '.txt')).write_bytes(license_bytes)
    raw = fetch('https://codeload.github.com/' + repo + '/zip/' + commit)
    archive_path = output / 'archives' / (name + '.zip'); archive_path.write_bytes(raw)
    documents = []; excluded = 0; total = 0
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        for member in sorted(archive.infolist(), key=lambda value: value.filename):
            path = PurePosixPath(member.filename); relative = PurePosixPath(*path.parts[1:])
            if '..' in path.parts or path.is_absolute(): raise ValueError('Unsafe original archive path')
            if member.is_dir() or path.suffix.lower() not in EXTENSIONS or member.file_size > 200_000: continue
            if set(part.lower() for part in relative.parts) & SKIP: continue
            if any(value in str(relative).lower() for value in ['min.js', 'lockfile', 'package-lock', 'license', 'copying']): continue
            source_bytes = archive.read(member)
            try: text = source_bytes.decode('utf-8')
            except UnicodeDecodeError: continue
            if '\x00' in text or len(text.strip()) < 200: continue
            if re.search(r'generated (?:file|code)|do not edit', text[:2000], re.I): excluded += 1; continue
            markers = re.findall(r'SPDX-License-Identifier:\s*([^\n]+)', text[:4000])
            if markers and any(expected not in marker for marker in markers): excluded += 1; continue
            total += len(text)
            if total > 100_000_000: break
            documents.append({'text': 'File: ' + str(relative) + '\nLanguage: ' + path.suffix + '\n\n' + text,
                              'domain': domain, 'kind': 'documentation' if path.suffix in {'.md', '.rst', '.txt'} else 'code',
                              'language_extension': path.suffix, 'source': 'GitHub/' + repo,
                              'source_url': 'https://github.com/' + repo + '/blob/' + commit + '/' + str(relative),
                              'source_file_sha256': hashlib.sha256(source_bytes).hexdigest(), 'source_file_bytes': len(source_bytes),
                              'revision': commit, 'license': expected, 'license_url': license_evidence['license_url'],
                              'group_id': 'github:' + repo + ':' + str(relative)})
    result = {**emit(output / (name + '.jsonl'), documents), 'name': name, 'repo': repo, 'revision': commit,
              'license': expected, 'license_sha256': hashlib.sha256(license_bytes).hexdigest(), **license_evidence,
              'archive_sha256': hashlib.sha256(raw).hexdigest(), 'archive_path': str(archive_path),
              'excluded_files': excluded, 'collected_at': datetime.now(timezone.utc).isoformat()}
    atomic_json(output / (name + '.json'), result); return result


def collect(output):
    output = Path(output).resolve()
    if output.exists(): raise ValueError('Use a fresh reviewed collection directory')
    output.mkdir(); (output / 'licenses').mkdir(); (output / 'archives').mkdir()
    initial = ROOT / 'data/public-systems-code-v1'; manifest = json.loads((initial / 'manifest.json').read_text())
    if file_hash(initial / 'all.jsonl') != manifest['all_sha256']: raise ValueError('Original strict collection changed')
    review_path = ROOT / 'configs/systems-license-reviews-v1.json'
    reviews = json.loads(review_path.read_text())['sources']; sources = []; failures = []
    for source in manifest['sources']:
        name = source['name']; path = initial / (name + '.jsonl')
        if file_hash(path) != source['sha256']: raise ValueError('Original accepted repository data changed')
        shutil.copy2(path, output / path.name)
        license_path = initial / 'licenses' / (source['repo'].replace('/', '--') + '.txt')
        shutil.copy2(license_path, output / 'licenses' / (name + '.txt'))
        sources.append({**source, 'license_sha256': file_hash(license_path), 'initial_collection': str(initial),
                        'initial_manifest_sha256': file_hash(initial / 'manifest.json')})
    for repo, expected, domain in RECOVERY:
        try:
            result = recover(repo, expected, domain, output, reviews); sources.append(result)
            print('Reviewed original source collected', repo, result['documents'], flush=True)
        except Exception as error:
            failures.append({'repo': repo, 'error': str(error)}); print('Reviewed original source failed', repo, str(error), flush=True)
    seen = set(); domains = {}; languages = {}; count = 0
    with (output / 'all.jsonl').open('w') as target:
        for source in sources:
            for row in jsonl(output / (source['name'] + '.jsonl')):
                if row['id'] in seen: continue
                seen.add(row['id']); count += 1; target.write(json.dumps(row, ensure_ascii=False) + '\n')
                domains[row['domain']] = domains.get(row['domain'], 0) + len(row['text'])
                extension = row['language_extension']; languages[extension] = languages.get(extension, 0) + len(row['text'])
    result = {'collected_at': datetime.now(timezone.utc).isoformat(), 'sources': sources, 'failures': failures,
              'initial_strict_collection_preserved': str(initial), 'publisher_reviews_sha256': file_hash(review_path),
              'implementation_sha256': file_hash(__file__), 'unique_documents': count, 'domain_characters': domains,
              'language_characters': languages, 'all_sha256': file_hash(output / 'all.jsonl'),
              'external_model_weights_used': False, 'downloaded_source_executed': False, 'goal_achieved': False,
              'limitations': ['Root licenses/publisher notices and exact-byte reviews retained; not an exhaustive per-file rights audit.',
                              'Public library code/data preparation is not a trained-model correctness or high-end capability score.',
                              'Authorship and full project tests were not independently verified; source code is read only as data.']}
    atomic_json(output / 'manifest.json', result); return result


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--output', required=True)
    args = parser.parse_args(); result = collect(args.output)
    print('Reviewed systems corpus', len(result['sources']), 'sources', result['unique_documents'], 'documents', len(result['failures']), 'failures', flush=True)


if __name__ == '__main__': main()
