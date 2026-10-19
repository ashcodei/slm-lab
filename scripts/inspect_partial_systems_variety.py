"""Reconstruct each stopped source collection from its pinned original archives."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json, jsonl
from slm_lab.masked_visual_pretraining import process_identity
from slm_lab.systems_variety import effective_license, LANGUAGES

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prior-launch', type=Path, required=True)
    parser.add_argument('--previous-configuration', type=Path, required=True)
    parser.add_argument('--previous-implementation', type=Path, required=True)
    args = parser.parse_args(); launch = json.loads(args.prior_launch.read_text())
    if process_identity(launch['pid']) == launch['start_ticks']: raise ValueError('Prior collector is still live; preserve it')
    output = Path(launch['output']); progress = json.loads((output / 'progress.json').read_text())
    if (output / 'manifest.json').exists() or 'requires inspection' not in progress['status']:
        raise ValueError('Require a verified stopped incomplete source collection')
    old = json.loads(args.previous_configuration.read_text()); configuration_path = ROOT / 'configs/systems-variety-sources-v1.json'
    current = json.loads(configuration_path.read_text()); count = progress['completed_repositories']
    if (file_hash(args.previous_configuration) != launch['configuration_sha256']
            or file_hash(args.previous_implementation) != launch['implementation_sha256']
            or old['repositories'][:count] != current['repositories'][:count]
            or {key: value for key, value in old.items() if key != 'repositories'} != {key: value for key, value in current.items() if key != 'repositories'}):
        raise ValueError('Previously collected source rules/budgets differ from the inspected implementation')
    expected = {spec['repo'].replace('/', '--') + '.jsonl' for spec in current['repositories'][:count]}
    if {path.name for path in output.glob('*.jsonl')} != expected: raise ValueError('Uninspected partial source populations')
    records = []
    for spec in current['repositories'][:count]:
        name = spec['repo'].replace('/', '--'); path = output / (name + '.jsonl'); rows = list(jsonl(path))
        if not rows or len({row['id'] for row in rows}) != len(rows): raise ValueError('Empty/repeated original source rows')
        revisions = {row['revision'] for row in rows}
        if len(revisions) != 1: raise ValueError('Mixed original source revisions')
        revision = next(iter(revisions)); archive_path = output / 'archives' / (name + '.zip')
        license_file = output / 'licenses' / (name + '.txt'); license_raw = license_file.read_bytes()
        with zipfile.ZipFile(archive_path) as archive:
            prefix = archive.namelist()[0].split('/')[0] + '/'
            license_paths = [p for p in spec['license_paths'] if prefix + p in archive.namelist() and archive.read(prefix + p) == license_raw]
            if len(license_paths) != 1: raise ValueError('Original archive/publisher license differs')
            for row in rows:
                raw = archive.read(prefix + row['source_file']); text = raw.decode(); digest = hashlib.sha256(raw).hexdigest()
                language = LANGUAGES[Path(row['source_file']).suffix.casefold()]
                licenses, scope = effective_license(text, spec['license'])
                if (len(raw) != row['source_file_bytes'] or digest != row['source_file_sha256']
                        or row['text'] != 'Programming language: ' + language + '\nFile: ' + row['source_file'] + '\n\n' + text
                        or row['license_expressions'] != licenses or row['license_scope'] != scope
                        or row['domain'] != spec['domain'] or row['source'] != 'GitHub/' + spec['repo']
                        or row['repository_license'] != spec['license'] or row['repository_license_sha256'] != file_hash(license_file)
                        or row['external_model_teacher_used'] is not False
                        or row['source_url'] != 'https://github.com/' + spec['repo'] + '/blob/' + revision + '/' + row['source_file']):
                    raise ValueError('Partial source row differs from complete original archive reconstruction')
        record = {'repo': spec['repo'], 'revision': revision, 'domain': spec['domain'], 'documents': len(rows),
            'source_bytes': sum(row['source_file_bytes'] for row in rows), 'language_files': dict(Counter(row['language'] for row in rows)),
            'jsonl': path.name, 'jsonl_sha256': file_hash(path), 'archive': str(archive_path.relative_to(output)),
            'archive_sha256': file_hash(archive_path), 'archive_bytes': archive_path.stat().st_size,
            'license': spec['license'], 'license_path': license_paths[0], 'license_sha256': file_hash(license_file)}
        records.append(record)
    target = output / 'partial-admission-before-resume-v1.json'
    if target.exists():
        previous = output / ('partial-admission-' + file_hash(target)[:16] + '.json')
        if not previous.exists(): shutil.copy2(target, previous)
    atomic_json(target, {'checked_at': datetime.now(timezone.utc).isoformat(), 'previous_actor_verified_terminal': True,
        'prior_pid': launch['pid'], 'prior_start_ticks': launch['start_ticks'],
        'old_implementation_snapshot': str(args.previous_implementation.resolve()),
        'old_implementation_sha256': file_hash(args.previous_implementation),
        'prior_configuration_snapshot': str(args.previous_configuration.resolve()), 'prior_configuration_sha256': launch['configuration_sha256'],
        'configuration_sha256': file_hash(configuration_path), 'sources': records,
        'all_prior_rows_joined_to_exact_original_archives': True, 'prior_unreported_exclusion_counts_not_reconstructed': True,
        'inspector_sha256': file_hash(__file__), 'trained': False, 'goal_achieved': False})
    print('Verified stopped original source collection', count, 'repositories', sum(item['documents'] for item in records), 'files', flush=True)


if __name__ == '__main__': main()
