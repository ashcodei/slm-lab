"""Reclaim unused public PyPIwheel download copies from the pip HTTPcache.

Only positively identified public PyPIwheel responses qualify. No installed file, model, dataset, built wheel,
source-bound path or currently open cache file can be removed.
"""
import argparse
from datetime import datetime, timezone
from email.parser import BytesParser
from importlib.metadata import distribution, PackageNotFoundError
import json
import os
from pathlib import Path
import shutil
import zipfile

from pip._vendor import msgpack
from reclaim_unused_GPU_wheel_cache_v2 import stamp, open_cache_files, protected_paths, normalized
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json

ROOT = Path(__file__).resolve().parents[1]
CACHE = Path('/home/jovyan/.cache/pip/http-v2')
OUT = ROOT / 'reports/public-PyPI-download-cache-v5'
ENV = ROOT / '.venv'


def owners():
    human = json.loads((ROOT / 'reports/phase-human-preference-v8/plan.json').read_text())
    source = json.loads((ROOT / 'reports/full-public-token-suffix-qualification-CPU-v1/plan.json').read_text())
    values = [human['owner'], source['owner']]
    values.extend(p for p in json.loads((ROOT / 'reports/phase-human-preference-v8/children.json').read_text())
        if p['subject'].startswith('train-chosen_CE-'))
    result = []
    for old in values:
        actual = Path('/proc', str(old['pid']), 'stat').read_text().split(') ', 1)[1].split()
        if actual[0] in ['Z', 'X'] or actual[19] != str(old['ticks']):
            raise ValueError('All exact original live human/source owners must remain unchanged')
        result.append(dict(pid=old['pid'], ticks=actual[19]))
    if len(result) != 7: raise ValueError('Exactly4original human training ranks, their launcher and human/source owners required')
    return result


def installed_metadata(project):
    try:
        value = distribution(project)
        return dict(version=value.version, root=str(value.locate_file('')))
    except PackageNotFoundError:
        return dict(version=None, root=None)


def observe(body):
    header = body.with_suffix('')
    body_state, header_state = stamp(body), stamp(header)
    form, raw = header.read_bytes().split(b',', 1)
    if form != b'cc=4': return None
    response = msgpack.loads(raw, raw=False)['response']
    headers = {k.lower(): v for k, v in response['headers'].items()}
    project, release = headers.get('x-pypi-file-project'), headers.get('x-pypi-file-version')
    if (response.get('status') != 200 or not project or not release
            or headers.get('x-pypi-file-package-type') != 'bdist_wheel'): return None
    installed = installed_metadata(project)
    with zipfile.ZipFile(body) as archive:
        names = [n for n in archive.namelist() if n.endswith('.dist-info/METADATA')]
        if len(names) != 1: return None
        metadata = BytesParser().parsebytes(archive.read(names[0]))
    if (normalized(metadata['Name']) != normalized(project) or metadata['Version'] != release
            or int(headers.get('content-length', '-1')) != body_state['bytes']):
        raise ValueError('Exact public wheel response and complete package metadata must reconcile')
    return dict(body=str(body), header=str(header), body_stamp=body_state, header_stamp=header_state,
        body_sha256=file_hash(body), header_sha256=file_hash(header), name=project, cached_version=release,
        installed_version=installed['version'], installed_distribution_root=installed['root'],
        positively_identified_public_PyPI_wheel=True, installed_package_files_not_targeted=True)


def prepare():
    if OUT.exists() or CACHE.resolve() != CACHE: raise ValueError('A new non-symlink cache plan is required')
    actual_owners = owners()
    protected, opened, selected = protected_paths(), open_cache_files(), []
    for body in sorted(CACHE.rglob('*.body'), key=lambda p: p.stat().st_size, reverse=True):
        if body.stat().st_size < 2 * 1024**2: continue
        if any(str(p) in protected or str(p) in opened for p in [body, body.with_suffix('')]): continue
        try: row = observe(body)
        except (zipfile.BadZipFile, KeyError, UnicodeError, PackageNotFoundError): continue
        if row is not None: selected.append(row)
    if not selected: raise ValueError('No unused positively identified public PyPIwheel download copies qualify')
    OUT.mkdir()
    atomic_json(OUT / 'plan.json', dict(at=datetime.now(timezone.utc).isoformat(), selected=selected,
        actual_live_original_owners=actual_owners, source_files_sha256={str(Path(__file__)): file_hash(__file__),
            str(ROOT / 'scripts/reclaim_unused_GPU_wheel_cache_v2.py'): file_hash(ROOT / 'scripts/reclaim_unused_GPU_wheel_cache_v2.py')},
        selected_download_bytes=sum(r['body_stamp']['bytes'] + r['header_stamp']['bytes'] for r in selected),
        selected_allocated_bytes=sum(r['body_stamp']['allocated_bytes'] + r['header_stamp']['allocated_bytes'] for r in selected),
        installed_models_data_built_wheels_source_bound_and_open_files_not_eligible=True, goal_achieved=False))
    print(json.dumps(dict(eligible_public_downloads=len(selected), eligible_allocated_bytes=sum(
        r['body_stamp']['allocated_bytes'] + r['header_stamp']['allocated_bytes'] for r in selected))))


def apply():
    plan = json.loads((OUT / 'plan.json').read_text())
    if (OUT / 'complete.json').exists() or (OUT / 'removed.jsonl').exists():
        raise ValueError('Preserve all earlier complete and partial cache maintenance')
    for path, digest in plan['source_files_sha256'].items():
        if file_hash(path) != digest: raise ValueError('Exact closed cache implementation required')
    if owners() != plan['actual_live_original_owners']: raise ValueError('Every exact original owner must remain live')
    protected, opened = protected_paths(), open_cache_files()
    for row in plan['selected']:
        if any(n in protected or n in opened for n in [row['body'], row['header']]) or observe(Path(row['body'])) != row:
            raise ValueError('Exact selected public cache bytes must remain unused and unbound before any deletion')
    before, removed = shutil.disk_usage(ROOT).free, []
    with (OUT / 'removed.jsonl').open('x') as journal:
        for row in plan['selected']:
            if (stamp(Path(row['body'])) != row['body_stamp'] or stamp(Path(row['header'])) != row['header_stamp']
                    or installed_metadata(row['name'])['version'] != row['installed_version']):
                raise ValueError('Selected public download identity and installed metadata changed')
            Path(row['header']).unlink()
            Path(row['body']).unlink()
            removed.append(row)
            journal.write(json.dumps(row) + '\n')
            journal.flush()
            os.fsync(journal.fileno())
    if owners() != plan['actual_live_original_owners']: raise ValueError('The original live work owners changed')
    if any(Path(r[k]).exists() for r in removed for k in ['body', 'header']): raise ValueError('Selected lookup/body pairs must be gone')
    for row in removed:
        if installed_metadata(row['name'])['version'] != row['installed_version']: raise ValueError('Installed package metadata must remain exact')
    atomic_json(OUT / 'complete.json', dict(complete=True, at=datetime.now(timezone.utc).isoformat(),
        removed_public_downloads=len(removed), removed_download_bytes=plan['selected_download_bytes'],
        removed_allocated_bytes=plan['selected_allocated_bytes'], free_bytes_before=before,
        free_bytes_after=shutil.disk_usage(ROOT).free, free_delta_may_include_concurrent_original_writes=True,
        installed_package_metadata_and_all_original_owners_unchanged=True,
        models_data_installed_files_source_bound_and_open_cache_entries_not_targeted=True,
        source_files_sha256={str(OUT / 'plan.json'): file_hash(OUT / 'plan.json'), str(OUT / 'removed.jsonl'): file_hash(OUT / 'removed.jsonl')},
        goal_achieved=False))
    print(json.dumps(dict(complete=True, removed_allocated_bytes=plan['selected_allocated_bytes'], actual_free_bytes=shutil.disk_usage(ROOT).free)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['prepare', 'apply'])
    args = parser.parse_args()
    prepare() if args.mode == 'prepare' else apply()
