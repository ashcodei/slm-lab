"""Prepare hash-verified PyPI wheels in an isolated CPU-only runtime tree."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tempfile
import urllib.request
import zipfile

from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json

ROOT = Path(__file__).resolve().parents[1]
VERSIONS = {'numpy': '1.26.4', 'numba': '0.59.1', 'llvmlite': '0.42.0'}


def main():
    folder = ROOT / 'reports/numba-runtime-source-v2'
    descriptor = ROOT / '.tools/numba-runtime-v1.json'
    if folder.exists() or descriptor.exists():
        raise ValueError('Preserve earlier isolated runtime and source evidence')
    folder.mkdir()
    cache = Path(tempfile.mkdtemp(prefix='slm-numba-runtime-v1-'))
    site = cache / 'site-packages'
    site.mkdir()
    durable = ROOT / '.tools/numba-wheel-archives-v1'
    if durable.exists():
        raise ValueError('Preserve exact earlier wheel archives')
    selections = []
    for name, version in VERSIONS.items():
        url = f'https://pypi.org/pypi/{name}/{version}/json'
        with urllib.request.urlopen(url, timeout=30) as response:
            raw = response.read(4 * 1024**2 + 1)
        if len(raw) > 4 * 1024**2:
            raise ValueError('Published package metadata size budget exceeded')
        (folder / (name + '-pypi.json')).write_bytes(raw)
        metadata = json.loads(raw)
        if metadata['info']['name'].lower() != name or metadata['info']['version'] != version:
            raise ValueError('Published exact package identity differs')
        wheels = [entry for entry in metadata['urls'] if entry['packagetype'] == 'bdist_wheel'
            and '-cp311-cp311-' in entry['filename']
            and 'manylinux_2_17_x86_64' in entry['filename'] and not entry.get('yanked')]
        if len(wheels) != 1:
            raise ValueError('Unique pinned CPython3.11 Linux wheel required')
        selections.append({'package': name, 'version': version, 'metadata_url': url,
            'metadata_sha256': hashlib.sha256(raw).hexdigest(), **wheels[0]})
    total = sum(entry['size'] for entry in selections)
    if total > 256 * 1024**2 or shutil.disk_usage(ROOT).free < 26 * 1024**3 + total + 5 * 1024**2:
        raise ValueError('Preserve persistent writer reserve and exact archive budget')
    if shutil.disk_usage(cache).free < 13 * 1024**3:
        raise ValueError('Preserve temporary reserve and extraction headroom')
    atomic_json(folder / 'plan.json', {'recorded_at': datetime.now(timezone.utc).isoformat(),
        'pinned_versions': VERSIONS, 'published_version_matrix': 'https://numba.readthedocs.io/en/0.59.1/user/installing.html#version-support-information',
        'selected_wheels': selections, 'runtime_cache': str(cache), 'package_root': str(site),
        'published_wheel_hashes_are_integrity_receipts_not_package_signatures': True,
        'existing_python_environment_changed': False, 'goal_achieved': False})
    durable.mkdir()
    for entry in selections:
        archive = durable / entry['filename']
        request = urllib.request.Request(entry['url'], headers={'User-Agent': 'slm-lab-isolated-runtime/1'})
        digest, received = hashlib.sha256(), 0
        with urllib.request.urlopen(request, timeout=30) as response, archive.open('xb') as output:
            while True:
                block = response.read(1024**2)
                if not block:
                    break
                received += len(block)
                if received > entry['size']:
                    raise ValueError('Wheel exceeds published byte count')
                digest.update(block)
                output.write(block)
        if received != entry['size'] or digest.hexdigest() != entry['digests']['sha256']:
            raise ValueError('Downloaded wheel differs from published PyPI identity')
        with zipfile.ZipFile(archive) as bundle:
            if sum(member.file_size for member in bundle.infolist()) > 1024**3:
                raise ValueError('Isolated wheel extraction budget exceeded')
            for member in bundle.infolist():
                path = PurePosixPath(member.filename)
                if path.is_absolute() or '..' in path.parts or (member.external_attr >> 16) & 0o170000 == 0o120000:
                    raise ValueError('Unexpected wheel path or symbolic link')
                target = site.joinpath(*path.parts)
                if target.exists() and not target.is_dir() and not member.is_dir():
                    raise ValueError('Pinned wheels overlap an existing file')
            bundle.extractall(site)
        print('verified_and_extracted', entry['package'], entry['version'], received, flush=True)
    manifest = folder / 'runtime-files.jsonl'
    files, retained = 0, 0
    with manifest.open('x') as output:
        for path in sorted(site.rglob('*')):
            if path.is_file():
                size = path.stat().st_size
                output.write(json.dumps({'relative_path': str(path.relative_to(site)),
                    'sha256': file_hash(path), 'bytes': size}) + '\n')
                files += 1
                retained += size
    atomic_json(folder / 'receipt.json', {'completed_at': datetime.now(timezone.utc).isoformat(),
        'versions': VERSIONS, 'published_wheel_hashes_verified': True, 'exact_wheel_archives_durable': str(durable),
        'archive_bytes': total, 'isolated_package_root': str(site), 'software_files': files,
        'extracted_bytes': retained, 'file_manifest_sha256': file_hash(manifest),
        'installer_source_sha256': file_hash(Path(__file__)),
        'runtime_import_or_confinement_tested': False, 'existing_python_environment_changed': False,
        'new_model_training_started': False, 'goal_achieved': False})
    atomic_json(descriptor, {'versions': VERSIONS, 'package_root': str(site),
        'file_manifest': str(manifest), 'file_manifest_sha256': file_hash(manifest),
        'receipt': str(folder / 'receipt.json'), 'receipt_sha256': file_hash(folder / 'receipt.json'),
        'published_wheel_hashes_verified': True, 'system_installation_performed': False})
    print('isolated_runtime_files', files, 'extracted_bytes', retained)


if __name__ == '__main__':
    main()
