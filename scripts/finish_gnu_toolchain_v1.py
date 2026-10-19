"""Verify signed Ubuntu index/package chains and extract an owned GNU profile."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json

ROOT = Path(__file__).resolve().parents[1]


def fields(text):
    return dict(line.split(': ', 1) for line in text.splitlines() if ': ' in line and not line.startswith(' '))


def main():
    folder = ROOT / 'reports/gnu-toolchain-source-v1'
    output = folder / 'verified-packages-and-extraction.json'
    if output.exists() or (ROOT / '.tools/gnu-toolchain-v1.json').exists():
        raise ValueError('Preserve prepared GNU software and receipts')
    metadata = json.loads((folder / 'verified-metadata.json').read_text())
    cache, keyring = Path(metadata['cache']), Path('/usr/share/keyrings/ubuntu-archive-keyring.gpg')
    env = {**os.environ, 'APT_CONFIG': metadata['apt_config']}
    indexes = []
    for record in metadata['signed_inrelease_files']:
        release = Path(record['path'])
        if file_hash(release) != record['sha256'] or file_hash(keyring) != metadata['host_keyring_sha256']:
            raise ValueError('Signed repository/key bytes differ')
        signature = subprocess.run(['gpgv', '--status-fd', '1', '--keyring', str(keyring), str(release)],
                                   capture_output=True, text=True, check=True)
        if '[GNUPG:] VALIDSIG ' not in signature.stdout:
            raise ValueError('Trusted Ubuntu signature required')
        in_sha256, expected = False, None
        for line in release.read_text().splitlines():
            if line == 'SHA256:':
                in_sha256 = True
                continue
            if in_sha256 and not line.startswith(' '):
                in_sha256 = False
            if in_sha256:
                parts = line.split()
                if len(parts) == 3 and parts[2] == 'main/binary-amd64/Packages':
                    expected = parts
        if expected is None:
            raise ValueError('Signed native package index digest missing')
        index_path = release.with_name(release.name.replace('InRelease', 'main_binary-amd64_Packages.lz4'))
        process = subprocess.Popen(['/usr/lib/apt/apt-helper', 'cat-file', str(index_path)], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        digest, length = hashlib.sha256(), 0
        for block in iter(lambda: process.stdout.read(1024**2), b''):
            digest.update(block)
            length += len(block)
            if length > int(expected[1]):
                raise ValueError('Repository index exceeds its signed size')
        error = process.stderr.read()
        if process.wait(timeout=10) or digest.hexdigest() != expected[0] or length != int(expected[1]):
            raise ValueError('Uncompressed repository package index differs: ' + error.decode(errors='replace'))
        indexes.append({'inrelease': str(release), 'index': str(index_path), 'uncompressed_sha256': digest.hexdigest(),
                        'uncompressed_bytes': length, 'trusted_signature': signature.stdout})
    archives = sorted((cache / 'archives').glob('*.deb'))
    if not archives or list((cache / 'archives/partial').glob('*')):
        raise ValueError('Completed official package downloads required')
    if shutil.disk_usage(cache).free < 14 * 1024**3:
        raise ValueError('Preserve temporary reserve and extraction headroom')
    if shutil.disk_usage(ROOT).free < 26 * 1024**3 + sum(p.stat().st_size for p in archives) + 10 * 1024**2:
        raise ValueError('Preserve persistent reserve while retaining exact package archives')
    receipts = []
    for archive in archives:
        package = fields(subprocess.check_output(['dpkg-deb', '--field', str(archive), 'Package', 'Version', 'Architecture'], text=True))
        info = subprocess.check_output(['apt-cache', *metadata['apt_options'], 'show', package['Package'] + '=' + package['Version']], env=env, text=True)
        candidates = [fields(paragraph) for paragraph in info.strip().split('\n\n')]
        matching = [row for row in candidates if row.get('Version') == package['Version'] and row.get('Architecture') == package['Architecture']]
        if not matching or any(row['SHA256'] != file_hash(archive) or int(row['Size']) != archive.stat().st_size for row in matching):
            raise ValueError('Downloaded archive differs from the signed package index')
        receipts.append({'archive': archive.name, **package, 'sha256': file_hash(archive),
                         'bytes': archive.stat().st_size, 'signed_index_filename': matching[0]['Filename']})
    durable_archives = ROOT / '.tools/gnu-toolchain-archives-v1'
    if durable_archives.exists():
        raise ValueError('Preserve earlier durable package copies')
    durable_archives.mkdir()
    root = cache / 'root'
    for archive, receipt in zip(archives, receipts):
        shutil.copyfile(archive, durable_archives / archive.name)
        if file_hash(durable_archives / archive.name) != receipt['sha256']:
            raise ValueError('Durable software archive copy differs')
        subprocess.run(['dpkg-deb', '--extract', str(archive), str(root)], check=True)
    relocated_links = []
    for path in root.rglob('*'):
        if path.is_symlink():
            target = os.readlink(path)
            if target.startswith('/'):
                replacement = os.path.relpath(root / target.lstrip('/'), path.parent)
                path.unlink()
                path.symlink_to(replacement)
                relocated_links.append({'path': str(path.relative_to(root)), 'package_absolute_target': target,
                                        'isolated_relative_target': replacement})
    manifest_path = folder / 'toolchain-files.jsonl'
    files, retained = 0, 0
    with manifest_path.open('x') as manifest:
        for path in sorted(root.rglob('*')):
            if path.is_file():
                if not path.resolve().is_relative_to(root):
                    raise ValueError('Extracted compiler file escapes isolated software tree')
                manifest.write(json.dumps({'relative_path': str(path.relative_to(root)), 'sha256': file_hash(path),
                                           'bytes': path.stat().st_size}) + '\n')
                files += 1
                retained += path.stat().st_size
    compiler = root / 'usr/bin/x86_64-linux-gnu-g++-11'
    version = subprocess.check_output([str(compiler), '--version'], text=True).splitlines()[0]
    atomic_json(output, {'completed_at': datetime.now(timezone.utc).isoformat(), 'verified_signed_package_indexes': indexes,
        'verified_package_archives': receipts, 'durable_archive_folder': str(durable_archives),
        'source_and_package_hash_chain_verified': True, 'isolated_root': str(root),
        'relocated_absolute_links': relocated_links, 'software_files': files, 'software_file_bytes': retained,
        'file_manifest_sha256': file_hash(manifest_path), 'source_script_sha256': file_hash(Path(__file__)),
        'system_packages_or_python_environment_changed': False,
        'compiler_runtime_confinement_tested': False, 'goal_achieved': False})
    atomic_json(ROOT / '.tools/gnu-toolchain-v1.json', {'version': version, 'compiler_path': str(compiler),
        'compiler_sha256': file_hash(compiler), 'toolchain_root': str(root),
        'file_manifest': str(manifest_path), 'file_manifest_sha256': file_hash(manifest_path),
        'signed_repository_and_package_hashes_verified': True,
        'package_receipt': str(output), 'package_receipt_sha256': file_hash(output),
        'extracted_tree_is_ephemeral_but_exact_source_archives_are_durable': True,
        'system_installation_performed': False})
    print('verified_official_packages', len(receipts), 'isolated_software_files', files, 'version', version)


if __name__ == '__main__':
    main()
