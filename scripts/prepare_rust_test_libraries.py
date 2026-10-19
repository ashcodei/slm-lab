"""Build pinned crates required by unchanged original Rust publisher references."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import tempfile
import tomllib
import urllib.request

from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.exercise_rust_runner import TARGET, compiler, confined

ROOT = Path(__file__).resolve().parents[1]
VERSIONS = {'diff': '0.1.13', 'yansi': '1.0.1', 'regex-lite': '0.1.9', 'pretty_assertions': '1.4.1'}


def fetch(url, maximum):
    request = urllib.request.Request(url, headers={'User-Agent': 'slm-lab-local-code-evaluation'})
    with urllib.request.urlopen(request, timeout=30) as stream: raw = stream.read(maximum + 1)
    if len(raw) > maximum: raise ValueError('Original Rust dependency exceeds download bound')
    return raw


def main():
    manifest_path = ROOT / '.tools/rust-toolchain-v1.json'
    if manifest_path.exists(): raise ValueError('Preserve installed pinned Rust compiler evidence')
    installation = json.loads((ROOT / 'reports/rust-toolchain-installation-location-v1.json').read_text())
    sdk = Path(installation['root']); sources = sdk / 'crate-source'; sources.mkdir(exist_ok=True)
    libraries = sdk / 'evaluation-libraries'; libraries.mkdir(exist_ok=True)
    receipts = {}; compile_receipts = {}; tools = {'toolchain_root': str(sdk), 'provided_crates': {}}
    for name, version in VERSIONS.items():
        metadata_raw = fetch('https://crates.io/api/v1/crates/' + name + '/' + version, 2_000_000)
        metadata = json.loads(metadata_raw)['version']; digest = metadata['checksum']
        archive = fetch('https://static.crates.io/crates/' + name + '/' + name + '-' + version + '.crate', 2_000_000)
        if hashlib.sha256(archive).hexdigest() != digest: raise ValueError('Original crate archive checksum differs')
        archive_path = sources / (name + '-' + version + '.crate'); archive_path.write_bytes(archive)
        (sources / (name + '-' + version + '.publisher.json')).write_bytes(metadata_raw)
        directory = sources / (name + '-' + version); directory.mkdir(exist_ok=True)
        with tarfile.open(archive_path) as source:
            for member in source:
                parts = Path(member.name).parts
                if not parts or parts[0] != directory.name or '..' in parts: raise ValueError('Unsafe crate package path')
                target = directory / Path(*parts[1:])
                if member.isdir(): target.mkdir(parents=True, exist_ok=True)
                elif member.isfile():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with source.extractfile(member) as stream: target.write_bytes(stream.read())
                else: raise ValueError('Unreviewed nonregular Rust crate entry')
        cargo = tomllib.loads((directory / 'Cargo.toml').read_text())
        if cargo.get('package', {}).get('build') or cargo.get('lib', {}).get('proc-macro'):
            raise ValueError('This explicit library provision does not run build scripts or procedural macros')
        name_in_rust = name.replace('-', '_'); edition = cargo['package'].get('edition', '2015')
        features = cargo.get('features', {}); enabled = {'default'}; pending = list(features.get('default', [])); optional_dependencies = set()
        while pending:
            feature = pending.pop()
            if feature.startswith('dep:'): optional_dependencies.add(feature[4:]); continue
            if '/' in feature:
                optional_dependencies.add(feature.split('/')[0].rstrip('?')); continue
            if feature in enabled: continue
            enabled.add(feature); pending.extend(features.get(feature, []))
        with tempfile.TemporaryDirectory(prefix='slm-rust-trusted-library-') as temporary:
            work = Path(temporary); shutil.copytree(directory, work / 'src'); (work / 'build').mkdir()
            destination = work / 'build' / ('lib' + name_in_rust + '.rlib')
            command = compiler(tools, edition) + ['--crate-name', name_in_rust, '--crate-type', 'rlib',
                cargo.get('lib', {}).get('path', 'src/lib.rs'), '-o', str(destination), '-L', 'dependency=' + str(libraries)]
            for feature in sorted(enabled): command += ['--cfg', 'feature="' + feature + '"']
            for dependency, specification in cargo.get('dependencies', {}).items():
                if isinstance(specification, dict) and specification.get('optional') and dependency not in optional_dependencies:
                    continue
                dep = dependency.replace('-', '_'); expected = libraries / ('lib' + dep + '.rlib')
                if not expected.exists(): raise ValueError('Explicit original Rust library dependency is not built: ' + dependency)
                command += ['--extern', dep + '=' + str(expected)]
            result = confined(work, tools, command, 'compile'); result.pop('output')
            if result['returncode'] != 0 or result['limit']: raise ValueError('Pinned crate failed confined compilation: ' + repr(result))
            shutil.copy2(destination, libraries / destination.name); compile_receipts[name] = result
        receipts[name_in_rust] = {'version': version, 'publisher_archive_sha256': digest,
            'publisher_archive_bytes': len(archive), 'source_Cargo_toml_sha256': file_hash(directory / 'Cargo.toml'),
            'enabled_default_features': sorted(enabled), 'compiled_library_sha256': file_hash(libraries / ('lib' + name_in_rust + '.rlib'))}
        print('Pinned native Rust library', name, version, 'compiled', flush=True)
    version = __import__('subprocess').check_output([str(sdk / 'bin/rustc'), '--version'], text=True).strip()
    if version != 'rustc 1.99.0 (b940084d7 2026-09-28)': raise ValueError('Pinned compiler version differs')
    metadata = {'created_at': datetime.now(timezone.utc).isoformat(), 'version': version, 'target': TARGET,
        'toolchain_root': str(sdk), 'compiler_components': installation['components'],
        'channel_manifest_sha256': file_hash(ROOT / 'reports/rust-stable-publisher-channel-v1.toml'),
        'provided_crates': {key: receipts[key] for key in ('regex_lite', 'pretty_assertions')},
        'transitive_crates': {key: receipts[key] for key in ('diff', 'yansi')},
        'trusted_library_compilation': compile_receipts,
        'files_sha256': {str(path.relative_to(sdk)): file_hash(path) for path in sdk.rglob('*') if path.is_file()},
        'scope': 'Pinned compiler, musl standard library and explicit regex-lite/pretty_assertions provision; unchanged publisher integration tests, no Cargo network resolution/build scripts.',
        'goal_achieved': False}
    atomic_json(manifest_path, metadata)
    print('Pinned Rust toolchain ready', version, len(metadata['files_sha256']), 'files', flush=True)


if __name__ == '__main__': main()
