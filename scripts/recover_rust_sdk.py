"""Attempt byte-exact recovery of the existing pinned Rust evaluator runtime."""
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
from slm_lab.exercise_rust_runner import compiler, confined

ROOT = Path(__file__).resolve().parents[1]


def download(url, destination, digest, size):
    if destination.exists():
        if destination.stat().st_size != size or file_hash(destination) != digest:
            raise ValueError('Existing recovery archive differs: ' + str(destination))
        return
    request = urllib.request.Request(url, headers={'User-Agent': 'slm-lab-pinned-runtime-recovery'})
    temporary = destination.with_suffix('.partial')
    with urllib.request.urlopen(request, timeout=45) as stream, temporary.open('xb') as target:
        total = 0
        while block := stream.read(1024**2):
            total += len(block)
            if total > size:
                raise ValueError('Recovery download exceeds original archive size')
            target.write(block)
    if total != size or file_hash(temporary) != digest:
        raise ValueError('Recovery download differs from original publisher archive')
    temporary.replace(destination)


def main():
    manifest = ROOT / '.tools/rust-toolchain-v1.json'
    original = manifest.read_bytes(); tools = json.loads(original)
    sdk = Path(tools['toolchain_root']); sdk.mkdir(exist_ok=True)
    durable = ROOT / '.tools/recovery-downloads'; durable.mkdir(exist_ok=True)
    expected = tools['files_sha256']
    for label, item in tools['compiler_components'].items():
        archive_path = durable / item['url'].rsplit('/', 1)[1]
        download(item['url'], archive_path, item['sha256'], item['bytes'])
        links = []
        with tarfile.open(archive_path) as archive:
            for member in archive:
                parts = Path(member.name).parts
                if len(parts) < 3 or '..' in parts:
                    continue
                relative = Path(*parts[2:]); name = relative.as_posix()
                if name not in expected:
                    continue
                destination = sdk / relative; destination.parent.mkdir(parents=True, exist_ok=True)
                if member.isfile():
                    with archive.extractfile(member) as source, destination.open('wb') as target:
                        shutil.copyfileobj(source, target)
                    destination.chmod(member.mode & 0o777)
                elif member.issym():
                    if not (destination.parent / member.linkname).resolve().is_relative_to(sdk):
                        raise ValueError('Compiler symlink escapes pinned runtime')
                    links.append((destination, member.linkname))
                elif member.islnk():
                    raise ValueError('Inspect compiler hard-link layout before restoring')
        for destination, target in links:
            if destination.is_symlink() and destination.readlink() == Path(target):
                continue
            if destination.exists() or destination.is_symlink():
                raise ValueError('Existing compiler symlink differs from publisher archive')
            destination.symlink_to(target)
        print('Pinned Rust compiler component restored', label, flush=True)
    sources = sdk / 'crate-source'; sources.mkdir(exist_ok=True)
    libraries = sdk / 'evaluation-libraries'; libraries.mkdir(exist_ok=True)
    recipes = {**tools['provided_crates'], **tools['transitive_crates']}
    compile_receipts = {}
    for rust_name in ('diff', 'yansi', 'regex_lite', 'pretty_assertions'):
        name = 'regex-lite' if rust_name == 'regex_lite' else rust_name
        recipe = recipes[rust_name]; version = recipe['version']
        archive_path = durable / (name + '-' + version + '.crate')
        download('https://static.crates.io/crates/' + name + '/' + archive_path.name, archive_path,
                 recipe['publisher_archive_sha256'], recipe['publisher_archive_bytes'])
        shutil.copy2(archive_path, sources / archive_path.name)
        directory = sources / (name + '-' + version)
        with tarfile.open(archive_path) as archive:
            archive.extractall(sources, filter='data')
        request = urllib.request.Request('https://crates.io/api/v1/crates/' + name + '/' + version,
                                        headers={'User-Agent': 'slm-lab-pinned-runtime-recovery'})
        with urllib.request.urlopen(request, timeout=30) as response:
            metadata = response.read(2_000_001)
        if len(metadata) > 2_000_000:
            raise ValueError('Crate publisher metadata exceeds original bound')
        (sources / (directory.name + '.publisher.json')).write_bytes(metadata)
        cargo = tomllib.loads((directory / 'Cargo.toml').read_text())
        if cargo.get('package', {}).get('build') or cargo.get('lib', {}).get('proc-macro'):
            raise ValueError('Pinned library requires unexpected build script or procedural macro')
        with tempfile.TemporaryDirectory(prefix='slm-rust-trusted-library-') as temporary:
            work = Path(temporary); shutil.copytree(directory, work / 'src'); (work / 'build').mkdir()
            destination = work / 'build' / ('lib' + rust_name + '.rlib')
            command = compiler(tools, cargo['package'].get('edition', '2015')) + [
                '--crate-name', rust_name, '--crate-type', 'rlib', cargo.get('lib', {}).get('path', 'src/lib.rs'),
                '-o', str(destination), '-L', 'dependency=' + str(libraries)]
            for feature in recipe['enabled_default_features']:
                command += ['--cfg', 'feature="' + feature + '"']
            for dependency, specification in cargo.get('dependencies', {}).items():
                if isinstance(specification, dict) and specification.get('optional'):
                    continue
                dep = dependency.replace('-', '_')
                command += ['--extern', dep + '=' + str(libraries / ('lib' + dep + '.rlib'))]
            result = confined(work, tools, command, 'compile')
            if result['returncode'] != 0 or result['limit']:
                raise ValueError('Pinned library recompilation failed: ' + result['diagnostic'])
            result.pop('output')
            compile_receipts[name] = result
            shutil.copy2(destination, libraries / destination.name)
        print('Pinned Rust library rebuilt', name, flush=True)
    missing = [name for name in expected if not (sdk / name).is_file()]
    changed = {name: {'expected': digest, 'actual': file_hash(sdk / name)} for name, digest in expected.items()
               if (sdk / name).is_file() and file_hash(sdk / name) != digest}
    record = {'verified_at': datetime.now(timezone.utc).isoformat(), 'original_manifest_unchanged': manifest.read_bytes() == original,
              'original_file_count': len(expected), 'missing_files': missing, 'changed_files': changed,
              'trusted_library_compilation': compile_receipts,
              'byte_exact_recovery': not missing and not changed, 'goal_achieved': False}
    (ROOT / 'reports/rust-runtime-recovery-v1.json').write_text(json.dumps(record, indent=2) + '\n')
    if missing or changed:
        raise ValueError('Rust bytes require further recovery; original manifest preserved: ' + repr(list(changed)))
    print('All original Rust evaluator files restored exactly', len(expected), flush=True)


if __name__ == '__main__':
    main()
