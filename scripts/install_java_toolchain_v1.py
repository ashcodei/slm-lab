"""Install a pinned official Temurin archive locally; do not change system Java."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import tarfile
import urllib.request

from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json


ROOT = Path(__file__).resolve().parents[1]
META = ROOT / 'reports/java-toolchain-v1-github-release.json'
DEST = ROOT / '.tools/java-toolchain-v1'
POINTER = ROOT / '.tools/java-toolchain-v1.json'
RESERVE = 26 * 1024**3
MAXIMUM_ARCHIVE = 256 * 1024**2


def fetch(url, accept='application/octet-stream'):
    return urllib.request.urlopen(urllib.request.Request(url, headers={
        'User-Agent': 'slm-lab-public-toolchain-research', 'Accept': accept}), timeout=45)


def main():
    if DEST.exists() or POINTER.exists():
        raise ValueError('Preserve the existing Java installation or incomplete attempt; inspect before retry')
    if shutil.disk_usage(ROOT).free < RESERVE + 1024**3:
        raise ValueError('Fund1GiB of local toolchain space above the original26GiB reserve')
    release = json.loads(META.read_text())
    if not release['html_url'].startswith('https://github.com/adoptium/temurin21-binaries/releases/tag/'):
        raise ValueError('Official Adoptium Temurin21 release metadata required')
    packages = [a for a in release['assets'] if a['name'].startswith('OpenJDK21U-jdk_x64_linux_hotspot_')
        and a['name'].endswith('.tar.gz')]
    if len(packages) != 1 or packages[0]['size'] > MAXIMUM_ARCHIVE:
        raise ValueError('One bounded original Linux/x64 JDK archive required')
    package = packages[0]
    checksum = next(a for a in release['assets'] if a['name'] == package['name'] + '.sha256.txt')
    DEST.mkdir()
    try:
        with fetch(checksum['browser_download_url']) as stream:
            published = stream.read(4096).decode()
        expected = published.split()[0]
        if not re.fullmatch('[0-9a-f]{64}', expected):
            raise ValueError('Original publisher SHA256 checksum required')
        if package.get('digest') not in (None, 'sha256:' + expected):
            raise ValueError('Original GitHub metadata and published archive checksum disagree')
        (DEST / 'publisher-sha256.txt').write_text(published)
        archive = DEST / package['name']
        digest, total = hashlib.sha256(), 0
        with fetch(package['browser_download_url']) as source, archive.open('xb') as target:
            while chunk := source.read(1024**2):
                total += len(chunk)
                if total > MAXIMUM_ARCHIVE or shutil.disk_usage(ROOT).free < RESERVE + 768 * 1024**2:
                    raise ValueError('Toolchain archive exceeds its bounded reserve')
                target.write(chunk)
                digest.update(chunk)
        if digest.hexdigest() != expected or total != package['size']:
            raise ValueError('Exact published archive digest and byte count required before extraction/execution')
        with tarfile.open(archive) as bundle:
            members = bundle.getmembers()
            roots = {Path(m.name).parts[0] for m in members}
            if len(roots) != 1:
                raise ValueError('One owned JDK directory required')
            expanded = sum(m.size for m in members if m.isfile())
            if expanded > 768 * 1024**2 or shutil.disk_usage(ROOT).free < RESERVE + expanded:
                raise ValueError('Full extracted toolchain reserve is unfunded')
            for member in members:
                path = DEST / member.name
                if not path.resolve().is_relative_to(DEST.resolve()):
                    raise ValueError('Archive path escapes the owned installation')
                if member.issym() and not (path.parent / member.linkname).resolve().is_relative_to(DEST.resolve()):
                    raise ValueError('Archive symlink escapes the owned installation')
            bundle.extractall(DEST, filter='data')
        jdk = DEST / roots.pop()
        java, javac = jdk / 'bin/java', jdk / 'bin/javac'
        smoke = DEST / 'smoke'
        smoke.mkdir()
        source = smoke / 'Smoke.java'
        source.write_text('public class Smoke { public static void main(String[] args) { System.out.println(6 * 7); } }\n')
        environment = dict(PATH='/usr/bin:/bin', LANG='C.UTF-8', TMPDIR=str(smoke))
        compile_result = subprocess.run([str(javac), '-J-XX:ActiveProcessorCount=1', '-J-XX:-UsePerfData',
            '-J-Xms16m', '-J-Xmx128m', str(source)], cwd=smoke, env=environment,
            capture_output=True, text=True, timeout=60, check=True)
        runtime_result = subprocess.run([str(java), '-XX:ActiveProcessorCount=1', '-XX:-UsePerfData',
            '-Xms16m', '-Xmx64m', '-cp', str(smoke), 'Smoke'], cwd=smoke, env=environment,
            capture_output=True, text=True, timeout=30, check=True)
        if runtime_result.stdout.strip() != '42':
            raise ValueError('Original compiler/runtime did not compile and execute the known local smoke program')
        version = subprocess.run([str(java), '-version'], env=environment,
            capture_output=True, text=True, timeout=20, check=True)
        inventory = {str(p): file_hash(p) for p in jdk.rglob('*') if p.is_file()}
        result = dict(installed_at=datetime.now(timezone.utc).isoformat(), format='slm-original-java-toolchain-v1',
            release_tag=release['tag_name'], release_url=release['html_url'],
            metadata_sha256=file_hash(META), metadata=str(META), archive=str(archive), archive_sha256=expected,
            archive_bytes=total, publisher_checksum_url=checksum['browser_download_url'],
            archive_url=package['browser_download_url'], jdk_root=str(jdk), java_path=str(java), javac_path=str(javac),
            java_sha256=file_hash(java), javac_sha256=file_hash(javac), version_output=version.stderr,
            all_installed_files_sha256=inventory, installed_files=len(inventory),
            isolated_compile_and_run_smoke_passed=True, smoke_source_sha256=file_hash(source),
            smoke_class_sha256=file_hash(smoke / 'Smoke.class'), smoke_stdout=runtime_result.stdout,
            smoke_compile_stderr=compile_result.stderr, system_PATH_modified=False,
            model_or_benchmark_training_updates=0, Java_published_suite_runner_completed=False,
            goal_achieved=False)
        atomic_json(POINTER, result)
        print(json.dumps({k: v for k, v in result.items() if k != 'all_installed_files_sha256'}, indent=2))
    except Exception as error:
        atomic_json(DEST / 'failure.json', dict(error=str(error), failed_at=datetime.now(timezone.utc).isoformat(),
            preserve_partial_download=True, automatic_retry=False, goal_achieved=False))
        raise


if __name__ == '__main__':
    main()
