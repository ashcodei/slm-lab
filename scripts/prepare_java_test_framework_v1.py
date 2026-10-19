"""Fetch the pinned publisher Java test dependencies into an isolated cache."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import urllib.error
import urllib.request

from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / '.tools/java-test-framework-v1'
MANIFEST = ROOT / '.tools/java-test-framework-v1.json'
ARTIFACTS = [('org/junit/platform', 'junit-platform-console-standalone', '1.10.0'),
    ('org/assertj', 'assertj-core', '3.25.1'), ('org/assertj', 'assertj-core', '3.15.0'),
    ('junit', 'junit', '4.13'), ('org/hamcrest', 'hamcrest-core', '1.3')]


def fetch(url, maximum):
    with urllib.request.urlopen(url, timeout=30) as stream:
        raw = stream.read(maximum + 1)
    if len(raw) > maximum:
        raise ValueError('Bounded Java test dependency exceeded')
    return raw


def main():
    if FOLDER.exists() or MANIFEST.exists():
        raise ValueError('Preserve existing dependency cache; no automatic retry')
    if shutil.disk_usage(ROOT).free < 26 * 1024**3 + 64 * 1024**2:
        raise ValueError('Preserve original26GiB reserve plus64MiB dependency capacity')
    FOLDER.mkdir()
    records = []
    for group, artifact, version in ARTIFACTS:
        base = f'https://repo.maven.apache.org/maven2/{group}/{artifact}/{version}/{artifact}-{version}'
        for algorithm in ['sha256', 'sha512', 'sha1']:
            try:
                checksum = fetch(base + '.jar.' + algorithm, 4096).decode().strip().split()[0]
                break
            except urllib.error.HTTPError as error:
                if error.code != 404:
                    raise
        else:
            raise ValueError('Original Maven publisher checksum missing')
        raw = fetch(base + '.jar', 8 * 1024**2)
        if hashlib.new(algorithm, raw).hexdigest() != checksum:
            raise ValueError('Original Maven dependency checksum differs')
        jar = FOLDER / (artifact + '-' + version + '.jar')
        jar.write_bytes(raw)
        pom = jar.with_suffix('.pom')
        pom.write_bytes(fetch(base + '.pom', 256 * 1024))
        (FOLDER / (jar.name + '.' + algorithm)).write_text(checksum + '\n')
        records.append(dict(artifact=artifact, version=version, jar=str(jar), pom=str(pom),
            original_jar_url=base + '.jar', original_pom_url=base + '.pom',
            publisher_checksum_algorithm=algorithm, publisher_checksum=checksum,
            sha256=file_hash(jar), bytes=len(raw), pom_sha256=file_hash(pom)))
    atomic_json(MANIFEST, dict(format='slm-original-java-test-framework-v1', prepared_at=datetime.now(timezone.utc).isoformat(),
        artifacts=records, files_sha256={str(p):file_hash(p) for p in FOLDER.iterdir()},
        publisher_versions='JUnit Jupiter5.10.0/platform1.10.0 and AssertJ3.25.1; legacyJUnit4.13/AssertJ3.15.0 with Hamcrest1.3. Explicit legacy JUnit jar precedes standalone bundled classes.',
        checksum_scope='Original Maven checksums match; older artifacts publish SHA1 only. All fetched jar/POM bytes additionally receive local pinned SHA256.',
        model_or_benchmark_updates=0, goal_achieved=False))
    print(json.dumps(dict(artifacts=len(records),total_bytes=sum(p.stat().st_size for p in FOLDER.iterdir()),manifest=str(MANIFEST))))


if __name__ == '__main__':
    main()
