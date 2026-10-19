"""Restore the exact pinned Go distribution after temporary-instance loss."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def main():
    manifest = ROOT / '.tools/go-toolchain-v1.json'; tools = json.loads(manifest.read_text())
    sdk = Path(tools['toolchain_root']); archive = Path(tools['archive'])
    durable = ROOT / '.tools/recovery-downloads' / archive.name; durable.parent.mkdir(exist_ok=True)
    if not durable.exists():
        request = urllib.request.Request(tools['archive_url'], headers={'User-Agent':'slm-lab-runtime-recovery'})
        with urllib.request.urlopen(request, timeout=60) as stream, durable.with_suffix('.partial').open('wb') as output:
            digest = hashlib.sha256(); total = 0
            while raw := stream.read(1024**2):
                total += len(raw)
                if total > tools['archive_bytes']: raise ValueError('Pinned Go download exceeds original byte budget')
                output.write(raw); digest.update(raw)
        if total != tools['archive_bytes'] or digest.hexdigest() != tools['archive_sha256']:
            raise ValueError('Pinned Go publisher archive differs from original bytes')
        durable.with_suffix('.partial').replace(durable)
    from slm_lab.corpus import file_hash
    if file_hash(durable) != tools['archive_sha256']: raise ValueError('Durable Go archive checksum changed')
    sdk.parent.mkdir(exist_ok=True)
    if not archive.exists(): shutil.copy2(durable, archive)
    if not sdk.exists():
        with tarfile.open(durable) as source:
            source.extractall(sdk.parent, filter='data')
    for name, expected in tools['files_sha256'].items():
        if file_hash(sdk / name) != expected: raise ValueError('Restored Go SDK file differs: ' + name)
    print('Exact pinned Go SDK bytes restored', len(tools['files_sha256']), flush=True)
    cache = Path(tools['compile_cache_root']); cache.mkdir(exist_ok=True)
    env = dict(os.environ, GOCACHE=str(cache), GOTOOLCHAIN='local', GOPROXY='off', GOSUMDB='off',
               GOMAXPROCS='2', GOFLAGS='-p=1', CGO_ENABLED='0')
    completed = subprocess.run([str(sdk / 'bin/go'), 'tool', 'test2json', '-h'], env=env,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=180)
    target = Path(tools['test2json_binary'])
    matched = target.is_file() and file_hash(target) == tools['test2json_sha256']
    record = {'verified_at':datetime.now(timezone.utc).isoformat(), 'archive':str(durable),
        'archive_sha256':tools['archive_sha256'], 'exact_sdk_files_verified':len(tools['files_sha256']),
        'test2json_original_bytes_restored':matched, 'test2json_command_returncode':completed.returncode,
        'test2json_diagnostic':completed.stderr[-1000:], 'manifest_unchanged':True,'goal_achieved':False}
    (ROOT / 'reports/go-runtime-recovery-v1.json').write_text(json.dumps(record,indent=2)+'\n')
    if not matched: raise ValueError('Go SDK restored, but original test2json cache still requires explicit reconstruction')
    print('Original Go test2json helper restored exactly', flush=True)


if __name__ == '__main__': main()
