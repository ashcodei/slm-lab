"""Pinned private CPU JIT runtime without changing active model dependencies."""
import json
from pathlib import Path
import sys

from .corpus import file_hash
from .native_code import ROOT, NativeBackendError, READ, EXEC, ALL_FS
from .standalone_program_v1 import PYTHON_BOOTSTRAP, prepare_python
from .jit_transport_v1 import run_jit_bounded


LAUNCHER = ROOT / 'scripts/numba_program_sandbox_v1.py'


def numba_runtime():
    descriptor = ROOT / '.tools/numba-runtime-v1.json'
    if not descriptor.exists():
        raise NativeBackendError('Pinned private Numba runtime is not prepared')
    data = json.loads(descriptor.read_text())
    if data.get('published_wheel_hashes_verified') is not True:
        raise NativeBackendError('Verified official package hashes required')
    manifest, receipt = Path(data['file_manifest']), Path(data['receipt'])
    if file_hash(manifest) != data['file_manifest_sha256'] or file_hash(receipt) != data['receipt_sha256']:
        raise NativeBackendError('Private JIT runtime provenance differs')
    root = Path(data['package_root']).resolve(strict=True)
    for entry in map(json.loads, manifest.open()):
        path = root / entry['relative_path']
        if not path.resolve(strict=True).is_relative_to(root) or file_hash(path) != entry['sha256']:
            raise NativeBackendError('Pinned JIT runtime file bytes differ')
    return data


def prepare_numba(code, work, runtime):
    work, root = Path(work).resolve(), Path(runtime['package_root']).resolve()
    inner = prepare_python(code, work)
    cache = work / 'jit-cache'
    cache.mkdir()
    inner['paths'].extend([(str(root), READ), (str(cache), ALL_FS & ~EXEC)])
    inner['command'][5] = 'import sys;sys.path.insert(0,' + repr(str(root)) + ');' + PYTHON_BOOTSTRAP
    inner['environment'].update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
        NUMBA_NUM_THREADS='1', NUMBA_CACHE_DIR=str(cache))
    inner.update(memory_bytes=2 * 1024**3, cpu_seconds=20, file_bytes=2 * 1024**2)
    # Reuse the already-tested bounded pipe transport unchanged. A trusted
    # bootstrap runs inside the outer compile-mode restrictions, then installs
    # the stricter JIT runtime filter before exec of the target interpreter.
    # Only bootstrap script reads are additional in the outer filesystem view;
    # the target inherits the intersection with the tighter inner ruleset.
    outer = {**inner, 'mode': 'compile',
        'paths': inner['paths'] + [(str(LAUNCHER), READ), (str(ROOT / 'scripts/native_sandbox.py'), READ)],
        'command': [str(Path(sys.executable).resolve()), '-I', '-S', '-B', str(LAUNCHER), json.dumps(inner)],
        'private_jit_cache': str(cache)}
    return outer


def run_numba(constraints, data, seconds=30, output_bytes=1024**2, cache_bytes=64 * 1024**2):
    return run_jit_bounded(constraints, data, seconds=seconds, output_bytes=output_bytes, cache_bytes=cache_bytes)
