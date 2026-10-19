"""Separate verified GNU C++ profile; preserve the live ISO/libc++ assessment."""
import json
from pathlib import Path

from .corpus import file_hash
from .native_code import ROOT, NativeBackendError, policy
from .standalone_program_v1 import run_bounded


def gnu_toolchain():
    descriptor = ROOT / '.tools/gnu-toolchain-v1.json'
    if not descriptor.exists():
        raise NativeBackendError('Verified isolated GNU toolchain is not prepared')
    tools = json.loads(descriptor.read_text())
    if tools.get('signed_repository_and_package_hashes_verified') is not True:
        raise NativeBackendError('Verified official GNU package provenance required')
    manifest = Path(tools['file_manifest'])
    if file_hash(manifest) != tools['file_manifest_sha256']:
        raise NativeBackendError('GNU runtime file manifest differs')
    root = Path(tools['toolchain_root']).resolve(strict=True)
    for line in manifest.open():
        entry = json.loads(line)
        path = root / entry['relative_path']
        if not path.resolve(strict=True).is_relative_to(root) or file_hash(path) != entry['sha256']:
            raise NativeBackendError('Isolated GNU software files differ')
    if file_hash(Path(tools['compiler_path'])) != tools['compiler_sha256']:
        raise NativeBackendError('GNU driver bytes differ')
    return tools


def compile_gnu_program(code, work, tools, seconds=90):
    if not isinstance(code, bytes) or not code or len(code) > 32768 or b'\0' in code:
        raise ValueError('Complete original C++ programme size/encoding invalid')
    root, work = Path(tools['toolchain_root']).resolve(), Path(work).resolve()
    (work / 'candidate.cpp').write_bytes(code)
    command = [tools['compiler_path'], '-B' + str(root / 'usr/lib/gcc/x86_64-linux-gnu/11') + '/',
        '-B' + str(root / 'usr/bin') + '/', '--sysroot=' + str(root),
        '-static', '-std=gnu++17', '-O2', '-fno-lto', 'candidate.cpp', '-o', 'program']
    constraints = policy('compile', command, work, root)
    # Executables/libraries are an explicitly read-only verified software tree.
    constraints['environment']['PATH'] = str(root / 'usr/bin')
    constraints['environment']['LD_LIBRARY_PATH'] = str(root / 'usr/lib/x86_64-linux-gnu')
    result = run_bounded(constraints, b'', seconds=seconds)
    result['compiler_profile'] = 'GNU C++17/libstdc++/glibc/static; original source bytes unchanged; 1536MiB runtime address-space cap'
    result['runtime_memory_limit_bytes'] = 1536 * 1024**2
    if result['status'] != 'completed':
        return None, result
    binary = work / 'program'
    if not binary.is_file():
        raise NativeBackendError('GNU compiler succeeded without the expected executable')
    runtime = policy('runtime', [binary], work)
    # Original p03058 contains a 2000*2000*2000-bit static array. Its virtual
    # mapping exceeds the small function-contract runner cap even on tiny
    # published samples. Keep one fixed bounded standalone profile; unchanged
    # filesystem/syscall restrictions still apply before program startup.
    runtime['memory_bytes'] = 1536 * 1024**2
    return runtime, result
