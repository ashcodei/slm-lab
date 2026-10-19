"""Bounded JIT transport with a private-cache quota, including unlinked files."""
import os
from pathlib import Path
import selectors
import subprocess
import time

from .native_code import NativeBackendError, launch, stop


def cache_usage(cache, pid, maximum_files):
    cache = Path(cache)
    inodes, count = {}, 0
    pending = [cache]
    while pending:
        directory = pending.pop()
        try:
            entries = os.scandir(directory)
        except FileNotFoundError:
            continue
        with entries:
            for entry in entries:
                count += 1
                if count > maximum_files:
                    return sum(inodes.values()), count, True
                try:
                    if entry.is_symlink():
                        raise ValueError('Private JIT cache must not contain symlinks')
                    if entry.is_dir(follow_symlinks=False):
                        pending.append(Path(entry.path))
                    else:
                        info = entry.stat(follow_symlinks=False)
                        inodes[(info.st_dev, info.st_ino)] = info.st_size
                except FileNotFoundError:
                    pass  # Atomic cache replacement can race this observation.
    # Unlinking an open cache file must not hide its allocated bytes. The target
    # cannot fork and its descriptor cap is32, so this scan also stays bounded.
    inspection_available = True
    try:
        descriptors = list(Path(f'/proc/{pid}/fd').iterdir())
    except FileNotFoundError:
        descriptors = []
    except PermissionError:
        # Linux may temporarily deny this ptrace-gated view during exec. The
        # caller permits a bounded startup grace, never persistent blindness.
        descriptors, inspection_available = [], False
    for descriptor in descriptors:
        try:
            target = os.readlink(descriptor)
            if target.startswith(str(cache) + '/'):
                info = descriptor.stat()
                inodes[(info.st_dev, info.st_ino)] = info.st_size
        except FileNotFoundError:
            pass
        except PermissionError:
            inspection_available = False
    return sum(inodes.values()), max(count, len(inodes)), inspection_available


def run_jit_bounded(constraints, data, seconds=30, output_bytes=1024**2, cache_bytes=64 * 1024**2):
    if (not isinstance(data, bytes) or len(data) > 4 * 1024**2 or not 0 < seconds <= 120
        or not 0 < output_bytes <= 4 * 1024**2 or not 0 < cache_bytes <= 64 * 1024**2):
        raise ValueError('JIT input/time/output/cache budget exceeded')
    cache = Path(constraints['private_jit_cache']).resolve(strict=True)
    if cache.parent != Path(constraints['cwd']).resolve() or cache.name != 'jit-cache':
        raise ValueError('Owned per-program JIT cache required')
    started, sent, total = time.perf_counter(), 0, 0
    deadline, reason, early, peak, unavailable_since = started + seconds, None, False, 0, None
    buffers = {'stdout': bytearray(), 'stderr': bytearray()}
    process = launch(constraints, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        with selectors.DefaultSelector() as selector:
            for stream, name in [(process.stdout, 'stdout'), (process.stderr, 'stderr')]:
                os.set_blocking(stream.fileno(), False)
                selector.register(stream, selectors.EVENT_READ, name)
            os.set_blocking(process.stdin.fileno(), False)
            if data:
                selector.register(process.stdin, selectors.EVENT_WRITE, 'stdin')
            else:
                process.stdin.close()
            while True:
                used, files, inspectable = cache_usage(cache, process.pid, 256)
                peak = max(peak, used)
                if inspectable:
                    unavailable_since = None
                elif unavailable_since is None:
                    unavailable_since = time.perf_counter()
                elif time.perf_counter() - unavailable_since > .5:
                    reason = 'private_cache_inspection_unavailable'
                    break
                if used > cache_bytes or files > 256:
                    reason = 'private_cache_budget_exceeded'
                    break
                if not selector.get_map() and process.poll() is not None:
                    break
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    reason = 'wall_deadline_exceeded'
                    break
                # Poll the private filesystem quota even if the target is silent
                # or closes both output streams before continuing to write files.
                for key, _ in selector.select(min(remaining, .02)):
                    if key.data == 'stdin':
                        try:
                            sent += os.write(key.fd, data[sent:sent + 65536])
                        except BlockingIOError:
                            continue
                        except BrokenPipeError:
                            early = True
                        if sent == len(data) or early:
                            selector.unregister(key.fileobj)
                            key.fileobj.close()
                    else:
                        try:
                            block = os.read(key.fd, 65536)
                        except BlockingIOError:
                            continue
                        if not block:
                            selector.unregister(key.fileobj)
                            key.fileobj.close()
                            continue
                        buffers[key.data].extend(block[:max(0, output_bytes - total)])
                        total += len(block)
                        if total > output_bytes:
                            reason = 'combined_output_budget_exceeded'
                            break
                if reason:
                    break
        if reason is None:
            reason = 'completed' if process.returncode == 0 else 'runtime_exit'
    finally:
        if process.poll() is None:
            stop(process)
        else:
            process.wait()
        for stream in [process.stdin, process.stdout, process.stderr]:
            if not stream.closed:
                stream.close()
    if process.returncode == 126 and b'SANDBOX_ERROR:' in buffers['stderr']:
        raise NativeBackendError(bytes(buffers['stderr']).decode(errors='replace'))
    return {'status': reason, 'returncode': process.returncode, 'stdout': bytes(buffers['stdout']),
        'stderr': bytes(buffers['stderr']), 'seconds': time.perf_counter() - started,
        'input_bytes': len(data), 'input_bytes_sent': sent, 'input_pipe_closed_early': early,
        'combined_output_bytes_observed': total, 'private_cache_peak_bytes': peak,
        'private_cache_budget_bytes': cache_bytes}
