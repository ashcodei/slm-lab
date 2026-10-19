"""Compile full Go modules, then execute exact original tests in confinement."""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
import zipfile

from .corpus import file_hash
from .native_code import NativeBackendError, stop

ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = ROOT / 'scripts/go_sandbox.py'
BLOCK = re.compile(r'^File: ([^\r\n]+)\r?\n```(?:go)?[ \t]*\r?\n(.*?)^```[ \t]*$', re.M | re.S)


def extract_files(response, expected=None):
    if len(response.encode()) > 100000: raise ValueError('Go module source budget exceeded')
    files = {}
    for match in BLOCK.finditer(response):
        name, source = match.groups()
        if (not re.fullmatch(r'[A-Za-z_][A-Za-z_0-9-]*\.go', name) or name.endswith('_test.go')
                or name in files or not source.strip()): raise ValueError('Unsafe, repeated or empty Go solution filename')
        files[name] = source
    if not files or BLOCK.sub('', response).strip(): raise ValueError('Return complete labeled Go files only')
    if expected is not None and set(files) != set(expected): raise ValueError('Go files differ from original requested interface')
    return files


def toolchain():
    tools = json.loads((ROOT / '.tools/go-toolchain-v1.json').read_text())
    sdk = Path(tools['toolchain_root'])
    for name, digest in tools['files_sha256'].items():
        if file_hash(sdk / name) != digest: raise NativeBackendError('Pinned Go SDK changed: ' + name)
    if file_hash(tools['test2json_binary']) != tools['test2json_sha256']:
        raise NativeBackendError('Trusted Go event converter changed')
    return tools


def policy(work, tools, command, mode):
    work = Path(work).resolve(); sdk = Path(tools['toolchain_root']).resolve()
    cache = Path(tools['compile_cache_root']); cache.mkdir(exist_ok=True)
    paths = [(str(work / 'src'), 12), (str(work / 'cache'), 32766)]
    if mode == 'compile':
        paths += [(str(work / 'build'), 32766), (str(cache), 32766), (str(sdk), 13)]
        for path in ('/lib/x86_64-linux-gnu', '/usr/lib/x86_64-linux-gnu', '/lib64', '/etc/ld.so.cache'):
            if Path(path).exists(): paths.append((str(Path(path).resolve()), 13))
    else: paths.append((str(work / 'build/publisher.test'), 5))
    for path in ('/dev/null', '/dev/urandom', '/proc/self/stat', '/proc/self/status', '/proc/self/maps',
                 '/proc/meminfo', '/proc/cpuinfo', '/proc/stat'):
        if Path(path).exists(): paths.append((path, 6 if path == '/dev/null' else 4))
    env = {'PATH': str(sdk / 'bin'), 'LANG': 'C.UTF-8', 'TZ': 'UTC', 'GOROOT': str(sdk), 'GOENV': 'off',
        'GOOS': 'linux', 'GOARCH': 'amd64', 'CGO_ENABLED': '0', 'GOTOOLCHAIN': 'local', 'GOWORK': 'off',
        'GOPROXY': 'off', 'GOSUMDB': 'off', 'GOCACHE': str(cache), 'GOTMPDIR': str(work / 'cache'),
        'TMPDIR': str(work / 'cache'), 'GOMAXPROCS': '1', 'GOMEMLIMIT': '192MiB'}
    if mode == 'compile': env['GODEBUG'] = 'asyncpreemptoff=1'
    return {'mode': mode, 'cwd': str(work / 'src'), 'command': command, 'paths': paths, 'environment': env,
        'address_bytes': 8 * 1024**3, 'cpu_seconds': 120 if mode == 'compile' else 10,
        'file_bytes': 64 * 1024**2 if mode == 'compile' else 4 * 1024**2}


def confined(work, tools, command, mode):
    output = Path(work) / ('stdout-' + mode); errors = Path(work) / ('stderr-' + mode)
    with output.open('wb') as stdout, errors.open('wb') as stderr:
        process = subprocess.Popen([sys.executable, '-I', str(LAUNCHER), json.dumps(policy(work, tools, command, mode))],
            stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr, start_new_session=True, close_fds=True,
            env={'PATH': '/usr/bin:/bin', 'LANG': 'C.UTF-8'})
        started = time.monotonic(); peak = 0; limit = None
        while process.poll() is None:
            try:
                text = Path(f'/proc/{process.pid}/status').read_text()
                rss = next(int(line.split()[1]) * 1024 for line in text.splitlines() if line.startswith('VmRSS:'))
                peak = max(peak, rss)
            except (OSError, StopIteration): pass
            if time.monotonic() - started > (180 if mode == 'compile' else 15): limit = 'Go wall-time budget exceeded'
            if mode == 'runtime' and peak > 256 * 1024**2: limit = 'Go runtime RSS budget exceeded'
            if limit: stop(process); break
            time.sleep(.02)
    diagnostic = errors.read_bytes()[-6000:].decode(errors='replace')
    if 'SANDBOX_ERROR:' in diagnostic: raise NativeBackendError(diagnostic)
    return {'returncode': process.returncode, 'diagnostic': diagnostic[-2500:], 'peak_rss_bytes': peak,
            'limit': limit, 'output': output.read_bytes()}


def inventory(events):
    runs = [r['Test'] for r in events if r['Action'] == 'run' and 'Test' in r]
    passed = [r['Test'] for r in events if r['Action'] == 'pass' and 'Test' in r]
    skipped = [r['Test'] for r in events if r['Action'] == 'skip' and 'Test' in r]
    if not runs or len(set(runs)) != len(runs): raise ValueError('Missing or repeated original Go test cases')
    package = [r for r in events if 'Test' not in r and r['Action'] in ('pass', 'fail', 'skip')]
    all_passed = len(package) == 1 and package[0]['Action'] == 'pass' and sorted(passed) == sorted(runs) and not skipped
    return {'cases': sorted(runs), 'passed_cases': sorted(passed), 'skipped': skipped, 'all_passed': all_passed}


class PublishedSuites:
    def __init__(self, cache):
        cache = Path(cache); self.source = json.loads((cache / 'source.json').read_text())
        if file_hash(cache / 'source.zip') != self.source['archive_sha256']: raise ValueError('Original Go archive changed')
        self.archive = zipfile.ZipFile(cache / 'source.zip'); self.prefix = self.archive.namelist()[0].split('/')[0] + '/'
        self.tools = toolchain()

    def close(self): self.archive.close()

    def reference(self, row):
        files = extract_files(row['response'])
        item = next(r for r in row['source_files'] if r['path'].endswith('/.meta/config.json'))
        raw = self.archive.read(self.prefix + item['path'])
        if hashlib.sha256(raw).hexdigest() != item['sha256']: raise ValueError('Original Go interface config changed')
        config = json.loads(raw)['files']; base = item['path'].split('/.meta/')[0] + '/'
        references = config.get('exemplar', config.get('example', [])); solutions = config['solution']
        if len(solutions) != len(references) or set(files) != set(solutions): raise ValueError('Original Go source/interface mapping differs')
        for name, source_name in zip(solutions, references):
            body = self.archive.read(self.prefix + base + source_name)
            record = next(r for r in row['source_files'] if r['path'] == base + source_name)
            if hashlib.sha256(body).hexdigest() != record['sha256'] or files[name].strip() != body.decode().strip():
                raise ValueError('Prepared Go teacher body differs from exact publisher implementation')
        return files

    def suite(self, row):
        if row['revision'] != self.source['revision'] or row['source_archive_sha256'] != self.source['archive_sha256']:
            raise ValueError('Original Go source revision differs')
        tests = {}
        for artifact in row['published_test_artifacts']:
            if not artifact['path'].endswith('.go'): continue
            body = self.archive.read(self.prefix + artifact['path'])
            if hashlib.sha256(body).hexdigest() != artifact['sha256']: raise ValueError('Original Go test source changed')
            name = Path(artifact['path']).name
            if name in tests: raise ValueError('Repeated Go publisher test filename')
            tests[name] = body
        config = next(r for r in row['source_files'] if r['path'].endswith('/.meta/config.json'))
        base = config['path'].split('/.meta/')[0]
        # Some original publisher configurations omit generated cases_test.go
        # helpers. Preserve every direct test file from the pinned archive.
        parent = self.prefix + base + '/'
        for member in self.archive.namelist():
            relative = member[len(parent):] if member.startswith(parent) else None
            if relative and '/' not in relative and relative.endswith('_test.go'):
                body = self.archive.read(member)
                if relative in tests and tests[relative] != body: raise ValueError('Recorded/original Go test bytes differ')
                tests[relative] = body
        module = self.archive.read(self.prefix + base + '/go.mod')
        if not tests or b'require' in module: raise ValueError('Original Go suite is missing or requires external dependencies')
        tests['go.mod'] = module
        return {'tests': tests, 'test_sha256': {name: hashlib.sha256(raw).hexdigest() for name, raw in tests.items()}}

    def run(self, files, suite, expected=None):
        with tempfile.TemporaryDirectory(prefix='slm-exercise-go-') as directory:
            work = Path(directory)
            for name in ('src', 'cache', 'build'): (work / name).mkdir()
            for name, source in files.items():
                if not re.fullmatch(r'[A-Za-z_][A-Za-z_0-9-]*\.go', name) or name in suite['tests']:
                    raise ValueError('Unsafe Go solution filename')
                (work / 'src' / name).write_text(source)
            for name, raw in suite['tests'].items(): (work / 'src' / name).write_bytes(raw)
            binary = work / 'build/publisher.test'; sdk = Path(self.tools['toolchain_root'])
            compile_result = confined(work, self.tools, [str(sdk / 'bin/go'), 'test', '-c', '-p', '1', '-vet=off', '-o', str(binary), '.'], 'compile')
            result = {key: value for key, value in compile_result.items() if key != 'output'}
            result.update(compilation_passed=compile_result['returncode'] == 0 and compile_result['limit'] is None,
                          published_tests_passed=False)
            if not result['compilation_passed']: return result
            runtime = confined(work, self.tools, [str(binary), '-test.v=test2json', '-test.timeout=10s', '-test.count=1', '-test.run=.'], 'runtime')
            result['execution'] = {key: value for key, value in runtime.items() if key != 'output'}
            for name, digest in suite['test_sha256'].items():
                if file_hash(work / 'src' / name) != digest: raise NativeBackendError('Original confined Go test source altered')
            converted = subprocess.run([self.tools['test2json_binary'], '-p', 'published'], input=runtime['output'],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=10, check=True)
            try: measured = inventory([json.loads(line) for line in converted.stdout.splitlines()])
            except (ValueError, KeyError, TypeError) as error:
                result['runtime_error'] = 'Incomplete Go test event inventory: ' + str(error); return result
            result['inventory'] = measured
            result['report_sha256'] = hashlib.sha256(converted.stdout).hexdigest()
            matched = expected is None or measured['cases'] == expected['cases']
            result['reference_inventory_matched'] = matched
            result['published_tests_passed'] = (runtime['returncode'] == 0 and runtime['limit'] is None
                                                and measured['all_passed'] and matched)
            return result
