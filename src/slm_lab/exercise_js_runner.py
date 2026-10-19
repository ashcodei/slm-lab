"""Original JavaScript Jest suites under a separate kernel-confined Node runtime."""
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile

from .corpus import file_hash
from .native_code import NativeBackendError, stop

ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = ROOT / 'scripts/javascript_sandbox.py'
READ = (1 << 2) | (1 << 3)
EXEC = 1
FILE_BLOCK = re.compile(r'^File: ([^\r\n]+)\r?\n```(?:javascript|js)?[ \t]*\r?\n(.*?)^```[ \t]*$', re.M | re.S)
ENABLE_ALL = '''module.exports = class FullPublisherEnvironment extends NodeEnvironment {
  handleTestEvent(event, state) {
    if (event.name !== 'run_start') return;
    function enable(block) {
      if (block.mode === 'skip' || block.mode === 'only') block.mode = undefined;
      for (const child of block.children) {
        if (child.type === 'describeBlock') enable(child);
        else if (child.mode === 'skip' || child.mode === 'only') child.mode = undefined;
      }
    }
    enable(state.rootDescribeBlock);
    state.hasFocusedTests = false;
  }
};
'''


def extract_files(response, expected=None):
    if len(response.encode()) > 100000: raise ValueError('Generated JavaScript module exceeds source budget')
    files = {}
    for match in FILE_BLOCK.finditer(response):
        name, source = match.groups()
        if not re.fullmatch(r'[a-zA-Z_][a-zA-Z_0-9-]*\.js', name) or name.endswith(('.spec.js', '.test.js')) or name in files or not source.strip():
            raise ValueError('Unsafe, repeated or empty JavaScript solution filename')
        files[name] = source
    if not files or FILE_BLOCK.sub('', response).strip(): raise ValueError('Return complete labeled JavaScript files only')
    if expected is not None and set(files) != set(expected): raise ValueError('JavaScript solution files differ from requested interface')
    return files


def toolchain():
    path = ROOT / '.tools/javascript-toolchain-v1.json'; metadata = json.loads(path.read_text())
    root = Path(metadata['toolchain_root'])
    if (file_hash(metadata['node_path']) != metadata['node_sha256']
            or file_hash(root / 'package-lock.json') != metadata['package_lock_sha256']): raise NativeBackendError('JavaScript toolchain identity changed')
    for name, digest in metadata['dependency_files_sha256'].items():
        if file_hash(root / name) != digest: raise NativeBackendError('Pinned Jest dependency changed: ' + name)
    return metadata


def policy(work, tools, command):
    work = Path(work).resolve(); node = Path(tools['node_path']).resolve()
    paths = [(str(work), READ), (str(work / 'cache'), (1 << 15) - 2),
             (str(Path(tools['toolchain_root']).resolve()), READ), (str(node), EXEC | (1 << 2))]
    for path in ['/lib/x86_64-linux-gnu', '/usr/lib/x86_64-linux-gnu', '/lib64', '/etc/ld.so.cache']:
        if Path(path).exists(): paths.append((str(Path(path).resolve()), READ | EXEC))
    for path in ['/dev/null', '/dev/urandom', '/proc/self/stat', '/proc/self/status', '/proc/self/maps', '/proc/meminfo', '/proc/cpuinfo', '/proc/stat']:
        if Path(path).exists(): paths.append((path, (1 << 2) | ((1 << 1) if path == '/dev/null' else 0)))
    return {'cwd': str(work), 'command': list(map(str, command)), 'paths': paths,
            'environment': {'PATH': str(node.parent), 'LANG': 'C.UTF-8', 'TZ': 'UTC', 'TMPDIR': str(work / 'cache'),
                            'NODE_DISABLE_COLORS': '1', 'CI': 'true'}}


def inventory(report):
    suites = report['testResults']
    if len(suites) != 1: raise ValueError('Expected one complete original JavaScript suite')
    cases = suites[0]['assertionResults']; names = [row['fullName'] for row in cases]
    if not names or len(set(names)) != len(names) or len(names) != report['numTotalTests']:
        raise ValueError('Missing or repeated JavaScript published cases')
    return {'cases': sorted(names), 'assertions': sum(row.get('numPassingAsserts', 0) for row in cases),
            'all_passed': report['success'] is True and report['numPassedTests'] == len(cases)
                          and all(row['status'] == 'passed' for row in cases),
            'skipped': sum(row['status'] in ['pending', 'todo', 'skipped', 'disabled'] for row in cases)}


class PublishedSuites:
    def __init__(self, cache):
        cache = Path(cache); self.source = json.loads((cache / 'source.json').read_text())
        if file_hash(cache / 'source.zip') != self.source['archive_sha256']: raise ValueError('Original JavaScript source archive changed')
        self.archive = zipfile.ZipFile(cache / 'source.zip'); self.prefix = self.archive.namelist()[0].split('/')[0] + '/'
        self.tools = toolchain()

    def close(self): self.archive.close()

    def reference(self, row):
        files = extract_files(row['response'])
        config_item = next(item for item in row['source_files'] if item['path'].endswith('/.meta/config.json'))
        base = config_item['path'].split('/.meta/')[0] + '/'
        raw = self.archive.read(self.prefix + config_item['path'])
        if hashlib.sha256(raw).hexdigest() != config_item['sha256']: raise ValueError('Original JavaScript interface config changed')
        config = json.loads(raw)['files']; solutions = config['solution']
        references = config.get('exemplar', config.get('example', []))
        if len(solutions) != len(references) or set(solutions) != set(files): raise ValueError('Original JavaScript reference/interface mapping differs')
        for name, reference in zip(solutions, references):
            original = self.archive.read(self.prefix + base + reference)
            original_item = next(item for item in row['source_files'] if item['path'] == base + reference)
            if hashlib.sha256(original).hexdigest() != original_item['sha256'] or files[name].strip() != original.decode().strip():
                raise ValueError('Prepared JavaScript teacher body differs from exact publisher reference')
        return files

    def suite(self, row):
        if row['revision'] != self.source['revision'] or row['source_archive_sha256'] != self.source['archive_sha256']:
            raise ValueError('Original JavaScript exercise source identity differs')
        tests = {}
        for artifact in row['published_test_artifacts']:
            if not artifact['path'].endswith('.spec.js'): continue
            raw = self.archive.read(self.prefix + artifact['path'])
            if hashlib.sha256(raw).hexdigest() != artifact['sha256']: raise ValueError('Original JavaScript test bytes changed')
            tests[Path(artifact['path']).name] = raw
        if len(tests) != 1: raise ValueError('Expected one original JavaScript publisher spec')
        return {'tests': tests, 'test_sha256': {name: hashlib.sha256(raw).hexdigest() for name, raw in tests.items()}}

    def run(self, files, suite, expected=None):
        result = {'published_tests_passed': False, 'full_file_delivery': True}
        with tempfile.TemporaryDirectory(prefix='slm-exercise-js-') as directory:
            work = Path(directory); (work / 'cache').mkdir()
            for name, source in files.items():
                if not re.fullmatch(r'[a-zA-Z_][a-zA-Z_0-9-]*\.js', name) or name in suite['tests']: raise ValueError('Unsafe solution filename')
                (work / name).write_text(source)
            for name, raw in suite['tests'].items(): (work / name).write_bytes(raw)
            (work / 'package.json').write_text('{"type":"module"}\n')
            environment = 'const {TestEnvironment: NodeEnvironment} = require(' + json.dumps(str(Path(self.tools['toolchain_root']) / 'node_modules/jest-environment-node')) + ');\n' + ENABLE_ALL
            (work / 'environment.cjs').write_text(environment)
            config = {'rootDir': str(work), 'testEnvironment': str(work / 'environment.cjs'), 'transform': {}, 'watchman': False,
                      'modulePaths': [str(Path(self.tools['toolchain_root']) / 'node_modules')],
                      'cacheDirectory': str(work / 'cache')}
            (work / 'jest.config.json').write_text(json.dumps(config))
            command = [self.tools['node_path'], '--openssl-config=/dev/null', '--max-old-space-size=128',
                       '--experimental-vm-modules', str(Path(self.tools['toolchain_root']) / 'node_modules/jest/bin/jest.js'),
                       '--config', str(work / 'jest.config.json'), '--runInBand', '--no-cache', '--ci', '--json',
                       '--runTestsByPath', str(work / next(iter(suite['tests'])))]
            output, errors = work / 'stdout.json', work / 'stderr.log'
            with output.open('wb') as stdout, errors.open('wb') as stderr:
                process = subprocess.Popen([sys.executable, '-I', str(LAUNCHER), json.dumps(policy(work, self.tools, command))],
                    stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr, env={'PATH': '/usr/bin:/bin', 'LANG': 'C.UTF-8'},
                    close_fds=True, start_new_session=True)
                started = time.monotonic(); peak = 0; limit = None
                while process.poll() is None:
                    try:
                        status = Path(f'/proc/{process.pid}/status').read_text()
                        rss = next(int(line.split()[1]) * 1024 for line in status.splitlines() if line.startswith('VmRSS:'))
                        peak = max(peak, rss)
                    except (OSError, StopIteration): pass
                    if peak > 384 * 1024**2: limit = 'RSS memory budget exceeded'
                    if time.monotonic() - started > 60: limit = 'Wall-time budget exceeded'
                    if limit: stop(process); break
                    time.sleep(.02)
            diagnostic = errors.read_bytes()[-4000:].decode(errors='replace')
            if 'SANDBOX_ERROR:' in diagnostic: raise NativeBackendError(diagnostic)
            result.update(returncode=process.returncode, diagnostic=diagnostic[-1600:], peak_rss_bytes=peak)
            if limit: return {**result, 'runtime_error': limit}
            for name, digest in suite['test_sha256'].items():
                if file_hash(work / name) != digest: raise NativeBackendError('Confined test source was altered')
            try:
                raw = output.read_bytes(); parsed = json.loads(raw); measured = inventory(parsed)
            except (ValueError, KeyError, TypeError) as error:
                return {**result, 'runtime_error': 'Incomplete published test report: ' + str(error)}
            result.update(inventory=measured, report_sha256=hashlib.sha256(raw).hexdigest())
            matched = expected is None or measured['cases'] == expected['cases'] and measured['assertions'] == expected['assertions']
            result['reference_inventory_matched'] = matched
            result['published_tests_passed'] = process.returncode == 0 and measured['all_passed'] and matched
            return result
