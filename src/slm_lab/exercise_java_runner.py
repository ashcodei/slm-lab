"""Execute complete original Java publisher tests in a pre-exec JVM sandbox."""
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
from .codenet_checkpoint_v1 import FileBindings
from .native_code import NativeBackendError, stop

ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = ROOT / 'scripts/java_sandbox.py'
HARNESS = ROOT / 'scripts/java/OwnedPublishedTests.java'
BLOCK = re.compile(r'^File: ([^\r\n]+)\r?\n```java[ \t]*\r?\n(.*?)^```[ \t]*$', re.M | re.S)
SOURCE_PATH = re.compile(r'src/main/java/(?:[A-Za-z_]\w*/)*[A-Za-z_]\w*\.java', re.ASCII)
PREFIX = b'SLM_JAVA_INVENTORY\t'


def extract_files(response, expected=None):
    if len(response.encode()) > 100000:
        raise ValueError('Complete Java source-size budget exceeded')
    files = {}
    for match in BLOCK.finditer(response):
        path, body = match.groups()
        if not SOURCE_PATH.fullmatch(path) or path in files or not body.strip():
            raise ValueError('Unsafe, duplicate or empty Java solution path')
        files[path] = body
    if not files or BLOCK.sub('', response).strip():
        raise ValueError('Return complete labeled Java solution files without extra content')
    if expected is not None and set(files) != set(expected):
        raise ValueError('Java solution files differ from the original requested interface')
    return files


def inventory(raw):
    if len(raw) > 4 * 1024**2:
        raise ValueError('Oversized Java inventory output')
    reports = [line[len(PREFIX):] for line in raw.splitlines() if line.startswith(PREFIX)]
    if len(reports) != 1:
        raise ValueError('Exactly one complete Java test inventory required')
    record = json.loads(reports[0])
    if (set(record) != {'mode', 'junit4_version', 'registered', 'started', 'passed', 'failed', 'skipped', 'container_errors'}
            or record['mode'] not in {'junit4', 'jupiter'} or record['junit4_version'] != '4.13'):
        raise ValueError('Original pinned JUnit inventory/version required')
    for key in ('registered', 'started', 'passed', 'failed', 'skipped', 'container_errors'):
        values = record[key]
        if not isinstance(values, list) or len(values) > 4096 or any(not isinstance(v, str) or not v or len(v) > 4096 for v in values):
            raise ValueError('Invalid original Java event identities')
        if len(values) != len(set(values)):
            raise ValueError('Repeated Java event identity')
    registered = set(record['registered'])
    if not registered or any(not set(record[k]) <= registered for k in ('started', 'passed', 'failed', 'skipped')):
        raise ValueError('Missing or undeclared Java test identities')
    if (set(record['passed']) & (set(record['failed']) | set(record['skipped']))
            or set(record['failed']) & set(record['skipped'])):
        raise ValueError('Conflicting Java test outcomes')
    return dict(cases=sorted(registered), started_cases=sorted(record['started']), passed_cases=sorted(record['passed']), failed=record['failed'],
        skipped=record['skipped'], container_errors=record['container_errors'], framework=record['mode'],
        junit4_version=record['junit4_version'], all_passed=(set(record['started']) == registered
            and set(record['passed']) == registered and not record['failed'] and not record['skipped'] and not record['container_errors']))


def jvm_flags(cache):
    return ['-XX:ActiveProcessorCount=1', '-XX:-UsePerfData', '-XX:-UseContainerSupport', '-XX:+UseSerialGC',
        '-XX:CompressedClassSpaceSize=64m', '-XX:ReservedCodeCacheSize=32m', '-XX:-CreateCoredumpOnCrash',
        '-Xss512k', '-Xms16m', '-Xmx192m', '-Dfile.encoding=UTF-8', '-Duser.language=en', '-Duser.country=US',
        '-Duser.timezone=UTC', '-Djava.io.tmpdir=' + str(cache), '-Duser.home=' + str(cache),
        '-XX:ErrorFile=' + str(cache / 'hs_err_pid%p.log')]


def policy(work, tools, jars, command, mode):
    work = Path(work).resolve()
    paths = [(str(Path(tools['jdk_root']).resolve()), 13), (str(work / 'cache'), 32766),
        (str(work / 'build'), 32766 if mode == 'compile' else 12)]
    if mode == 'compile':
        paths.append((str(work / 'src'), 12))
    paths += [(str(Path(p).resolve()), 4) for p in jars]
    for path in ('/lib/x86_64-linux-gnu', '/usr/lib/x86_64-linux-gnu', '/lib64', '/etc/ld.so.cache',
            '/dev/null', '/dev/urandom', '/proc/self/stat', '/proc/self/status', '/proc/self/maps',
            '/proc/self/task', '/proc/meminfo', '/proc/cpuinfo', '/proc/stat', '/sys/devices/system/cpu'):
        if Path(path).exists():
            paths.append((path, 6 if path == '/dev/null' else 13 if Path(path).is_dir() and '/lib' in path else 12))
    return dict(mode=mode, cwd=str(work / 'cache'), command=command, paths=paths,
        environment=dict(PATH='/usr/bin:/bin', LANG='C.UTF-8', TZ='UTC', TMPDIR=str(work / 'cache')),
        address_bytes=3 * 1024**3, cpu_seconds=90 if mode == 'compile' else 20, file_bytes=8 * 1024**2)


def confined(work, tools, jars, command, mode):
    output, errors = work / ('stdout-' + mode), work / ('stderr-' + mode)
    with output.open('wb') as stdout, errors.open('wb') as stderr:
        process = subprocess.Popen([sys.executable, '-I', str(LAUNCHER), json.dumps(policy(work, tools, jars, command, mode))],
            stdout=stdout, stderr=stderr, stdin=subprocess.DEVNULL, close_fds=True, start_new_session=True,
            env={'PATH': '/usr/bin:/bin', 'LANG': 'C.UTF-8'})
        started, peak, limit = time.monotonic(), 0, None
        while process.poll() is None:
            try:
                status = Path(f'/proc/{process.pid}/status').read_text()
                peak = max(peak, next(int(s.split()[1]) * 1024 for s in status.splitlines() if s.startswith('VmRSS:')))
            except (OSError, StopIteration):
                pass
            if time.monotonic() - started > (120 if mode == 'compile' else 30):
                limit = 'Java wall-time budget exceeded'
            if peak > 512 * 1024**2:
                limit = 'Java RSS budget exceeded'
            if output.stat().st_size + errors.stat().st_size > 8 * 1024**2:
                limit = 'Java output budget exceeded'
            if limit:
                stop(process)
                break
            time.sleep(.02)
    diagnostic = errors.read_bytes()[-6000:].decode(errors='replace')
    if 'SANDBOX_ERROR:' in diagnostic:
        raise NativeBackendError(diagnostic)
    return dict(returncode=process.returncode, diagnostic=diagnostic[-3000:],
        peak_rss_bytes=peak, limit=limit, output=output.read_bytes())


class PublishedSuites:
    def __init__(self, cache):
        self.cache = Path(cache)
        self.source = json.loads((self.cache / 'source.json').read_text())
        if file_hash(self.cache / 'source.zip') != self.source['archive_sha256']:
            raise ValueError('Original Java archive changed')
        self.archive = zipfile.ZipFile(self.cache / 'source.zip')
        self.prefix = self.archive.namelist()[0].split('/')[0] + '/'
        self.tools = json.loads((ROOT / '.tools/java-toolchain-v1.json').read_text())
        self.framework = json.loads((ROOT / '.tools/java-test-framework-v1.json').read_text())
        files = {**self.tools['all_installed_files_sha256'], **self.framework['files_sha256']}
        for path in [self.cache / 'source.json', self.cache / 'source.zip',
                ROOT / '.tools/java-toolchain-v1.json', ROOT / '.tools/java-test-framework-v1.json',
                HARNESS, LAUNCHER, ROOT / 'scripts/native_sandbox.py', Path(__file__)]:
            files[str(path)] = file_hash(path)
        self.guard = FileBindings(files)
        self.guard.check(full=True)

    def close(self):
        self.archive.close()

    def read(self, name):
        if Path(name).is_absolute() or '..' in Path(name).parts:
            raise ValueError('Unsafe original Java archive path')
        info = self.archive.getinfo(self.prefix + name)
        if info.file_size > 1024**2:
            raise ValueError('Oversized original Java source')
        return self.archive.read(info)

    def config(self, row):
        if row['revision'] != self.source['revision'] or row['source_archive_sha256'] != self.source['archive_sha256']:
            raise ValueError('Original Java revision/archive differs')
        item = next(r for r in row['source_files'] if r['path'].endswith('/.meta/config.json'))
        raw = self.read(item['path'])
        if hashlib.sha256(raw).hexdigest() != item['sha256']:
            raise ValueError('Original Java interface config differs')
        return item['path'].split('/.meta/')[0] + '/', json.loads(raw)['files']

    def layout(self, row):
        base, config = self.config(row)
        originals = config.get('exemplar', config.get('example', []))
        references = {Path(name).name: name for name in originals}
        if len(references) != len(originals):
            raise ValueError('Ambiguous original Java reference filename')
        expected, provided = {}, {}
        for name in config['solution']:
            original = references.pop(Path(name).name, None)
            if original is None:
                # The publisher includes Signal.java among solution files but
                # omits it from the example list. It is already complete in the
                # supplied starter, byte-identical to its archived reference.
                raw = self.read(base + name)
                archived_reference = self.read(base + '.meta/src/reference/java/' + Path(name).name)
                item = next(r for r in row['source_files'] if r['path'] == base + name)
                if raw != archived_reference or hashlib.sha256(raw).hexdigest() != item['sha256']:
                    raise ValueError('Unlisted Java solution must be an unchanged complete publisher starter')
                provided[name] = raw
                continue
            expected[name] = original
        if references or not expected:
            raise ValueError('Every original Java reference must map to its declared solution basename')
        return base, config, expected, provided

    def reference(self, row):
        base, config, expected, provided = self.layout(row)
        delivered = extract_files(row['response'], expected)
        for name, original in expected.items():
            raw = self.read(base + original)
            item = next(r for r in row['source_files'] if r['path'] == base + original)
            if hashlib.sha256(raw).hexdigest() != item['sha256'] or delivered[name].strip() != raw.decode().strip():
                raise ValueError('Prepared Java reference differs from its original publisher body')
        return delivered

    def suite(self, row):
        base, config, expected, provided = self.layout(row)
        tests = {}
        for item in row['published_test_artifacts']:
            raw = self.read(item['path'])
            if hashlib.sha256(raw).hexdigest() != item['sha256']:
                raise ValueError('Original published Java tests changed')
            relative = item['path'][len(base):] if item['path'].startswith(base) else ''
            if not re.fullmatch(r'src/test/java/(?:[A-Za-z_]\w*/)*[A-Za-z_]\w*\.java', relative, re.ASCII):
                raise ValueError('Original Java test path differs')
            tests[relative] = raw
        # Preserve every publisher test/support source under src/test/java.
        for name in self.archive.namelist():
            if name.startswith(self.prefix + base + 'src/test/java/') and name.endswith('.java'):
                relative = name[len(self.prefix + base):]
                raw = self.read(base + relative)
                if relative in tests and tests[relative] != raw:
                    raise ValueError('Recorded and complete original Java test bytes differ')
                tests[relative] = raw
        editors = {**{name: self.read(base + name) for name in config.get('editor', [])}, **provided}
        if any(not SOURCE_PATH.fullmatch(name) for name in editors):
            raise ValueError('Unsafe original Java provided supporting file')
        build = self.read(base + 'build.gradle')
        legacy = b'junit:junit:4.13' in build
        if ((legacy and b'assertj-core:3.15.0' not in build)
                or (not legacy and (b'junit-bom:5.10.0' not in build or b'assertj-core:3.25.1' not in build))):
            raise ValueError('Original Java publisher dependency declarations differ')
        classes = []
        for name, raw in tests.items():
            if re.search(rb'@Test\b', raw):
                package = re.search(rb'^package\s+([\w.]+)\s*;', raw, re.M)
                classes.append((package[1].decode() + '.' if package else '') + Path(name).stem)
        if not classes or len(set(classes)) != len(classes) or not tests:
            raise ValueError('Every complete original Java test class required')
        return dict(tests=tests, editors=editors, expected_solution_files=list(expected),
            test_sha256={name: hashlib.sha256(raw).hexdigest() for name, raw in tests.items()},
            editor_sha256={name: hashlib.sha256(raw).hexdigest() for name, raw in editors.items()},
            build_sha256=hashlib.sha256(build).hexdigest(), test_classes=sorted(classes),
            mode='junit4' if legacy else 'jupiter', assertj_version='3.15.0' if legacy else '3.25.1')

    def run(self, files, suite, expected=None):
        self.guard.check()
        if set(files) != set(suite['expected_solution_files']) or set(files) & (suite['editors'].keys() | suite['tests'].keys()):
            raise ValueError('Only original declared Java solution files may be delivered')
        if any(not SOURCE_PATH.fullmatch(name) for name in files) or sum(len(s.encode()) for s in files.values()) > 100000:
            raise ValueError('Invalid original Java solution paths/size')
        artifacts = self.framework['artifacts']
        def jar(name, version):
            return next(a['jar'] for a in artifacts if a['artifact'] == name and a['version'] == version)
        jars = [jar('junit', '4.13'), jar('hamcrest-core', '1.3'), jar('assertj-core', suite['assertj_version']),
            jar('junit-platform-console-standalone', '1.10.0')]
        with tempfile.TemporaryDirectory(prefix='slm-exercise-java-') as directory:
            work = Path(directory)
            for name in ('src', 'cache', 'build'):
                (work / name).mkdir()
            bodies = {**{name: text.encode() for name, text in files.items()}, **suite['editors'], **suite['tests']}
            for name, raw in bodies.items():
                path = work / 'src' / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(raw)
            harness = work / 'src/OwnedPublishedTests.java'
            harness.write_bytes(HARNESS.read_bytes())
            sources = [str(work / 'src' / name) for name in bodies] + [str(harness)]
            classpath = os.pathsep.join(jars)
            compile_command = [self.tools['javac_path']] + ['-J' + flag for flag in jvm_flags(work / 'cache')]
            compile_command += ['-proc:none', '-encoding', 'UTF-8', '--release', '21', '-cp', classpath, '-d', str(work / 'build')] + sources
            compile_result = confined(work, self.tools, jars, compile_command, 'compile')
            result = {k: v for k, v in compile_result.items() if k != 'output'}
            result.update(compilation_passed=compile_result['returncode'] == 0 and compile_result['limit'] is None,
                published_tests_passed=False)
            if not result['compilation_passed']:
                result['compiler_stdout'] = compile_result['output'][-3000:].decode(errors='replace')
                return result
            command = [self.tools['java_path']] + jvm_flags(work / 'cache') + ['-cp', classpath + os.pathsep + str(work / 'build'),
                'slm.owned.OwnedPublishedTests', suite['mode']] + suite['test_classes']
            runtime = confined(work, self.tools, jars, command, 'runtime')
            result['execution'] = {k: v for k, v in runtime.items() if k != 'output'}
            for name, raw in {**suite['tests'], **suite['editors']}.items():
                if file_hash(work / 'src' / name) != hashlib.sha256(raw).hexdigest():
                    raise NativeBackendError('Original confined Java test/support bytes altered')
            try:
                measured = inventory(runtime['output'])
            except (ValueError, KeyError, TypeError) as error:
                result['runtime_error'] = 'Incomplete Java test inventory: ' + str(error)
                result['runtime_stdout'] = runtime['output'][-3000:].decode(errors='replace')
                return result
            matched = expected is None or measured['cases'] == expected['cases']
            result.update(inventory=measured, raw_inventory_output=runtime['output'].decode(errors='replace'),
                report_sha256=hashlib.sha256(runtime['output']).hexdigest(),
                reference_inventory_matched=matched, published_tests_passed=(runtime['returncode'] == 0
                    and runtime['limit'] is None and measured['all_passed'] and matched))
            self.guard.check()
            return result
