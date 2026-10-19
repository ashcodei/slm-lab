"""Compile complete Rust solution crates and execute every original integration test."""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
import tomllib
import zipfile

from .corpus import file_hash
from .native_code import NativeBackendError, stop
from .exercise_code import target_for

ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = ROOT / 'scripts/rust_sandbox.py'
TARGET = 'x86_64-unknown-linux-musl'
BLOCK = re.compile(r'^File: ([^\r\n]+)\r?\n```(?:rust|toml)?[ \t]*\r?\n(.*?)^```[ \t]*$', re.M | re.S)


def extract_files(response, expected=None):
    if len(response.encode()) > 100000: raise ValueError('Rust source budget exceeded')
    files = {}
    for match in BLOCK.finditer(response):
        name, source = match.groups()
        if (not (re.fullmatch(r'src/[A-Za-z_][A-Za-z_0-9]*\.rs', name) or name == 'Cargo.toml') or name in files or not source.strip()):
            raise ValueError('Unsafe, repeated or empty Rust solution file')
        files[name] = source
    if not files or BLOCK.sub('', response).strip(): raise ValueError('Return complete labeled Rust files only')
    if expected is not None and set(files) != set(expected): raise ValueError('Rust files differ from original requested interface')
    return files


def toolchain():
    tools = json.loads((ROOT / '.tools/rust-toolchain-v1.json').read_text()); sdk = Path(tools['toolchain_root'])
    if tools['target'] != TARGET: raise NativeBackendError('Pinned Rust target differs')
    for name, digest in tools['files_sha256'].items():
        if file_hash(sdk / name) != digest: raise NativeBackendError('Pinned Rust compiler/library changed: ' + name)
    return tools


def policy(work, tools, command, mode):
    work = Path(work).resolve(); sdk = Path(tools['toolchain_root']).resolve()
    paths = [(str(work / 'src'), 12)]
    if mode == 'compile':
        paths += [(str(work / 'build'), 32766), (str(sdk), 13)]
        for path in ('/lib/x86_64-linux-gnu', '/usr/lib/x86_64-linux-gnu', '/lib64', '/etc/ld.so.cache'):
            if Path(path).exists(): paths.append((str(Path(path).resolve()), 13))
    else: paths += [(str(work / 'build'), 12), (str(Path(command[0]).resolve()), 5)]
    for path in ('/dev/null', '/dev/urandom', '/proc/self/stat', '/proc/self/status', '/proc/self/maps', '/proc/cpuinfo'):
        if Path(path).exists(): paths.append((path, 6 if path == '/dev/null' else 4))
    return {'mode': mode, 'cwd': str(work / 'src'), 'command': list(map(str, command)), 'paths': paths,
            'environment': {'PATH': str(sdk / 'bin'), 'LANG': 'C.UTF-8', 'TZ': 'UTC', 'TMPDIR': str(work / 'build'),
                            'RUST_BACKTRACE': '0', 'RUST_TEST_THREADS': '1'},
            'memory_bytes': 3 * 1024**3 if mode == 'compile' else 384 * 1024**2,
            'cpu_seconds': 60 if mode == 'compile' else 30,
            'file_bytes': 64 * 1024**2 if mode == 'compile' else 2 * 1024**2}


def confined(work, tools, command, mode):
    work = Path(work); output = work / ('stdout-' + mode); errors = work / ('stderr-' + mode)
    with output.open('wb') as stdout, errors.open('wb') as stderr:
        process = subprocess.Popen([sys.executable, '-I', str(LAUNCHER), json.dumps(policy(work, tools, command, mode))],
            stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr, start_new_session=True, close_fds=True,
            env={'PATH': '/usr/bin:/bin', 'LANG': 'C.UTF-8'})
        started = time.monotonic(); peak = 0; limit = None
        while process.poll() is None:
            try:
                status = Path(f'/proc/{process.pid}/status').read_text()
                rss = next(int(line.split()[1]) * 1024 for line in status.splitlines() if line.startswith('VmRSS:'))
                peak = max(peak, rss)
            except (OSError, StopIteration): pass
            if time.monotonic() - started > (90 if mode == 'compile' else 45): limit = 'Rust wall-time budget exceeded'
            if mode == 'runtime' and peak > 256 * 1024**2: limit = 'Rust runtime RSS budget exceeded'
            if limit: stop(process); break
            time.sleep(.02)
    diagnostic = errors.read_bytes()[-6000:].decode(errors='replace')
    if 'SANDBOX_ERROR:' in diagnostic: raise NativeBackendError(diagnostic)
    return {'returncode': process.returncode, 'diagnostic': diagnostic[-2500:], 'peak_rss_bytes': peak,
            'limit': limit, 'output': output.read_bytes()}


def compiler(tools, edition='2024'):
    sdk = Path(tools['toolchain_root'])
    return [str(sdk / 'bin/rustc'), '--target', TARGET, '--edition=' + edition,
            '-C', 'opt-level=2', '-C', 'codegen-units=1', '-C', 'link-self-contained=yes',
            '-C', 'linker=' + str(sdk / 'lib/rustlib/x86_64-unknown-linux-gnu/bin/rust-lld'),
            '-C', 'linker-flavor=ld.lld', '-C', 'link-arg=--threads=1']


def dependencies(tools):
    directory = Path(tools['toolchain_root']) / 'evaluation-libraries'; result = ['-L', 'dependency=' + str(directory)]
    for name in tools['provided_crates']:
        result += ['--extern', name + '=' + str(directory / ('lib' + name + '.rlib'))]
    return result


def inventory(output):
    text = output.decode('utf-8')
    cases = re.findall(r'^test ([A-Za-z_0-9:]+)(?: - should panic)? \.\.\. (ok|FAILED|ignored)\s*$', text, re.M)
    summary = re.findall(r'^test result: (ok|FAILED)\. (\d+) passed; (\d+) failed; (\d+) ignored; (\d+) measured; (\d+) filtered out;', text, re.M)
    if not cases or len({name for name, _ in cases}) != len(cases) or len(summary) != 1:
        raise ValueError('Missing, repeated or incomplete Rust native test inventory')
    state, passed, failed, ignored, measured, filtered = summary[0]
    statuses = {name: status for name, status in cases}
    if ([int(passed), int(failed), int(ignored)] != [sum(status == value for status in statuses.values()) for value in ('ok', 'FAILED', 'ignored')]
            or int(measured) or int(filtered)):
        raise ValueError('Rust native case counts differ or tests were filtered')
    return {'cases': sorted(statuses), 'passed_cases': sorted(name for name, status in cases if status == 'ok'),
            'skipped': sorted(name for name, status in cases if status == 'ignored'),
            'all_passed': state == 'ok' and int(passed) == len(cases) and not int(failed) and not int(ignored)}


class PublishedSuites:
    def __init__(self, cache, tools=None):
        cache = Path(cache); self.source = json.loads((cache / 'source.json').read_text())
        if file_hash(cache / 'source.zip') != self.source['archive_sha256']: raise ValueError('Original Rust archive changed')
        self.archive = zipfile.ZipFile(cache / 'source.zip'); self.prefix = self.archive.namelist()[0].split('/')[0] + '/'
        self.tools = toolchain() if tools is None else tools

    def close(self): self.archive.close()

    def reference(self, row):
        files = extract_files(row['response']); item = next(x for x in row['source_files'] if x['path'].endswith('/.meta/config.json'))
        raw = self.archive.read(self.prefix + item['path']); base = item['path'].split('/.meta/')[0] + '/'
        if hashlib.sha256(raw).hexdigest() != item['sha256']: raise ValueError('Original Rust interface config changed')
        config = json.loads(raw)['files']; references = config.get('example', config.get('exemplar', []))
        targets = [target_for(reference, config['solution']) for reference in references]
        if len(targets) != len(set(targets)) or set(targets) != set(files):
            raise ValueError('Original Rust reference/interface mapping differs')
        for name, reference in zip(targets, references):
            original = self.archive.read(self.prefix + base + reference)
            evidence = next(x for x in row['source_files'] if x['path'] == base + reference)
            if hashlib.sha256(original).hexdigest() != evidence['sha256'] or files[name].strip() != original.decode().strip():
                raise ValueError('Rust target differs from exact publisher reference')
        return files

    def suite(self, row):
        if row['revision'] != self.source['revision'] or row['source_archive_sha256'] != self.source['archive_sha256']:
            raise ValueError('Original Rust exercise source identity differs')
        item = next(x for x in row['source_files'] if x['path'].endswith('/.meta/config.json'))
        base = item['path'].split('/.meta/')[0] + '/'; tests = {}
        for path in self.archive.namelist():
            relative = path.removeprefix(self.prefix + base)
            if path.startswith(self.prefix + base + 'tests/') and relative.endswith('.rs'):
                tests[relative] = self.archive.read(path)
        for artifact in row['published_test_artifacts']:
            relative = artifact['path'].removeprefix(base)
            if relative not in tests or hashlib.sha256(tests[relative]).hexdigest() != artifact['sha256']:
                raise ValueError('Original Rust published test bytes differ')
        references = self.reference(row)
        cargo = references['Cargo.toml'].encode() if 'Cargo.toml' in references else self.archive.read(self.prefix + base + 'Cargo.toml')
        tests['Cargo.toml'] = cargo
        manifest = tomllib.loads(cargo.decode()); package = manifest['package']
        if not re.fullmatch(r'[a-z_][a-z_0-9]*', package['name']) or package.get('edition') != '2024':
            raise ValueError('Unverified original Rust package/edition')
        crates = [path for path in tests if path.startswith('tests/') and len(Path(path).parts) == 2]
        if not crates: raise ValueError('No complete original Rust integration test crate')
        return {'tests': tests, 'crates': sorted(crates), 'package': package['name'], 'edition': package['edition'],
                'test_sha256': {name: hashlib.sha256(raw).hexdigest() for name, raw in tests.items()}}

    def run(self, files, suite, expected=None):
        result = {'full_file_delivery': True, 'compilation_passed': False, 'published_tests_passed': False}
        with tempfile.TemporaryDirectory(prefix='slm-exercise-rust-') as directory:
            work = Path(directory); (work / 'src').mkdir(); (work / 'build').mkdir()
            for name, source in files.items():
                if not (re.fullmatch(r'src/[A-Za-z_][A-Za-z_0-9]*\.rs', name) or name == 'Cargo.toml'):
                    raise ValueError('Unsafe Rust source path')
                if name == 'Cargo.toml' and source.encode() != suite['tests']['Cargo.toml']:
                    raise ValueError('Prepared Rust package target differs from its verified original reference')
                path = work / 'src' / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_text(source)
            for name, raw in suite['tests'].items():
                path = work / 'src' / name
                if not path.resolve().is_relative_to((work / 'src').resolve()): raise ValueError('Rust test path escapes module')
                path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(raw)
            library = work / 'build' / ('lib' + suite['package'] + '.rlib')
            base = compiler(self.tools, suite['edition']); extern = dependencies(self.tools)
            compile_steps = [base + ['--crate-name', suite['package'], '--crate-type', 'rlib', 'src/lib.rs', '-o', str(library), *extern]]
            binaries = []
            for index, path in enumerate(suite['crates']):
                binary = work / 'build' / ('publisher-' + str(index)); binaries.append((path, binary))
                compile_steps.append(base + ['--crate-name', 'publisher_test_' + str(index), '--test', path,
                    '-o', str(binary), '--extern', suite['package'] + '=' + str(library), *extern])
            logs = []
            for command in compile_steps:
                step = confined(work, self.tools, command, 'compile'); step.pop('output'); logs.append(step)
                if step['returncode'] != 0 or step['limit']:
                    return {**result, 'compilation': logs, 'diagnostic': step['diagnostic'], 'limit': step['limit']}
            result.update(compilation_passed=True, compilation=logs)
            measured = []; executions = []
            for path, binary in binaries:
                run = confined(work, self.tools, [str(binary), '--include-ignored', '--test-threads=1', '--format=pretty', '--color=never'], 'runtime')
                raw = run.pop('output'); executions.append({**run, 'test_crate': path, 'output_sha256': hashlib.sha256(raw).hexdigest()})
                try: items = inventory(raw)
                except (ValueError, UnicodeError) as error:
                    return {**result, 'execution': executions, 'runtime_error': str(error)}
                measured.append({**items, 'test_crate': path})
            for name, digest in suite['test_sha256'].items():
                if file_hash(work / 'src' / name) != digest: raise NativeBackendError('Original Rust source tests/manifest altered')
            cases = sorted(path['test_crate'] + '::' + name for path in measured for name in path['cases'])
            passed = sorted(path['test_crate'] + '::' + name for path in measured for name in path['passed_cases'])
            skipped = sorted(path['test_crate'] + '::' + name for path in measured for name in path['skipped'])
            full = {'cases': cases, 'passed_cases': passed, 'skipped': skipped,
                    'all_passed': all(path['all_passed'] for path in measured)}
            matched = expected is None or cases == expected['cases']
            result.update(execution=executions, inventory=full, reference_inventory_matched=matched,
                published_tests_passed=full['all_passed'] and matched and all(x['returncode'] == 0 and not x['limit'] for x in executions))
            return result
