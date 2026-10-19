"""Published C++ library tests compiled and run inside the native kernel sandbox."""
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET
import zipfile

from .corpus import file_hash
from .native_code import NativeBackendError, launch, policy, stop, toolchain

STANDARD_HEADERS = set(('algorithm array bitset cassert cctype cerrno cfloat climits cmath complex '
                        'cstddef cstdint cstdlib cstring deque exception functional initializer_list '
                        'iomanip ios iosfwd iostream istream iterator limits list map memory numeric '
                        'optional ostream queue random ratio regex set sstream stack stdexcept string '
                        'string_view tuple type_traits unordered_map unordered_set utility valarray '
                        'variant vector').split())
FILE_BLOCK = re.compile(r'^File: ([^\r\n]+)\r?\n```(?:cpp|c\+\+|cc|cxx)?[ \t]*\r?\n(.*?)^```[ \t]*$', re.M | re.S)


def extract_files(response, expected=None):
    if len(response.encode()) > 100_000:
        raise ValueError('Generated module exceeds source-size limit')
    files = {}
    for match in FILE_BLOCK.finditer(response):
        name, code = match.groups()
        if not re.fullmatch(r'[a-zA-Z_][a-zA-Z0-9_]*\.(?:cpp|h|hpp)', name) or name in files:
            raise ValueError('Unsafe or repeated solution filename')
        if not code.strip(): raise ValueError('Empty solution file')
        files[name] = code
    if not files or FILE_BLOCK.sub('', response).strip():
        raise ValueError('Return complete labeled C++ file blocks, without extra content')
    if expected is not None and set(files) != set(expected):
        raise ValueError('Generated solution filenames do not match the requested interface')
    return files


def validate_sources(files):
    """Restrict preprocessor interaction with publisher tests; kernel policy is separate."""
    if not files or sum(len(code.encode()) for code in files.values()) > 100_000:
        raise ValueError('Invalid solution source size')
    for name, code in files.items():
        if not re.fullmatch(r'[a-zA-Z_][a-zA-Z0-9_]*\.(?:cpp|h|hpp)', name):
            raise ValueError('Unsafe solution filename')
        clean = re.sub(r'/\*.*?\*/|//[^\n]*', '', code, flags=re.S).replace('\\\n', '')
        if re.search(r'\b(?:Catch|CATCH_\w*|EXERCISM_\w*|NDEBUG|TEST_CASE|REQUIRE\w*|CHECK\w*)\b', clean):
            raise ValueError('Solution may not alter the test framework')
        if re.search(r'\bmain\s*\(|\b(?:asm|__asm__|__attribute__)\b|%:|\?\?=', clean):
            raise ValueError('Library solutions may not define main or compiler escape directives')
        guard = None
        allowed_guards = {name.replace('.', '_').upper(), name.replace('.', '_').upper() + '_INCLUDED'}
        for line in clean.splitlines():
            if not line.lstrip().startswith('#'): continue
            directive = line.strip()
            include = re.fullmatch(r'#\s*include\s*(?:<([a-z_0-9]+)>|"([a-zA-Z_][a-zA-Z0-9_]*\.(?:h|hpp|cpp))")\s*', directive)
            if include:
                standard, local = include.groups()
                if standard and standard not in STANDARD_HEADERS or local and local not in files:
                    raise ValueError('Only declared solution files and allowed standard headers may be included')
            elif re.fullmatch(r'#\s*pragma\s+once', directive): pass
            elif re.fullmatch(r'#\s*ifndef\s+\w+', directive):
                symbol = directive.split()[-1]
                if guard is not None or symbol not in allowed_guards: raise ValueError('Invalid solution include guard')
                guard = (symbol, False)
            elif re.fullmatch(r'#\s*define\s+\w+', directive):
                symbol = directive.split()[-1]
                if guard != (symbol, False): raise ValueError('Only the solution include guard may be defined')
                guard = (symbol, True)
            elif re.fullmatch(r'#\s*endif', directive):
                if guard is None or not guard[1]: raise ValueError('Unmatched solution include guard')
                guard = None
            else: raise ValueError('Unsupported solution preprocessor directive')
        if guard is not None: raise ValueError('Unclosed solution include guard')


def parse_junit(raw):
    if len(raw) > 1024**2 or b'<!DOCTYPE' in raw or b'<!ENTITY' in raw:
        raise ValueError('Invalid or oversized test report')
    root = ET.fromstring(raw)
    if root.tag != 'testsuites' or len(root.findall('testsuite')) != 1:
        raise ValueError('Expected one complete published test suite')
    suite = root.find('testsuite')
    assertions = int(suite.attrib['tests'])
    cases = [(node.attrib['classname'], node.attrib['name']) for node in suite.findall('testcase')]
    if assertions <= 0 or not cases or len(cases) != len(set(cases)):
        raise ValueError('Empty or duplicate published test inventory')
    failed = int(suite.attrib['failures']) + int(suite.attrib['errors'])
    skipped = int(suite.get('skipped', '0'))
    details_failed = bool(suite.findall('.//failure') or suite.findall('.//error') or suite.findall('.//skipped'))
    return {'assertions': assertions, 'cases': [list(case) for case in sorted(cases)],
            'all_passed': failed == 0 and skipped == 0 and not details_failed}


class PublishedSuites:
    """One verified source archive and one confined Catch main compilation per run."""
    def __init__(self, cache):
        cache = Path(cache)
        self.source = json.loads((cache / 'source.json').read_text())
        archive = cache / 'source.zip'
        if file_hash(archive) != self.source['archive_sha256']:
            raise ValueError('Original C++ source archive changed')
        self.archive = zipfile.ZipFile(archive)
        self.prefix = self.archive.namelist()[0].split('/')[0] + '/'
        self.tools = toolchain()
        self.temporary = tempfile.TemporaryDirectory(prefix='slm-exercise-cpp-')
        self.work = Path(self.temporary.name)
        shutil.copytree(self.tools['cache_path'], self.work / 'global-cache')
        self.common = [self.tools['compiler_path'], 'c++', '-target', 'x86_64-linux-musl', '-static',
                       '-std=c++17', '-O2', '-fno-lto', '-DEXERCISM_RUN_ALL_TESTS']
        self.catch_sha = None

    def close(self):
        self.archive.close(); self.temporary.cleanup()

    def read(self, path):
        if '..' in Path(path).parts or Path(path).is_absolute(): raise ValueError('Unsafe original source path')
        info = self.archive.getinfo(self.prefix + path)
        if info.file_size > 1024**2: raise ValueError('Oversized original source file')
        return self.archive.read(info)

    def suite(self, row):
        if row['revision'] != self.source['revision'] or row['source_archive_sha256'] != self.source['archive_sha256']:
            raise ValueError('Exercise source revision/archive mismatch')
        base = row['source_files'][0]['path'].split('/.meta/')[0] + '/'
        tests = {}
        for artifact in row['published_test_artifacts']:
            if artifact['path'].endswith('_test.cpp'):
                raw = self.read(artifact['path'])
                if hashlib.sha256(raw).hexdigest() != artifact['sha256']: raise ValueError('Publisher test bytes changed')
                tests[Path(artifact['path']).name] = raw
        if len(tests) != 1: raise ValueError('Expected one original C++ test translation unit')
        catch = self.read(base + 'test/catch.hpp'); main = self.read(base + 'test/tests-main.cpp')
        digest = hashlib.sha256(catch).hexdigest()
        if self.catch_sha is None:
            (self.work / 'catch.hpp').write_bytes(catch)
            (self.work / 'tests-main.cpp').write_bytes(main)
            error = self.compile(['-c', 'tests-main.cpp', '-o', 'catch-main.o'], self.work)
            if error: raise NativeBackendError('Original Catch harness compilation failed: ' + error)
            self.catch_sha = digest
        elif digest != self.catch_sha or main != (self.work / 'tests-main.cpp').read_bytes():
            raise ValueError('Publisher test harness differs between exercises')
        return {'tests': tests, 'catch': catch, 'catch_sha256': digest,
                'main_sha256': hashlib.sha256(main).hexdigest(),
                'test_sha256': {name: hashlib.sha256(raw).hexdigest() for name, raw in tests.items()}}

    def compile(self, args, work):
        log = self.work / 'compiler-output.txt'
        with log.open('wb') as output:
            configuration = policy('compile', self.common + args, work, self.tools['toolchain_root'])
            # Reuse only our trusted prewarmed/per-run compiler cache, not another case's source files.
            configuration['paths'].append((str(self.work / 'global-cache'), (1 << 15) - 2))
            configuration['environment']['ZIG_GLOBAL_CACHE_DIR'] = str(self.work / 'global-cache')
            process = launch(configuration, stdout=output, stderr=output)
            try: result = process.wait(timeout=120)
            except subprocess.TimeoutExpired:
                stop(process); return 'Compiler deadline exceeded'
        diagnostic = log.read_bytes()[-4000:].decode(errors='replace')
        if 'SANDBOX_ERROR:' in diagnostic: raise NativeBackendError(diagnostic)
        return (diagnostic[-1600:] or 'Compiler failed') if result else None

    def run(self, files, suite, expected=None):
        result = {'compilation_succeeded': False, 'published_tests_passed': False}
        try: validate_sources(files)
        except ValueError as error: return {**result, 'source_rejected': str(error)}
        with tempfile.TemporaryDirectory(prefix='case-', dir=self.work) as directory:
            work = Path(directory); (work / 'test').mkdir()
            for name, code in files.items(): (work / name).write_text(code)
            for name, raw in suite['tests'].items(): (work / name).write_bytes(raw)
            (work / 'test/catch.hpp').write_bytes(suite['catch'])
            shutil.copyfile(self.work / 'catch-main.o', work / 'catch-main.o')
            # Some concept exercises include their implementation directly in the test file.
            included = set(re.findall(r'#\s*include\s*"([^"\n]+\.cpp)"', b'\n'.join(suite['tests'].values()).decode()))
            sources = [name for name in files if name.endswith('.cpp') and name not in included] + list(suite['tests'])
            objects = []
            for index, source in enumerate(sources):
                target = f'unit-{index}.o'; error = self.compile(['-c', source, '-o', target], work)
                if error: return {**result, 'compile_error': error}
                objects.append(target)
            error = self.compile(objects + ['catch-main.o', '-o', 'program'], work)
            if error: return {**result, 'compile_error': error}
            result['compilation_succeeded'] = True
            stdout, stderr = work / 'junit.xml', work / 'runtime-errors.txt'
            with stdout.open('wb') as output, stderr.open('wb') as errors:
                process = launch(policy('runtime', [work / 'program', '--reporter', 'junit', '--rng-seed', '42'], work),
                                 stdin=subprocess.DEVNULL, stdout=output, stderr=errors)
                try: returncode = process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    stop(process); return {**result, 'runtime_error': 'Test execution deadline exceeded'}
            diagnostic = stderr.read_bytes()[-1200:].decode(errors='replace')
            if 'SANDBOX_ERROR:' in diagnostic: raise NativeBackendError(diagnostic)
            result.update(returncode=returncode, runtime_diagnostic=diagnostic)
            raw = stdout.read_bytes(); result['report_sha256'] = hashlib.sha256(raw).hexdigest()
            try: inventory = parse_junit(raw)
            except (ValueError, ET.ParseError, KeyError) as error:
                return {**result, 'runtime_error': 'Incomplete test report: ' + str(error)}
            result['inventory'] = inventory
            matched = expected is None or (inventory['assertions'] == expected['assertions'] and inventory['cases'] == expected['cases'])
            result['reference_inventory_matched'] = matched
            result['published_tests_passed'] = returncode == 0 and inventory['all_passed'] and matched
            return result
