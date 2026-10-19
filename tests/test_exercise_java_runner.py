"""Real JVM/JUnit scoring and kernel confinement before class initialization."""
import hashlib
import json
from pathlib import Path
import socket

import pytest

from slm_lab.exercise_java_runner import PublishedSuites, extract_files, inventory


@pytest.fixture(scope='module')
def suites():
    root = Path(__file__).resolve().parents[1]
    value = PublishedSuites(root / 'data/exercism-source-cache-v1/java')
    yield value
    value.close()


def fixture_suite(mode='jupiter'):
    if mode == 'jupiter':
        body = ('import org.junit.jupiter.api.Test; import org.junit.jupiter.api.Disabled; '
            'import static org.junit.jupiter.api.Assertions.assertEquals; '
            'public class ProbeTest { @Test void first() throws Exception { assertEquals(42, Probe.answer()); } '
            '@Disabled @Test void second() throws Exception { assertEquals(42, Probe.answer()); } }')
    else:
        body = ('import org.junit.Test; import org.junit.Ignore; import static org.junit.Assert.assertEquals; '
            'public class ProbeTest { @Test public void first() throws Exception { assertEquals(42, Probe.answer()); } '
            '@Ignore @Test public void second() throws Exception { assertEquals(42, Probe.answer()); } }')
    raw = body.encode()
    return dict(tests={'src/test/java/ProbeTest.java': raw}, editors={},
        expected_solution_files=['src/main/java/Probe.java'], test_classes=['ProbeTest'], mode=mode,
        assertj_version='3.15.0' if mode=='junit4' else '3.25.1',
        test_sha256={'src/test/java/ProbeTest.java': hashlib.sha256(raw).hexdigest()}, editor_sha256={})


def program(body):
    return {'src/main/java/Probe.java': 'public class Probe { ' + body + ' }'}


@pytest.mark.parametrize('mode', ['jupiter', 'junit4'])
def test_real_pinned_framework_runs_disabled_tests_and_rejects_wrong_results(suites, tmp_path, mode):
    suite = fixture_suite(mode)
    good = suites.run(program('public static int answer() { return 42; }'), suite)
    assert good['compilation_passed'] and good['published_tests_passed']
    assert len(good['inventory']['cases']) == 2 and len(good['inventory']['started_cases']) == 2
    assert not good['inventory']['skipped'] and good['inventory']['junit4_version']=='4.13'
    assert inventory(good['raw_inventory_output'].encode()) == good['inventory']
    bad = suites.run(program('public static int answer() { return 0; }'), suite, good['inventory'])
    assert bad['compilation_passed'] and not bad['published_tests_passed'] and len(bad['inventory']['failed']) == 2
    (tmp_path / (mode + '-known-scoring.json')).write_text(json.dumps(dict(good=good, bad=bad)))


@pytest.mark.parametrize('attempt', ['initializer_read', 'thread_read', 'host_write', 'network', 'process'])
def test_confinement_precedes_static_initializers_and_applies_to_threads(suites, tmp_path, attempt):
    sentinel = tmp_path / 'outside-sentinel.txt'
    sentinel.write_text('OWNED TEST SENTINEL')
    outside = tmp_path / 'outside-write.txt'
    source_path = json.dumps(str(sentinel))
    output_path = json.dumps(str(outside))
    with socket.socket() as listening:
        listening.bind(('127.0.0.1',0))
        listening.listen(4)
        port = listening.getsockname()[1]
        if attempt == 'initializer_read':
            body = ('static int stored; static { try { java.nio.file.Files.readString(java.nio.file.Path.of(' + source_path + ')); '
                'stored=0; } catch(java.io.IOException ex) { stored=42; } } public static int answer() { return stored; }')
        elif attempt == 'thread_read':
            body = ('public static int answer() throws Exception { java.util.concurrent.atomic.AtomicInteger result = '
                'new java.util.concurrent.atomic.AtomicInteger(0); Thread t = new Thread(() -> { try { '
                'java.nio.file.Files.readString(java.nio.file.Path.of(' + source_path + ')); result.set(0); } '
                'catch(java.io.IOException ex) { result.set(42); } }); t.start(); t.join(); return result.get(); }')
        elif attempt == 'host_write':
            body = ('public static int answer() { try { java.nio.file.Files.writeString(java.nio.file.Path.of(' + output_path
                + '), "escaped"); return 0; } catch(java.io.IOException ex) { return 42; } }')
        elif attempt == 'network':
            body = ('public static int answer() { try(java.net.Socket s = new java.net.Socket("127.0.0.1", '
                + str(port) + ')) { return 0; } catch(java.io.IOException ex) { return 42; } }')
        else:
            body = ('public static int answer() throws Exception { try { new ProcessBuilder("/usr/bin/touch", '
                + output_path + ').start().waitFor(); return 0; } catch(java.io.IOException ex) { return 42; } }')
        measured = suites.run(program(body), fixture_suite())
        if attempt == 'network':
            # HotSpot wraps the kernel EPERM from NIO socket initialization in
            # an initializer error. This denied program must fail its tests.
            assert measured['compilation_passed'] and not measured['published_tests_passed'], measured
            assert 'Operation not permitted' in measured['execution']['diagnostic'], measured
        else:
            assert measured['published_tests_passed'], measured
        assert not outside.exists() and sentinel.read_text()=='OWNED TEST SENTINEL'
        listening.settimeout(.01)
        with pytest.raises(TimeoutError):
            listening.accept()
    (tmp_path / (attempt + '-confinement.json')).write_text(json.dumps(measured))


def test_zero_exit_without_inventory_and_mismatched_registered_cases_cannot_pass(suites, tmp_path):
    early = suites.run(program('public static int answer() { System.exit(0); return 42; }'), fixture_suite())
    assert early['compilation_passed'] and not early['published_tests_passed']
    assert early['execution']['returncode']==0 and 'runtime_error' in early
    unmatched = suites.run(program('public static int answer() { return 42; }'), fixture_suite(), {'cases':['unregistered']})
    assert unmatched['inventory']['all_passed'] and not unmatched['reference_inventory_matched'] and not unmatched['published_tests_passed']
    (tmp_path / 'early-and-unmatched.json').write_text(json.dumps(dict(early=early, unmatched=unmatched)))


def test_original_multifile_identity_and_complete_provided_enum_are_preserved(suites):
    rows = [json.loads(line) for line in (Path(__file__).resolve().parents[1] / 'data/exercism-support-context-v3/valid.jsonl').open()]
    queen = next(r for r in rows if r['language']=='Java' and r['exercise']=='queen-attack')
    assert set(suites.reference(queen))=={'src/main/java/Queen.java','src/main/java/QueenAttackCalculator.java'}
    handshake = next(r for r in rows if r['language']=='Java' and r['exercise']=='secret-handshake')
    suite = suites.suite(handshake)
    assert set(suites.reference(handshake))=={'src/main/java/HandshakeCalculator.java'}
    assert suite['editors']['src/main/java/Signal.java']==suites.read('exercises/practice/secret-handshake/src/main/java/Signal.java')


def test_undeclared_or_repeated_files_and_prose_are_rejected_before_compilation():
    good='File: src/main/java/Probe.java\n```java\npublic class Probe {}\n```'
    for bad in [good.replace('src/main/java/Probe.java','../Probe.java'),good.replace('src/main/java','src/test/java'),good+'\n'+good,good+'\nall tests passed']:
        with pytest.raises(ValueError):
            extract_files(bad, ['src/main/java/Probe.java'])
