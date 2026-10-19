"""Exercise actual compiled Rust restrictions and incomplete-report rejection."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tempfile

from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.evaluation_provenance import capture
from slm_lab.exercise_rust_runner import PublishedSuites

ROOT = Path(__file__).resolve().parents[1]


def suite():
    tests = {
        'tests/contract.rs': b'#[test]\nfn complete_contract() { assert!(probe::check()); }\n',
        'Cargo.toml': b'[package]\nname = "probe"\nversion = "0.1.0"\nedition = "2024"\n',
    }
    return {'tests': tests, 'crates': ['tests/contract.rs'], 'package': 'probe', 'edition': '2024',
            'test_sha256': {name: hashlib.sha256(raw).hexdigest() for name, raw in tests.items()}}


def main():
    output = ROOT / 'reports/rust-kernel-and-report-verification-v1.json'
    if output.exists(): raise ValueError('Preserve completed Rust kernel evidence')
    sources = [Path(__file__), ROOT / 'src/slm_lab/exercise_rust_runner.py', ROOT / 'scripts/rust_sandbox.py',
               ROOT / 'scripts/javascript_sandbox.py', ROOT / 'scripts/native_sandbox.py',
               ROOT / '.tools/rust-toolchain-v1.json']
    runtime = capture(sources); findings = {}
    runner = PublishedSuites(ROOT / 'data/exercism-source-cache-v1/rust')

    def run(name, source):
        result = runner.run({'src/lib.rs': source}, suite())
        findings[name] = result
        atomic_json(ROOT / 'reports/rust-kernel-probes-progress-v1.json', {
            'updated_at': datetime.now(timezone.utc).isoformat(), 'findings': findings,
            'all_planned_probes_completed': False, 'goal_achieved': False})
        print(json.dumps({'probe': name, 'compiled': result['compilation_passed'],
                          'published_tests_passed': result['published_tests_passed'],
                          'execution': result.get('execution')}), flush=True)
        return result

    try:
        with tempfile.TemporaryDirectory(prefix='slm-rust-outside-marker-') as directory:
            marker = Path(directory) / 'marker'; marker.write_text('harmless kernel probe marker')
            accepted = {
                'outside_file_read': 'pub fn check() -> bool { std::fs::read(' + json.dumps(str(marker)) + ').unwrap_err().raw_os_error() == Some(13) }',
                'child_process_creation': 'pub fn check() -> bool { matches!(std::process::Command::new("/usr/bin/true").status().unwrap_err().raw_os_error(), Some(1) | Some(13)) }',
                'network_socket': 'pub fn check() -> bool { std::net::TcpStream::connect("127.0.0.1:9").unwrap_err().raw_os_error() == Some(1) }',
                'original_test_write': 'pub fn check() -> bool { std::fs::write("tests/contract.rs", b"changed").unwrap_err().raw_os_error() == Some(13) }',
                'native_threads_allowed': 'pub fn check() -> bool { std::thread::spawn(|| 7).join().unwrap() == 7 }',
            }
            for name, source in accepted.items():
                result = run(name, source)
                if not result['published_tests_passed']: raise ValueError('Actual Rust restriction/thread contract failed: ' + name)
            rejected = {
                'wrong_output': ('pub fn check() -> bool { false }', True),
                'premature_successful_exit': ('pub fn check() -> bool { std::process::exit(0) }', True),
                'invalid_source': ('pub fn check( {', False),
                'address_space_limit': ('pub fn check() -> bool { let mut b = vec![0u8; 512*1024*1024]; for v in b.iter_mut() { *v=1; } std::hint::black_box(b); true }', True),
                'cpu_limit': ('pub fn check() -> bool { loop { std::hint::black_box(1234); } }', True),
                'wall_time_limit': ('pub fn check() -> bool { std::thread::sleep(std::time::Duration::from_secs(60)); true }', True),
            }
            for name, (source, should_compile) in rejected.items():
                result = run(name, source)
                if result['published_tests_passed'] or result['compilation_passed'] != should_compile:
                    raise ValueError('Bad/limited Rust program incorrectly accepted: ' + name)
                if name == 'address_space_limit' and not any(x['returncode'] for x in result.get('execution', [])):
                    raise ValueError('Actual Rust address-space allocation was not rejected')
                if name == 'cpu_limit' and not any(x['returncode'] == -9 for x in result.get('execution', [])):
                    raise ValueError('Actual Rust CPU limit did not terminate execution')
                if name == 'wall_time_limit' and not any(x['limit'] == 'Rust wall-time budget exceeded' for x in result.get('execution', [])):
                    raise ValueError('Actual Rust wall deadline did not terminate execution')
        if capture(sources) != runtime: raise ValueError('Bound Rust probe runtime changed')
        atomic_json(output, {'passed': True, 'actual_probe_count': len(findings), 'findings': findings,
            'evaluation_provenance': runtime, 'toolchain_manifest_sha256': file_hash(ROOT / '.tools/rust-toolchain-v1.json'),
            'goal_achieved': False,
            'scope': 'Actual compiled Rust file/process/socket restrictions, native threads, unchanged-test protection, finite CPU/AS/wall limits and rejection of early/wrong/incomplete output. Same-process score forgery remains unproved; no model score.'})
        print('Actual Rust kernel/report probes passed', len(findings), flush=True)
    finally: runner.close()


if __name__ == '__main__': main()
