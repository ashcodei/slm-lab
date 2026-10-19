"""Exercise actual Node/Jest isolation and score/report failure behavior."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tempfile

from slm_lab.diverse_images import atomic_json
from slm_lab.exercise_js_runner import PublishedSuites

ROOT = Path(__file__).resolve().parents[1]


def run(suites, code, expectation):
    test = ("import {test,expect} from '@jest/globals';\nimport {probe} from './probe';\n"
            "test('real runtime contract',async()=>{expect(await probe()).toBe(" + json.dumps(expectation) + ");});\n").encode()
    suite = {'tests': {'probe.spec.js': test}, 'test_sha256': {'probe.spec.js': hashlib.sha256(test).hexdigest()}}
    return suites.run({'probe.js': code}, suite)


def main():
    output = ROOT / 'reports/javascript-kernel-and-report-verification-v1.json'
    if output.exists(): raise ValueError('Preserve existing JavaScript confinement evidence')
    suites = PublishedSuites(ROOT / 'data/exercism-source-cache-v1/javascript'); findings = {}
    try:
        with tempfile.TemporaryDirectory(prefix='slm-js-outside-probe-') as outside:
            private = Path(outside) / 'unshared.txt'; private.write_text('OWNED PROBE VALUE')
            findings['outside_file_read'] = run(suites, "import fs from 'node:fs'; export function probe(){try{return fs.readFileSync(" + json.dumps(str(private)) + ", 'utf8')}catch(e){return e.code}}", 'EACCES')
        findings['subprocess_creation'] = run(suites, "import {spawnSync} from 'node:child_process';export function probe(){return spawnSync('/bin/sh',['-c','exit 0']).error.code}", 'EPERM')
        findings['network_socket'] = run(suites, "import {Socket} from 'node:net';export function probe(){return new Promise(resolve=>{const s=new Socket();s.on('error',e=>resolve(e.code));s.connect({host:'127.0.0.1',port:9});});}", 'EPERM')
        findings['test_source_write'] = run(suites, "import fs from 'node:fs';export function probe(){try{fs.writeFileSync(new URL('./probe.spec.js',import.meta.url),'changed');return 'written'}catch(e){return e.code}}", 'EACCES')
        findings['wrong_algorithm'] = run(suites, 'export function probe(){return 2}', 1)
        findings['early_successful_exit'] = run(suites, 'export function probe(){process.exit(0)}', 1)
        findings['invalid_javascript'] = run(suites, 'export function probe( {', 1)
        findings['memory_budget'] = run(suites, "export function probe(){const b=Buffer.alloc(512*1024*1024);b.fill(1);return 1;}", 1)
        expected_success = ['outside_file_read', 'subprocess_creation', 'network_socket', 'test_source_write']
        if any(not findings[name]['published_tests_passed'] for name in expected_success): raise ValueError('Required kernel isolation probe failed: ' + json.dumps(findings))
        if any(findings[name]['published_tests_passed'] for name in ['wrong_algorithm', 'early_successful_exit', 'invalid_javascript', 'memory_budget']):
            raise ValueError('Invalid runtime/report behavior incorrectly passed')
        if findings['memory_budget'].get('runtime_error') != 'RSS memory budget exceeded': raise ValueError('RSS limit was not actually exercised')
        report = {'completed_at': datetime.now(timezone.utc).isoformat(), 'passed': True, 'findings': findings,
                  'goal_achieved': False, 'scope': 'Actual kernel path/network/process confinement, source write protection, incorrect output/report handling and physical RSS limit. Not adversarial same-process score-forgery proof or a model score.'}
        atomic_json(output, report); print(json.dumps({'passed': True, 'actual_checks': len(findings)}), flush=True)
    finally: suites.close()


if __name__ == '__main__': main()
