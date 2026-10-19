"""Probe actual compiled Go execution restrictions and complete test reports."""
import hashlib
from pathlib import Path
import tempfile

from slm_lab.diverse_images import atomic_json
from slm_lab.exercise_go_runner import PublishedSuites

ROOT = Path(__file__).resolve().parents[1]


def suite(body):
    tests = {'probe_test.go': ('package probe\nimport "testing"\nfunc TestContract(t *testing.T) { ' + body + ' }\n').encode(),
             'go.mod': b'module probe\n\ngo 1.27\n'}
    return {'tests': tests, 'test_sha256': {name: hashlib.sha256(raw).hexdigest() for name, raw in tests.items()}}


def main():
    runner = PublishedSuites(ROOT / 'data/exercism-source-cache-v1/go')
    with tempfile.TemporaryDirectory(prefix='slm-go-outside-marker-') as directory:
        outside = Path(directory) / 'marker'; outside.write_text('harmless confinement marker')
        probes = {
            'outside_file_read': 'import ("os";"errors";"syscall")\nfunc Probe() bool { _,e:=os.ReadFile("' + str(outside) + '"); return errors.Is(e,syscall.EACCES) }',
            'child_process_creation': 'import ("os/exec";"errors";"syscall")\nfunc Probe() bool { e:=exec.Command("/usr/bin/true").Run(); return errors.Is(e,syscall.EPERM)||errors.Is(e,syscall.EACCES) }',
            'network_socket': 'import ("net";"errors";"syscall")\nfunc Probe() bool { _,e:=net.Dial("tcp","127.0.0.1:9"); return errors.Is(e,syscall.EPERM) }',
            'original_test_write': 'import ("os";"errors";"syscall")\nfunc Probe() bool { e:=os.WriteFile("probe_test.go",[]byte("changed"),0600); return errors.Is(e,syscall.EACCES) }',
            'other_process_signal': 'import ("os";"errors";"syscall")\nfunc Probe() bool { e:=syscall.Tgkill(os.Getppid(),os.Getppid(),0); return errors.Is(e,syscall.EPERM) }',
        }
        findings = {}
        for name, source in probes.items():
            result = runner.run({'probe.go': 'package probe\n' + source + '\n'}, suite('if !Probe(){t.Fatal("restriction not applied")}'))
            if not result['published_tests_passed']: raise ValueError('Actual Go confinement probe failed: ' + name)
            findings[name] = result
        rejected = {
            'wrong_output': ('func Probe() bool { return false }', True),
            'premature_successful_exit': ('import "os"\nfunc init(){os.Exit(0)}\nfunc Probe() bool{return true}', True),
            'invalid_source': ('func Probe( {', False),
        }
        for name, (source, compilation) in rejected.items():
            result = runner.run({'probe.go': 'package probe\n' + source + '\n'}, suite('if !Probe(){t.Fatal("wrong result")}'))
            if result['published_tests_passed'] or result['compilation_passed'] != compilation:
                raise ValueError('Actual Go bad-output/source probe incorrectly accepted: ' + name)
            findings[name] = result
        source = 'package probe\nimport "runtime"\nfunc Probe() bool { b:=make([]byte,512*1024*1024); for i:=range b {b[i]=1}; runtime.KeepAlive(b); return true }'
        result = runner.run({'probe.go': source}, suite('if !Probe(){t.Fatal("wrong result")}'))
        if not result['compilation_passed'] or result.get('execution', {}).get('limit') != 'Go runtime RSS budget exceeded':
            raise ValueError('Actual Go excessive resident memory was not stopped')
        findings['resident_memory_limit'] = result
        atomic_json(ROOT / 'reports/go-kernel-and-report-verification-v1.json', {'passed': True,
            'actual_probe_count': len(findings), 'findings': findings, 'goal_achieved': False,
            'scope': 'Actual compiled Go file/process/socket/signal restrictions and rejection of incomplete/wrong outputs; no adversarial score-forgery or model capability claim.'})
        print('Actual Go confinement/report probes passed', len(findings), flush=True)
    runner.close()


if __name__ == '__main__': main()
