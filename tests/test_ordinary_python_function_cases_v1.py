from slm_lab.ordinary_python_function_cases_v1 import execute


def test_normal_main_guard_module_constants_and_helpers_use_real_python_semantics():
    program = '''from itertools import accumulate
FACTOR = 1
class Prefix:
    def run(self, values):
        return list(accumulate(values))
def prefix(values):
    return [FACTOR*x for x in Prefix().run(values)]
if __name__ == "__main__":
    raise RuntimeError("Main must not run when testing a module function")
'''
    result = execute(program, 'prefix', [dict(args=[[1, 2, -1]], expected=[1, 3, 2]), dict(args=[[]], expected=[])])
    assert result['syntax_valid'] and result['functional_passed'] and result['passed_cases'] == 2
    assert result['source_bytes_not_repaired'] and result['expected_outputs_not_sent_to_confined_worker']
    assert not execute('def prefix(values):\n    return 1\n', 'prefix', [dict(args=[[1, 2]], expected=[1, 3])])['functional_passed']


def test_candidate_can_observe_inputs_but_never_expected_answers_in_settings():
    program = '''import json
def fields(value):
    return sorted(json.load(open("settings.json")))
'''
    result = execute(program, 'fields', [dict(args=[7], expected=['function', 'inputs', 'program'])])
    assert result['functional_passed']


def test_host_files_network_and_unbounded_runtime_are_still_confined(tmp_path):
    secret = tmp_path / 'outside-sandbox-private-control'; secret.write_text('not granted to candidate')
    result = execute('def f():\n    return open(' + repr(str(secret)) + ').read()\n', 'f', [dict(args=[], expected='not granted to candidate')])
    assert not result['functional_passed'] and result['status'] == 'runtime_or_resource_failure'
    result = execute('import socket\ndef f():\n    return socket.socket().fileno()\n', 'f', [dict(args=[], expected=0)])
    assert not result['functional_passed'] and result['status'] == 'runtime_or_resource_failure'
    result = execute('def f():\n    while True: pass\n', 'f', [dict(args=[], expected=0)], seconds=.5)
    assert not result['functional_passed'] and result['status'] == 'runtime_or_resource_failure'
