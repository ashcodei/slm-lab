import base64
from copy import deepcopy
import json
from pathlib import Path
import tempfile

import pytest

from slm_lab.public_python_state_trace_v1 import capture,validate
from slm_lab.standalone_program_v1 import prepare_python,run_bounded


def baseline(program,stdin):
    with tempfile.TemporaryDirectory() as temporary:
        constraints=prepare_python(program.encode(),Path(temporary));constraints['memory_bytes']=512*1024**2
        return run_bounded(constraints,stdin.encode(),seconds=3,python=True)


def test_actual_nested_call_loop_and_buffer_stdout_have_real_before_line_values():
    source=('import sys\n'
        'def add(n):\n'
        '    total = 0\n'
        '    for i in range(n):\n'
        '        total += i\n'
        '    return total\n'
        'n = int(sys.stdin.buffer.readline())\n'
        'result = add(n)\n'
        'sys.stdout.buffer.write((str(result)+"\\n").encode())\n')
    actual=capture(source,'4\n');assert actual['eligible_complete_trace'],actual
    raw=actual['actual_observations'];assert base64.b64decode(raw['actual_stdout_b64'])==b'6\n'
    ordinary=baseline(source,'4\n');assert ordinary['status']=='completed' and ordinary['stdout']==base64.b64decode(raw['actual_stdout_b64'])
    assert raw['source_frame_activations']==2
    before_add=[e for e in raw['events'] if e['event']=='line' and e['line']==5]
    assert [e['changes'].get('i') for e in before_add]==[['int',str(i)] for i in range(4)]
    returned=[e for e in raw['events'] if e['event']=='return' and e['function']=='add']
    assert len(returned)==1 and returned[0]['return_value']==['int','6']
    assert actual['expected_or_private_values_not_supplied_to_worker']
    assert validate(raw,source,maximum_events=256,maximum_trace_bytes=256*1024)
    damaged=deepcopy(raw);damaged['events'].pop()
    with pytest.raises(ValueError):validate(damaged,source,maximum_events=256,maximum_trace_bytes=256*1024)


def test_real_container_deltas_deletions_and_caught_exception_are_preserved():
    source='a = [1, 2]\na.append(3)\nb = {"x": tuple(a)}\ndel a\ntry:\n    missing_name\nexcept NameError:\n    print(b["x"])\n'
    actual=capture(source,'');assert actual['eligible_complete_trace'],actual
    raw=actual['actual_observations'];events=raw['events']
    assert any(e['changes'].get('a')==['list',[['int','1'],['int','2'],['int','3']]] for e in events)
    assert any('a' in e['deleted'] for e in events)
    assert any(e.get('exception_type')=='NameError' for e in events)
    assert base64.b64decode(raw['actual_stdout_b64'])==b'(1, 2, 3)\n'


def test_whole_trace_bounds_and_unsupported_cycles_never_become_complete_targets():
    source='n = int(input())\ntotal = 0\nfor i in range(n):\n    total += i\nprint(total)\n'
    actual=capture(source,'10\n',maximum_events=3)
    assert not actual['eligible_complete_trace'] and actual['actual_observations']['whole_trace_complete'] is False
    assert len(actual['actual_observations']['events'])==3
    cycle=capture('a=[]\na.append(a)\nprint("done")\n','')
    assert not cycle['eligible_complete_trace']
    bounded=capture('while True: pass\n','',seconds=.2)
    assert not bounded['eligible_complete_trace']


def test_only_original_source_and_public_input_reach_confined_observer():
    source='import json, pathlib\ns=json.loads(pathlib.Path("settings.json").read_text())\nprint(sorted(s))\nprint(sorted(p.name for p in pathlib.Path(".").iterdir()))\n'
    actual=capture(source,'only-public-input\n');assert actual['eligible_complete_trace'],actual
    output=base64.b64decode(actual['actual_observations']['actual_stdout_b64']).decode()
    assert "['maximum_events', 'maximum_trace_bytes', 'program']" in output
    assert "['observer.py', 'program.py', 'settings.json', 'values.py']" in output
