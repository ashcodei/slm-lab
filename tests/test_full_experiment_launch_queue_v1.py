"""Actual CPU child handoff with synthetic preparation/completion metadata.

Fixture scripts construct no model, query no GPU and perform no training.
The tests establish ownership, normal scheduling, ordering and rejection;
their declared full-scope metadata is not neural completion evidence.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

from slm_lab.full_experiment_launch_queue_v1 import (STAGES, identity, sha, run, completion, host_resources)

ROOT = Path(__file__).resolve().parents[1]


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def fixture(root, *, fail=None):
    interpreter = root / '.venv/bin/python'; interpreter.parent.mkdir(parents=True)
    interpreter.symlink_to(sys.executable)
    source = root / 'original-source'; source.write_bytes(b'original')
    captures = {}
    for index, spec in enumerate(STAGES):
        folder = root / 'reports' / spec['preparation']
        owner = dict(pid=99999970 + index, ticks='1')
        write(folder / 'plan.json', dict(owner=owner, fixture_metadata_only=True))
        write(folder / 'ready.json', dict(complete=True, GPU_work_started=False, production_updates=0,
            required_free_bytes_before_launch=40 * 1024**3 + 4096,
            source_files_sha256={str(source): sha(source)}, fixture_metadata_only=True))
        write(folder / 'complete.json', dict(complete=True, ready_sha256=sha(folder / 'ready.json'), fixture_metadata_only=True))
        captures[spec['name']] = dict(owner=owner, plan_sha256=sha(folder / 'plan.json'))
        path = root / 'scripts' / spec['driver']; path.parent.mkdir(exist_ok=True)
        code = """import json,os,sys
from pathlib import Path
assert os.environ['CUDA_VISIBLE_DEVICES']=='' and os.nice(0)==0 and os.sched_getscheduler(0)==os.SCHED_OTHER
assert 'torch' not in sys.modules
root=Path.cwd()
spec=json.loads(SPEC)
with (root/'actual-child-order').open('a') as stream:stream.write(spec['name']+'\\n')
if spec['name']==FAIL:sys.exit(5)
folder=root/'reports'/spec['output'];folder.mkdir()
fields=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()
data={'plan.json':{'owner':{'pid':os.getpid(),'ticks':fields[19]}},'children.json':[],
 'restored.json':{'all_registered_point_actors_terminal':True},
 'complete.json':{'complete':True,spec['count_key']:984399690,'original_nine_endpoint_pipeline_preserved':True,
  **{key:True for key in spec['scope']},'fixture_metadata_only':True}}
for name,value in data.items():(folder/name).write_text(json.dumps(value))
""".replace('SPEC', repr(json.dumps(spec))).replace('FAIL', repr(fail))
        path.write_text(code)
    directory = root / 'observer'; directory.mkdir()
    def publish(name, value): write(directory / name, value)
    return captures, directory, publish, source


def execute(root, captures, directory, publish, **kwargs):
    return run(root, directory, captures, guard=lambda: None, publish=publish,
        resources=lambda *args: dict(ready=True), pause=lambda seconds: time.sleep(.01), **kwargs)


def test_actual_four_serial_CPU_children_inherit_normal_context_and_exact_owned_completion(tmp_path):
    captures, directory, publish, _ = fixture(tmp_path)
    result = execute(tmp_path, captures, directory, publish)
    assert result['complete'] and result['goal_achieved'] is False
    assert (tmp_path / 'actual-child-order').read_text().splitlines() == [row['name'] for row in STAGES]
    for row, spec in zip(result['all_original_experiments'], STAGES):
        assert row['stage'] == spec['name'] and row['final_model_capability_not_certified']
        launched = json.loads((directory / (spec['name'] + '-launched.json')).read_text())
        assert row['owner'] == launched['actual_owned_driver']
        assert len(row['original_complete_receipts_sha256']) == 4
    assert result['final_shared_model_integration_and_verified_high_end_capability_still_required']


def test_live_exact_preparer_remains_waiting_even_with_complete_fixture_metadata(tmp_path):
    captures, directory, publish, _ = fixture(tmp_path)
    spec = STAGES[0]; folder = tmp_path / 'reports' / spec['preparation']
    actual = identity(os.getpid()); write(folder / 'plan.json', dict(owner=actual))
    captures[spec['name']] = dict(owner=actual, plan_sha256=sha(folder / 'plan.json'))
    def no_query(*args): raise AssertionError('No GPU/resource query may precede actual terminal source preparation')
    with pytest.raises(ValueError, match='observation lifetime expired'):
        run(tmp_path, directory, captures, guard=lambda: None, publish=publish, resources=no_query,
            maximum_wait_seconds=0, pause=lambda n: None)
    assert not (tmp_path / 'actual-child-order').exists()


@pytest.mark.parametrize('kind', ['changed_ready', 'changed_same_mtime_source', 'failed_preparer', 'existing_outputs'])
def test_changed_failed_or_foreign_inputs_cannot_launch_or_overwrite_an_experiment(tmp_path, kind):
    captures, directory, publish, source = fixture(tmp_path)
    spec = STAGES[0]; folder = tmp_path / 'reports' / spec['preparation']
    protected = None
    if kind == 'changed_ready':
        row = json.loads((folder / 'ready.json').read_text()); row['production_updates'] = 1; write(folder / 'ready.json', row)
    elif kind == 'changed_same_mtime_source':
        stat = source.stat(); source.write_bytes(b'changed!'); os.utime(source, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    elif kind == 'failed_preparer': write(folder / 'failure.json', dict(error='original controlled failure'))
    else:
        protected = tmp_path / 'reports' / spec['output'] / 'original-owner-evidence'; protected.parent.mkdir(); protected.write_bytes(b'keep')
    with pytest.raises(ValueError): execute(tmp_path, captures, directory, publish)
    assert not (tmp_path / 'actual-child-order').exists()
    if protected: assert protected.read_bytes() == b'keep'


def test_insufficient_actual_storage_never_queries_GPUs_or_takes_a_lease(tmp_path, monkeypatch):
    from slm_lab import full_experiment_launch_queue_v1 as module
    monkeypatch.setattr(module.shutil, 'disk_usage', lambda root: SimpleNamespace(free=40 * 1024**3))
    def forbidden(*args, **kwargs): raise AssertionError('No GPU query may precede complete storage funding')
    monkeypatch.setattr(module.subprocess, 'check_output', forbidden)
    result = host_resources(tmp_path, 40 * 1024**3 + 1)
    assert not result['ready'] and result['waiting_for'] == 'complete physical storage funding'
    assert not list(tmp_path.iterdir())


def test_actual_failed_CPU_child_is_not_retried_and_later_stages_never_start(tmp_path):
    captures, directory, publish, _ = fixture(tmp_path, fail=STAGES[0]['name'])
    with pytest.raises(ValueError, match='actual owned driver failed'): execute(tmp_path, captures, directory, publish)
    assert (tmp_path / 'actual-child-order').read_text().splitlines() == [STAGES[0]['name']]
    assert len(list(directory.glob('*-launched.json'))) == 1
    assert (directory / 'visual-owned-driver.log').is_file()


def test_partial_rehashed_completion_cannot_release_the_next_full_experiment(tmp_path):
    captures, directory, publish, _ = fixture(tmp_path)
    result = execute(tmp_path, captures, directory, publish)
    spec = STAGES[0]; path = tmp_path / 'reports' / spec['output'] / 'complete.json'
    row = json.loads(path.read_text()); row[spec['scope'][0]] = False; write(path, row)
    with pytest.raises(ValueError): completion(tmp_path, spec, result['all_original_experiments'][0]['owner'])


def test_observer_import_does_not_import_a_neural_runtime_or_launch_a_job():
    code = "import runpy,sys;runpy.run_path('scripts/queue_full_owned_experiments_CPU_v1.py');assert 'torch' not in sys.modules;print('non-neural observer import verified')"
    result = subprocess.run([sys.executable, '-c', code], cwd=ROOT, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0 and result.stdout.strip() == 'non-neural observer import verified'
