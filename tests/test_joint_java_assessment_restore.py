"""Exercise the real comparison watchdog on separately owned idle processes."""
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import time

import pytest

ROOT=Path(__file__).resolve().parents[1]
DRIVER=ROOT/'scripts/assess_joint_java_comparison_v1.py'


def loaded(tmp_path):
    spec=importlib.util.spec_from_file_location('fixture_comparison_driver',DRIVER)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.REPORTS=tmp_path
    return module


def sleeper():
    return subprocess.Popen([str(ROOT/'.venv/bin/python'),'-c','import time; time.sleep(120)'])


def stopped(module,process):
    actor=module.identity(process.pid)
    os.kill(process.pid,signal.SIGSTOP)
    deadline=time.monotonic()+5
    while (module.same(actor) or {}).get('state') not in {'T','t'}:
        assert time.monotonic()<deadline
        time.sleep(.01)
    return actor


def cleanup(*processes):
    for process in processes:
        if process.poll() is None:
            os.kill(process.pid,signal.SIGCONT)
            process.terminate()
        process.wait(timeout=10)


def test_owner_death_stops_registered_reader_then_restores_exact_original_idle_process(tmp_path):
    module=loaded(tmp_path)
    owner,waiter,reader=sleeper(),sleeper(),sleeper()
    watch=None
    try:
        actor=stopped(module,waiter)
        (tmp_path/'plan.json').write_text(json.dumps(dict(owner=module.identity(owner.pid),waiter=actor)))
        (tmp_path/'children.json').write_text(json.dumps([module.identity(reader.pid)]))
        shim=('import importlib.util; from pathlib import Path; '
            's=importlib.util.spec_from_file_location("fixture_watchdog",'+repr(str(DRIVER))+'); '
            'm=importlib.util.module_from_spec(s); s.loader.exec_module(m); '
            'm.REPORTS=Path('+repr(str(tmp_path))+'); m.watchdog()')
        watch=subprocess.Popen([str(ROOT/'.venv/bin/python'),'-c',shim])
        owner.terminate()
        owner.wait(timeout=10)
        assert watch.wait(timeout=15)==0
        receipt=json.loads((tmp_path/'restored.json').read_text())
        assert reader.poll() is not None
        assert module.same(actor)['state'] not in {'T','t'}
        assert receipt['same_original_waiter_resumed']
        assert receipt['waiter']['ticks']==actor['ticks']
    finally:
        if watch and watch.poll() is None:
            watch.terminate();watch.wait(timeout=10)
        cleanup(owner,waiter,reader)


def test_stale_process_birth_does_not_resume_an_unrelated_process(tmp_path):
    module=loaded(tmp_path)
    waiter=sleeper()
    try:
        actor=stopped(module,waiter)
        (tmp_path/'children.json').write_text('[]')
        module.restore(dict(waiter={**actor,'ticks':'0'}),'fixture stale identity')
        assert module.same(actor)['state'] in {'T','t'}
        assert not json.loads((tmp_path/'restored.json').read_text())['same_original_waiter_resumed']
    finally:
        cleanup(waiter)


def test_unregistered_reader_with_missing_owner_exits_before_model_loading(tmp_path,monkeypatch):
    module=loaded(tmp_path)
    (tmp_path/'plan.json').write_text(json.dumps(dict(owner={'pid':os.getpid(),'ticks':'0'})))
    with pytest.raises(ValueError,match='owner ended before worker registration'):
        module.worker('language')


def test_summary_assembly_preserves_real_complete_scores_without_duplicate_scope(tmp_path):
    module=loaded(tmp_path)
    reports=ROOT/'reports/joint-vs-language-Java-v1'
    before=json.loads((reports/'language.json').read_text())
    after=json.loads((reports/'joint.json').read_text())
    result=module.comparison_report(before,after,'a'*64,'b'*64,'c'*64)
    assert result['passed'] and result['before_module_pass_rate']==result['after_module_pass_rate']==0.
    assert result['base_checkpoint_sha256']==before['checkpoint_sha256']
    assert result['candidate_checkpoint_sha256']==after['checkpoint_sha256']
    assert result['driver_plan_sha256']=='a'*64 and result['baseline_report_sha256']=='b'*64
    assert result['joint_report_sha256']=='c'*64 and not result['candidate_promoted'] and not result['goal_achieved']
