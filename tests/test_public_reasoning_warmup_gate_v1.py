"""Real disposable process births cannot be replaced by a stale receipt."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

import slm_lab.public_reasoning_warmup_gate_v1 as gate


def child():
    process=subprocess.Popen([sys.executable,'-c','import sys;sys.stdin.read(1)'],stdin=subprocess.PIPE)
    owner=dict(pid=process.pid,ticks=(Path('/proc')/str(process.pid)/'stat').read_text().split(') ',1)[1].split()[19])
    return process,owner


def stop(process):
    process.communicate(b'x',timeout=10)


def receipt(root,owner):
    folder=root/'reports/public-edge-preference-comparison-CPU-v1';folder.mkdir(parents=True,exist_ok=True)
    (folder/'complete.json').write_text(json.dumps(dict(complete=True,owner=owner,
        all77machine_pairs_three_epochs_all_original_contexts_and_EOS_preserved=True,
        all_baseline_and_final15source48code_and_broad_local_suites_complete=True)))
    return folder


def test_actual_live_predecessors_keep_waiting_even_with_completed_receipt(tmp_path,monkeypatch):
    edge,edge_owner=child();main,main_owner=child();calls=[]
    monkeypatch.setattr(gate,'public_ready',lambda *args:calls.append(args) or True)
    try:
        receipt(tmp_path,edge_owner)
        assert gate.ready(tmp_path,edge_owner,main_owner) is False and not calls
        stop(edge)
        assert gate.ready(tmp_path,edge_owner,main_owner) is False and not calls
        stop(main)
        assert gate.ready(tmp_path,edge_owner,main_owner) is True and len(calls)==1
    finally:
        for process in [edge,main]:
            if process.poll() is None:stop(process)


def test_terminal_predecessor_without_complete_scope_rejects_and_never_admits(tmp_path,monkeypatch):
    edge,owner=child();stop(edge);calls=[]
    monkeypatch.setattr(gate,'public_ready',lambda *args:calls.append(args) or True)
    with pytest.raises(ValueError,match='ended without'):gate.ready(tmp_path,owner,owner)
    folder=receipt(tmp_path,owner);done=json.loads((folder/'complete.json').read_text())
    done['all_baseline_and_final15source48code_and_broad_local_suites_complete']=False
    (folder/'complete.json').write_text(json.dumps(done))
    with pytest.raises(ValueError,match='scope'):gate.ready(tmp_path,owner,owner)
    receipt(tmp_path,owner);(folder/'failure.json').write_text('{}')
    with pytest.raises(ValueError,match='ended without'):gate.ready(tmp_path,owner,owner)
    assert calls==[]
