import importlib.util
import json
import os
from pathlib import Path
import sys

import pytest


def driver():
    root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'scripts'))
    spec=importlib.util.spec_from_file_location('code_first_finalizer_unit',root/'scripts/finalize_code_first_student_CPU_v2.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def test_actual_live_owner_waits_and_missing_terminal_owner_never_substitutes_old_actor(tmp_path):
    m=driver();m.SOURCE=tmp_path;m.CHECKPOINT=tmp_path/'actor.pt'
    fields=Path('/proc/self/stat').read_text().split(') ',1)[1].split();identity=dict(pid=os.getpid(),ticks=fields[19])
    assert m.source_ready(identity) is False
    with pytest.raises(ValueError,match='ended'):m.source_ready(dict(identity,ticks='0'))


def test_complete_original_epoch_and_all_four_private_states_required_before_finalizing(tmp_path):
    m=driver();m.SOURCE=tmp_path;m.CHECKPOINT=tmp_path/'actor.pt';m.CHECKPOINT.write_bytes(b'unit-metadata-only')
    def save(name,value):(tmp_path/name).write_text(json.dumps(value))
    save('plan.json',dict(planned_steps=3721,original_coding_sources=3721))
    save('anchor_execution-training-complete.json',dict(actual_updates=3721,all_3721_original_coding_sources_once=True))
    save('train-anchor_execution-resume-distributed-state.json',dict(completed=3721,all_four_private_states=[dict(rank=r) for r in range(4)]))
    assert m.source_ready(dict(pid=0,ticks='0'))
    save('train-anchor_execution-resume-distributed-state.json',dict(completed=3721,all_four_private_states=[dict(rank=r) for r in range(3)]))
    with pytest.raises(ValueError,match='all four'):m.source_ready(dict(pid=0,ticks='0'))
