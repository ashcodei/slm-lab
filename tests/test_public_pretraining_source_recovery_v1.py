import importlib
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
raw=importlib.import_module('prepare_public_pretraining_sources_CPU_v3')
functions=importlib.import_module('prepare_public_pretraining_sources_CPU_v4')


def test_recovery_routes_require_exact_new_reader_terminal_full_admission(tmp_path,monkeypatch):
    reader=tmp_path/'reader';reader.mkdir()
    pack=tmp_path/'pack';pack.mkdir()
    (reader/'plan.json').write_text(json.dumps(dict(owner=raw.READER_OWNER)))
    for module in (raw,functions):
        monkeypatch.setattr(module,'READER',reader)
        monkeypatch.setattr(module,'PACK',pack)
        monkeypatch.setattr(module,'terminal',lambda identity:False)
        assert not module.dependencies_ready()
        monkeypatch.setattr(module,'terminal',lambda identity:True)
        with pytest.raises(ValueError,match='must finish'):module.dependencies_ready()
    (reader/'complete.json').write_text(json.dumps(dict(complete=True,data_admitted_to_training=True,actual_train_targets_admitted=10**9)))
    (pack/'manifest.json').write_text('{}')
    admission=dict(admitted_to_training=True,full_source_reader_complete=True,manifest_sha256=raw.file_hash(pack/'manifest.json'))
    (pack/'admission.json').write_text(json.dumps(admission))
    assert raw.dependencies_ready()
    calls=[]
    monkeypatch.setattr(functions,'stage_ready',lambda path,flag:calls.append((path,flag)) or True)
    assert functions.dependencies_ready()
    assert calls==[(raw.OUT,'all_new_admitted_raw_train_windows_once'),(functions.FUNCTION_ADMISSION,'dataset_admitted_to_function_CE_training')]
    admission['admitted_to_training']=False
    (pack/'admission.json').write_text(json.dumps(admission))
    for module in (raw,functions):
        with pytest.raises(ValueError,match='complete data admission'):module.dependencies_ready()


def test_original_failed_reader_identity_is_rejected_even_with_same_paths(tmp_path,monkeypatch):
    (tmp_path/'plan.json').write_text(json.dumps(dict(owner=dict(pid=3815661,ticks='43628858'))))
    for module in (raw,functions):
        monkeypatch.setattr(module,'READER',tmp_path)
        with pytest.raises(ValueError,match='exact complete'):module.dependencies_ready()
    assert raw.READER_OWNER==functions.READER_OWNER==dict(pid=3824880,ticks='44031474')
    assert functions.PREVIOUS_PREPARATION==raw.OUT
