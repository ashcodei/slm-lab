import copy

import pytest
import torch

from slm_lab.corpus import file_hash
from slm_lab.own_native_query_v1 import inspect_package,load_package
from slm_lab.own_native_training_factory_v1 import load_package_for_training
from test_own_dense_standalone_v1 import fixture


def test_validated_CPU_graph_load_works_with_initialized_flag_without_relaxing_CPU_query_contract(tmp_path,monkeypatch):
    model,ids,path,sources=fixture(tmp_path)
    package=inspect_package(path,expected_sha256=file_hash(path),expected_sources=sources)
    monkeypatch.setattr(torch.cuda,'is_initialized',lambda:True)
    with pytest.raises(ValueError,match='complete CPU'):load_package(package)
    rng=torch.get_rng_state().clone();cold,metadata=load_package_for_training(package)
    assert metadata['construction_on_CPU_before_explicit_training_device_transfer']
    assert torch.equal(rng,torch.get_rng_state()) and all(p.device.type=='cpu' for p in cold.parameters())
    assert all(torch.equal(a,b) for a,b in zip(model.parameters(),cold.parameters()))
    assert not {p.data_ptr() for p in model.parameters()} & {p.data_ptr() for p in cold.parameters()}
    with torch.no_grad():assert torch.equal(model(ids)['logits'],cold(ids)['logits'])


def test_changed_declared_count_source_or_tokenizer_is_still_rejected(tmp_path):
    _,_,path,sources=fixture(tmp_path);package=inspect_package(path,expected_sha256=file_hash(path),expected_sources=sources)
    bad=copy.deepcopy(package);bad['declared_counts']['total_unique']+=1
    with pytest.raises(ValueError):load_package_for_training(bad)
    bad=copy.deepcopy(package);bad['source_identities']={'different_own_source':'b'*64}
    with pytest.raises(ValueError):load_package_for_training(bad)
    (tmp_path/'tokenizer.json').write_text('{}')
    with pytest.raises(ValueError):load_package_for_training(package)
