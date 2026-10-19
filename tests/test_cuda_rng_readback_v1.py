import pytest
import torch

from slm_lab.cuda_rng_readback_v1 import review_cuda_rng_layout


def test_opaque_sixteen_byte_CUDA_state_layout_is_accepted_without_any_GPU_initialization(monkeypatch):
    monkeypatch.setattr(torch.cuda,'_lazy_init',lambda:pytest.fail('CPU readback must not initialize a GPU'))
    value=torch.arange(16,dtype=torch.uint8);before=value.clone()
    result=review_cuda_rng_layout(value)
    assert result['serialized_bytes']==16 and result['CPU_byte_layout_checked'] and torch.equal(value,before)
    assert result['semantic_CUDA_state_validation_requires_owned_GPU_private_generator']
    # A structural reader deliberately does not claim that an arbitrary byte
    # payload is a valid CUDA generator state; the private GPU check does that.
    assert review_cuda_rng_layout(torch.zeros(4,dtype=torch.uint8))['serialized_bytes']==4


def test_empty_wrong_dtype_wrong_rank_and_missing_RNG_serializations_reject_without_mutation():
    for value in (None,[],torch.zeros(0,dtype=torch.uint8),torch.zeros(16),torch.zeros(1,16,dtype=torch.uint8)):
        with pytest.raises(ValueError,match='opaque CUDA RNG'):review_cuda_rng_layout(value)
