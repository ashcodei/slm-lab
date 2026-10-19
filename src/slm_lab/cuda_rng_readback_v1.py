"""CPU-only structural readback of opaque CUDA generator serialization.

State byte counts depend on the runtime. CPU readers check an intact nonempty
byte vector; the owned GPU worker must validate contents with a private CUDA
generator and demonstrate exact next-update recovery. Layout is not validity.
"""
import torch


def review_cuda_rng_layout(state):
    if (not isinstance(state,torch.Tensor) or state.device.type!='cpu' or state.dtype!=torch.uint8
            or state.ndim!=1 or not state.numel()):
        raise ValueError('A retained nonempty CPU byte vector for the original opaque CUDA RNG state is required')
    return dict(serialized_bytes=state.numel(),CPU_byte_layout_checked=True,
        semantic_CUDA_state_validation_requires_owned_GPU_private_generator=True)
