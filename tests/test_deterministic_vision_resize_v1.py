import copy
import json
import math
import os
from pathlib import Path

import pytest
import torch
from torch import distributed as dist
from torch.multiprocessing import spawn
from torch.nn import functional as F

from slm_lab.deterministic_vision_resize_v1 import DeterministicVisionEncoder, _NativeForwardCPUBackward
from slm_lab.model import ModelConfig, VisionEncoder, TinyMultimodalLM
from slm_lab.point_grounding_head_v1 import PointGroundingHead
from slm_lab.point_grounding_stage_v1 import PointerStage, make_optimizer, frozen_digest


def test_bicubic_cpu_vjp_matches_native_and_finite_difference():
    torch.set_num_threads(1)
    torch.manual_seed(617)
    original = torch.randn(1, 3, 3, 5, requires_grad=True)
    source = original.detach().clone().requires_grad_()
    expected = F.interpolate(original, size=(7, 9), mode='bicubic', align_corners=False)
    actual = _NativeForwardCPUBackward.apply(source, (7, 9))
    weights = torch.randn_like(expected)
    assert torch.equal(actual, expected)
    (expected * weights).sum().backward()
    (actual * weights).sum().backward()
    assert torch.equal(source.grad, original.grad)
    coordinate = (0, 1, 2, 3)
    eps = .002
    plus = source.detach().clone(); plus[coordinate] += eps
    minus = source.detach().clone(); minus[coordinate] -= eps
    finite = ((_NativeForwardCPUBackward.apply(plus, (7, 9)) -
               _NativeForwardCPUBackward.apply(minus, (7, 9))) * weights).sum() / (2 * eps)
    assert float(finite) == pytest.approx(float(source.grad[coordinate]), abs=.002)


@pytest.mark.parametrize('side', [16, 32])
@pytest.mark.parametrize('recompute', [False, True])
def test_complete_vision_preserves_parameter_identity_rng_native_forward_and_all_gradients(side, recompute):
    torch.set_num_threads(1)
    torch.manual_seed(618)
    config = ModelConfig(vocab_size=32, hidden_size=16, intermediate_size=32, num_layers=1,
        num_heads=4, num_kv_heads=2, max_text_length=64, vision_hidden_size=8,
        vision_intermediate_size=16, vision_layers=1, vision_heads=2, image_size=16, patch_size=4, pool_size=2)
    native = VisionEncoder(config)
    independent = copy.deepcopy(native)
    state = torch.get_rng_state().clone()
    wrapped = DeterministicVisionEncoder(native)
    assert torch.equal(state, torch.get_rng_state())
    assert wrapped.state_dict().keys() == native.state_dict().keys()
    assert [id(p) for p in wrapped.parameters()] == [id(p) for p in native.parameters()]
    pixels = torch.randn(2, 3, side, side)
    previous = torch.are_deterministic_algorithms_enabled()
    try:
        torch.use_deterministic_algorithms(True)
        expected = independent(pixels, gradient_checkpointing=recompute)
        actual = wrapped(pixels, gradient_checkpointing=recompute)
        assert torch.equal(expected, actual)
        expected.square().mean().backward(); actual.square().mean().backward()
        for (name, parameter), (_, reference) in zip(wrapped.named_parameters(), independent.named_parameters()):
            assert torch.equal(parameter.grad, reference.grad), name
        with torch.no_grad():
            assert torch.equal(wrapped(pixels), independent(pixels))
        with pytest.raises(ValueError, match='exactly one'):
            DeterministicVisionEncoder(wrapped)
    finally:
        torch.use_deterministic_algorithms(previous)


class ResizeFixture:
    role = 'train'
    purpose = 'optimization'

    def __len__(self):
        return 37

    def batch(self, positions, device):
        ids = torch.tensor([[1, 4 + p % 8, 6, 2] for p in positions], device=device)
        pixels = torch.stack([torch.linspace(-1, 1, 3 * 32 * 32).reshape(3, 32, 32) + p % 3 * .1
                              for p in positions]).to(device)
        target = torch.zeros(len(positions), 16, device=device)
        for i, p in enumerate(positions):
            target[i, p % 16] = 4 + p % 3
        return dict(input_ids=ids, attention_mask=torch.ones_like(ids, dtype=torch.bool), pixel_values=pixels,
                    targets=target, valid_cells=torch.ones_like(target, dtype=torch.bool))


def resize_worker(rank, folder):
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    folder = Path(folder)
    dist.init_process_group('gloo', init_method='file://' + str(folder / 'rendezvous'), rank=rank, world_size=4)
    try:
        options = dict(seed=13977, world_size=4, microbatch_per_rank=2, accumulation=4,
            global_queries_per_update=32, epochs=1, steps=math.ceil(37 / 32), planned_complete_query_exposures=64,
            rounding_repeated_queries=27, vision_learning_rate=2e-4, head_learning_rate=2e-3,
            betas=[.9, .95], weight_decay=.01, gradient_norm_clip=1., warmup_fraction=.1,
            cosine_floor_fraction=.1, image_size=32, maximum_whole_query_tokens=34)
        def fresh():
            torch.manual_seed(1093)
            config = ModelConfig(vocab_size=32, hidden_size=16, intermediate_size=32, num_layers=1,
                num_heads=4, num_kv_heads=2, max_text_length=64, vision_hidden_size=8,
                vision_intermediate_size=16, vision_layers=1, vision_heads=2, image_size=16, patch_size=4, pool_size=2)
            head = PointGroundingHead(TinyMultimodalLM(config), rank=8)
            head.base.vision = DeterministicVisionEncoder(head.base.vision)
            return head
        head = fresh(); before = frozen_digest(head)
        kernel = PointerStage(head, make_optimizer(head, options), ResizeFixture(), options, {'bridge': 'v1'}, 'cpu')
        kernel.update(); kernel.save(folder / 'resume.pt', 20 * 1024**2, 0)
        expected = kernel.update()
        restored = fresh()
        again = PointerStage(restored, make_optimizer(restored, options), ResizeFixture(), options, {'bridge': 'v1'}, 'cpu')
        again.resume(torch.load(folder / 'resume.pt', map_location='cpu', weights_only=True))
        assert again.update() == expected
        assert all(torch.equal(t, restored.state_dict()[n]) for n, t in head.state_dict().items())
        assert frozen_digest(restored) == before
        if rank == 0:
            (folder / 'evidence.json').write_text(json.dumps(dict(actual_CPU_workers=4, resized_image_side=32,
                native_image_side=16, deterministic_algorithms=True, exact_next_metrics_and_all_parameters=True)))
    finally:
        dist.destroy_process_group()


def test_four_workers_exact_complete_query_resume_with_resized_position_gradients(tmp_path):
    spawn(resize_worker, args=(str(tmp_path),), nprocs=4, join=True)
    assert json.loads((tmp_path / 'evidence.json').read_text())['exact_next_metrics_and_all_parameters']
