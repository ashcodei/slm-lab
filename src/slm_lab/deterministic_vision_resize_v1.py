"""Keep native vision forwards and use a CPU bicubic VJP during training.

PyTorch rejects CUDA bicubic backward with deterministic algorithms enabled.
Only the small learned position grid's vector-Jacobian product runs on CPU;
the forward, image encoder, parameters, state keys and exports stay native.
This first-order training bridge does not support higher-order gradients.
"""
import torch
from torch import nn
from torch.autograd.function import once_differentiable
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .model import VisionEncoder


class _NativeForwardCPUBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, grid, size):
        if grid.ndim != 4 or grid.dtype != torch.float32:
            raise ValueError('Native FP32 learned position grid required')
        ctx.shape = tuple(grid.shape)
        ctx.device = grid.device
        ctx.dtype = grid.dtype
        ctx.size = tuple(size)
        return F.interpolate(grid, size=ctx.size, mode='bicubic', align_corners=False)

    @staticmethod
    @once_differentiable
    def backward(ctx, gradient):
        # Bicubic interpolation is linear in its input, so this VJP depends
        # only on grid shape, interpolation settings and the output gradient.
        with torch.enable_grad():
            source = torch.zeros(ctx.shape, dtype=ctx.dtype, device='cpu', requires_grad=True)
            resized = F.interpolate(source, size=ctx.size, mode='bicubic', align_corners=False)
            result, = torch.autograd.grad(resized, source, gradient.detach().to(device='cpu', dtype=ctx.dtype))
        return result.to(device=ctx.device, dtype=ctx.dtype), None


class DeterministicVisionEncoder(VisionEncoder):
    def __init__(self, native):
        if type(native) is not VisionEncoder:
            raise ValueError('Wrap exactly one original own vision encoder')
        # Share every original parameter/module. Allocate no replacement
        # weights, consume no RNG, and preserve all native state_dict names.
        nn.Module.__init__(self)
        self.config = native.config
        self.position = native.position
        for name in ['patch', 'blocks', 'norm', 'project']:
            setattr(self, name, getattr(native, name))
        self.train(native.training)

    def forward(self, pixels, gradient_checkpointing=False):
        c = self.config
        if pixels.ndim != 4 or pixels.shape[1] != 3 or pixels.shape[2] != pixels.shape[3]:
            raise ValueError('Images must have shape [batch,3,side,side].')
        size = pixels.shape[2]
        c.image_token_count(size)
        side = size // c.patch_size
        position = self.position
        if size != c.image_size:
            base = c.image_size // c.patch_size
            grid = position.float().reshape(1, base, base, c.vision_hidden_size).permute(0, 3, 1, 2)
            if torch.is_grad_enabled() and grid.requires_grad:
                position = _NativeForwardCPUBackward.apply(grid, (side, side))
            else:
                position = F.interpolate(grid, size=(side, side), mode='bicubic', align_corners=False)
            position = position.flatten(2).transpose(1, 2).to(self.position.dtype)
        x = self.patch(pixels).flatten(2).transpose(1, 2) + position
        for block in self.blocks:
            if gradient_checkpointing and self.training:
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        x = self.norm(x).transpose(1, 2).reshape(pixels.shape[0], c.vision_hidden_size, side, side)
        x = F.avg_pool2d(x, c.pool_size, c.pool_size).flatten(2).transpose(1, 2)
        return self.project(x)
