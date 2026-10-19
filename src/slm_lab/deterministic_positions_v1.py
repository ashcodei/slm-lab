"""Differentiable separable bicubic position grids using deterministic matmuls.

The coefficients are derived from PyTorch's CPU align_corners=False bicubic
operator on basis vectors. They are geometric constants, not learned weights.
Checkpoint tensor names/shapes/counts and the base-resolution path are intact.
"""
from functools import lru_cache
from types import MethodType

import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

PROTOCOL='separable-bicubic-CPU-basis-FP32-matmul-align-corners-false-v1'


@lru_cache(maxsize=32)
def coefficients(original,target):
    basis=torch.eye(original,dtype=torch.float32).reshape(original,1,original,1)
    return F.interpolate(basis,size=(target,1),mode='bicubic',align_corners=False)[:,0,:,0].T.contiguous()


def resize(grid,size):
    if grid.ndim!=4 or len(size)!=2 or any(type(n) is not int or n<1 for n in size):raise ValueError('Complete NCHW position grid and positive target shape required')
    h,w=grid.shape[-2:]
    if (h,w)==tuple(size):return grid
    with torch.autocast(device_type=grid.device.type,enabled=False):
        sh=coefficients(h,size[0]).to(grid.device);sw=coefficients(w,size[1]).to(grid.device)
        return torch.matmul(torch.matmul(sh,grid.float()),sw.T).to(grid.dtype)


def vision_forward(self,pixels,gradient_checkpointing=False):
    c=self.config
    if pixels.ndim!=4 or pixels.shape[1]!=3 or pixels.shape[2]!=pixels.shape[3]:raise ValueError('Images must have shape [batch,3,side,side].')
    size=pixels.shape[2];c.image_token_count(size);side=size//c.patch_size;position=self.position
    if size!=c.image_size:
        base=c.image_size//c.patch_size;grid=position.float().reshape(1,base,base,c.vision_hidden_size).permute(0,3,1,2)
        position=resize(grid,(side,side)).flatten(2).transpose(1,2).to(self.position.dtype)
    x=self.patch(pixels).flatten(2).transpose(1,2)+position
    for block in self.blocks:x=checkpoint(block,x,use_reentrant=False) if gradient_checkpointing and self.training else block(x)
    x=self.norm(x).transpose(1,2).reshape(pixels.shape[0],c.vision_hidden_size,side,side)
    return self.project(F.avg_pool2d(x,c.pool_size,c.pool_size).flatten(2).transpose(1,2))


def patch_grid(self,pixels):
    c=self.config;vision=self.network.vision
    if pixels.ndim!=4 or pixels.shape[1]!=3 or any(side<c.patch_size or side>c.max_image_size or side%c.patch_size for side in pixels.shape[-2:]):
        raise ValueError('Actual bounded patch-aligned complete RGB image required')
    h,w=(side//c.patch_size for side in pixels.shape[-2:]);base=c.image_size//c.patch_size;position=vision.position
    if (h,w)!=(base,base):
        position=resize(position.float().reshape(1,base,base,c.vision_hidden_size).permute(0,3,1,2),(h,w)).flatten(2).transpose(1,2).to(position.dtype)
    x=vision.patch(pixels).flatten(2).transpose(1,2)+position
    for block in vision.blocks:x=checkpoint(block,x,use_reentrant=False) if self.network.gradient_checkpointing and self.training else block(x)
    return vision.norm(x).reshape(pixels.shape[0],h,w,c.vision_hidden_size)


def install(model):
    before=model.counts();model.network.vision.forward=MethodType(vision_forward,model.network.vision)
    model.patch_grid=MethodType(patch_grid,model)
    if model.counts()!=before:raise ValueError('Position arithmetic must add no learned parameter or extra model')
    return model


def check_operator(device):
    """Actual 224->448 and 64x448 word shapes, with CPU gradient oracle."""
    rows=[]
    for size in [(28,28),(4,28)]:
        values=torch.linspace(-1,1,512*14*14).reshape(1,512,14,14)
        x=values.to(device).detach().requires_grad_();y=resize(x,size)
        weight=torch.linspace(-.5,.5,y.numel(),device=device).reshape_as(y);(y*weight).sum().backward()
        reference=values.detach().clone().requires_grad_();expected=F.interpolate(reference,size=size,mode='bicubic',align_corners=False)
        (expected*weight.cpu()).sum().backward()
        forward_error=float((y.detach().cpu()-expected.detach()).abs().max());gradient_error=float((x.grad.cpu()-reference.grad).abs().max())
        if not torch.allclose(y.detach().cpu(),expected.detach(),atol=2e-5,rtol=2e-5) or not torch.allclose(x.grad.cpu(),reference.grad,atol=2e-5,rtol=2e-5):
            raise ValueError('Full-size position forward/backward must match the independent CPU bicubic oracle')
        rows.append(dict(target_grid=list(size),forward_max_absolute_error=forward_error,position_gradient_max_absolute_error=gradient_error,
            finite_actual_position_gradient=bool(torch.isfinite(x.grad).all())))
    return rows
