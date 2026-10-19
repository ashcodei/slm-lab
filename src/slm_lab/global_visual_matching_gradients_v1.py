"""Global paired matching through small Gloo host feature/gradient tables.

The full batch determines both directions' positives, negatives and means.
Each rank receives only its feature derivatives; the caller subsequently SUMs
parameter gradients exactly once. There is no DDP/world averaging here. Empty
local shards participate in collectives and perform no fabricated forward.
This primitive neither loads sources nor admits a full-size/GPU training run.
"""
from dataclasses import dataclass
import hashlib
import json
import math

import torch
from torch import distributed as dist

from .own_actor_vision_alignment_v1 import pair_positives,symmetric_pair_loss


def digest(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def signature(value):
    return (id(value),value._version,value.data_ptr(),tuple(value.shape),str(value.dtype),str(value.device))


@dataclass(frozen=True)
class GlobalMatchingGradients:
    rank: int
    world: int
    global_positions: tuple
    local_positions: tuple
    unweighted_loss: float
    weighted_loss: float
    matching_weight: float
    temperature: float
    actual_global_positive_cells: int
    global_features_sha256: str
    image_signature: tuple
    text_signature: tuple
    image_gradients: torch.Tensor
    text_gradients: torch.Tensor
    image_gradients_sha256: str
    text_gradients_sha256: str

    def record(self):
        return dict(rank=self.rank,world=self.world,global_positions=list(self.global_positions),
            local_positions=list(self.local_positions),unweighted_loss=self.unweighted_loss,
            weighted_loss=self.weighted_loss,matching_weight=self.matching_weight,temperature=self.temperature,
            actual_global_positive_cells=self.actual_global_positive_cells,global_features_sha256=self.global_features_sha256,
            local_image_gradient_shape=list(self.image_gradients.shape),local_text_gradient_shape=list(self.text_gradients.shape),
            complete_global_pairs_and_normalization=True,parameter_gradients_require_one_SUM_not_world_average=True,
            full_size_GPU_capacity_and_quality_unverified=True)


def global_matching_gradients(images,texts,*,global_positions,local_positions,pixel_keys,captions,
                              matching_weight=.1,temperature=.07,group=None):
    """Create the exact host-FP32 full-batch matching gradient for each shard.

    Inputs retain their local autograd graphs. Source positions are distinct
    original positions, ordered exactly as the proposed complete batch. The
    graph-to-source bindings and source admission belong to the calling stage.
    """
    distributed = dist.is_initialized()
    if distributed and dist.get_backend(group) != 'gloo':
        raise ValueError('An explicitly selected actual Gloo host/control group is required')
    if not distributed and group is not None:
        raise ValueError('An uninitialized process group cannot stand in for a serial call')
    world = dist.get_world_size(group) if distributed else 1
    rank = dist.get_rank(group) if distributed else 0
    def gather(value):
        if world == 1:return [value]
        result = [None]*world;dist.all_gather_object(result,value,group=group);return result
    error = None;specification = None
    try:
        if (not 1 <= world <= 4 or not isinstance(global_positions,list) or not 2 <= len(global_positions) <= 64
                or any(type(p) is not int or p < 0 for p in global_positions)
                or len(set(global_positions)) != len(global_positions)
                or local_positions != global_positions[rank::world]
                or not isinstance(images,torch.Tensor) or not isinstance(texts,torch.Tensor)
                or images.ndim != 2 or images.shape != texts.shape or images.shape[0] != len(local_positions)
                or not 1 <= images.shape[1] <= 1024 or images.device != texts.device or images.dtype != texts.dtype
                or not images.is_floating_point() or not texts.is_floating_point()
                or not bool(torch.isfinite(images).all()) or not bool(torch.isfinite(texts).all())
                or len(local_positions) and (not images.requires_grad or not texts.requires_grad)
                or len(pixel_keys) != len(global_positions) or len(captions) != len(global_positions)
                or any(type(v) not in (int,float) or not math.isfinite(v) or v <= 0 for v in [matching_weight,temperature])):
            raise ValueError('Complete bounded global sources, exact strided local features and positive finite objective settings required')
        positive = pair_positives(pixel_keys,captions)
        if bool(positive.all()):
            raise ValueError('This matching branch needs observed negatives; the calling stage must still retain every source CE target')
        specification = dict(global_positions=global_positions,pixel_keys=pixel_keys,captions=captions,
            feature_width=images.shape[1],feature_dtype=str(images.dtype),matching_weight=matching_weight,temperature=temperature)
        specification = hashlib.sha256(json.dumps(specification,sort_keys=True,ensure_ascii=True).encode()).hexdigest()
    except Exception as exc:error = type(exc).__name__+': '+str(exc)
    preflight = gather(dict(error=error,specification=specification))
    if any(item['error'] is not None for item in preflight):
        raise ValueError('Coordinated global matching preflight rejection: '+repr(preflight))
    if any(item['specification'] != specification for item in preflight):
        raise ValueError('Coordinated global matching source/order/precision/settings mismatch')
    width = images.shape[1];maximum_local = math.ceil(len(global_positions)/world)
    payload = torch.zeros(maximum_local,2*width,dtype=torch.float32,device='cpu')
    payload[:len(images),:width] = images.detach().to(device='cpu',dtype=torch.float32)
    payload[:len(images),width:] = texts.detach().to(device='cpu',dtype=torch.float32)
    if world == 1:pieces = [payload]
    else:
        pieces = [torch.empty_like(payload) for _ in range(world)]
        dist.all_gather(pieces,payload,group=group)
    features = torch.empty(len(global_positions),2*width,dtype=torch.float32,device='cpu')
    for source_rank,piece in enumerate(pieces):
        features[source_rank::world] = piece[:len(global_positions[source_rank::world])]
    with torch.enable_grad(),torch.autocast('cpu',enabled=False):
        all_images = features[:,:width].detach().clone().requires_grad_()
        all_texts = features[:,width:].detach().clone().requires_grad_()
        objective = symmetric_pair_loss(all_images,all_texts,positive,temperature=temperature)
        weighted = matching_weight*objective
        image_gradients,text_gradients = torch.autograd.grad(weighted,[all_images,all_texts])
    local_image = image_gradients[rank::world].detach().contiguous()
    local_text = text_gradients[rank::world].detach().contiguous()
    if not bool(torch.isfinite(local_image).all()) or not bool(torch.isfinite(local_text).all()):
        raise ValueError('The complete global objective must yield finite actual local feature derivatives')
    return GlobalMatchingGradients(rank,world,tuple(global_positions),tuple(local_positions),
        float(objective.detach()),float(weighted.detach()),matching_weight,temperature,int(positive.sum()),digest(features),
        signature(images),signature(texts),local_image,local_text,digest(local_image),digest(local_text))


def backward_global_matching(images,texts,gradients,*,parameters,retain_graph=False):
    """Backpropagate already globally normalized matching derivatives locally."""
    if (type(gradients) is not GlobalMatchingGradients or signature(images) != gradients.image_signature
            or signature(texts) != gradients.text_signature or gradients.image_gradients.shape != images.shape
            or gradients.text_gradients.shape != texts.shape
            or digest(gradients.image_gradients) != gradients.image_gradients_sha256
            or digest(gradients.text_gradients) != gradients.text_gradients_sha256
            or not isinstance(parameters,list) or not parameters or len({id(p) for p in parameters}) != len(parameters)
            or any(not isinstance(p,torch.nn.Parameter) or not p.requires_grad for p in parameters)
            or type(retain_graph) is not bool):
        raise ValueError('The exact untouched local feature/gradient plan and explicit distinct selected parameters required')
    if not gradients.local_positions:
        return dict(local_sources=0,local_backward_calls=0,no_fabricated_forward_or_gradient=True)
    torch.autograd.backward([images,texts],
        grad_tensors=[gradients.image_gradients.to(device=images.device,dtype=images.dtype),
                      gradients.text_gradients.to(device=texts.device,dtype=texts.dtype)],
        inputs=parameters,retain_graph=retain_graph)
    return dict(local_sources=len(gradients.local_positions),local_backward_calls=1,
        already_global_normalized=True,parameter_gradients_still_require_one_SUM=True)
