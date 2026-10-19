"""Nonmutating CPU FP32 folding of our own dense policy adapters.

For each projection W, replace the copied weight by W + up @ down; copy
trained policy norms and the separate policy critic. All image/word/reward/
pointing/router/detector parameters remain in the single resulting backbone.

Weight merging and repeated adapter resets are established techniques, not a
novel algorithm. See https://arxiv.org/abs/2307.05695 . SLM evidence is mixed:
https://arxiv.org/abs/2509.12960 . This utility establishes neither a quality
gain nor FP32/BF16/CUDA generation equivalence at production scale. It never
installs a model, changes a live lineage, or reuses an old optimizer after reset.
"""
import copy

import torch

from .own_dense_actor_v1 import OwnDenseActor
from .own_code_delta_v1 import matrix_names,norm_names
from .whole_human_sft_evidence_v1 import tensor_digest

FORMAT='own-nonmutating-FP32-dense-policy-fold-v1'


def signature(model):
    return tuple((n,id(p),p._version,p.data_ptr(),p.dtype,p.device,p.requires_grad)
        for n,p in model.named_parameters())


def sources_valid(sources):
    return (isinstance(sources,dict) and bool(sources)
        and all(isinstance(k,str) and bool(k) and isinstance(v,str) and len(v)==64
            and all(c in '0123456789abcdef' for c in v) for k,v in sources.items()))


@torch.no_grad()
def fold_actor(model,*,sources,working_budget_bytes):
    if type(model) is not OwnDenseActor or not sources_valid(sources):
        raise ValueError('Exact own dense actor and explicit source SHA bindings required')
    model.assert_shared()
    if (model.training or any(p.device.type!='cpu' or p.dtype!=torch.float32 or p.grad is not None for p in model.parameters())
            or any(m._forward_hooks or m._forward_pre_hooks or m._backward_hooks for m in model.modules())):
        raise ValueError('Completed eval-mode CPU FP32 actor with no gradients or active hooks required')
    count=model.base.counts()['total_unique']
    names=matrix_names(model.config.num_layers)
    biggest=max(model.base.network.get_parameter(n).numel() for n in names)
    biggest_source=max(p.numel() for p in model.parameters())
    # Output clone, matrix product plus addition scratch, finite masks, and
    # interpreter/allocator headroom. Existing source ownership is additional.
    required=4*count+8*biggest+biggest_source+128*1024**2
    if type(working_budget_bytes) is not int or working_budget_bytes<required:
        raise ValueError('An explicit sufficient additional CPU working-memory budget is required')
    if any(not bool(torch.isfinite(p).all()) for p in model.parameters()):
        raise ValueError('Every original dense/head/actor/critic value must be finite')
    before=signature(model);rng=torch.get_rng_state().clone();digest=tensor_digest(model.named_parameters())
    # No source module, source tensor, learned head or cache is registered in
    # the output. Deepcopy preserves the existing tied vocabulary internally.
    folded=copy.deepcopy(model.base)
    with torch.autocast('cpu',enabled=False):
        for name,parts in zip(names,model.parts['linears']):
            folded.network.get_parameter(name).add_(parts.up@parts.down)
        for name,norm in zip(norm_names(model.config.num_layers),model.parts['norms']):
            folded.network.get_parameter(name).copy_(norm.weight)
        folded.network.value_head.weight.copy_(model.parts['critic'].weight)
    folded.eval();folded.network.gradient_checkpointing=False;folded.assert_inventory()
    if (folded.counts()['total_unique']!=count
            or folded.network.embedding.weight is not folded.network.lm_head.weight
            or set(p.data_ptr() for p in folded.parameters()) & set(p.data_ptr() for p in model.parameters())
            or any(not bool(torch.isfinite(p).all()) for p in folded.parameters())):
        raise ValueError('One complete finite tied dense model with no source aliases required')
    if before!=signature(model) or digest!=tensor_digest(model.named_parameters()) or not torch.equal(rng,torch.get_rng_state()):
        raise ValueError('Source values, graph, metadata and caller RNG must remain unchanged')
    report=dict(format=FORMAT,sources=dict(sources),source_counts=model.counts(),folded_counts=folded.counts(),
        source_parameters_sha256=digest,folded_parameters_sha256=tensor_digest(folded.named_parameters()),
        merged_projection_matrices=len(names),copied_policy_norms=len(norm_names(model.config.num_layers)),
        policy_critic_folded_into_existing_dense_value_head=True,all_existing_multimodal_and_auxiliary_heads_retained=True,
        removed_adapter_and_separate_critic_values=model.counts()['actual_unique']-count,
        source_unchanged=True,source_and_result_share_no_parameter_storage=True,arithmetic='CPU FP32 with ambient autocast disabled',
        required_additional_working_bytes=required,declared_additional_working_bytes=working_budget_bytes,
        FP32_arithmetic_order_changes=True,BF16_and_generation_equivalence_not_certified=True,
        full_size_capacity_and_quality_and_throughput_pending=True,candidate_promoted=False,goal_achieved=False)
    return folded,report


@torch.no_grad()
def recycle_actor(model,*,sources,working_budget_bytes,seed):
    if type(seed) is not int or seed<1:raise ValueError('Explicit positive next-cycle seed required')
    if type(model) is not OwnDenseActor or type(working_budget_bytes) is not int:
        raise ValueError('Exact own actor and explicit whole-cycle additional memory budget required')
    fresh_part_bytes=4*sum(p.numel() for p in model.parts.parameters())
    folded,report=fold_actor(model,sources=sources,working_budget_bytes=working_budget_bytes-fresh_part_bytes)
    # The original optimizer belongs to old factor tensors and must not be
    # attached to this graph. A training controller must make a fresh optimizer
    # and explicitly bind this changed dense reference and cycle schedule.
    recycled=OwnDenseActor(folded,rank=model.settings['rank'],seed=seed,
        parameter_limit=model.settings['parameter_limit']).eval()
    recycled.parts['critic'].weight.copy_(folded.network.value_head.weight)
    report.update(next_cycle_seed=seed,next_cycle_counts=recycled.counts(),
        required_additional_working_bytes=report['required_additional_working_bytes']+fresh_part_bytes,
        declared_additional_working_bytes=working_budget_bytes,new_actor_parameter_bytes=fresh_part_bytes,
        fresh_zero_up_factors=True,previous_learned_critic_preserved=True,
        new_optimizer_and_new_reference_bindings_required=True,production_cycle_controller_not_installed=True)
    return recycled,report
