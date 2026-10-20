"""Materialize a complete coding epoch without resuming Adam or caller RNG.

Actual source order, whole masks, objective arithmetic and private-state
inventories must agree before any selected value is copied. CUDA RNG bytes
are structurally read on CPU; their semantic recovery belongs to GPU proofs.
"""
import math

import torch

from .code_residual_checkpoint_v1 import _validate_adam, _validate_rng
from .cuda_rng_readback_v1 import review_cuda_rng_layout
from .own_dense_actor_checkpoint_v1 import FORMAT
from .own_dense_actor_v1 import OwnDenseActor
from .parallel_function_repair_stage_v1 import value_digest
from .serial_public_code_epoch_v1 import reconcile, schedule
from .whole_human_sft_evidence_v1 import tensor_digest


def install_completed_policy(model, data, state, proof, *, profile, expected_sources,
        optimizer_options, expected_capacity_rng=None, allow_mechanics=False):
    if (type(model) is not OwnDenseActor or model.training
            or any(p.device.type != 'cpu' or p.dtype != torch.float32 or p.grad is not None for p in model.parameters())):
        raise ValueError('A fresh complete eval-mode own CPU policy is required')
    entries = schedule(data)
    planned = len(entries)
    small = allow_mechanics is True and model.counts()['actual_unique'] < 10**7
    if not small and (allow_mechanics or model.counts()['actual_unique'] != 999706442
            or model.base.counts()['total_unique'] != 984399690 or len(data) != 3675
            or planned != 230 or len(entries[-1]) != 11
            or optimizer_options != dict(lr=3e-5,betas=(.9,.95),eps=1e-8,weight_decay=.01,foreach=False,fused=False)):
        raise ValueError('The complete real own999M coding epoch and its original fixed optimizer are required')
    named = model.named_selected(('actor',))
    if (state.get('format') != FORMAT or state.get('world_size') != 1 or state.get('groups') != ['actor']
            or state.get('optimized_names') != [n for n,_ in named]
            or type(state.get('completed')) is not int or state['completed'] != planned or state.get('planned') != planned
            or state.get('cursor') != {'next_global_batch':planned}
            or state.get('candidate_promoted') is not False or state.get('goal_achieved') is not False
            or proof.get('actual_registered_parameters') != model.counts()['actual_unique']
            or proof.get('actual_selected_Adam_states') != len(named)):
        raise ValueError('The complete exact source cursor, selected inventory and full-size training proof are required')
    summary = reconcile(data, entries, state['metrics'], profile)
    if any(proof.get(k) != v for k,v in summary.items()):
        raise ValueError('Every independently reconstructed whole-epoch result must match the completed producer')
    for row in state['metrics']:
        meta = row['source_metadata']
        ce = math.fsum(m['original_CE_sum'] for m in row['actual_microbatches'])
        eos = math.fsum(m['EOS_CE_sum'] for m in row['actual_microbatches'])
        expected = dict(original_token_mean_NLL=ce/meta['response_EOS_targets'],
            EOS_token_mean_NLL=eos/meta['genuine_EOS_targets'],
            weighted_token_mean_CE=(ce+(meta['EOS_weight']-1)*eos)/meta['actual_global_weighted_targets'])
        if (not math.isfinite(row['gradient_norm_before_clip']) or row['gradient_norm_before_clip'] < 0
                or row['selected_Adam_states'] != len(named)
                or any(not math.isfinite(value) or value < 0 or not math.isclose(row[k],value,rel_tol=1e-12,abs_tol=1e-12)
                    for k,value in expected.items())
                or any(m['forward_dtype'] != 'torch.bfloat16' or m['loss_dtype'] != 'torch.float32'
                    or not math.isfinite(m['original_CE_sum']) or not math.isfinite(m['EOS_CE_sum'])
                    or m['EOS_CE_sum'] < 0 or m['original_CE_sum'] < m['EOS_CE_sum']
                    for m in row['actual_microbatches'])):
            raise ValueError('Every actual complete source objective/precision/gradient record must reconstruct')
    initial = tensor_digest(model.parts.named_parameters())
    fixed = tensor_digest(model.base.named_parameters())
    if (proof.get('initial_selected_policy_sha256') != initial or proof.get('frozen_native_sha256') != fixed
            or proof.get('final_selected_policy_sha256') != tensor_digest(state['model']['parts'].items())
            or proof['final_selected_policy_sha256'] == initial):
        raise ValueError('The identical original initialization, frozen native and actually learned final policy must agree')
    model.validate_export(state['model'], expected_sources=expected_sources)
    adam = torch.optim.AdamW([p for _,p in named], **optimizer_options)
    _validate_adam(state['optimizer'], [p for _,p in named], planned)
    if state['optimizer']['param_groups'] != adam.state_dict()['param_groups']:
        raise ValueError('The exact original ordinary selected Adam options and parameter order are required')
    selected_names = {n for n,_ in named}
    if any(not torch.equal(t,state['model']['parts'][n]) for n,t in model.parts.state_dict().items() if n not in selected_names):
        raise ValueError('The original unused critic and every unoptimized part must remain exact')
    cuda = state['rng'].get('cuda')
    _validate_rng({**state['rng'],'cuda':None}, 'cpu')
    cuda_review = None if small and cuda is None else review_cuda_rng_layout(cuda)
    if not small and (expected_capacity_rng is None or value_digest(state['rng']) != value_digest(expected_capacity_rng)):
        raise ValueError('The complete unchanged private RNG must equal the actual semantically recovered capacity state')
    nonzero = sum(bool(v['exp_avg'].abs().sum() > 0) for v in state['optimizer']['state'].values())
    if nonzero == 0: raise ValueError('Actual nonzero learned first moments are required beyond weight decay')
    model.import_export(state['model'], expected_sources=expected_sources)
    return dict(complete=True, original_complete_epoch=summary,
        completed_updates=planned, actual_registered_parameters=model.counts()['actual_unique'],
        all_actual_selected_Adam_states=len(named), Adam_clock=planned, nonzero_first_moment_tensors=nonzero,
        opaque_CUDA_RNG_layout=cuda_review, private_CPU_RNG_inventory_checked_without_restoring_caller=True,
        private_RNG_matches_GPU_recovered_capacity_state=not small,
        actual_GPU_RNG_semantic_recovery_not_replayed=True, selected_values_installed_without_Adam_or_RNG_resume=True,
        original_critic_and_every_frozen_native_value_preserved=True, candidate_promoted=False, goal_achieved=False)
