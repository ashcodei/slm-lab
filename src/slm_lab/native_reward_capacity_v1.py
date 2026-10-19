"""Actual train-only feature batch extremes for each original physical L40."""
import torch

from .codenet_checkpoint_v1 import capture_rng
from .native_human_reward_v1 import native_features, scope, signature
from .parallel_function_repair_stage_v1 import value_digest
from .whole_human_reward_v1 import feature_batches
from .whole_human_sft_evidence_v1 import tensor_digest

FORMAT = 'own-current-native-whole-human-FP32-feature-GPU-capacity-v1'


def cases(data, rank, world, options):
    if ((data.family, data.role, data.purpose) != ('preferences', 'train', 'optimization')
            or type(world) is not int or world < 1 or type(rank) is not int or not 0 <= rank < world):
        raise ValueError('Only original train-only whole-pair feature shards can supply capacity inputs')
    batches = list(feature_batches(data, list(range(rank, len(data), world)),
        token_budget=options['token_budget'], max_sequences=options['maximum_sequences']))
    if not batches: raise ValueError('Every actual feature capacity shard must be nonempty')
    selected = [max(batches, key=lambda b: len(b)*max(x[2] for x in b)), max(batches, key=lambda b: max(x[2] for x in b))]
    result = []
    for batch in selected:
        if batch not in result: result.append(batch)
    return result


def batch(data, items, device):
    length = max(r[2] for r in items); ids = torch.zeros(len(items), length, dtype=torch.long, device=device)
    mask = torch.zeros_like(ids, dtype=torch.bool)
    for i, (position, field, size) in enumerate(items):
        original, _ = data.sequence(position, field)
        if len(original) != size or original[-1] != 2: raise ValueError('Every complete original response/EOS must reach capacity')
        ids[i, :size] = torch.tensor(original, device=device); mask[i, :size] = True
    return ids, mask


@torch.no_grad()
def prove(base, data, rank, world, sources, options, *, physical_GPU, checked=lambda: None, allow_mechanics=False):
    full, guard = scope(base, data, sources, allow_mechanics); device = next(base.parameters()).device
    if full and (device.type != 'cuda' or device.index != 0 or world != 4 or physical_GPU != rank
            or torch.cuda.get_device_name(device) != 'NVIDIA L40'):
        raise ValueError('Every full-size capacity proof requires its actual isolated original L40')
    if not full and device.type != 'cpu': raise ValueError('Small capacity controls are CPU only')
    before = signature(base.parameters()); rng = value_digest(capture_rng(device)); results = []
    if full: torch.cuda.reset_peak_memory_stats(device)
    for items in cases(data, rank, world, options):
        checked(); ids, mask = batch(data, items, device); values = native_features(base, ids, mask)
        results.append(dict(original_source_items=[list(r) for r in items], original_padded_shape=list(ids.shape),
            complete_original_context_tokens=int(mask.sum()), actual_feature_shape=list(values.shape),
            actual_FP32_features_sha256=value_digest(values), all_finite=True))
    checked()
    if guard is not None: guard.check(full=True)
    if (before != signature(base.parameters()) or value_digest(capture_rng(device)) != rng
            or tensor_digest(base.network.named_parameters()) != sources['frozen_native_encoder_sha256']):
        raise ValueError('Actual feature extremes must preserve every native value, graph and private generator')
    return dict(format=FORMAT, complete=True, physical_GPU=physical_GPU, original_shard_rank=rank, original_world=world,
        actual_device_type=device.type, actual_device_index=device.index, sources=dict(sources), options=dict(options),
        actual_native_values=base.counts()['total_unique'], actual_complete_capacity_cases=results,
        maximum_padded_and_longest_whole_original_batches_checked=True, all_values_metadata_and_private_RNG_unchanged=True,
        full_size_GPU_feature_capacity_passed=full,
        actual_peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if full else 0,
        actual_peak_reserved_bytes=torch.cuda.max_memory_reserved(device) if full else 0,
        reward_head_optimizer_updates=0, PPO_training_admitted=False, goal_achieved=False)
