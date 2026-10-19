"""Bounded whole-pair CE/DPO gradients for a shared mutable-vision policy.

The reference is strictly text-only and shares unchanged own dense language.
All original pair and chosen-token denominators are global before replica
SUM. Microbatching changes no source, response whitespace, target or EOS.
This primitive performs no optimizer step or production training admission.
"""
from contextlib import nullcontext
import math

import torch
from torch.nn import functional as F

from .dense_human_preference_v1 import pair_information
from .frozen_own_text_reference_v1 import FrozenOwnTextReferenceV1
from .parallel_function_repair_stage_v1 import value_digest
from .shared_joint_actor_vision_v1 import SharedJointActorVisionV1
from .whole_human_data_v2 import WholeHumanData

FORMAT = 'own-batched-joint-text-only-human-DPO-global-pair-token-gradient-v1'


def row_logp(logits, labels):
    if logits.shape[:2] != labels.shape or logits.ndim != 3 or labels.ndim != 2:
        raise ValueError('Complete native sequence logits and exact response/EOS labels are required')
    targets = labels[:, 1:]; mask = targets != -100
    counts = mask.sum(1)
    if not bool((counts > 0).all()): raise ValueError('Every complete response must retain at least its true EOS')
    selected = logits[:, :-1][mask].float()
    losses = F.cross_entropy(selected, targets[mask], reduction='none')
    rows = torch.arange(labels.shape[0], device=labels.device)[:, None].expand_as(targets)[mask]
    logp = torch.zeros(labels.shape[0], device=logits.device, dtype=torch.float32).scatter_add(0, rows, -losses)
    if not bool(torch.isfinite(logp).all()): raise ValueError('Every complete sequence log probability must be finite')
    return logp, counts


def plan(data, positions, *, rank, world, DPO_weight, maximum_examples, maximum_padded_tokens):
    if (type(data) is not WholeHumanData or (data.family, data.role) != ('preferences', 'train')
            or data.purpose not in ['optimization', 'capacity'] or type(positions) is not list or not positions
            or len(set(positions)) != len(positions) or any(type(p) is not int or not 0 <= p < len(data) for p in positions)
            or type(rank) is not int or type(world) is not int or not 0 <= rank < world <= 4
            or type(DPO_weight) is not int or DPO_weight not in [0, 1]
            or any(type(v) is not int or v < 1 for v in [maximum_examples, maximum_padded_tokens])):
        raise ValueError('Exact distinct train-only complete human pairs and bounded actual replica/microbatch roles are required')
    information = [pair_information(data, p) for p in positions]
    def cost(batch):
        chosen = max(data.rows[p]['native']['chosen']['length'] for p in batch)
        rejected = max(data.rows[p]['native']['rejected']['length'] for p in batch) if DPO_weight else 0
        return len(batch) * (chosen + rejected)
    # Every global extreme is checked even on empty ranks, before any forward.
    if any(cost([p]) > maximum_padded_tokens for p in positions):
        raise ValueError('The original complete pair exceeds the declared microbatch bound; clipping is forbidden')
    local = positions[rank::world]
    order = sorted(local, key=lambda p: (cost([p]), p))
    batches = []; batch = []
    for position in order:
        if batch and (len(batch) == maximum_examples or cost(batch + [position]) > maximum_padded_tokens):
            batches.append(batch); batch = []
        batch.append(position)
    if batch: batches.append(batch)
    return information, batches, [cost(batch) for batch in batches]


def gradients(view, reference, data, positions, *, DPO_weight, beta=.1, chosen_CE_weight=.1,
              maximum_examples=4, maximum_padded_tokens=16384, rank=0, world=1, device='cpu', amp=None):
    info, batches, padded = plan(data, positions, rank=rank, world=world, DPO_weight=DPO_weight,
        maximum_examples=maximum_examples, maximum_padded_tokens=maximum_padded_tokens)
    if (type(view) is not SharedJointActorVisionV1 or type(beta) not in [int, float] or not math.isfinite(beta) or not 0 < beta <= 1
            or type(chosen_CE_weight) not in [int, float] or not math.isfinite(chosen_CE_weight)
            or not 0 <= chosen_CE_weight <= 1 or not DPO_weight and chosen_CE_weight == 0
            or DPO_weight and (type(reference) is not FrozenOwnTextReferenceV1 or reference.base is not view.actor.base)
            or not DPO_weight and reference is not None):
        raise ValueError('An exact own joint policy and active shared text reference or reference-free CE control are required')
    view.assert_shared()
    if any(p.device != torch.device(device) or p.dtype != torch.float32 or p.grad is not None for p in view.parameters()):
        raise ValueError('The actual complete FP32 joint graph must begin at a clear-gradient boundary')
    if DPO_weight: reference.assert_bound()
    actor = [('actor.' + n, p) for n, p in view.actor.named_selected(('actor',))]
    result = {n: None for n, _ in view.named_selected()}; records = []; amp = amp or nullcontext
    chosen_denominator = sum(row['chosen_response_EOS_targets'] for row in info)
    by_position = {row['original_train_position']: row for row in info}
    fields = ['chosen', 'rejected'] if DPO_weight else ['chosen']
    for batch_positions in batches:
        values = {}; anchors = {}
        for field in fields:
            batch = data.batch(batch_positions, field, device)
            inputs = {k: v for k, v in batch.items() if k != 'labels'}
            with amp():
                output = view.text(**inputs)['logits']
                if DPO_weight:
                    with torch.no_grad(): original = reference(**inputs)
            values[field], counts = row_logp(output, batch['labels'])
            if counts.tolist() != [by_position[p][field + '_response_EOS_targets'] for p in batch_positions]:
                raise ValueError('Every complete original chosen/rejected mask and true EOS must enter the loss')
            if DPO_weight:
                anchors[field], anchor_counts = row_logp(original, batch['labels'])
                if not torch.equal(counts, anchor_counts) or anchors[field].requires_grad:
                    raise ValueError('Reference and policy must score identical complete text with detached reference logits')
        losses = -chosen_CE_weight * values['chosen'] / chosen_denominator
        if DPO_weight:
            margins = (values['chosen'] - values['rejected']) - (anchors['chosen'] - anchors['rejected'])
            pairs = F.softplus(-beta * margins)
            losses = losses + pairs / len(positions)
        if not bool(torch.isfinite(losses).all()): raise ValueError('Every globally normalized human objective must remain finite')
        partial = torch.autograd.grad(losses.sum(), [p for _, p in actor])
        for (name, _), value in zip(actor, partial):
            if not bool(torch.isfinite(value).all()): raise ValueError('Every complete local policy gradient must be finite')
            result[name] = value if result[name] is None else result[name] + value
        for index, position in enumerate(batch_positions):
            records.append(dict(**by_position[position], policy_chosen_logp=float(values['chosen'][index].detach()),
                policy_rejected_logp=float(values['rejected'][index].detach()) if DPO_weight else None,
                reference_chosen_logp=float(anchors['chosen'][index]) if DPO_weight else None,
                reference_rejected_logp=float(anchors['rejected'][index]) if DPO_weight else None,
                pair_DPO=float(pairs[index].detach()) if DPO_weight else 0.,
                complete_global_normalized_contribution=float(losses[index].detach())))
    if DPO_weight: reference.assert_bound()
    if any(p.grad is not None for p in view.parameters()): raise ValueError('A task-gradient primitive cannot leave model accumulation state')
    by_record = {row['original_train_position']: row for row in records}
    local = positions[rank::world]; records = [by_record[p] for p in local]
    chosen_context = sum(data.rows[p]['native']['chosen']['length'] for p in local)
    rejected_context = sum(data.rows[p]['native']['rejected']['length'] for p in local) if DPO_weight else 0
    return result, dict(format=FORMAT, rank=rank, world=world, DPO_weight=DPO_weight, beta=beta, chosen_CE_weight=chosen_CE_weight,
        global_original_source_positions=list(positions), actual_local_original_source_positions=local,
        complete_global_human_source_sha256=value_digest(info), original_global_pair_denominator=len(positions),
        original_global_chosen_response_EOS_denominator=chosen_denominator, actual_local_records=records,
        actual_microbatch_positions=batches, actual_padded_policy_context_slots=padded,
        actual_policy_sequence_forwards=len(local) * len(fields), actual_policy_model_calls=len(batches) * len(fields),
        actual_reference_model_calls=len(batches) * 2 if DPO_weight else 0,
        actual_policy_context_tokens=chosen_context + rejected_context,
        actual_reference_context_tokens=chosen_context + rejected_context if DPO_weight else 0,
        local_global_normalized_objective=math.fsum(r['complete_global_normalized_contribution'] for r in records),
        empty_rank_performs_no_neural_forward=not local,
        all_original_complete_responses_and_EOS_preserved=True, global_gradient_SUM_still_required=world > 1,
        human_ranked_other_model_responses_not_fresh_own_output_judgments=True,
        reference_queried=bool(DPO_weight), reference_is_shared_own_text_only_and_adds_no_parameters=bool(DPO_weight),
        production_training_admitted=False, goal_achieved=False)
