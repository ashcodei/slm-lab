"""Bounded world-one verified-code CE/DPO ablation; not final production.

Both arms retain complete CodeNet positives, verified original Java/Python CE
and identical earlier rehearsal. Only the extra execution preference gradient
differs. Finite publisher tests provide machine labels, not human RLHF.
"""
import math

import torch

from .code_residual_pilot_v1 import PROFILES, learning_rate
from .code_residual_stage_v3 import replay_field
from .codenet_runner_v1 import FixedReductionModel
from .codenet_stage_v1 import Objective, objective_gradients
from .execution_preference_v1 import preference_gradients, loss


class SequenceEncoder:
    truncated = 0

    def __init__(self, sequences, config):
        self.sequences = sequences
        for ids, labels in sequences:
            if (not isinstance(ids, list) or not isinstance(labels, list) or not 2 <= len(ids) <= config.max_text_length
                    or len(labels) != len(ids) or ids[0] != 1 or ids[-1] != 2
                    or any(type(v) is not int or not 0 <= v < config.vocab_size for v in ids)
                    or any(type(v) is not int for v in labels)):
                raise ValueError('Complete original own-BPE input/program/EOS required')
            start = next((i for i, v in enumerate(labels) if v != -100), len(ids))
            if not 1 <= start < len(ids) or labels != [-100] * start + ids[start:]:
                raise ValueError('Complete original contiguous response/EOS mask required')

    def batch(self, rows, field, device):
        if field != 'response' or len(rows) != 1:
            raise ValueError('One complete original program per microbatch required')
        ids, labels = self.sequences[rows[0]['index']]
        return dict(input_ids=torch.tensor([ids], device=device), labels=torch.tensor([labels], device=device), pixel_values=None)


class _ReplayBatch:
    def __init__(self, batch): self.value = batch
    def batch(self, rows, field, device): return self.value


def update(model, optimizer, code, positives, pairs, reference, *, policy_sha256,
        replay_encoder, replay_row, family, step, alpha, device, amp):
    model.assert_partition()
    if (type(step) is not int or not 0 <= step < 128 or not code or not positives
            or any(e['role'] != 'train' for e in code)
            or any(e['split'] != 'train' or e['human_preferences'] is not False
                or e['feedback_source'] != 'machine_execution_of_original_publisher_tests' for e in positives)
            or any(p.grad is not None for p in model.parameters())):
        raise ValueError('Clean bounded pilot with complete original train-only code and execution positives required')
    source = SequenceEncoder([e['positive'] for e in code], model.config)
    verified = SequenceEncoder([e['sequence'] for e in positives], model.config)
    for pair in pairs:
        SequenceEncoder([pair['chosen'], pair['rejected']], model.config)
    # Validate and measure each exposure separately. Epoch-boundary repeated
    # source IDs remain separate terms in the pair mean, never deduplicated.
    measured = [preference_gradients(model.code, [pair], reference, expected_policy_sha256=policy_sha256,
        alpha=0., beta=.1, device=device, amp=amp)['records'][0] for pair in pairs]
    if not measured: raise ValueError('Complete nonempty execution pair exposures required')
    if type(alpha) not in {int, float} or not math.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError('Explicit finite execution preference coefficient required')
    replay_batch = replay_encoder.batch([replay_row], replay_field(family), device)
    if replay_encoder.truncated: raise ValueError('Every original rehearsal input must remain complete')
    model.train(); model.code.gradient_checkpointing = True
    lr = learning_rate(PROFILES[1], step); optimizer.param_groups[0]['lr'] = lr
    optimizer.zero_grad(set_to_none=True)
    try:
        ce = objective_gradients(FixedReductionModel(model.code), [
            Objective('complete_CodeNet', source, [[dict(index=i)] for i in range(len(code))], 'response', 1.),
            Objective('verified_original_program', verified, [[dict(index=i)] for i in range(len(positives))], 'response', 1.),
            Objective(family, _ReplayBatch(replay_batch), [[replay_row]], replay_field(family), .5)], device, amp)
        for pair, record in zip(pairs, measured):
            x = .1 * record['reference_anchored_margin']
            sigmoid = math.exp(-x) / (1 + math.exp(-x)) if x >= 0 else 1 / (1 + math.exp(x))
            coefficient = alpha * .1 * sigmoid / len(pairs)
            record['gradient_coefficient'] = coefficient
            if coefficient:
                (loss(model.code, pair['chosen'], device, amp) * (coefficient * record['chosen_targets'])).backward()
                (loss(model.code, pair['rejected'], device, amp) * (-coefficient * record['rejected_targets'])).backward()
        if (any(p.grad is not None for p in model.base.parameters())
                or any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in model.adapter_parameters())):
            raise ValueError('All and only own code residuals require finite gradients')
        norm = torch.nn.utils.clip_grad_norm_(model.adapter_parameters(), 1., error_if_nonfinite=True)
        optimizer.step(); optimizer.zero_grad(set_to_none=True)
    except BaseException:
        optimizer.zero_grad(set_to_none=True)
        raise
    dpo = sum(r['unscaled_DPO'] for r in measured) / len(measured)
    return dict(step=step + 1, CE=ce, DPO=dict(alpha=alpha, beta=.1, unscaled_mean_DPO=dpo,
        weighted_DPO=alpha * dpo, records=measured), summed_objective=ce['summed_objective'] + alpha * dpo,
        learning_rate=lr, gradient_norm_before_clipping=float(norm), replay_family=family,
        code_source_ids=[e['id'] for e in code], verified_source_ids=[e['id'] for e in positives],
        pair_source_ids=[e['id'] for e in pairs], machine_feedback_not_human_RLHF=True)
