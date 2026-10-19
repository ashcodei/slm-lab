"""Disposable visual capacity probes; no candidate or checkpoint is exported."""
import torch

from .code_residual_collective_checkpoint_v2 import _same, _validated, fingerprint
from .code_residual_stage_v3 import _EncodedBatches
from .codenet_runner_v1 import average_gradients
from .execution_training_v5 import global_CE_gradients
from .paired_visual_contrast_v1 import eligible, gradients
from .recurrent_ablation import state_digest


def select_extremes(pairs, replay, bundle):
    """Select complete original sequences by CPU encoding; never invent inputs."""
    lengths = {qid: len(pairs.encoder.sequence(row, 'response')[0]) for qid, row in pairs.rows.items()}
    targets = {qid: sum(v != -100 for v in pairs.encoder.sequence(row, 'response')[1][1:]) for qid, row in pairs.rows.items()}
    selected = {max(range(len(pairs)), key=lambda i: max(lengths[q] for q in pairs.pairs[i]['question_ids'])),
        max(range(len(pairs)), key=lambda i: sum(targets[q] for q in pairs.pairs[i]['question_ids']))}
    binary = []
    for i, pair in enumerate(pairs.pairs):
        sides = [pairs.rows[q] for q in pair['question_ids']]
        if (set(r['response'] for r in sides) == {'yes', 'no'}
                and all(r['original_human_annotation']['answer_type'] == 'yes/no' and r['exact_human_agreement'] >= 9 for r in sides)
                and len({r['group_id'] for r in sides}) == 2
                and len({r['model_pixel_sha256_448'] for r in sides}) == 2):
            binary.append(i)
    if binary:
        selected.add(max(binary, key=lambda i: max(lengths[q] for q in pairs.pairs[i]['question_ids'])))
    maxima = {}
    positions = {}
    for name in replay.families:
        family = next(f for f in bundle['selected_actual_full_encoder_checks'] if f['name'] == name)
        check = max(family['checks'], key=lambda c: c['context_tokens'])
        if not check['actual448px_golden_pixels_checked']:
            raise ValueError('Capacity requires actual original golden448px visual rows')
        positions[name] = check['position']; maxima[name] = check['context_tokens']
    return sorted(selected), positions, dict(max_paired_context=max(lengths.values()),
        max_paired_total_targets=max(sum(targets[q] for q in p['question_ids']) for p in pairs.pairs),
        high_consensus_binary_exists=bool(binary), replay_maximum_contexts=maxima)


def exercise_capacity(kernel, pair_positions, replay_positions):
    """Allocate real Adam in one update, then exercise every original extreme.

Each probe repeats an admitted original example to fill the configured local
layout. Probes backward and clear gradients, without further optimizer steps.
Their disposable weights are never a resumable training candidate.
"""
    def preflight():
        kernel._boundary(full=True)
        if (kernel.completed or not pair_positions or len(set(pair_positions)) != len(pair_positions)
                or any(type(i) is not int or not 0 <= i < len(kernel.pairs) for i in pair_positions)
                or set(replay_positions) != set(kernel.replay.families)
                or any(type(i) is not int or not 0 <= i < kernel.replay.size(n) for n, i in replay_positions.items())):
            raise ValueError('Fresh disposable policy and all actual original extreme positions required')
        return fingerprint(dict(pairs=pair_positions, replay=replay_positions, identity=kernel.identity))
    _same(_validated(preflight))
    before = state_digest(kernel.model.vision.state_dict())
    first = kernel.update()
    probes = []
    opts = kernel.options
    for name in kernel.replay.families:
        for position in pair_positions:
            def prepare():
                example = kernel.pairs.example(position)
                row = kernel.replay.row(name, replay_positions[name])
                groups = [[row for _ in range(opts['replay_batch_size'])] for _ in range(opts['replay_accumulation'])]
                encoded = _EncodedBatches(kernel.encoder, groups, 'response', kernel.device)
                if (kernel.encoder.truncated or any(tuple(image.shape[-2:]) != (opts['image_size'], opts['image_size']) for image in example['images'])
                        or any(tuple(batch['pixel_values'].shape[-2:]) != (opts['image_size'], opts['image_size']) for batch in encoded.batches)):
                    raise ValueError('Original full-resolution complete visual inputs required')
                return example, [example for _ in range(opts['pair_batch_size'] * opts['pair_accumulation'])], encoded
            example, examples, encoded = _validated(prepare)
            kernel.optimizer.zero_grad(set_to_none=True)
            try:
                paired = gradients(kernel.model, examples, contract_sha256=kernel.pairs.contract_sha256,
                    alpha=opts['alpha'], beta=opts['beta'], margin=opts['margin'], world=4, device=kernel.device, amp=kernel.amp)
                replay = global_CE_gradients(kernel.model, [(name, encoded, opts['replay_coefficient'])], kernel.model.config, kernel.device, kernel.amp)
                def finite():
                    if (any(p.grad is not None for n, p in kernel.model.named_parameters() if not n.startswith('vision.'))
                            or any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in kernel.model.vision.parameters())):
                        raise ValueError('Every capacity probe requires all-and-only finite vision gradients')
                _validated(finite)
                average_gradients(kernel.model, kernel.device, 4)
                norm = _validated(lambda: float(torch.nn.utils.clip_grad_norm_(kernel.model.vision.parameters(),
                    opts['gradient_clip'], error_if_nonfinite=True)))
                probes.append(dict(family=name, replay_position=replay_positions[name], pair_position=position,
                    pair_id=example['id'], paired_context_tokens=max(len(ids) for ids, _ in example['sequences']),
                    paired_total_targets=sum(sum(v != -100 for v in labels[1:]) for _, labels in example['sequences']),
                    contrast_eligible=eligible(example), image_size=opts['image_size'],
                    replay_context_tokens=max(batch['input_ids'].shape[1] for batch in encoded.batches),
                    gradient_norm=norm, paired=paired, replay=replay,
                    finite_all_and_only_vision_gradients=True))
            finally:
                kernel.optimizer.zero_grad(set_to_none=True)
    _validated(lambda: kernel._boundary(full=True))
    if state_digest(kernel.model.vision.state_dict()) == before:
        raise ValueError('Disposable real kernel update must change vision weights')
    return dict(actual_full_kernel_optimizer_updates=1, first_kernel_metric=first,
        probes=probes, all_eight_visual_families_exercised=sorted({p['family'] for p in probes}),
        actual_max_paired_context=max(p['paired_context_tokens'] for p in probes),
        actual_max_paired_total_targets=max(p['paired_total_targets'] for p in probes),
        actual_replay_maximum_contexts={n: max(p['replay_context_tokens'] for p in probes if p['family'] == n) for n in kernel.replay.families},
        high_consensus_binary_contrast_exercised=any(p['contrast_eligible'] for p in probes),
        original_language_critic_unchanged=True, vision_parameters_changed=True,
        candidate_weights_saved=False, capability_result=False)
