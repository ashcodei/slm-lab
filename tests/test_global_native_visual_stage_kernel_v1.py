"""Small actual own-neural mechanics, never a visual-quality benchmark."""
from contextlib import nullcontext
import copy
from datetime import timedelta
import io
import json
import os
from pathlib import Path

import pytest
import torch
from torch import distributed as dist
from torch.multiprocessing import spawn

from slm_lab.deterministic_positions_v1 import install
from slm_lab.global_native_visual_stage_kernel_v1 import (
    accumulate_global_native_visual, named_native_visual_selected, native_microbatches)
from slm_lab.global_visual_matching_gradients_v1 import digest
from slm_lab.model import ModelConfig
from slm_lab.own_actor_vision_prefix_split_backward_v1 import OwnActorVisionPrefixSplitBackward
from slm_lab.own_dense_actor_v1 import OwnDenseActor
from slm_lab.own_dense_growth_v1 import OwnDenseAssistant, new_backbone


def fixture(lengths=(8, 12, 16, 20, 24, 28, 10, 14, 18)):
    torch.set_num_threads(1)
    config = ModelConfig(vocab_size=64, hidden_size=16, intermediate_size=32, num_layers=1,
        num_heads=2, num_kv_heads=1, max_text_length=4096, vision_hidden_size=16,
        vision_intermediate_size=32, vision_layers=1, vision_heads=2, image_size=224, patch_size=16)
    base = OwnDenseAssistant(new_backbone(config, 998117), reward_hidden=4, word_temporal=4,
        pointing_rank=4, router_hidden=4, detector_hidden=4, word_expansion=5)
    install(base)
    model = OwnDenseActor(base, rank=2, seed=998217)
    view = OwnActorVisionPrefixSplitBackward(model).train()
    view.native.gradient_checkpointing = True
    view.native.vision.train()
    maximum = max(lengths)
    ids = torch.zeros(len(lengths), maximum, dtype=torch.long)
    labels = torch.full_like(ids, -100)
    mask = torch.zeros_like(ids, dtype=torch.bool)
    for index, length in enumerate(lengths):
        sequence = [1] + [4 + (i + index) % 59 for i in range(length - 2)] + [2]
        ids[index, :length] = torch.tensor(sequence)
        start = max(2, length - 4)
        labels[index, start:length] = ids[index, start:length]
        mask[index, :length] = True
    generator = torch.Generator().manual_seed(998317)
    return view, dict(input_ids=ids, labels=labels, attention_mask=mask,
        pixel_values=torch.rand(len(lengths), 3, 448, 448, generator=generator) * 2 - 1)


def parameters(view):
    return [p for _, p in named_native_visual_selected(view)]


def optimizer(view):
    return torch.optim.AdamW(parameters(view), lr=1e-4, eps=1e-5, weight_decay=.01, foreach=False)


def specification(batch, positions):
    return dict(global_positions=positions, global_context_lengths=batch['attention_mask'].sum(1).tolist(),
        global_response_EOS_targets=(batch['labels'][:, 1:] != -100).sum(1).tolist())


def subset(batch, slots):
    return {name: value[slots] for name, value in batch.items()}


def independent_backward(view, batch, chunks, global_targets, amp=None):
    total = 0.
    for slots in chunks:
        width = int(batch['attention_mask'][slots].sum(1).max())
        current = {name: value[slots] if name == 'pixel_values' else value[slots, :width]
            for name, value in batch.items()}
        with (amp or nullcontext)():
            output = view.native(**current)
        count = int((current['labels'][:, 1:] != -100).sum())
        # Independently use the decoder's existing mean CE rather than the
        # kernel's selected-logit sum implementation.
        (output['loss'] * count / global_targets).backward(inputs=parameters(view))
        total += float(output['loss'].detach()) * count
    return total


def test_microbatch_limits_keep_longest_source_and_original_order_without_clipping():
    lengths = [8, 12, 2952, 7, 10, 24]
    chunks = native_microbatches(lengths, prefix_tokens=196, maximum_examples=4, maximum_padded_tokens=4096)
    assert [i for chunk in chunks for i in chunk] == list(range(len(lengths)))
    assert [2] in chunks
    assert all((max(lengths[i] for i in chunk) + 196) * len(chunk) <= 4096 for chunk in chunks)
    assert native_microbatches([], prefix_tokens=196) == []
    with pytest.raises(ValueError, match='clipping or omission is forbidden'):
        native_microbatches([4096], prefix_tokens=196)


def test_serial_microbatches_match_independent_original_decoder_objective():
    view, batch = fixture()
    reference = copy.deepcopy(view)
    positions = list(range(len(batch['input_ids'])))
    spec = specification(batch, positions)
    record = accumulate_global_native_visual(view, batch, local_positions=positions,
        maximum_examples=4, maximum_padded_tokens=450, **spec)
    chunks = [row['local_slots'] for row in record['actual_microbatches']]
    expected = independent_backward(reference, batch, chunks, sum(spec['global_response_EOS_targets']))
    assert abs(record['local_native_CE_sum'] - expected) < 5e-5
    for actual, gold in zip(parameters(view), parameters(reference)):
        torch.testing.assert_close(actual.grad, gold.grad, rtol=2e-5, atol=2e-6)
    selected = {id(p) for p in parameters(view)}
    assert all(p.grad is None for p in view.parameters() if id(p) not in selected)
    assert [p for chunk in record['actual_microbatches'] for p in chunk['original_positions']] == positions


@pytest.mark.parametrize('bf16', [False, True])
def test_next_Adam_update_recovers_exactly_and_text_other_heads_stay_fixed(bf16):
    view, batch = fixture((8, 12, 16))
    adam = optimizer(view)
    spec = specification(batch, [17, 23, 29])
    amp = (lambda: torch.autocast('cpu', dtype=torch.bfloat16)) if bf16 else None

    def update(current, opt, source):
        record = accumulate_global_native_visual(current, source, local_positions=[17, 23, 29],
            maximum_padded_tokens=450, native_logits_dtype=torch.bfloat16 if bf16 else torch.float32, amp=amp, **spec)
        torch.nn.utils.clip_grad_norm_(parameters(current), 1., error_if_nonfinite=True)
        opt.step()
        opt.zero_grad(set_to_none=True)
        return record

    chosen = {id(p) for p in parameters(view)}
    fixed = {name: p.detach().clone() for name, p in view.actor.named_parameters() if id(p) not in chosen}
    with torch.no_grad():
        text = view.actor(batch['input_ids'], attention_mask=batch['attention_mask'])['logits'].clone()
    update(view, adam, batch)
    stream = io.BytesIO()
    torch.save(dict(model=view.actor.state_dict(), optimizer=adam.state_dict()), stream)
    stream.seek(0)
    saved = torch.load(stream, map_location='cpu', weights_only=True)
    resumed, _ = fixture((8, 12, 16))
    resumed.actor.load_state_dict(saved['model'], strict=True)
    resumed_adam = optimizer(resumed)
    resumed_adam.load_state_dict(saved['optimizer'])
    changed = {**batch, 'pixel_values': batch['pixel_values'] * .8}
    assert update(view, adam, changed) == update(resumed, resumed_adam, changed)
    for name, value in view.actor.state_dict().items():
        assert torch.equal(value, resumed.actor.state_dict()[name])
        if name in fixed:
            assert torch.equal(value, fixed[name])
    for left, right in zip(adam.state.values(), resumed_adam.state.values()):
        for name, value in left.items():
            assert torch.equal(value, right[name])
    with torch.no_grad():
        assert torch.equal(text, view.actor(batch['input_ids'], attention_mask=batch['attention_mask'])['logits'])


def test_actual_small_model_backward_keeps_full2952_token448px_source():
    view, batch = fixture((2952,))
    spec = specification(batch, [21055])
    record = accumulate_global_native_visual(view, batch, local_positions=[21055], **spec)
    assert record['local_original_context_tokens'] == 2952
    assert record['local_response_EOS_targets'] == 4
    assert record['actual_microbatches'][0]['padded_decoder_tokens'] == 3148
    assert all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in parameters(view))
    assert not torch.cuda.is_initialized()


def worker(rank, folder):
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '' and os.nice(0) >= 19
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    dist.init_process_group('gloo', init_method='file://' + str(Path(folder)/'rendezvous'),
        rank=rank, world_size=4, timeout=timedelta(minutes=4))
    try:
        view, batch = fixture()
        reference = copy.deepcopy(view)
        adam, refadam = optimizer(view), optimizer(reference)
        selected = parameters(view)
        chosen = {id(p) for p in selected}
        original = {name: p.detach().clone() for name, p in view.actor.named_parameters() if id(p) not in chosen}
        proofs = []
        for slots, positions in [(list(range(9)), [17 + 3*i for i in range(9)]), ([8], [41])]:
            whole = subset(batch, slots)
            local = list(range(len(slots)))[rank::4]
            payload = subset(whole, local) if local else None
            spec = specification(whole, positions)
            calls = []
            hook = view.native.vision.register_forward_hook(lambda *args: calls.append(1))
            record = accumulate_global_native_visual(view, payload, local_positions=positions[rank::4],
                maximum_padded_tokens=450, **spec)
            hook.remove()
            assert len(calls) == len(record['actual_microbatches'])
            if not local:
                assert not calls and all(p.grad is None for p in selected)
            for p in selected:
                if p.grad is None:
                    p.grad = torch.zeros_like(p)
                dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
            # Independent serial shard-shaped oracle preserves the actual
            # batch geometry; no full-batch bitidentity claim is made.
            gold_sum = 0.
            for owner in range(4):
                owned = list(range(len(slots)))[owner::4]
                if not owned:
                    continue
                lengths = [spec['global_context_lengths'][i] for i in owned]
                # Independent exhaustive greedy construction for this small
                # finite fixture, separate from native_microbatches.
                chunks = []
                remaining = list(range(len(owned)))
                while remaining:
                    valid = [n for n in range(1, min(4, len(remaining)) + 1)
                        if n * (196 + max(lengths[i] for i in remaining[:n])) <= 450]
                    take = max(valid)
                    chunks.append(remaining[:take])
                    remaining = remaining[take:]
                gold_sum += independent_backward(reference, subset(whole, owned), chunks,
                    sum(spec['global_response_EOS_targets']))
            gradient_error = max(float((a.grad - b.grad).abs().max()) for a,b in zip(selected, parameters(reference)))
            assert gradient_error < 5e-6
            sums = [None] * 4
            dist.all_gather_object(sums, record['local_native_CE_sum'])
            assert abs(sum(sums) - gold_sum) < 5e-5
            torch.nn.utils.clip_grad_norm_(selected, 1., error_if_nonfinite=True)
            torch.nn.utils.clip_grad_norm_(parameters(reference), 1., error_if_nonfinite=True)
            adam.step()
            refadam.step()
            adam.zero_grad(set_to_none=True)
            refadam.zero_grad(set_to_none=True)
            weight_error = max(float((a-b).detach().abs().max()) for a,b in zip(selected, parameters(reference)))
            assert weight_error < 5e-6
            for name,p in view.actor.named_parameters():
                if name in original:
                    assert torch.equal(original[name], p) and p.grad is None
            hashes = [None] * 4
            dist.all_gather_object(hashes, digest(torch.cat([p.detach().reshape(-1) for p in selected])))
            assert len(set(hashes)) == 1
            proofs.append(dict(rank=rank, global_sources=len(slots), local_sources=len(local),
                actual_own_unique_parameters=view.actor.counts()['actual_unique'], actual_selected_values=sum(p.numel() for p in selected),
                serial_shard_max_gradient_absolute_difference=gradient_error, serial_shard_max_Adam_weight_absolute_difference=weight_error,
                serial_original_CE_sum=gold_sum, actual_native_encoder_calls=len(calls),
                all_replicas_bitexact=True, all_nonvision_weights_fixed=True, record=record))
        rejected = []
        for fault in ['missing_EOS', 'wrong_global_target', 'nan_pixels', 'different_valid_limits']:
            whole = subset(batch, [0, 1, 2])
            current = subset(whole, [rank]) if rank < 3 else None
            spec = specification(whole, [17, 20, 23])
            limit = 450
            if rank == 1:
                if fault == 'missing_EOS':
                    current['labels'][current['labels'] == 2] = -100
                elif fault == 'wrong_global_target':
                    spec['global_response_EOS_targets'][1] += 1
                elif fault == 'nan_pixels':
                    current['pixel_values'][0, 0, 0, 0] = float('nan')
                else:
                    limit = 500
            before = [p.detach().clone() for p in selected]
            calls = []
            hook = view.native.vision.register_forward_hook(lambda *args: calls.append(1))
            with pytest.raises(ValueError, match='Coordinated native visual'):
                accumulate_global_native_visual(view, current, local_positions=[17,20,23][rank::4],
                    maximum_padded_tokens=limit, **spec)
            hook.remove()
            assert not calls and all(p.grad is None for p in selected)
            assert all(torch.equal(a,b) for a,b in zip(before, selected))
            rejected.append(fault)
        assert not torch.cuda.is_initialized()
        (Path(folder)/(str(rank)+'.json')).write_text(json.dumps(dict(complete=True, proofs=proofs,
            coordinated_rejections_before_forward_and_optimizer=rejected, CUDA_hidden=True,
            production_optimizer_updates=0, full_size_GPU_and_quality_pending=True), indent=2)+'\n')
    finally:
        dist.destroy_process_group()


def test_four_own_replicas_complete_microbatches_empty_tail_and_coordinated_rejections(tmp_path):
    spawn(worker, args=(str(tmp_path),), nprocs=4, join=True)
    rows = [json.loads((tmp_path/(str(rank)+'.json')).read_text()) for rank in range(4)]
    assert all(row['complete'] and len(row['proofs']) == 2 for row in rows)
    assert all(row['proofs'][1]['actual_native_encoder_calls'] == 0 for row in rows[1:])
