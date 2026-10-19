"""Independent native cold capacity using the original base plus selected values.

Capacity recovery need not duplicate a4GBfull export on disk. The separately
loaded complete original student receives exactly the current selected vision,
pointing and word values. Final inference still requires its own complete
standalone export and cold-load proof; an overlay is not a final deliverable.
"""
import torch

from .joint_visual_native_export_v1 import assert_identity
from .own_dense_growth_v1 import OwnDenseAssistant
from .whole_human_sft_evidence_v1 import tensor_digest


def selected(model):
    if type(model) is not OwnDenseAssistant:
        raise ValueError('One exact own complete native student is required')
    return {**{'vision.' + n: p for n, p in model.network.vision.named_parameters()},
        **{'pointing.' + n: p for n, p in model.heads['pointing'].named_parameters()},
        **{'word.' + n: p for n, p in model.heads['word'].named_parameters()}}


def fixed_digest(model):
    ids = {id(p) for p in selected(model).values()}
    return tensor_digest((n, p) for n, p in model.named_parameters() if id(p) not in ids)


def apply_selected(model, values, *, expected_fixed_sha256):
    parameters = selected(model)
    if (not isinstance(values, dict) or list(values) != list(parameters)
            or fixed_digest(model) != expected_fixed_sha256
            or any(p.grad is not None for p in model.parameters())):
        raise ValueError('Every exact selected name and unchanged independent native fixed value is required')
    for name, p in parameters.items():
        value = values[name]
        if (not isinstance(value, torch.Tensor) or value.shape != p.shape or value.dtype != p.dtype
                or value.dtype != torch.float32 or not bool(torch.isfinite(value).all())):
            raise ValueError('Every finite exact-shape FP32 selected value must validate before any native copy')
    with torch.no_grad():
        for name, p in parameters.items():
            p.copy_(values[name])
    if fixed_digest(model) != expected_fixed_sha256:
        raise ValueError('Selected visual copies must preserve every fixed native value')


def prove(view, independent, inputs, *, mode, precision, expected_fixed_sha256):
    identity = assert_identity(view)
    original = view.actor.base
    if (type(independent) is not OwnDenseAssistant or mode not in ['mechanics', 'capacity']
            or precision not in ['fp32', 'bf16'] or independent.counts() != original.counts()
            or independent.config != original.config or independent.settings != original.settings
            or set(inputs) != {'text', 'photo', 'word'}
            or set(inputs['text']) != {'input_ids', 'attention_mask'}
            or set(inputs['photo']) != {'input_ids', 'attention_mask', 'pixel_values'}
            or set(inputs['word']) != {'word_pixels'}
            or {p.data_ptr() for p in independent.parameters()} & {p.data_ptr() for p in original.parameters()}):
        raise ValueError('A separate complete own student and every whole text/photo/word cold input are required')
    device = next(original.parameters()).device
    text = inputs['text']['input_ids']
    photo = inputs['photo']['pixel_values']
    word = inputs['word']['word_pixels']
    if (text.ndim != 2 or text.shape[0] != 1 or inputs['text']['attention_mask'].shape != text.shape
            or inputs['text']['attention_mask'].dtype != torch.bool or not bool(inputs['text']['attention_mask'].all())
            or photo.shape != (1, 3, 448, 448) or word.shape != (1, 3, 64, 448)
            or any(p.device != device or p.dtype != torch.float32 or p.grad is not None for p in independent.parameters())
            or fixed_digest(original) != expected_fixed_sha256
            or mode == 'mechanics' and (device.type != 'cpu' or torch.cuda.is_initialized())
            or mode == 'capacity' and (device.type != 'cuda' or text.shape[1] != 4096
                or original.counts()['total_unique'] != 984399690 or identity['actual_research_actor_parameters'] != 999706442)):
        raise ValueError('Whole actual4096text448photo64x448word inputs and exact full-size runtime are required for production capacity')
    current = {n: p.detach() for n, p in selected(original).items()}
    apply_selected(independent, current, expected_fixed_sha256=expected_fixed_sha256)
    if any(not torch.equal(p, dict(independent.named_parameters())[n]) for n, p in original.named_parameters()):
        raise ValueError('Every cold native value, including all untouched heads and language, must equal the warm native student')
    view.eval()
    independent.eval()
    batches = {n: {k: v.to(device) for k, v in batch.items()} for n, batch in inputs.items()}
    torch.clear_autocast_cache()
    with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=precision == 'bf16'):
        for name in ['text', 'photo']:
            wanted = view.native(**batches[name])['logits']
            actual = independent(**batches[name])['logits']
            if not torch.equal(wanted, actual):
                raise ValueError('Actual complete native cold text/photo logits must be bit exact')
        wanted = original.word_logits(batches['word']['word_pixels'])
        actual = independent.word_logits(batches['word']['word_pixels'])
        if not torch.equal(wanted, actual) or actual.dtype != torch.float32:
            raise ValueError('The complete original FP32 word readout must survive independent cold capacity')
    torch.clear_autocast_cache()
    return dict(complete=True, mode=mode, actual_text_context_tokens=text.shape[1], actual_photo_side=448,
        actual_word_shape=[64, 448], actual_word_readout_dtype='torch.float32',
        every_original_fixed_and_selected_native_value_equal=True,
        actual_text_photo_logits_and_word_readout_bit_exact=True,
        independent_native_storage_without_aliasing=True, capacity_uses_original_base_plus_verified_selected_values=True,
        final_complete_standalone_export_and_cold_load_still_required=True,
        actual_native_parameters=independent.counts()['total_unique'],
        mechanics_only=mode == 'mechanics', new_training_updates=0, goal_achieved=False)
