"""Compare independent selected-state and exported natives on actual devices.

Both full-size models must have identical values and independent storage.
Preserve the actual4096text,448photo and64x448/140frame FP32 word checks.
Explicit own-small CPU controls can exercise the flow without GPU admission.
This checks the folded native; hot-actor BF16 generation equivalence and
final model quality are not implied.
"""
import copy

import torch

from .codenet_checkpoint_v1 import capture_rng
from .joint_selected_native_state_v1 import FORMAT as RECONSTRUCTION
from .joint_native_fold_v1 import FORMAT as FOLD
from .own_dense_growth_v1 import OwnDenseAssistant
from .parallel_function_repair_stage_v1 import value_digest
from .whole_human_sft_evidence_v1 import tensor_digest

FORMAT = 'own-independent-joint-selected-native-versus-exported-GPU-cold-v1'


def prove(expected, cold, source_proof, *, checkpoint_sha256, sources, inputs, physical_GPU, allow_mechanics=False):
    if (type(expected) is not OwnDenseAssistant or type(cold) is not OwnDenseAssistant or expected is cold
            or expected.training or cold.training or expected.config != cold.config or expected.settings != cold.settings
            or expected.counts() != cold.counts() or set(inputs) != {'text', 'photo', 'word'}
            or len(checkpoint_sha256) != 64 or any(c not in '0123456789abcdef' for c in checkpoint_sha256)):
        raise ValueError('Two independent compatible actual natives,original cold inputs and exact candidate identity are required')
    device = next(expected.parameters()).device; count = expected.counts()['total_unique']
    full = device.type == 'cuda' and count == 984399690
    if (not full and not (allow_mechanics is True and device.type == 'cpu' and count < 10**7)
            or any(p.device != device or p.dtype != torch.float32 or p.grad is not None for m in [expected, cold] for p in m.parameters())
            or full and (type(physical_GPU) is not int or not 0 <= physical_GPU < 4 or torch.cuda.get_device_name(device) != 'NVIDIA L40')):
        raise ValueError('Full-size actual original L40 natives or an explicitly small CPU control are required')
    reconstruction = source_proof['actual_reconstruction']; folded = source_proof['actual_fold']
    if (reconstruction.get('format') != RECONSTRUCTION or folded.get('format') != FOLD
            or reconstruction.get('sources') != sources or folded.get('sources') != sources
            or reconstruction.get('every_original_metric_and_sparse_Adam_clock_reconstructed') is not True
            or folded.get('original_native_critic_preserved') is not True
            or folded.get('current_joint_visual_word_and_auxiliary_values_preserved') is not True
            or source_proof.get('exported_candidate_weights_not_loaded_for_source_native') is not True
            or full and (reconstruction.get('original_world') != 4 or reconstruction.get('mode') not in ['capacity', 'train'])):
        raise ValueError('The complete independently rebuilt selected-state native and original critic/visual values must be bound')
    digest = tensor_digest(expected.named_parameters()); original_rng = value_digest(capture_rng(device))
    if (digest != folded['folded_parameters_sha256'] or tensor_digest(cold.named_parameters()) != digest
            or {p.data_ptr() for p in expected.parameters()} & {p.data_ptr() for p in cold.parameters()}
            or expected.network.embedding.weight is not expected.network.lm_head.weight
            or cold.network.embedding.weight is not cold.network.lm_head.weight):
        raise ValueError('Every actual native value must be identical with independent storage and one tied vocabulary per native')
    text, photo, word = inputs['text'], inputs['photo'], inputs['word']['word_pixels']
    if (text['input_ids'].device != device or text.get('pixel_values') is not None or photo.get('pixel_values') is None
            or any(v.device != device for batch in inputs.values() for v in batch.values() if isinstance(v, torch.Tensor))
            or full and (text['input_ids'].shape[1] != 4096 or tuple(photo['pixel_values'].shape[-2:]) != (448, 448)
                or tuple(word.shape[-2:]) != (64, 448))):
        raise ValueError('The exact original native text/photo/whole-word geometry and devices must remain')
    signatures = [[(n, id(p), p.data_ptr(), p._version, p.requires_grad) for n, p in m.named_parameters()] for m in [expected, cold]]
    observations = {}
    with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16):
        for name in ['text', 'photo']:
            batch = {k: v for k, v in inputs[name].items() if k in ['input_ids', 'attention_mask', 'pixel_values']}
            a, b = expected(**batch)['logits'], cold(**batch)['logits']
            if not bool(torch.isfinite(a).all()) or not torch.equal(a, b):
                raise ValueError('Actual complete independent native text/photo BF16 outputs must be bit exact')
            observations[name] = dict(actual_output_dtype=str(a.dtype), actual_output_shape=list(a.shape),
                actual_original_input_shape=list(batch['input_ids'].shape),
                actual_original_photo_shape=None if batch.get('pixel_values') is None else list(batch['pixel_values'].shape),
                original_input_tensors_sha256=value_digest(batch), bit_exact=True)
            del a, b
        a, b = expected.word_logits(word), cold.word_logits(word)
        if (a.dtype != torch.float32 or b.dtype != torch.float32 or not bool(torch.isfinite(a).all())
                or not torch.equal(a, b) or full and a.shape[1] != 140):
            raise ValueError('The actual original140frame FP32 word readout must be finite and bit exact')
        observations['word'] = dict(actual_output_dtype=str(a.dtype), actual_output_shape=list(a.shape),
            original_word_pixels_sha256=value_digest(word), bit_exact=True)
    if (signatures != [[(n, id(p), p.data_ptr(), p._version, p.requires_grad) for n, p in m.named_parameters()] for m in [expected, cold]]
            or tensor_digest(expected.named_parameters()) != digest or tensor_digest(cold.named_parameters()) != digest
            or value_digest(capture_rng(device)) != original_rng):
        raise ValueError('Every original native value,graph and private generator must remain unchanged during cold assessment')
    return dict(format=FORMAT, complete=True, actual_device_type=device.type, actual_device_index=device.index,
        physical_GPU=physical_GPU, actual_native_values=count, actual_checkpoint_sha256=checkpoint_sha256,
        actual_source_selected_checkpoint_sha256=reconstruction['actual_checkpoint_sha256'], sources=copy.deepcopy(sources),
        actual_native_parameters_sha256=digest, actual_observations=observations,
        all_original_native_values_and_text_photo_FP32word_cold_outputs_bit_exact=True,
        original_native_critic_and_current_joint_visual_word_values_preserved=True,
        independent_source_and_candidate_storage=True, source_native_derived_without_exported_candidate_weights=True,
        private_RNG_and_both_model_values_and_metadata_unchanged=True,
        full_size_GPU_native_cold_passed=full, hot_actor_BF16_generation_equivalence_or_quality_proved=False,
        optimizer_updates=0, candidate_promoted=False, goal_achieved=False)
