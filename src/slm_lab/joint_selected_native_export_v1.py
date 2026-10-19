"""Export one complete own native from verified joint selected weights.

Reconstruct every original checkpoint observation and sparse Adam clock,
fold only the learned language adapters, retain the native critic/current
vision and word values, then independently cold-load every exported value.
GPU cold parity, completed-source ownership and all quality remain separate.
"""
import math
from pathlib import Path
import shutil

import torch

from .codenet_checkpoint_v1 import atomic_checkpoint, capture_rng, FileBindings
from .corpus import file_hash
from .joint_native_fold_v1 import fold
from .joint_selected_native_state_v1 import install
from .own_dense_standalone_v1 import export as native_payload, load
from .parallel_function_repair_stage_v1 import value_digest
from .whole_human_sft_evidence_v1 import tensor_digest

FORMAT = 'own-complete-joint-selected-state-native-export-and-CPU-cold-v1'


def export(view, optimizer, directory, folder, tokenizer_path, *, sources, entries, source_files,
           expected_world, expected_completed, expected_pointer_sha256, mode, probes, word_pixels,
           working_budget_bytes, maximum_checkpoint_bytes, minimum_free_bytes,
           maximum_logit_error=.002, allow_mechanics=False):
    folder, tokenizer_path = Path(folder), Path(tokenizer_path)
    if (folder.exists() or not probes or minimum_free_bytes < 40 * 1024**3
            or type(maximum_logit_error) not in [int, float] or not math.isfinite(maximum_logit_error)
            or not 0 < maximum_logit_error <= .002 or file_hash(tokenizer_path) != sources['tokenizer_sha256']
            or shutil.disk_usage(folder.parent).free < minimum_free_bytes + maximum_checkpoint_bytes):
        raise ValueError('Fresh fully funded complete native export,original tokenizer and explicit numerical bound are required')
    full = view.actor.base.counts()['total_unique'] == 984399690
    if full and (not any(b['input_ids'].shape[1] == 4096 and b.get('pixel_values') is None for _, b in probes)
            or not any(b.get('pixel_values') is not None and tuple(b['pixel_values'].shape[-2:]) == (448, 448) for _, b in probes)
            or tuple(word_pixels.shape[-2:]) != (64, 448)):
        raise ValueError('The full4096text,448pxphoto and64x448whole-word native cold inputs must remain')
    guard = FileBindings(source_files); before_rng = value_digest(capture_rng('cpu'))
    reconstruction = install(view, optimizer, directory, sources=sources, entries=entries, source_files=source_files,
        expected_world=expected_world, expected_completed=expected_completed, expected_pointer_sha256=expected_pointer_sha256,
        mode=mode, allow_mechanics=allow_mechanics)
    source_parameters = tensor_digest(view.named_parameters())
    native, fold_proof = fold(view.actor, sources=sources, working_budget_bytes=working_budget_bytes)
    observations = []
    with torch.no_grad(), torch.autocast('cpu', enabled=False):
        for name, batch in probes:
            inputs = {k: batch[k] for k in ['input_ids', 'attention_mask', 'pixel_values'] if k in batch}
            hot = view.actor(**inputs)['logits']; merged = native(**inputs)['logits']
            maximum = float((hot - merged).abs().max())
            if not math.isfinite(maximum) or maximum > maximum_logit_error:
                raise ValueError('Actual complete native actor-to-fold FP32 error exceeds the original declared bound')
            observations.append(dict(name=name, actual_input_tensors_sha256=value_digest(inputs),
                complete_text_context_tokens=batch['input_ids'].shape[1],
                actual_photo_side=None if inputs.get('pixel_values') is None else inputs['pixel_values'].shape[-1],
                actual_hot_to_fold_FP32_maximum_logit_error=maximum,
                actual_hot_to_fold_greedy_argmax_differences=int((hot.argmax(-1) != merged.argmax(-1)).sum())))
            del hot, merged
        expected_word = native.word_logits(word_pixels)
        if (not torch.equal(expected_word, view.actor.base.word_logits(word_pixels))
                or expected_word.dtype != torch.float32 or full and expected_word.shape[1] != 140):
            raise ValueError('The complete actual joint FP32 word reader must survive the language fold exactly')
    folder.mkdir(); shutil.copyfile(tokenizer_path, folder / 'tokenizer.json'); checkpoint = folder / 'model.pt'
    provenance = dict(kind=FORMAT, mode=mode, actual_joint_reconstruction=reconstruction, actual_joint_fold=fold_proof,
        completed_original_joint_source_epoch=mode == 'train' and expected_completed == 5362,
        hot_policy_BF16_generation_equivalence_and_quality_not_assumed=True)
    atomic_checkpoint(checkpoint, native_payload(native, sources=sources, provenance=provenance,
        completed_updates=expected_completed, tokenizer_sha256=sources['tokenizer_sha256']), maximum_checkpoint_bytes, minimum_free_bytes)
    cold, metadata = load(checkpoint, expected_checkpoint_sha256=file_hash(checkpoint), expected_sources=sources)
    if tensor_digest(cold.named_parameters()) != tensor_digest(native.named_parameters()):
        raise ValueError('Every actually selected/folded native value must independently cold-load exactly')
    with torch.no_grad(), torch.autocast('cpu', enabled=False):
        for _, batch in probes:
            inputs = {k: batch[k] for k in ['input_ids', 'attention_mask', 'pixel_values'] if k in batch}
            if not torch.equal(native(**inputs)['logits'], cold(**inputs)['logits']):
                raise ValueError('Every complete actual folded text/photo native output must independently cold-load bit exactly')
        if not torch.equal(expected_word, cold.word_logits(word_pixels)):
            raise ValueError('The whole original joint word readout must independently cold-load bit exactly')
    guard.check(full=True)
    if (tensor_digest(view.named_parameters()) != source_parameters or optimizer.state
            or value_digest(capture_rng('cpu')) != before_rng or torch.cuda.is_initialized()):
        raise ValueError('Export cannot change reconstructed source values,private generators,Adam or acquire CUDA')
    return dict(format=FORMAT, complete=True, mode=mode, actual_reconstruction=reconstruction, actual_fold=fold_proof,
        actual_native_probes=observations, actual_word_pixels_sha256=value_digest(word_pixels),
        actual_word_logits_shape=list(expected_word.shape), actual_word_logits_dtype=str(expected_word.dtype),
        checkpoint=str(checkpoint), checkpoint_sha256=file_hash(checkpoint), tokenizer_sha256=file_hash(folder / 'tokenizer.json'),
        sources=dict(sources), actual_counts=metadata['actual_counts'], completed_updates=expected_completed,
        every_complete_native_parameter_and_text_photo_word_output_cold_load_bit_exact=True,
        original_nonzero_native_critic_and_current_joint_visual_word_values_preserved=True,
        original_selected_checkpoint_Adam_private_RNG_and_source_files_unchanged=True,
        standalone_inference_requires_only_native_and_tokenizer=True, full_size_GPU_cold_or_quality_proved=False,
        optimizer_updates=0, candidate_promoted=False, goal_achieved=False)
